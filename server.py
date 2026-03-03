import os
import json
import asyncio
from datetime import datetime
from uuid import UUID

import asyncpg
import httpx
from fastmcp import FastMCP, Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import ToolError

# ─── Configuration ───────────────────────────────────────────────────────

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MCP_API_KEY = os.environ["MCP_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"

# ─── Authentication Middleware ───────────────────────────────────────────


class BearerAuthMiddleware(Middleware):
    """Validates Bearer token on every tool call."""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        headers = get_http_headers()
        auth_header = headers.get("authorization", "")

        if not auth_header.startswith("Bearer "):
            raise ToolError("Unauthorized: missing or malformed Bearer token")

        token = auth_header.removeprefix("Bearer ").strip()
        if token != MCP_API_KEY:
            raise ToolError("Unauthorized: invalid API key")

        return await call_next(context)


# ─── MCP Server ──────────────────────────────────────────────────────────

mcp = FastMCP(
    "Memory Server",
    description="Personal semantic memory layer. Store and search memories across all your AI tools.",
    middleware=[BearerAuthMiddleware()],
)

# ─── Database Pool ───────────────────────────────────────────────────────

db_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return db_pool


# ─── Embedding Helper ───────────────────────────────────────────────────


async def get_embedding(text: str) -> list[float]:
    """Call OpenAI API to generate embedding for the given text."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


# ─── JSON Encoder ────────────────────────────────────────────────────────


class MemoryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# ─── Tools ───────────────────────────────────────────────────────────────


@mcp.tool()
async def store_memory(
    content: str,
    source: str = "manual",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """
    Store a new memory with automatic semantic embedding.

    Args:
        content: The text content to remember. Be descriptive — this is what gets searched.
        source: Where this memory came from (e.g. 'claude-code', 'user', 'meeting').
        tags: Optional list of tags for categorical filtering (e.g. ['project-x', 'decision']).
        metadata: Optional JSON metadata (e.g. {'project': 'website-redesign', 'priority': 'high'}).

    Returns:
        Confirmation with the memory ID.
    """
    db = await get_pool()
    embedding = await get_embedding(content)
    tags = tags or []
    metadata = metadata or {}

    row = await db.fetchrow(
        """
        INSERT INTO memories (content, embedding, source, tags, metadata)
        VALUES ($1, $2::vector, $3, $4, $5::jsonb)
        RETURNING id, created_at
        """,
        content,
        str(embedding),
        source,
        tags,
        json.dumps(metadata),
    )

    return json.dumps(
        {
            "status": "stored",
            "id": str(row["id"]),
            "created_at": row["created_at"].isoformat(),
            "tags": tags,
            "content_preview": content[:100],
        }
    )


@mcp.tool()
async def search_memory(
    query: str,
    limit: int = 10,
    tags: list[str] | None = None,
    source: str | None = None,
) -> str:
    """
    Search memories by semantic similarity. Finds memories by meaning, not just keywords.

    Args:
        query: Natural language search query. Describe what you're looking for.
        limit: Maximum number of results to return (default 10, max 50).
        tags: Optional tag filter — only return memories with ALL of these tags.
        source: Optional source filter (e.g. 'claude-code').

    Returns:
        List of matching memories ranked by relevance, with similarity scores.
    """
    db = await get_pool()
    limit = min(limit, 50)
    embedding = await get_embedding(query)

    conditions = []
    params = [str(embedding), limit]
    param_idx = 3

    if tags:
        conditions.append(f"tags @> ${param_idx}::text[]")
        params.append(tags)
        param_idx += 1

    if source:
        conditions.append(f"source = ${param_idx}")
        params.append(source)
        param_idx += 1

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    rows = await db.fetch(
        f"""
        SELECT id, content, source, tags, metadata, created_at,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memories
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        *params,
    )

    results = [
        {
            "id": str(row["id"]),
            "content": row["content"],
            "source": row["source"],
            "tags": row["tags"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "similarity": round(float(row["similarity"]), 4),
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]

    return json.dumps(
        {"query": query, "count": len(results), "results": results},
        cls=MemoryEncoder,
    )


@mcp.tool()
async def list_recent(limit: int = 20, source: str | None = None) -> str:
    """
    List the most recent memories.

    Args:
        limit: Number of memories to return (default 20, max 100).
        source: Optional filter by source.

    Returns:
        List of recent memories in reverse chronological order.
    """
    db = await get_pool()
    limit = min(limit, 100)

    if source:
        rows = await db.fetch(
            "SELECT id, content, source, tags, metadata, created_at FROM memories WHERE source = $1 ORDER BY created_at DESC LIMIT $2",
            source,
            limit,
        )
    else:
        rows = await db.fetch(
            "SELECT id, content, source, tags, metadata, created_at FROM memories ORDER BY created_at DESC LIMIT $1",
            limit,
        )

    results = [
        {
            "id": str(row["id"]),
            "content": row["content"][:200],
            "source": row["source"],
            "tags": row["tags"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]

    return json.dumps(
        {"count": len(results), "results": results}, cls=MemoryEncoder
    )


@mcp.tool()
async def delete_memory(memory_id: str) -> str:
    """
    Delete a specific memory by ID.

    Args:
        memory_id: The UUID of the memory to delete.

    Returns:
        Confirmation of deletion.
    """
    db = await get_pool()
    result = await db.execute("DELETE FROM memories WHERE id = $1", memory_id)

    if result == "DELETE 1":
        return json.dumps({"status": "deleted", "id": memory_id})
    else:
        return json.dumps({"status": "not_found", "id": memory_id})


# ─── Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
