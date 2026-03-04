import os
import json
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

import asyncpg
import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware import Middleware as StarletteMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# ─── Configuration ───────────────────────────────────────────────────────

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MCP_API_KEY = os.environ["MCP_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"

logger = logging.getLogger("memory-server")

# ─── Authentication Middleware (Starlette layer) ────────────────────────


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validates Bearer token at the HTTP layer, before FastMCP processes the request.

    Works around a FastMCP bug where get_http_headers() returns stale/missing
    headers during tool execution with the Streamable HTTP transport.
    See: https://github.com/jlowin/fastmcp/issues/1233
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks / OPTIONS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Unauthorized: missing or malformed Bearer token"},
                status_code=401,
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token != MCP_API_KEY:
            return JSONResponse(
                {"error": "Unauthorized: invalid API key"}, status_code=401
            )

        return await call_next(request)


# ─── MCP Server ──────────────────────────────────────────────────────────

mcp = FastMCP(
    "Memory Server",
    instructions="Personal semantic memory layer. Store and search memories across all your AI tools.",
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


# ─── AI Metadata Extraction ─────────────────────────────────────────────


async def extract_metadata(content: str) -> dict:
    """Call GPT-4o-mini to extract structured metadata from content.

    Best-effort: returns {} on any failure. Never blocks storage.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You extract structured metadata from text. "
                                "Respond with JSON containing:\n"
                                '- "type": one of "observation", "task", "idea", "reference", "person_note"\n'
                                '- "topic_tags": 1-3 short lowercase tags (e.g. ["meeting", "acme-corp"])\n'
                                '- "entities": {"people": [], "places": [], "organizations": []}\n'
                                '- "action_items": list of actionable items (empty list if none)\n'
                                "Be concise. Tags should be kebab-case."
                            ),
                        },
                        {"role": "user", "content": content},
                    ],
                },
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            return json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:
        logger.warning("Metadata extraction failed: %s", e)
        return {}


# ─── JSON Encoder ────────────────────────────────────────────────────────


class MemoryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# ─── Shared Storage Implementation ──────────────────────────────────────


async def _store_memory_impl(
    content: str,
    source: str = "manual",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Core storage logic shared by the MCP tool and webhook endpoint."""
    db = await get_pool()
    tags = tags or []
    metadata = metadata or {}

    # Run embedding and AI metadata extraction in parallel
    embedding, ai_metadata = await asyncio.gather(
        get_embedding(content),
        extract_metadata(content),
    )

    # Merge AI-generated tags with user-supplied tags (deduplicated)
    ai_tags = ai_metadata.pop("topic_tags", [])
    merged_tags = list(dict.fromkeys(tags + ai_tags))

    # Store AI metadata under 'ai' key (never conflicts with user keys)
    if ai_metadata:
        metadata["ai"] = ai_metadata

    row = await db.fetchrow(
        """
        INSERT INTO memories (content, embedding, source, tags, metadata)
        VALUES ($1, $2::vector, $3, $4, $5::jsonb)
        RETURNING id, created_at
        """,
        content,
        str(embedding),
        source,
        merged_tags,
        json.dumps(metadata),
    )

    return {
        "status": "stored",
        "id": str(row["id"]),
        "created_at": row["created_at"].isoformat(),
        "tags": merged_tags,
        "ai_metadata": ai_metadata if ai_metadata else None,
        "content_preview": content[:100],
    }


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
    result = await _store_memory_impl(content, source, tags, metadata)
    return json.dumps(result)


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


@mcp.tool()
async def weekly_review(days: int = 7) -> str:
    """
    Review memories from the last N days, grouped by date with type/tag distribution and action items.

    Args:
        days: Number of days to look back (default 7).

    Returns:
        Structured summary for the LLM to synthesize themes and insights.
    """
    db = await get_pool()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await db.fetch(
        """
        SELECT id, content, source, tags, metadata, created_at
        FROM memories
        WHERE created_at >= $1
        ORDER BY created_at DESC
        """,
        since,
    )

    # Group by date
    by_date: dict[str, list] = {}
    type_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    action_items: list[str] = []

    for row in rows:
        date_key = row["created_at"].strftime("%Y-%m-%d")
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        ai = meta.get("ai", {})

        memory_summary = {
            "id": str(row["id"]),
            "content": row["content"][:200],
            "source": row["source"],
            "tags": row["tags"],
            "type": ai.get("type", "unknown"),
        }
        by_date.setdefault(date_key, []).append(memory_summary)

        # Count types
        mem_type = ai.get("type", "unknown")
        type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        # Count tags
        for tag in row["tags"] or []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Collect action items
        for item in ai.get("action_items", []):
            action_items.append(f"[{date_key}] {item}")

    return json.dumps(
        {
            "period": f"Last {days} days",
            "total_memories": len(rows),
            "by_date": by_date,
            "type_distribution": type_counts,
            "tag_distribution": dict(
                sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            "action_items": action_items,
        },
        cls=MemoryEncoder,
    )


@mcp.tool()
async def memory_stats() -> str:
    """
    Aggregate statistics: total count, source distribution, top tags, and daily activity.

    Returns:
        Dashboard-style stats about your memory store.
    """
    db = await get_pool()

    total = await db.fetchval("SELECT COUNT(*) FROM memories")

    source_rows = await db.fetch(
        "SELECT source, COUNT(*) as cnt FROM memories GROUP BY source ORDER BY cnt DESC"
    )
    sources = {row["source"]: row["cnt"] for row in source_rows}

    tag_rows = await db.fetch(
        """
        SELECT tag, COUNT(*) as cnt
        FROM memories, unnest(tags) AS tag
        GROUP BY tag
        ORDER BY cnt DESC
        LIMIT 20
        """
    )
    top_tags = {row["tag"]: row["cnt"] for row in tag_rows}

    activity_rows = await db.fetch(
        """
        SELECT DATE(created_at) as day, COUNT(*) as cnt
        FROM memories
        WHERE created_at >= NOW() - INTERVAL '30 days'
        GROUP BY day
        ORDER BY day DESC
        """
    )
    daily_activity = {row["day"].isoformat(): row["cnt"] for row in activity_rows}

    return json.dumps(
        {
            "total_memories": total,
            "sources": sources,
            "top_tags": top_tags,
            "daily_activity_30d": daily_activity,
        }
    )


# ─── Webhook Endpoint ───────────────────────────────────────────────────


@mcp.custom_route("/webhook/capture", methods=["POST"])
async def capture_webhook(request: Request) -> JSONResponse:
    """REST endpoint for external capture sources (web form, scripts, etc.).

    Auth is handled by the Starlette BearerAuthMiddleware.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    content = body.get("content")
    if not content or not isinstance(content, str):
        return JSONResponse(
            {"error": "Missing or invalid 'content' field"}, status_code=400
        )

    result = await _store_memory_impl(
        content=content,
        source=body.get("source", "webhook"),
        tags=body.get("tags"),
        metadata=body.get("metadata"),
    )
    return JSONResponse(result)


# ─── Prompts ─────────────────────────────────────────────────────────────


@mcp.prompt()
def memory_migration() -> str:
    """Instructs the LLM to extract and store every meaningful piece of information from the current conversation."""
    return (
        "Review the entire conversation above. Extract every meaningful piece of information, including:\n"
        "- User preferences and decisions\n"
        "- Technical details and configurations\n"
        "- Action items and commitments\n"
        "- People mentioned and their roles/context\n"
        "- Key facts, observations, and insights\n\n"
        "For EACH piece of information, call `store_memory` with:\n"
        "- `content`: A self-contained description (should make sense without conversation context)\n"
        "- `source`: 'conversation-migration'\n"
        "- `tags`: Relevant categorical tags\n"
        "- `metadata`: Any structured data (e.g. project name, priority, person's role)\n\n"
        "Store each item as a separate memory. Prefer more granular memories over fewer large ones.\n"
        "Do not store trivial small talk or tool output — focus on durable, reusable knowledge."
    )


@mcp.prompt()
def quick_capture(text: str, context: str = "") -> str:
    """Takes raw text and optional context, instructs the LLM to store an optimally-phrased memory."""
    prompt = (
        f"The user wants to capture this as a memory:\n\n"
        f"Text: {text}\n"
    )
    if context:
        prompt += f"Context: {context}\n"
    prompt += (
        "\nDetermine the optimal:\n"
        "- `content`: Rewrite the text to be self-contained, clear, and searchable\n"
        "- `source`: Best source label for this type of information\n"
        "- `tags`: 2-4 relevant tags\n"
        "- `metadata`: Any structured fields worth extracting\n\n"
        "Then call `store_memory` with these values. Return the result to the user."
    )
    return prompt


# ─── Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
        middleware=[StarletteMiddleware(BearerAuthMiddleware)],
    )
