import os
import json
import asyncio
import logging
import contextvars
from datetime import datetime, timedelta, timezone
from typing import Annotated
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
DUPLICATE_THRESHOLD = float(os.environ.get("DUPLICATE_THRESHOLD", "0.95"))

# Embedding provider (defaults to OpenAI — any OpenAI-compatible API works)
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

logger = logging.getLogger("memory-server")

# ─── Multi-Token Auth ───────────────────────────────────────────────────
# Build a mapping of token → forced source.
#   MCP_API_KEY          → primary key, no source override (None)
#   MCP_API_KEY_<NAME>   → scoped key, forces source to lowercase <NAME>
#
# Example: MCP_API_KEY_EXAMPLE=abc123 means token "abc123" forces source="example"

TOKEN_SOURCE_MAP: dict[str, str | None] = {}

_primary_key = os.environ.get("MCP_API_KEY", "")
if _primary_key:
    TOKEN_SOURCE_MAP[_primary_key] = None  # No source override

for key, value in os.environ.items():
    if key.startswith("MCP_API_KEY_") and value:
        source_name = key.removeprefix("MCP_API_KEY_").lower()
        TOKEN_SOURCE_MAP[value] = source_name

logger.info(
    "Loaded %d API key(s): %s",
    len(TOKEN_SOURCE_MAP),
    ", ".join(
        f"{s or 'primary'}" for s in TOKEN_SOURCE_MAP.values()
    ),
)

# Context variable to carry the forced source from middleware → tool
auth_source_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "auth_source_override", default=None
)

# ─── Authentication Middleware (Starlette layer) ────────────────────────


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validates Bearer token at the HTTP layer, before FastMCP processes the request.

    Supports multiple API keys with per-key source overrides.
    Works around a FastMCP bug where get_http_headers() returns stale/missing
    headers during tool execution with the Streamable HTTP transport.
    See: https://github.com/jlowin/fastmcp/issues/1233
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks / OPTIONS preflight
        if request.method == "OPTIONS" or request.url.path == "/health":
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Unauthorized: missing or malformed Bearer token"},
                status_code=401,
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token not in TOKEN_SOURCE_MAP:
            return JSONResponse(
                {"error": "Unauthorized: invalid API key"}, status_code=401
            )

        # Set the source override for this request's async context
        auth_source_override.set(TOKEN_SOURCE_MAP[token])
        return await call_next(request)


# ─── MCP Server ──────────────────────────────────────────────────────────

mcp = FastMCP(
    "Memory Server",
    instructions="Personal semantic memory layer. Store and search memories across all your AI tools.",
)

# ─── Database Pool ───────────────────────────────────────────────────────

db_pool: asyncpg.Pool | None = None
_http_client: httpx.AsyncClient | None = None


async def get_pool() -> asyncpg.Pool:
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return db_pool


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
    return _http_client


# ─── Embedding Helper ───────────────────────────────────────────────────


async def get_embedding(text: str) -> list[float]:
    """Generate embedding via the configured provider (OpenAI-compatible API)."""
    client = get_http_client()
    response = await client.post(
        EMBEDDING_API_URL,
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
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
        client = get_http_client()
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
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


# ─── MCP Parameter Coercion ─────────────────────────────────────────────
# Some MCP clients (including Claude Code) double-serialise structured
# parameters, sending '["a","b"]' (a JSON string) instead of ["a","b"]
# (a native array).  Pydantic BeforeValidators intercept the raw value
# before type-checking, so JSON strings are transparently parsed.

from pydantic import BeforeValidator


def _coerce_json_list(v: object) -> object:
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return v


def _coerce_json_dict(v: object) -> object:
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return v


TagList = Annotated[list[str] | None, BeforeValidator(_coerce_json_list)]
MetaDict = Annotated[dict | None, BeforeValidator(_coerce_json_dict)]
MemoryList = Annotated[list[dict], BeforeValidator(_coerce_json_list)]
IntList = Annotated[list[int], BeforeValidator(_coerce_json_list)]


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
    force: bool = False,
) -> dict:
    """Core storage logic shared by the MCP tool and webhook endpoint."""
    # Apply source override from scoped API key (if any)
    forced_source = auth_source_override.get()
    if forced_source is not None:
        source = forced_source

    db = await get_pool()
    tags = [t.lower().strip() for t in (tags or [])]
    metadata = metadata or {}

    # Embed first — needed for the duplicate check below.
    embedding = await get_embedding(content)

    # Check for near-duplicate before spending on metadata extraction or inserting.
    if not force:
        existing = await db.fetchrow(
            """
            SELECT id, content, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            ORDER BY embedding <=> $1::vector
            LIMIT 1
            """,
            str(embedding),
        )
        if existing and float(existing["similarity"]) >= DUPLICATE_THRESHOLD:
            return {
                "status": "duplicate_detected",
                "existing_id": str(existing["id"]),
                "similarity": round(float(existing["similarity"]), 4),
                "existing_content_preview": existing["content"][:200],
                "created_at": existing["created_at"].isoformat(),
            }

    # Not a duplicate — only now pay for AI metadata extraction.
    ai_metadata = await extract_metadata(content)

    # Merge AI-generated tags with user-supplied tags (deduplicated)
    ai_tags = [t.lower().strip() for t in ai_metadata.pop("topic_tags", [])]
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
    tags: TagList = None,
    metadata: MetaDict = None,
    force: bool = False,
) -> str:
    """
    Store a new memory with automatic semantic embedding.

    Args:
        content: The text content to remember. Be descriptive — this is what gets searched.
        source: Where this memory came from (e.g. 'claude-code', 'user', 'meeting').
        tags: Optional list of tags for categorical filtering (e.g. ['project-x', 'decision']).
        metadata: Optional JSON metadata (e.g. {'project': 'website-redesign', 'priority': 'high'}).
        force: Skip duplicate detection and store regardless (default False).

    Returns:
        Confirmation with the memory ID.
    """
    result = await _store_memory_impl(content, source, tags, metadata, force=force)
    return json.dumps(result)


@mcp.tool()
async def store_memories(
    memories: MemoryList,
    force: bool = False,
) -> str:
    """
    Store multiple memories in one call. Each memory is processed concurrently.

    Args:
        memories: List of memory objects, each with 'content' (required) and optional
                  'source', 'tags', and 'metadata' fields.
        force: Skip duplicate detection for all memories (default False).

    Returns:
        List of results (one per memory, in order).
    """
    if not memories or not isinstance(memories, list):
        return json.dumps({"error": "memories must be a non-empty list"})
    if len(memories) > 20:
        return json.dumps({"error": "Maximum 20 memories per batch"})

    async def _store_one(mem: dict) -> dict:
        content = mem.get("content")
        if not content or not isinstance(content, str):
            return {"status": "error", "error": "Missing or invalid 'content'"}
        try:
            return await _store_memory_impl(
                content=content,
                source=mem.get("source", "manual"),
                tags=mem.get("tags"),
                metadata=mem.get("metadata"),
                force=force,
            )
        except Exception as e:
            return {"status": "error", "error": str(e), "content_preview": content[:100]}

    results = await asyncio.gather(*[_store_one(m) for m in memories])
    stored = sum(1 for r in results if r.get("status") == "stored")
    dupes = sum(1 for r in results if r.get("status") == "duplicate_detected")
    errors = sum(1 for r in results if r.get("status") == "error")

    return json.dumps({
        "summary": {"stored": stored, "duplicates": dupes, "errors": errors, "total": len(memories)},
        "results": list(results),
    }, cls=MemoryEncoder)


@mcp.tool()
async def search_memory(
    query: str,
    limit: int = 10,
    tags: TagList = None,
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
    params = [str(embedding), limit, query]
    param_idx = 4

    if tags:
        conditions.append(f"tags @> ${param_idx}::text[]")
        params.append([t.lower().strip() for t in tags])
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
        WITH candidates AS (
            SELECT id, content, source, tags, metadata, created_at,
                   1 - (embedding <=> $1::vector) AS vector_score,
                   ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $3)) AS fts_score
            FROM memories
            {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2 * 3
        )
        SELECT *,
               vector_score * 0.7 + fts_score * 0.3 AS combined_score
        FROM candidates
        ORDER BY combined_score DESC
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
            "similarity": round(float(row["combined_score"]), 4),
            "vector_score": round(float(row["vector_score"]), 4),
            "fts_score": round(float(row["fts_score"]), 4),
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
async def update_memory(
    memory_id: str,
    content: str | None = None,
    tags: TagList = None,
    metadata: MetaDict = None,
) -> str:
    """
    Update an existing memory. Re-embeds automatically if content changes.

    Args:
        memory_id: The UUID of the memory to update.
        content: New content (triggers re-embedding). Leave empty to keep existing.
        tags: New tags (replaces existing). Leave empty to keep existing.
        metadata: New metadata (merged with existing). Leave empty to keep existing.

    Returns:
        Updated memory details, or not_found if the ID doesn't exist.
    """
    db = await get_pool()

    existing = await db.fetchrow(
        "SELECT id, content, tags, metadata FROM memories WHERE id = $1",
        memory_id,
    )
    if not existing:
        return json.dumps({"status": "not_found", "id": memory_id})

    set_clauses = ["updated_at = NOW()"]
    params: list = [memory_id]  # $1
    idx = 2

    if content is not None and content != existing["content"]:
        embedding, ai_metadata = await asyncio.gather(
            get_embedding(content),
            extract_metadata(content),
        )
        ai_tags = [t.lower().strip() for t in ai_metadata.pop("topic_tags", [])]

        # If caller also provided tags, use those + AI tags; otherwise use existing + AI tags
        if tags is not None:
            new_tags = list(dict.fromkeys(
                [t.lower().strip() for t in tags] + ai_tags
            ))
        else:
            new_tags = list(dict.fromkeys(
                (existing["tags"] or []) + ai_tags
            ))

        set_clauses.append(f"content = ${idx}")
        params.append(content)
        idx += 1
        set_clauses.append(f"embedding = ${idx}::vector")
        params.append(str(embedding))
        idx += 1
        set_clauses.append(f"tags = ${idx}")
        params.append(new_tags)
        idx += 1

        # Merge AI metadata into existing
        existing_meta = json.loads(existing["metadata"]) if existing["metadata"] else {}
        if metadata:
            existing_meta.update(metadata)
        if ai_metadata:
            existing_meta["ai"] = ai_metadata
        set_clauses.append(f"metadata = ${idx}::jsonb")
        params.append(json.dumps(existing_meta))
        idx += 1
    else:
        # No content change — handle tags and metadata independently
        if tags is not None:
            set_clauses.append(f"tags = ${idx}")
            params.append([t.lower().strip() for t in tags])
            idx += 1

        if metadata is not None:
            existing_meta = json.loads(existing["metadata"]) if existing["metadata"] else {}
            existing_meta.update(metadata)
            set_clauses.append(f"metadata = ${idx}::jsonb")
            params.append(json.dumps(existing_meta))
            idx += 1

    if len(set_clauses) == 1:
        return json.dumps({"status": "no_changes", "id": memory_id})

    row = await db.fetchrow(
        f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = $1 RETURNING id, content, source, tags, metadata, created_at, updated_at",
        *params,
    )

    return json.dumps(
        {
            "status": "updated",
            "id": str(row["id"]),
            "content_preview": row["content"][:200],
            "source": row["source"],
            "tags": row["tags"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        },
        cls=MemoryEncoder,
    )


@mcp.tool()
async def resolve_action_items(memory_id: str, indices: IntList) -> str:
    """
    Mark action items on a memory as resolved (done / no longer open).

    Resolution state is stored in metadata.ai.resolved_indices as a list of
    integer indices into the memory's action_items array. Indices are 0-based.
    Call at end of a session when you have completed work matching a prior TODO
    surfaced by weekly_review.

    Args:
        memory_id: UUID of the memory whose action items to resolve.
        indices: 0-based indices into metadata.ai.action_items to mark resolved.

    Returns:
        Updated open/resolved state for that memory, plus any invalid indices ignored.
    """
    db = await get_pool()
    row = await db.fetchrow(
        "SELECT id, metadata FROM memories WHERE id = $1",
        memory_id,
    )
    if not row:
        return json.dumps({"status": "not_found", "id": memory_id})

    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    ai = meta.get("ai") or {}
    action_items = ai.get("action_items") or []

    if not action_items:
        return json.dumps({
            "status": "no_action_items",
            "id": memory_id,
        })

    existing_resolved = set(ai.get("resolved_indices") or [])
    valid_new = {i for i in indices if 0 <= i < len(action_items)}
    invalid = sorted({i for i in indices if not (0 <= i < len(action_items))})
    merged = sorted(existing_resolved | valid_new)

    ai["resolved_indices"] = merged
    meta["ai"] = ai

    await db.execute(
        "UPDATE memories SET metadata = $2::jsonb, updated_at = NOW() WHERE id = $1",
        memory_id,
        json.dumps(meta),
    )

    resolved_set = set(merged)
    open_items = [
        {"index": i, "text": t} for i, t in enumerate(action_items) if i not in resolved_set
    ]
    resolved_items = [
        {"index": i, "text": action_items[i]} for i in merged
    ]

    return json.dumps({
        "status": "updated",
        "id": memory_id,
        "open": open_items,
        "resolved": resolved_items,
        "invalid_indices": invalid,
    })


@mcp.tool()
async def unresolve_action_items(memory_id: str, indices: IntList) -> str:
    """
    Reverse resolve_action_items — remove indices from metadata.ai.resolved_indices.

    Args:
        memory_id: UUID of the memory.
        indices: 0-based indices to remove from the resolved list.

    Returns:
        Updated open/resolved state for that memory.
    """
    db = await get_pool()
    row = await db.fetchrow(
        "SELECT id, metadata FROM memories WHERE id = $1",
        memory_id,
    )
    if not row:
        return json.dumps({"status": "not_found", "id": memory_id})

    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    ai = meta.get("ai") or {}
    action_items = ai.get("action_items") or []

    existing_resolved = set(ai.get("resolved_indices") or [])
    merged = sorted(existing_resolved - set(indices))

    ai["resolved_indices"] = merged
    meta["ai"] = ai

    await db.execute(
        "UPDATE memories SET metadata = $2::jsonb, updated_at = NOW() WHERE id = $1",
        memory_id,
        json.dumps(meta),
    )

    resolved_set = set(merged)
    open_items = [
        {"index": i, "text": t} for i, t in enumerate(action_items) if i not in resolved_set
    ]
    resolved_items = [
        {"index": i, "text": action_items[i]} for i in merged
    ]

    return json.dumps({
        "status": "updated",
        "id": memory_id,
        "open": open_items,
        "resolved": resolved_items,
    })


@mcp.tool()
async def weekly_review(
    days: int = 7,
    include_memories: bool = False,
    memory_limit: int = 50,
    include_resolved: bool = False,
) -> str:
    """
    Review memories from the last N days: daily counts, type/tag distribution, and action items.

    Action items are returned as structured dicts ({memory_id, index, date, text})
    so they can be passed directly to resolve_action_items. By default only
    unresolved items are returned.

    Args:
        days: Number of days to look back (default 7).
        include_memories: If True, include a per-memory list grouped by date (truncated).
                          Default False — the digest alone is usually enough and keeps output small.
        memory_limit: When include_memories=True, cap total memories returned (default 50,
                      most recent first). Use list_recent for fuller pagination.
        include_resolved: If True, also include already-resolved action items (flagged
                          resolved: true). Default False.

    Returns:
        Compact digest for the LLM to synthesize themes and insights.
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

    daily_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    open_action_items: list[dict] = []
    resolved_action_items: list[dict] = []

    for row in rows:
        date_key = row["created_at"].strftime("%Y-%m-%d")
        daily_counts[date_key] = daily_counts.get(date_key, 0) + 1

        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        ai = meta.get("ai", {})

        mem_type = ai.get("type", "unknown")
        type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        for tag in row["tags"] or []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        items = ai.get("action_items") or []
        resolved_indices = set(ai.get("resolved_indices") or [])
        mem_id = str(row["id"])
        for idx, text in enumerate(items):
            record = {
                "memory_id": mem_id,
                "index": idx,
                "date": date_key,
                "text": text,
            }
            if idx in resolved_indices:
                record["resolved"] = True
                resolved_action_items.append(record)
            else:
                open_action_items.append(record)

    displayed = open_action_items + resolved_action_items if include_resolved else open_action_items

    ACTION_ITEM_CAP = 50
    action_items_truncated = len(displayed) > ACTION_ITEM_CAP

    result: dict = {
        "period": f"Last {days} days",
        "total_memories": len(rows),
        "daily_counts": dict(sorted(daily_counts.items(), reverse=True)),
        "type_distribution": type_counts,
        "tag_distribution": dict(
            sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
        "action_items": displayed[:ACTION_ITEM_CAP],
        "action_items_truncated": action_items_truncated,
        "action_items_total": len(displayed),
        "open_action_items_count": len(open_action_items),
        "resolved_action_items_count": len(resolved_action_items),
    }

    if include_memories:
        by_date: dict[str, list] = {}
        for row in rows[:memory_limit]:
            date_key = row["created_at"].strftime("%Y-%m-%d")
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            ai = meta.get("ai", {})
            by_date.setdefault(date_key, []).append({
                "id": str(row["id"]),
                "content": row["content"][:100],
                "source": row["source"],
                "tags": row["tags"],
                "type": ai.get("type", "unknown"),
            })
        result["by_date"] = by_date
        result["memories_returned"] = min(memory_limit, len(rows))
        result["memories_truncated"] = len(rows) > memory_limit

    return json.dumps(result, cls=MemoryEncoder)


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


@mcp.tool()
async def find_related(
    threshold: float = 0.85,
    tags: TagList = None,
    limit: int = 50,
) -> str:
    """
    Find clusters of related memories that may be candidates for consolidation.

    Args:
        threshold: Minimum cosine similarity to consider related (default 0.85, range 0.7-0.95).
        tags: Optional tag filter — only consider memories with ALL of these tags.
        limit: Maximum number of pairs to analyze (default 50).

    Returns:
        Clusters of related memories with similarity scores and content previews.
        Use update_memory to merge and delete_memory to remove redundant entries.
    """
    db = await get_pool()
    threshold = max(0.7, min(0.95, threshold))
    limit = min(limit, 200)

    conditions = ["m1.id < m2.id"]
    params: list = [threshold, limit]
    param_idx = 3

    if tags:
        normalized = [t.lower().strip() for t in tags]
        conditions.append(f"m1.tags @> ${param_idx}::text[]")
        conditions.append(f"m2.tags @> ${param_idx}::text[]")
        params.append(normalized)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    rows = await db.fetch(
        f"""
        SELECT
            m1.id AS id_a, m1.content AS content_a, m1.tags AS tags_a, m1.created_at AS created_a,
            m2.id AS id_b, m2.content AS content_b, m2.tags AS tags_b, m2.created_at AS created_b,
            1 - (m1.embedding <=> m2.embedding) AS similarity
        FROM memories m1
        JOIN memories m2 ON {where_clause}
            AND 1 - (m1.embedding <=> m2.embedding) >= $1
        ORDER BY similarity DESC
        LIMIT $2
        """,
        *params,
    )

    if not rows:
        return json.dumps({"threshold": threshold, "clusters": [], "total_clusters": 0, "total_memories_in_clusters": 0})

    # Union-find clustering
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    # Collect memory details and pairs
    memory_details: dict[str, dict] = {}
    pair_list: list[dict] = []

    for row in rows:
        id_a, id_b = str(row["id_a"]), str(row["id_b"])
        union(id_a, id_b)
        pair_list.append({"id_a": id_a, "id_b": id_b, "similarity": round(float(row["similarity"]), 4)})

        if id_a not in memory_details:
            memory_details[id_a] = {
                "id": id_a, "content_preview": row["content_a"][:200],
                "tags": row["tags_a"], "created_at": row["created_a"].isoformat(),
            }
        if id_b not in memory_details:
            memory_details[id_b] = {
                "id": id_b, "content_preview": row["content_b"][:200],
                "tags": row["tags_b"], "created_at": row["created_b"].isoformat(),
            }

    # Group by cluster root
    clusters_map: dict[str, list[str]] = {}
    for mem_id in memory_details:
        root = find(mem_id)
        clusters_map.setdefault(root, []).append(mem_id)

    # Build output, sorted by cluster size descending
    clusters = []
    for members in sorted(clusters_map.values(), key=len, reverse=True):
        if len(members) < 2:
            continue
        cluster_pairs = [p for p in pair_list if p["id_a"] in members and p["id_b"] in members]
        clusters.append({
            "size": len(members),
            "memories": [memory_details[m] for m in sorted(members, key=lambda m: memory_details[m]["created_at"])],
            "pairs": sorted(cluster_pairs, key=lambda p: p["similarity"], reverse=True),
        })

    return json.dumps(
        {
            "threshold": threshold,
            "clusters": clusters,
            "total_clusters": len(clusters),
            "total_memories_in_clusters": sum(c["size"] for c in clusters),
        },
        cls=MemoryEncoder,
    )


# ─── Health Check ─────────────────────────────────────────────────────────


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness + DB-connectivity probe. Exempt from auth (see middleware)."""
    try:
        db = await get_pool()
        await db.fetchval("SELECT 1")
        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.warning("Health check failed: %s", e)
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)


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
