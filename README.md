# MCP Memory Server

A self-hosted semantic memory layer for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and other MCP clients. Store memories as text, search them by meaning, and capture thoughts from your phone — all on your own infrastructure.

Built with [FastMCP](https://github.com/jlowin/fastmcp), Postgres + [pgvector](https://github.com/pgvector/pgvector), and OpenAI embeddings.

## Architecture

```
Claude Code ──HTTPS──▶ Reverse proxy ──▶ FastMCP server ──▶ Postgres + pgvector
                            ▲                  │
Phone/browser ──HTTPS───────┘                  ▼
  (capture form)                         OpenAI API
                                   (embeddings + metadata)
```

- **Database**: Postgres 17 with pgvector — stores text alongside 1536-dimension vector embeddings
- **Server**: Python 3.13 + FastMCP — Streamable HTTP transport with Bearer token auth
- **Embeddings**: OpenAI `text-embedding-3-small` for vector generation
- **Metadata**: GPT-4o-mini extracts structured metadata (type, tags, entities, action items) in parallel with embedding — best-effort, never blocks storage
- **Indexing**: HNSW (not IVFFlat) — works on empty tables

## Tools

| Tool | Description |
|---|---|
| `store_memory` | Save text with auto-generated embedding and AI-extracted metadata |
| `search_memory` | Semantic similarity search — finds memories by meaning, not keywords |
| `list_recent` | List the last N memories, optionally filtered by source |
| `delete_memory` | Remove a memory by UUID |
| `weekly_review` | Summarize the last N days: grouped by date, type/tag distribution, action items |
| `memory_stats` | Aggregate dashboard: totals, sources, top tags, 30-day activity |

## Prompts

| Prompt | Description |
|---|---|
| `memory_migration` | Instructs the LLM to extract everything worth keeping from the current conversation and store each piece as a separate memory |
| `quick_capture` | Takes raw text, has the LLM determine optimal phrasing/tags/metadata, then stores it |

## Setup

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key (for embeddings and metadata extraction)
- A reverse proxy that handles TLS (e.g. Caddy, Nginx, Traefik)

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
POSTGRES_USER=memory
POSTGRES_PASSWORD=<generate-a-strong-password>
POSTGRES_DB=memory
DATABASE_URL=postgresql://memory:<password>@mcp-memory-db:5432/memory
OPENAI_API_KEY=sk-...
MCP_API_KEY=<generate-with-openssl-rand-hex-32>
```

The `DATABASE_URL` must use the container name (`mcp-memory-db`), not the service name (`db`), to avoid DNS collisions if the server container is on a shared Docker network.

### 2. Start the stack

```bash
docker compose up -d
```

This starts two containers:
- `mcp-memory-db` — Postgres with pgvector, initialised by `init.sql`
- `mcp-memory-server` — FastMCP server on port 8000

### 3. Configure your reverse proxy

Point your domain at the `mcp-memory-server` container on port 8000. The server exposes:

- `/mcp` — MCP endpoint (Bearer token auth handled by the server)
- `/webhook/capture` — REST endpoint for external capture (Bearer token auth handled by the server)

### 4. Connect Claude Code

```bash
claude mcp add memory-server \
    --transport http \
    --scope user \
    --header "Authorization: Bearer <your-MCP_API_KEY>" \
    -- https://your-domain.example.com/mcp
```

Restart Claude Code. The six tools and two prompts will be available in every project.

## Webhook / Capture Form

The `/webhook/capture` endpoint accepts POST requests for storing memories from external sources:

```bash
curl -X POST https://your-domain.example.com/webhook/capture \
  -H "Authorization: Bearer <your-MCP_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"content": "Remember this", "source": "curl", "tags": ["test"]}'
```

You can build a mobile-friendly web form that POSTs to this endpoint. If you use a reverse proxy like Caddy, you can handle authentication at the proxy layer and inject the Bearer token server-side so the API key never reaches the browser.

## Backups

The included `backup.sh` script dumps the database with 14-day retention:

```bash
# Add to cron (e.g. daily at 3am)
0 3 * * * /path/to/mcp-memory-server/backup.sh
```

## AI Metadata Extraction

Every `store_memory` call runs GPT-4o-mini in parallel with embedding generation to extract:

- **Type**: `observation`, `task`, `idea`, `reference`, or `person_note`
- **Topic tags**: 1–3 kebab-case tags, merged with any user-supplied tags
- **Entities**: people, places, organizations mentioned
- **Action items**: anything actionable

AI metadata is stored under `metadata.ai` in the JSONB column, keeping it separate from user-supplied metadata. If extraction fails for any reason, the memory is still stored normally.

## Authentication note

Authentication uses a Starlette `BaseHTTPMiddleware` that validates the Bearer token at the HTTP layer, before FastMCP processes the request. This works around a [known FastMCP bug](https://github.com/jlowin/fastmcp/issues/1233) where `get_http_headers()` returns stale or missing headers during tool execution with the Streamable HTTP transport. If FastMCP's own middleware sees your headers correctly in a future release, you could switch back — but the Starlette approach is arguably more correct anyway since auth belongs at the transport layer.

## Docker networking note

The server container needs to be on your reverse proxy's Docker network so the proxy can reach it. The database container should stay on the default (internal) network only. If both containers share a network with other stacks, use explicit `container_name` values and reference those in `DATABASE_URL` to avoid DNS collisions with common service names like `db`.

## Credit

Inspired by [Nate B Jones](https://www.youtube.com/@NateBJones) and his [Open Brain guide](https://natesnewsletter.substack.com/p/every-ai-you-use-forgets-you-heres). His approach uses Supabase and OpenRouter — managed services that are faster to set up. This project takes the same concept and makes it fully self-hosted.

## License

MIT
