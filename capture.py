"""SimpleX Chat capture service — monitors a SimpleX group and stores messages as memories."""

import os
import json
import asyncio
import logging

import httpx
import websockets

# ─── Configuration ───────────────────────────────────────────────────────

SIMPLEX_WS_URL = os.environ["SIMPLEX_WS_URL"]  # ws://simplex-cli:5225
SIMPLEX_HTTP_URL = os.environ["SIMPLEX_HTTP_URL"]  # http://simplex-cli:5226
MEMORY_WEBHOOK_URL = os.environ["MEMORY_WEBHOOK_URL"]  # http://mcp-memory-server:8000/webhook/capture
MCP_API_KEY = os.environ["MCP_API_KEY"]
BRAIN_GROUP = os.environ.get("BRAIN_GROUP", "Brain")
DEBUG = os.environ.get("DEBUG", "0") == "1"
RECONNECT_DELAY = 5  # seconds

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("capture")

# ─── Helpers ─────────────────────────────────────────────────────────────


async def store_memory(content: str, source: str = "simplex") -> dict | None:
    """POST content to the memory webhook."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MEMORY_WEBHOOK_URL,
                headers={
                    "Authorization": f"Bearer {MCP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"content": content, "source": source, "tags": ["simplex"]},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("Failed to store memory: %s", e)
        return None


async def send_confirmation(message: str) -> None:
    """Send a confirmation message back to the Brain group via SimpleX HTTP API.

    Note: The exact SimpleX HTTP API format may need adjustment based on the
    CLI version. This uses the /send endpoint pattern.
    """
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{SIMPLEX_HTTP_URL}/send",
                json={"group": BRAIN_GROUP, "message": message},
                timeout=10.0,
            )
    except Exception as e:
        logger.warning("Failed to send confirmation: %s", e)


def format_confirmation(result: dict) -> str:
    """Format a stored memory result as a confirmation message."""
    ai_meta = result.get("ai_metadata") or {}
    mem_type = ai_meta.get("type", "memory")
    tags = result.get("tags", [])
    tag_str = f" [{', '.join(tags)}]" if tags else ""
    return f"Stored ({mem_type}){tag_str}"


def extract_message(event: dict) -> tuple[str, str] | None:
    """Extract (group_name, message_text) from a SimpleX WebSocket event.

    Confirmed event structure (SimpleX CLI v6, WebSocket API):
    {
      "resp": {
        "type": "newChatItems",
        "chatItems": [{
          "chatInfo": {"type": "group", "groupInfo": {"localDisplayName": "Brain", ...}},
          "chatItem": {"content": {"msgContent": {"type": "text", "text": "..."}}}
        }]
      }
    }
    """
    resp = event.get("resp", {})

    if resp.get("type") != "newChatItems":
        return None

    for item in resp.get("chatItems", []):
        chat_info = item.get("chatInfo", {})
        if chat_info.get("type") != "group":
            continue

        group_name = chat_info.get("groupInfo", {}).get("localDisplayName", "")

        chat_item = item.get("chatItem", {})
        content = chat_item.get("content", {})
        msg_content = content.get("msgContent", {})
        text = msg_content.get("text", "")

        if group_name and text:
            return (group_name, text)

    return None


# ─── Main Loop ───────────────────────────────────────────────────────────


async def monitor():
    """Connect to SimpleX WebSocket and process messages."""
    while True:
        try:
            logger.info("Connecting to SimpleX WebSocket at %s", SIMPLEX_WS_URL)
            async with websockets.connect(SIMPLEX_WS_URL) as ws:
                logger.info("Connected to SimpleX WebSocket")

                async for raw in ws:
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message: %s", raw[:200])
                        continue

                    if DEBUG:
                        logger.debug("RAW EVENT: %s", json.dumps(event, indent=2)[:2000])

                    parsed = extract_message(event)
                    if parsed is None:
                        continue

                    group_name, text = parsed
                    if group_name != BRAIN_GROUP:
                        continue

                    text = text.strip()
                    if not text:
                        continue

                    logger.info("Brain message: %s", text[:100])

                    result = await store_memory(text)
                    if result:
                        confirmation = format_confirmation(result)
                        logger.info("Stored: %s", confirmation)
                        await send_confirmation(confirmation)

        except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
            logger.warning("WebSocket disconnected (%s), reconnecting in %ds", e, RECONNECT_DELAY)
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)

        await asyncio.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    asyncio.run(monitor())
