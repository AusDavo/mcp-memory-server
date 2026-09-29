#!/usr/bin/env python3
"""Fill the staging column embedding_768 for migration 002 (see migrations/).

Embeds every row whose staged vector is missing or was made from different
content (md5 mismatch), so it is safe to rerun: phase A fills everything,
the phase B rerun only touches rows written or edited since.

Vectors from an earlier run (e.g. the 2026-09-29 eval) can be reused with
--reuse VECTORS.jsonl --corpus CORPUS.jsonl. A vector is reused only when the
row's current content is byte-identical to the corpus copy it was made from,
and it must have been made with the same model, prefix and settings.

Runs inside the server container, which has asyncpg/httpx and DATABASE_URL:

  docker cp backfill_embeddings.py mcp-memory-server:/tmp/
  docker exec mcp-memory-server python /tmp/backfill_embeddings.py [--reuse ...]
"""
import argparse, asyncio, hashlib, json, os, sys, time

import asyncpg, httpx

TEMP_ZONE = "/sys/class/thermal/thermal_zone1/temp"  # mithril's x86_pkg_temp


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def cpu_temp() -> int:
    try:
        return int(open(TEMP_ZONE).read()) // 1000
    except OSError:
        return 0  # not readable (other host, or masked): guard disabled


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://ollama:11434/api/embed")
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--prefix", default="search_document: ")
    ap.add_argument("--num-thread", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reuse", help="jsonl of {id, tokens, vec}")
    ap.add_argument("--corpus", help="jsonl of {id, content} the reused vectors were made from")
    ap.add_argument("--temp-high", type=int, default=95, help="pause above this CPU temp (C)")
    ap.add_argument("--temp-resume", type=int, default=88)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if bool(a.reuse) != bool(a.corpus):
        sys.exit("--reuse and --corpus go together")

    reuse = {}
    if a.reuse:
        src = {}
        for line in open(a.corpus):
            c = json.loads(line)
            src[c["id"]] = md5(c["content"])
        for line in open(a.reuse):
            v = json.loads(line)
            if v["id"] in src:
                reuse[v["id"]] = (src[v["id"]], v["vec"], v["tokens"] >= a.max_tokens)
        print(f"loaded {len(reuse)} reusable vectors", flush=True)

    db = await asyncpg.connect(os.environ["DATABASE_URL"])
    rows = await db.fetch(
        "SELECT id::text, content, md5(content) AS h FROM memories "
        "WHERE embedding_768 IS NULL OR embedding_768_src IS DISTINCT FROM md5(content) "
        "ORDER BY created_at"
    )
    reused = [r for r in rows if r["id"] in reuse and reuse[r["id"]][0] == r["h"]]
    reused_ids = {r["id"] for r in reused}
    todo = [r for r in rows if r["id"] not in reused_ids]
    print(f"{len(rows)} rows need vectors: {len(reused)} reusable, {len(todo)} to embed", flush=True)
    if a.dry_run:
        await db.close()
        return

    async def write(r, vec, truncated):
        # The md5 guard skips a row whose content changed mid-run; the next
        # run picks it up.
        await db.execute(
            "UPDATE memories SET embedding_768 = $2::vector, embedding_768_src = $3, "
            "embedding_768_truncated = $4 WHERE id = $1::uuid AND md5(content) = $3",
            r["id"], str(vec), r["h"], truncated,
        )

    for r in reused:
        _, vec, truncated = reuse[r["id"]]
        await write(r, vec, truncated)
    print(f"wrote {len(reused)} reused vectors", flush=True)

    async with httpx.AsyncClient(timeout=600) as http:
        start = time.time()
        for i, r in enumerate(todo, 1):
            if cpu_temp() >= a.temp_high:
                print(f"CPU {cpu_temp()}C, pausing", flush=True)
                while cpu_temp() >= a.temp_resume:
                    await asyncio.sleep(30)
            resp = await http.post(a.url, json={
                "model": a.model, "input": a.prefix + r["content"], "truncate": True,
                "options": {"num_thread": a.num_thread},
            })
            resp.raise_for_status()
            d = resp.json()
            await write(r, d["embeddings"][0], d.get("prompt_eval_count", 0) >= a.max_tokens)
            if i % 50 == 0 or i == len(todo):
                print(f"embedded {i}/{len(todo)} ({time.time() - start:.0f}s, cpu {cpu_temp()}C)", flush=True)

    left = await db.fetchval(
        "SELECT COUNT(*) FROM memories "
        "WHERE embedding_768 IS NULL OR embedding_768_src IS DISTINCT FROM md5(content)"
    )
    print(f"done; {left} row(s) still lack a current vector", flush=True)
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
