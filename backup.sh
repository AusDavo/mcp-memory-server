#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="/srv/docker/mcp-memory-server/backups"
RETENTION_DAYS=14

# Optional alerting config (untracked, gitignored). Defines KUMA_PUSH_URL
# for an Uptime Kuma "Push" monitor — a dead-man's switch that alerts if no
# heartbeat arrives. Absent on dev machines, which is fine (no ping is sent).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.backup-env" ] && . "$SCRIPT_DIR/.backup-env"

mkdir -p "$BACKUP_DIR"

# pipefail (set above) makes the script exit non-zero if pg_dump fails, so the
# heartbeat below is only reached when the dump AND rotation both succeed.
docker exec mcp-memory-db pg_dump -U memory memory \
  | gzip > "$BACKUP_DIR/memory-$(date +%Y%m%d-%H%M%S).sql.gz"

find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Success heartbeat. If this never fires (cron didn't run, dump failed, host
# down), Kuma sees a missing heartbeat past the interval and alerts.
if [ -n "${KUMA_PUSH_URL:-}" ]; then
  curl -fsS -m 10 "$KUMA_PUSH_URL" >/dev/null \
    || echo "$(date): WARN: backup succeeded but Kuma heartbeat ping failed"
fi

echo "$(date): Backup completed, old backups cleaned"
