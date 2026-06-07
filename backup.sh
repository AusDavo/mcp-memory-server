#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="/srv/docker/mcp-memory-server/backups"
RETENTION_DAYS=14

mkdir -p "$BACKUP_DIR"

docker exec mcp-memory-db pg_dump -U memory memory \
  | gzip > "$BACKUP_DIR/memory-$(date +%Y%m%d-%H%M%S).sql.gz"

find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "$(date): Backup completed, old backups cleaned"
