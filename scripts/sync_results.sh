#!/usr/bin/env bash
# Pull results + logs back from the GPU server to the laptop.
# Usage: bash scripts/sync_results.sh [PORT] [HOST]
set -eu
PORT="${1:-25372}"
HOST="${2:-154.64.230.67}"
KEY="$HOME/.ssh/id_ed25519_new"
cd "$(dirname "$0")/.."
mkdir -p crcl/results logs
scp -i "$KEY" -P "$PORT" -o BatchMode=yes -r \
  "root@$HOST:~/Continual_learning/crcl/results/" crcl/ 2>/dev/null || true
scp -i "$KEY" -P "$PORT" -o BatchMode=yes -r \
  "root@$HOST:~/Continual_learning/logs/" . 2>/dev/null || true
echo "synced results/ and logs/ from $HOST:$PORT"
