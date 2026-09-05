#!/usr/bin/env bash
# Flash every board.
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: $0 A=/dev/ttyUSB0 [B=/dev/ttyUSB1] [C=/dev/ttyUSB2]"; exit 1; }

for spec in "$@"; do
    tier="${spec%%=*}"
    port="${spec#*=}"
    echo "=== tier $tier -> $port"
    bash "$(dirname "$0")/idf_env.sh" "$tier" flash -p "$port"
done
echo "All boards flashed. Bring up the Pi AP, then follow README.md."
