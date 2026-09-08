#!/usr/bin/env bash
# Flash every board.
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: $0 esp32s3=/dev/ttyACM0 [esp32cam=/dev/ttyUSB0] [esp32=/dev/ttyUSB1]"; exit 1; }

for spec in "$@"; do
    board="${spec%%=*}"
    port="${spec#*=}"
    echo "=== $board -> $port"
    bash "$(dirname "$0")/idf_env.sh" "$board" flash -p "$port"
done
echo "All boards flashed. Read each boot log (gate G3), then bring up the host AP."
