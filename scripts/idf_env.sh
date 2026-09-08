#!/usr/bin/env bash
# Run idf.py inside one board's project folder
set -euo pipefail

BOARD="${1:?usage: idf_env.sh <esp32s3|esp32cam|esp32> [idf.py args]}"
shift || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/firmware/$BOARD/CMakeLists.txt" ] || {
    echo "no project at firmware/$BOARD (expected esp32s3, esp32cam or esp32)"; exit 1; }

# fleet_config.h is generated from config.yaml before every build
(cd "$ROOT" && uv run python scripts/gen_config_header.py)

export _FLEET_DIR="$ROOT/firmware/$BOARD" _FLEET_ARGS="${*:-build}"
export _FLEET_EIM="${EIM_ACTIVATE:-$HOME/.espressif/tools/activate_idf_v5.5.3.sh}"
export _FLEET_IDF_DIR="${IDF_DIR:-$HOME/.espressif/v5.5.3/esp-idf}"

exec bash -c '
  set +eu
  if [ -f "$_FLEET_EIM" ]; then
    . "$_FLEET_EIM" >/dev/null
  elif [ -f "$_FLEET_IDF_DIR/export.sh" ]; then
    . "$_FLEET_IDF_DIR/export.sh" >/dev/null
  else
    echo "ESP-IDF not found (set EIM_ACTIVATE or IDF_DIR)"; exit 1
  fi
  set -e
  [ -n "$IDF_PATH" ] || { echo "IDF activation failed"; exit 1; }
  export PATH="$IDF_PATH/tools:$PATH"    # eim does not add idf.py's dir
  cd "$_FLEET_DIR"
  # shellcheck disable=SC2086
  idf.py $_FLEET_ARGS
'
