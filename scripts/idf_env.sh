#!/usr/bin/env bash
# Build FleetSplit firmware for one tier against the local ESP-IDF install.
set -euo pipefail

TIER="${1:?usage: idf_env.sh A|B|C [idf.py args]}"
shift || true

case "$TIER" in
  A) TARGET=esp32s3 ;;
  B|C) TARGET=esp32 ;;
  *) echo "tier must be A, B or C"; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# fleet_config.h is generated from config.yaml before every build
(cd "$ROOT" && uv run python scripts/gen_config_header.py)

export _FLEET_TIER="$TIER" _FLEET_TARGET="$TARGET" _FLEET_ROOT="$ROOT" _FLEET_ARGS="${*:-build}"
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
  cd "$_FLEET_ROOT/firmware"
  DEFS="sdkconfig.defaults;sdkconfig.tier$_FLEET_TIER"
  idf.py -B "build_$_FLEET_TIER" -DSDKCONFIG="build_$_FLEET_TIER/sdkconfig" \
         -DSDKCONFIG_DEFAULTS="$DEFS" set-target "$_FLEET_TARGET"
  # shellcheck disable=SC2086
  idf.py -B "build_$_FLEET_TIER" -DSDKCONFIG="build_$_FLEET_TIER/sdkconfig" \
         -DSDKCONFIG_DEFAULTS="$DEFS" $_FLEET_ARGS
'
