#!/usr/bin/env bash
# Bring the fleet AP up and hold it
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/scripts/host/out"
IF=$(sed -n 's/^ *ap_interface: *\([^ #]*\).*/\1/p' "$ROOT/config.yaml")
IP=$(sed -n 's/^ *server_ip: *\([^ #]*\).*/\1/p' "$ROOT/config.yaml")

[ "$(id -u)" -eq 0 ] || { echo "needs root: sudo $0"; exit 1; }
[ -f "$OUT/hostapd.conf" ] || { echo "missing $OUT/hostapd.conf - run: uv run python scripts/host/gen_ap_configs.py"; exit 1; }
command -v hostapd >/dev/null || { echo "hostapd not installed: pacman -S hostapd"; exit 1; }

cleanup() {
    echo
    echo "[ap] stopping"
    kill ${HOSTAPD_PID:-} ${DNSMASQ_PID:-} 2>/dev/null || true
    ip addr flush dev "$IF" 2>/dev/null || true
    nmcli device set "$IF" managed yes 2>/dev/null || true
    echo "[ap] $IF handed back to NetworkManager"
}
trap cleanup EXIT INT TERM

echo "[ap] taking $IF from NetworkManager"
nmcli device disconnect "$IF" 2>/dev/null || true
nmcli device set "$IF" managed no
ip addr flush dev "$IF"
ip addr add "$IP/24" dev "$IF"
ip link set "$IF" up

echo "[ap] hostapd on $IF (2.4 GHz - the ESP32s have no 5 GHz radio)"
hostapd "$OUT/hostapd.conf" &
HOSTAPD_PID=$!
sleep 2
kill -0 "$HOSTAPD_PID" 2>/dev/null || { echo "[ap] hostapd died - see its output above"; exit 1; }

echo "[ap] dnsmasq (DHCP only, port=0)"
dnsmasq -C "$OUT/dnsmasq.conf" -d &
DNSMASQ_PID=$!

echo "[ap] up. Expect per board: AP-STA-CONNECTED then DHCPACK ... node11/21/31"
wait -n
echo "[ap] one of the two exited"
