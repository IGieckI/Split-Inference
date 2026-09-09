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
    if [ "${FIREWALLD_MOVED:-0}" = 1 ]; then
        firewall-cmd --zone=trusted --remove-interface="$IF" >/dev/null 2>&1 || true
        echo "[ap] $IF removed from the firewalld trusted zone"
    fi
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

# firewalld's default zone (public) allows only ssh and dhcpv6-client
if command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1; then
    firewall-cmd --zone=trusted --change-interface="$IF" >/dev/null
    FIREWALLD_MOVED=1
    echo "[ap] $IF moved to the firewalld trusted zone (restored on exit)"
fi

# dnsmasq FIRST.
echo "[ap] dnsmasq (DHCP only, port=0)"
dnsmasq -C "$OUT/dnsmasq.conf" -d &
DNSMASQ_PID=$!
sleep 1
kill -0 "$DNSMASQ_PID" 2>/dev/null || { echo "[ap] dnsmasq died - see its output above"; exit 1; }

echo "[ap] hostapd on $IF (2.4 GHz - the ESP32s have no 5 GHz radio)"
hostapd "$OUT/hostapd.conf" &
HOSTAPD_PID=$!
sleep 2
kill -0 "$HOSTAPD_PID" 2>/dev/null || { echo "[ap] hostapd died - see its output above"; exit 1; }

echo "[ap] up. Expect per board: AP-STA-CONNECTED then DHCPACK ... node11/21/31"
wait -n
echo "[ap] one of the two exited"
