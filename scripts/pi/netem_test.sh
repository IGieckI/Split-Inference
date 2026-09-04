#!/usr/bin/env bash
# Injected loss/delay on the Pi AP interface
set -euo pipefail
IF=${IF:-wlan0}

case "${1:?usage: netem_test.sh <loss-percent|off>}" in
  off)
    tc qdisc del dev "$IF" root 2>/dev/null || true
    echo "netem cleared on $IF"
    ;;
  *)
    tc qdisc replace dev "$IF" root netem loss "$1%" delay 10ms 5ms
    echo "netem on $IF: loss $1%, delay 10ms +/- 5ms"
    tc qdisc show dev "$IF"
    ;;
esac
