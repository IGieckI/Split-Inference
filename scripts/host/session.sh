#!/usr/bin/env bash
# The S2.2 policy runs plus figures, unattended.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
LOG="runs/logs/session_$(date +%Y%m%d-%H%M%S).log"
mkdir -p runs/logs

say() { echo "[session] $*" | tee -a "$LOG"; }

# preconditions, all fatal
KHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
if [ "$KHZ" -gt 500000 ]; then
    say "ABORT: CPU ceiling is $((KHZ / 1000)) MHz, expected 400."
    say "       Run: sudo cpupower frequency-set -u 400MHz"
    exit 1
fi
pgrep -x hostapd >/dev/null || { say "ABORT: hostapd is not running - start scripts/host/ap_up.sh first"; exit 1; }
pgrep -x dnsmasq >/dev/null || { say "ABORT: dnsmasq is not running - start scripts/host/ap_up.sh first"; exit 1; }
ls runs/*_sweep.db >/dev/null 2>&1 || { say "ABORT: no sweep in runs/ - --policy best needs best_table.json"; exit 1; }
[ -f runs/best_table.json ] || { say "ABORT: runs/best_table.json missing - re-run the sweep"; exit 1; }

say "CPU ceiling $((KHZ / 1000)) MHz, AP up, sweep present. Starting."
say "best_table: $(tr -d ' \n' < runs/best_table.json)"

for policy in k0 k_shallow k_deep best; do
    say "$policy"
    if ! taskset -c 0-3 uv run python -m orchestrator.experiment \
            --policy "$policy" --wait-timeout 300 --max-duration 1800 >>"$LOG" 2>&1; then
        say "FAILED on $policy - stopping. Tail of the log:"
        tail -5 "$LOG"
        exit 1
    fi
    tail -1 "$LOG" | tee -a /dev/null
    say "$(grep -h 'done:' "$LOG" | tail -1)"
done

say "figures"
uv run python analysis/figures.py >>"$LOG" 2>&1 \
    && say "figures + results.md written to analysis/out/" \
    || say "figures FAILED - see $LOG"

say "Session complete. Log: $LOG"
