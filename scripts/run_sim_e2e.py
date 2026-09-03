"""Full Phase-1 experiment sequence against simulated nodes."""

import argparse
import json
import pathlib
import sqlite3
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LOGS = RUNS / "logs"


def experiment(args_list, log_name, timeout):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{log_name}.log"
    cmd = [sys.executable, "-m", "orchestrator.experiment", "--bind", "127.0.0.1",
           "--sim-tail", "--settle", "1"] + args_list
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    text = log.read_text()
    if r.returncode != 0:
        print(text[-3000:])
        raise SystemExit(f"step {log_name} failed (see {log})")
    db = [ln.split("-> ")[-1].strip() for ln in text.splitlines() if "-> " in ln]
    print(f"  {log_name}: {text.splitlines()[-1].strip()}")
    return pathlib.Path(db[-1]) if db else None


def q(db, sql):
    conn = sqlite3.connect(db)
    rows = conn.execute(sql).fetchall()
    conn.close()
    return rows


def node_means(db):
    return {int(n): (avg, p) for n, avg, p in q(
        db, "SELECT node_id, AVG(t_total_ms), COUNT(*) FROM requests "
            "WHERE status='OK' GROUP BY node_id")}


def late_half_means(db):
    """Mean OK latency over each node's second half of rounds"""
    out = {}
    conn = sqlite3.connect(db)
    for (n,) in conn.execute("SELECT DISTINCT node_id FROM requests"):
        rows = [r[0] for r in conn.execute(
            "SELECT t_total_ms FROM requests WHERE node_id=? AND status='OK' "
            "ORDER BY req_id", (n,))]
        half = rows[len(rows) // 2:]
        out[int(n)] = sum(half) / len(half)
    conn.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for req in ["model/artifacts/cuts.json", "model/assets_dev/manifest.json"]:
        assert (ROOT / req).exists(), f"{req} missing - run `make model-dev slice assets-dev`"
    RUNS.mkdir(exist_ok=True)
    (RUNS / "t_ref.json").unlink(missing_ok=True)
    (RUNS / "b3_table.json").unlink(missing_ok=True)

    print("[e2e] starting sim nodes ...")
    LOGS.mkdir(parents=True, exist_ok=True)
    sim_log = open(LOGS / "sim_nodes.log", "w")
    sim = subprocess.Popen(
        [sys.executable, "scripts/sim_node.py", "--server", "127.0.0.1",
         "--loss", str(args.loss), "--seed", "1"],
        cwd=ROOT, stdout=sim_log, stderr=subprocess.STDOUT)
    dbs = {}
    try:
        # Every trace run is aligned to whole trace periods so all policies see the identical phase mix
        period = 120  # traces/trace_dev.yaml
        trace = ["--trace", "traces/trace_dev.yaml", "--min-ok", "0"]
        experiment(["--mode", "warmup", "--max-duration", "300"], "warmup", 400)
        dbs["sweep"] = experiment(["--mode", "sweep", "--max-duration", "600"], "sweep", 700)
        for b in ("b0", "b2", "b3"):
            dbs[b] = experiment(["--mode", "baseline", "--policy", b] + trace
                                + ["--max-duration", str(period)], b, 300)
        dbs["learn"] = experiment(
            ["--mode", "learn", "--policy", "eps", "--seed", str(args.seed),
             "--max-duration", str(2 * period)] + trace, "learn", 600)
    finally:
        sim.terminate()
        sim.wait(10)
        sim_log.close()

    # verification
    print("\n[e2e] verification")
    failures, report = [], {"dbs": {k: str(v) for k, v in dbs.items()}}

    learned, b3m = node_means(dbs["learn"]), node_means(dbs["b3"])
    late = late_half_means(dbs["learn"])
    report["mean_ok_ms"] = {"learned_whole_run": learned, "learned_late_half": late,
                            "b3": b3m, "b0": node_means(dbs["b0"]), "b2": node_means(dbs["b2"])}
    print(f"  {'node':>5} {'B0':>8} {'B2':>8} {'B3':>8} {'lrn-all':>8} {'lrn-late':>9}")
    for n in sorted(learned):
        row = [report["mean_ok_ms"][k].get(n, (float('nan'),))[0] for k in ("b0", "b2", "b3")]
        print(f"  {n:>5} " + " ".join(f"{v:>8.1f}" for v in row)
              + f" {learned[n][0]:>8.1f} {late[n]:>9.1f}")
        if late[n] > b3m[n][0] * 1.03:  # V1 on the post-burn-in policy (3% noise slack)
            failures.append(f"V1 node {n}: learned(late) {late[n]:.1f} ms > B3 {b3m[n][0]:.1f} ms")

    # V2: dominant arm per tier over the exploitation half of the learning run
    hist = q(dbs["learn"],
             "SELECT tier, action, COUNT(*) FROM requests WHERE status='OK' AND req_id > "
             "(SELECT MAX(req_id)/2 FROM requests) GROUP BY tier, action")
    by_tier = {}
    for tier, action, cnt in hist:
        by_tier.setdefault(tier, {})[action] = cnt
    dominant = {t: max(d, key=d.get) for t, d in by_tier.items()}
    report["arm_histograms_late_half"] = by_tier
    report["dominant_arm"] = dominant
    print(f"  dominant arms (late half): {dominant}")
    print(f"  histograms: {by_tier}")
    if len(set(dominant.values())) < 2:
        failures.append(f"V2: all tiers converged to the same arm {dominant}")

    # V3: timeout rate on the learning run
    (n_all,) = q(dbs["learn"], "SELECT COUNT(*) FROM requests")[0]
    (n_to,) = q(dbs["learn"], "SELECT COUNT(*) FROM requests WHERE status='TIMEOUT'")[0]
    rate = n_to / n_all if n_all else 0
    report["timeout_rate"] = rate
    print(f"  timeouts: {n_to}/{n_all} = {rate:.2%}")
    if rate >= 0.03:
        failures.append(f"V3: timeout rate {rate:.2%} >= 3%")

    # V4: no mid-run reboots
    (boots,) = q(dbs["learn"], "SELECT MAX(boot_count) - MIN(boot_count) FROM heartbeats")[0]
    if boots:
        failures.append(f"V4: boot_count changed by {boots} during the learning run")

    report["failures"] = failures
    (RUNS / "sim_e2e_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[e2e] report -> {RUNS / 'sim_e2e_report.json'}")
    if failures:
        print("[e2e] FAIL:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("[e2e] PASS - mechanism verified end-to-end in simulation")


if __name__ == "__main__":
    main()
