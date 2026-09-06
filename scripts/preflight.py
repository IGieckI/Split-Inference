"""Pre-flight: run the whole measurement sequence against simulated nodes."""

import argparse
import json
import pathlib
import sqlite3
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LOGS = RUNS / "logs"
POLICIES = ["k0", "k_shallow", "k_deep", "best"]


def run_step(args_list, log_name, timeout):
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
    print(f"  {log_name}: {text.splitlines()[-1].strip()}")
    return pathlib.Path([ln.split("-> ")[-1].strip() for ln in text.splitlines() if "-> " in ln][-1])


def q(db, sql):
    conn = sqlite3.connect(db)
    rows = conn.execute(sql).fetchall()
    conn.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", type=float, default=0.01, help="uplink fragment loss probability")
    ap.add_argument("--reqs", type=int, default=60, help="OK requests per node per policy run")
    args = ap.parse_args()

    for req in ["model/artifacts/cuts.json", "model/assets_dev/manifest.json"]:
        assert (ROOT / req).exists(), f"{req} missing - run `make model-dev slice assets-dev`"

    # A leftover sim fleet from an interrupted run answers as the same node ids
    stray = subprocess.run(["pgrep", "-f", "scripts/sim_node.py"],
                           capture_output=True, text=True).stdout.split()
    assert not stray, (f"sim nodes already running (pids {' '.join(stray)}) - "
                       "kill them first: pkill -f scripts/sim_node.py")
    RUNS.mkdir(exist_ok=True)
    (RUNS / "best_table.json").unlink(missing_ok=True)

    print("[preflight] starting sim nodes ...")
    LOGS.mkdir(parents=True, exist_ok=True)
    sim_log = open(LOGS / "sim_nodes.log", "w")
    sim = subprocess.Popen(
        [sys.executable, "scripts/sim_node.py", "--server", "127.0.0.1",
         "--loss", str(args.loss), "--seed", "1"],
        cwd=ROOT, stdout=sim_log, stderr=subprocess.STDOUT)
    dbs = {}
    try:
        dbs["sweep"] = run_step(["--policy", "sweep", "--max-duration", "900"], "sweep", 1000)
        for p in POLICIES:
            dbs[p] = run_step(["--policy", p, "--min-ok", str(args.reqs),
                               "--max-duration", "600"], p, 700)
    finally:
        sim.terminate()
        sim.wait(10)
        sim_log.close()

    print("\n[preflight] verification")
    failures, report = [], {"dbs": {k: str(v) for k, v in dbs.items()}}

    # V1 - every feasible (node, cut) was actually exercised
    sys.path.insert(0, str(ROOT))
    from orchestrator.config import load_config
    cfg = load_config()
    covered = {(n, a) for n, a in q(dbs["sweep"],
               "SELECT DISTINCT node_id, action FROM requests WHERE status='OK'")}
    expected = {(n.node_id, c) for n in cfg.nodes for c in cfg.cuts_for_tier(n.tier)}
    report["sweep_coverage"] = sorted(f"{n}:{a}" for n, a in covered)
    if expected - covered:
        failures.append(f"V1: no OK rows for {sorted(expected - covered)}")

    # V2/V3 - integrity and failure rate across every run
    crc = fails = total = 0
    for db in dbs.values():
        (n_all,) = q(db, "SELECT COUNT(*) FROM requests")[0]
        (n_crc,) = q(db, "SELECT COUNT(*) FROM requests WHERE status='CRC'")[0]
        (n_bad,) = q(db, "SELECT COUNT(*) FROM requests WHERE status!='OK'")[0]
        total, crc, fails = total + n_all, crc + n_crc, fails + n_bad
    report["crc_failures"], report["failure_rate"] = crc, fails / max(total, 1)
    print(f"  requests {total}, failures {fails} ({fails / max(total, 1):.2%}), CRC {crc}")
    if crc:
        failures.append(f"V2: {crc} CRC failures - the ARQ path delivered corrupt tensors")
    if fails / max(total, 1) >= 0.03:
        failures.append(f"V3: failure rate {fails / max(total, 1):.2%} >= 3%")

    # V4 - no reboots
    for name, db in dbs.items():
        rows = q(db, "SELECT MAX(boot_count) - MIN(boot_count) FROM heartbeats")
        if rows and rows[0][0]:
            failures.append(f"V4: node rebooted during run {name}")

    # V5 - best_table is the sweep argmin
    table = {int(k): v for k, v in json.loads((RUNS / "best_table.json").read_text()).items()}
    argmin = {}
    for n, a, avg in q(dbs["sweep"], "SELECT node_id, action, AVG(t_total_ms) FROM requests "
                                     "WHERE status='OK' GROUP BY node_id, action"):
        if n not in argmin or avg < argmin[n][1]:
            argmin[n] = (a, avg)
    report["best_table"] = table
    if table != {n: a for n, (a, _) in argmin.items()}:
        failures.append(f"V5: best_table {table} != sweep argmin {argmin}")
    dispatched = {int(n): a for n, a in q(
        dbs["best"], "SELECT node_id, action FROM requests GROUP BY node_id")}
    multi = q(dbs["best"], "SELECT node_id, COUNT(DISTINCT action) FROM requests "
                           "GROUP BY node_id HAVING COUNT(DISTINCT action) > 1")
    if dispatched != table or multi:
        failures.append(f"V5: best run dispatched {dispatched} (multi-cut nodes: {multi}), "
                        f"expected {table}")

    # Reported, deliberately NOT a gate
    means = {}
    for p in POLICIES:
        means[p] = {int(n): m for n, m in q(
            dbs[p], "SELECT node_id, AVG(t_total_ms) FROM requests WHERE status='OK' "
                    "GROUP BY node_id")}
    sweep_means = {(int(n), a): m for n, a, m in q(
        dbs["sweep"], "SELECT node_id, action, AVG(t_total_ms) FROM requests "
                      "WHERE status='OK' GROUP BY node_id, action")}
    report["mean_ok_ms"] = means
    report["sweep_isolated_mean_ok_ms"] = {f"{n}:{a}": round(m, 1) for (n, a), m in sweep_means.items()}
    print("  mean ms per node - policy runs are fleet-concurrent, sweep is isolated")
    print(f"  {'node':>5} " + " ".join(f"{p:>12}" for p in POLICIES))
    for n in sorted(means["best"]):
        print(f"  {n:>5} " + " ".join(f"{means[p].get(n, float('nan')):>12.1f}" for p in POLICIES))

    # Coupling magnitude: same node
    print("  same (node, cut), isolated sweep vs fleet-concurrent policy run:")
    coupling = {}
    for p in POLICIES:
        cut_by_node = {int(n): a for n, a in q(
            dbs[p], "SELECT node_id, action FROM requests GROUP BY node_id")}
        for n, cut in cut_by_node.items():
            iso = sweep_means.get((n, cut))
            conc = means[p].get(n)
            if iso and conc:
                pct = (conc - iso) / iso * 100
                coupling[f"{p}/{n}:{cut}"] = round(pct, 1)
                print(f"    {p:10} node {n} {cut:10} {iso:7.1f} -> {conc:7.1f}  ({pct:+5.1f}%)")
    report["concurrency_overhead_pct"] = coupling

    # V6 - figures regenerate
    r = subprocess.run([sys.executable, "analysis/figures.py"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        failures.append("V6: analysis/figures.py failed")

    report["failures"] = failures
    (RUNS / "preflight_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[preflight] report -> {RUNS / 'preflight_report.json'}")
    if failures:
        print("[preflight] FAIL:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("[preflight] PASS - the measurement harness works end to end (synthetic latencies)")


if __name__ == "__main__":
    main()
