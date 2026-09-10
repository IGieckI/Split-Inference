"""Experiment driver: run one split-point policy against the fleet and write one SQLite run DB."""

import argparse
import asyncio
import json
import sqlite3
import subprocess
import time

import yaml

from .config import ROOT, load_config
from .logger import Logger, cpu_max_khz
from .policy import Sweep, make_policy
from .reassembly import Reassembler
from .registry import Registry
from .scheduler import Scheduler
from .server import FleetServer
from .tail import Tail


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def derive_best_table(db_path) -> dict[int, str]:
    """Best static cut per node = argmin of mean OK latency in the sweep DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT node_id, action, AVG(t_total_ms) FROM requests "
        "WHERE status='OK' GROUP BY node_id, action").fetchall()
    conn.close()
    best = {}
    for node_id, action, avg in rows:
        if node_id not in best or avg < best[node_id][1]:
            best[node_id] = (action, avg)
    return {int(n): a for n, (a, _) in best.items()}


async def amain(args):
    cfg = load_config()
    db_dir = ROOT / cfg.logging.db_dir
    db_dir.mkdir(exist_ok=True)

    best = None
    if args.policy == "best":
        best_path = db_dir / "best_table.json"
        assert best_path.exists(), f"{best_path} missing - run `--policy sweep` first"
        best = {int(k): v for k, v in json.loads(best_path.read_text()).items()}
    policy = make_policy(args.policy, cfg, best_table=best)
    if args.sweep_reqs and isinstance(policy, Sweep):
        policy.reqs_per_cut = args.sweep_reqs

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{policy.name}"
    logger = Logger(db_dir / f"{run_id}.db")
    logger.start_run(run_id, policy.name,
                     json.dumps(yaml.safe_load(open(ROOT / "config.yaml"))), git_hash())

    registry = Registry(cfg)
    server = FleetServer(cfg, registry, logger)
    await server.start(args.bind)
    reassembler = Reassembler(cfg, server.send_data)
    tail = Tail(cfg)
    tail.start()
    scheduler = Scheduler(cfg, registry, policy, reassembler, tail, server, logger)
    server.scheduler, server.reassembler = scheduler, reassembler

    khz = cpu_max_khz()
    print(f"[experiment] server CPU ceiling {khz // 1000 if khz else '?'} MHz "
          f"(capped for the session - see report/fleetsplit.tex, 'Threats to validity')")
    print(f"[experiment] run {run_id}: waiting for {len(cfg.nodes)} nodes ...")
    deadline = time.monotonic() + args.wait_timeout
    while len(registry.alive()) < len(cfg.nodes):
        if time.monotonic() > deadline:
            missing = sorted({n.node_id for n in cfg.nodes} - {s.node_id for s in registry.alive()})
            print(f"[experiment] ABORT: nodes {missing} never appeared within "
                  f"{args.wait_timeout:.0f} s - check their serial logs")
            tail.stop()
            server.close()
            logger.close()
            return None
        await asyncio.sleep(0.2)
    await asyncio.sleep(args.settle)
    print(f"[experiment] all nodes up; policy {policy.name}")

    node_ids = [n.node_id for n in cfg.nodes]
    if isinstance(policy, Sweep):
        stop = lambda: all(policy.done(n) for n in node_ids)
    else:
        target = args.min_ok or cfg.experiment.min_ok_per_node
        scheduler.target_ok = target
        stop = lambda: all(scheduler.ok_count.get(n, 0) >= target for n in node_ids)

    try:
        await scheduler.run(stop, max_duration_s=args.max_duration)
    finally:
        tail.stop()
        server.close()
    logger.close()

    if isinstance(policy, Sweep):
        table = derive_best_table(db_dir / f"{run_id}.db")
        (db_dir / "best_table.json").write_text(json.dumps(table, indent=2))
        print(f"[experiment] best static cut per node: {table}")

    ok = sum(scheduler.ok_count.values())
    att = sum(scheduler.attempts.values())
    print(f"[experiment] done: {ok}/{att} OK ({att - ok} failed) -> {db_dir / (run_id + '.db')}")
    return run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["sweep", "k0", "k_shallow", "k_deep", "best"])
    ap.add_argument("--min-ok", type=int, default=None, help="OK requests per node to stop at")
    ap.add_argument("--sweep-reqs", type=int, default=None,
                    help="requests per (node, cut) in a sweep run")
    ap.add_argument("--max-duration", type=float, default=None, help="hard wall-clock cap (s)")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--wait-timeout", type=float, default=180.0,
                    help="give up if the fleet is not complete within this many seconds")
    if asyncio.run(amain(ap.parse_args())) is None:
        raise SystemExit(1)  # incomplete fleet: let a driving script stop here


if __name__ == "__main__":
    main()
