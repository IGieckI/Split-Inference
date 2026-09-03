"""Experiment driver"""

import argparse
import asyncio
import json
import sqlite3
import subprocess
import time

import yaml

from .config import ROOT, load_config
from . import protocol as P
from .logger import Logger
from .policy import Sweep, make_policy
from .reassembly import Reassembler
from .registry import Registry
from .reward import RewardComputer
from .scheduler import Scheduler
from .server import FleetServer
from .tail import Tail
from .timesync import TimeSync


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_trace(path):
    if path is None:
        return {"phases": [{"t": 0, "cpu_mhz": 240}], "period": None}
    with open(path) as f:
        trace = yaml.safe_load(f)
    assert trace["phases"][0]["t"] == 0, "trace must start at t=0"
    return trace


async def trace_task(trace, cfg, registry, server, scheduler):
    """Broadcast THROTTLE at each phase boundary"""
    period = trace.get("period")
    phases = trace["phases"]
    t0 = time.monotonic()
    current = -1
    while True:
        elapsed = time.monotonic() - t0
        if period:
            elapsed %= period
        idx = max(i for i, p in enumerate(phases) if p["t"] <= elapsed)
        if idx != current:
            current = idx
            mhz = phases[idx]["cpu_mhz"]
            scheduler.current_phase = idx
            for st in registry.alive():
                server.send_ctrl(st.ctrl_addr, P.pack(
                    P.THROTTLE, st.node_id, 0, P.THROTTLE_S.pack(mhz)))
            print(f"[trace] phase {idx}: cpu_mhz={mhz}")
        await asyncio.sleep(0.1)


def derive_b3_table(db_path) -> dict[int, str]:
    """Best static arm per node = argmin of mean OK latency in the sweep DB."""
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
    reward = RewardComputer(cfg, db_dir / "t_ref.json")

    if args.mode == "warmup":
        policy = make_policy("b0", cfg)
    elif args.mode == "sweep":
        policy = make_policy("sweep", cfg)
    elif args.mode == "baseline":
        b3 = None
        if args.policy == "b3":
            b3 = {int(k): v for k, v in json.loads((db_dir / "b3_table.json").read_text()).items()}
        policy = make_policy(args.policy, cfg, b3_table=b3)
    elif args.mode == "learn":
        policy = make_policy(args.policy, cfg, seed=args.seed)
    if args.mode != "warmup":
        assert reward.t_ref, "no t_ref.json - run --mode warmup first (section 11.8 step 4)"

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{args.mode}_{policy.name}"
    logger = Logger(db_dir / f"{run_id}.db")
    logger.start_run(run_id, policy.name, args.trace or "flat",
                     json.dumps({"seed": args.seed, "config": yaml.safe_load(
                         open(ROOT / "config.yaml"))}), git_hash())

    registry = Registry(cfg)
    server = FleetServer(cfg, registry, logger)
    await server.start(args.bind)
    reassembler = Reassembler(cfg, server.send_data)
    tail = Tail(cfg)
    tail.start()
    timesync = TimeSync(cfg, server.send_sync)
    scheduler = Scheduler(cfg, registry, policy, reward, reassembler, tail, server, logger)
    scheduler.collect_warmup = args.mode == "warmup"
    server.scheduler, server.reassembler, server.timesync = scheduler, reassembler, timesync
    timesync.start(registry)

    print(f"[experiment] run {run_id}: waiting for {len(cfg.nodes)} nodes ...")
    while len(registry.alive()) < len(cfg.nodes):
        await asyncio.sleep(0.2)
    await asyncio.sleep(args.settle)
    print(f"[experiment] all nodes up; starting {args.mode}")

    trace = load_trace(args.trace)
    ttask = asyncio.ensure_future(trace_task(trace, cfg, registry, server, scheduler))

    node_ids = [n.node_id for n in cfg.nodes]
    if args.mode == "warmup":
        stop = lambda: reward.warmup_complete(node_ids)
    elif args.mode == "sweep":
        stop = lambda: all(policy.done(n) for n in node_ids)
    else:
        target = args.min_ok or cfg.experiment.learning_min_ok_per_node
        stop = lambda: all(scheduler.ok_count.get(n, 0) >= target for n in node_ids)

    try:
        await scheduler.run(stop, max_duration_s=args.max_duration)
    finally:
        ttask.cancel()
        timesync.stop()
        tail.stop()
        server.close()

    if args.mode == "warmup":
        reward.finalize_warmup()
        print(f"[experiment] T_ref written: {reward.t_ref}")
    if args.mode == "sweep":
        logger.close()
        table = derive_b3_table(db_dir / f"{run_id}.db")
        (db_dir / "b3_table.json").write_text(json.dumps(table, indent=2))
        print(f"[experiment] B3 best-static table: {table}")
    else:
        logger.close()

    ok = sum(scheduler.ok_count.values())
    att = sum(scheduler.attempts.values())
    print(f"[experiment] done: {ok}/{att} OK ({att - ok} failed) -> {db_dir / (run_id + '.db')}")
    return run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["warmup", "sweep", "baseline", "learn"])
    ap.add_argument("--policy", default="eps", choices=["eps", "b0", "b2", "b3"])
    ap.add_argument("--trace", default=None, help="traces/*.yaml (D2 phases)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-ok", type=int, default=None, help="OK requests per node to stop at")
    ap.add_argument("--max-duration", type=float, default=None, help="hard wall-clock cap (s)")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--settle", type=float, default=2.0)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
