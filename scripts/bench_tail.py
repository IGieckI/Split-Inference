"""What the server tail costs on this host, per cut."""

import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.config import ROOT, load_config  # noqa: E402
from orchestrator.tail import Tail  # noqa: E402

WARMUP, ITERS = 10, 100


def cpu_cap_mhz():
    """The frequency ceiling the tail actually runs under"""
    caps = {int(f.read_text()) // 1000 for f in
            pathlib.Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_max_freq")}
    return f"{min(caps)}-{max(caps)} MHz" if caps else "unknown"


def main():
    cfg = load_config()
    tail = Tail(cfg)
    jpeg = (ROOT / "model/assets_dev/img00.jpg").read_bytes()
    cuts = json.loads((ROOT / cfg.model.artifacts_dir / "cuts.json").read_text())["cuts"]

    print(f"cores available: {len(os.sched_getaffinity(0))}, "
          f"cpu max freq: {cpu_cap_mhz()}, {ITERS} iterations per cut")
    for c in cuts:
        blob = jpeg if c["boundary_op"] is None else bytes(c["tensor_bytes"])
        for _ in range(WARMUP):
            tail._infer_sync(c["name"], blob)
        ts = []
        for _ in range(ITERS):
            t0 = time.monotonic_ns()
            tail._infer_sync(c["name"], blob)
            ts.append((time.monotonic_ns() - t0) / 1e6)
        ts.sort()
        print(f"  {c['name']:10} mean {statistics.mean(ts):6.2f} ms  "
              f"p95 {ts[int(0.95 * ITERS)]:6.2f} ms  min {ts[0]:6.2f} ms")


if __name__ == "__main__":
    main()
