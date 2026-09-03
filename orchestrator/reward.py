"""Reward"""

import json
import pathlib
import statistics

from .config import Config

# Before T_ref exists (during the warm-up run itself) requests still need an abort bound
WARMUP_T_MAX_MS = 10_000.0


class RewardComputer:
    def __init__(self, cfg: Config, t_ref_path: pathlib.Path):
        self.cfg = cfg
        self.path = t_ref_path
        self.t_ref: dict[int, float] = {}
        self.warmup_samples: dict[int, list[float]] = {}
        if t_ref_path.exists():
            self.t_ref = {int(k): float(v) for k, v in json.loads(t_ref_path.read_text()).items()}

    # warm-up collection
    def observe_warmup(self, node_id: int, t_total_ms: float):
        self.warmup_samples.setdefault(node_id, []).append(t_total_ms)

    def warmup_complete(self, node_ids: list[int]) -> bool:
        need = self.cfg.reward.warmup_runs
        return all(len(self.warmup_samples.get(n, [])) >= need for n in node_ids)

    def finalize_warmup(self):
        self.t_ref = {n: statistics.median(v) for n, v in self.warmup_samples.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.t_ref, indent=2))

    # online use
    def reward(self, node_id: int, t_total_ms: float | None, ok: bool) -> float:
        if not ok:
            return self.cfg.reward.failure_penalty
        return -(t_total_ms / self.t_ref[node_id])

    def t_max_ms(self, node_id: int) -> float:
        if node_id not in self.t_ref:
            return WARMUP_T_MAX_MS
        return self.cfg.protocol.t_max_multiplier * self.t_ref[node_id]
