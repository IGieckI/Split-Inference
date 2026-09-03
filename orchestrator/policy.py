"""Policies behind the 2-method interface"""

import random
from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Context:
    tier: str
    rssi_bin: int
    load: int = 0  # fleet in-flight at decision time; logged (ctx_load), not a
                   # Phase-1 policy feature (becomes one in Phase 2, section 11.11)

    def cell(self):
        """Phase-1 context cell: tier x RSSI bin => <= 6 cells."""
        return (self.tier, self.rssi_bin)


class Policy:
    name = "base"

    def select(self, node_id: int, ctx: Context) -> str:
        raise NotImplementedError

    def update(self, node_id: int, ctx: Context, arm: str, reward: float):
        pass


class EpsGreedy(Policy):
    """Per-node epsilon-greedy contextual bandit"""

    def __init__(self, cfg: Config, seed: int = 0):
        self.name = f"eps_greedy_s{seed}"
        self.cfg = cfg
        self.eps = cfg.policy.epsilon
        self.rng = random.Random(seed)
        self.t: dict[int, int] = {}          # per-node round counter
        self.q: dict[tuple, dict[str, tuple[float, int]]] = {}  # cell -> arm -> (mean, n)
        self.arms_by_node = {n.node_id: cfg.arms_for_tier(n.tier) for n in cfg.nodes}

    def _eps(self, node_id: int) -> float:
        t = self.t.get(node_id, 0)
        return max(self.eps.floor, self.eps.start * self.eps.decay ** t)

    def select(self, node_id: int, ctx: Context) -> str:
        arms = self.arms_by_node[node_id]
        self.t[node_id] = self.t.get(node_id, 0) + 1
        cell = self.q.setdefault(ctx.cell(), {})
        untried = [a for a in arms if a not in cell]
        if untried:
            return self.rng.choice(untried)
        if self.rng.random() < self._eps(node_id):
            return self.rng.choice(arms)
        return max(arms, key=lambda a: cell[a][0])

    def update(self, node_id: int, ctx: Context, arm: str, reward: float):
        cell = self.q.setdefault(ctx.cell(), {})
        mean, n = cell.get(arm, (0.0, 0))
        cell[arm] = (mean + (reward - mean) / (n + 1), n + 1)


class FixedPerNode(Policy):
    """One fixed arm per node"""

    def __init__(self, name: str, arm_by_node: dict[int, str]):
        self.name = name
        self.arm_by_node = arm_by_node

    def select(self, node_id: int, ctx: Context) -> str:
        return self.arm_by_node[node_id]


class Sweep(Policy):
    """E1-lite static sweep"""

    def __init__(self, cfg: Config):
        self.name = "sweep"
        self.reqs_per_arm = cfg.experiment.sweep_reqs_per_arm
        self.arms_by_node = {n.node_id: cfg.arms_for_tier(n.tier) for n in cfg.nodes}
        self.count: dict[int, int] = {}

    def select(self, node_id: int, ctx: Context) -> str:
        i = self.count.get(node_id, 0)
        self.count[node_id] = i + 1
        arms = self.arms_by_node[node_id]
        return arms[min(i // self.reqs_per_arm, len(arms) - 1)]

    def done(self, node_id: int) -> bool:
        return self.count.get(node_id, 0) >= self.reqs_per_arm * len(self.arms_by_node[node_id])


def make_policy(kind: str, cfg: Config, seed: int = 0, b3_table: dict[int, str] | None = None) -> Policy:
    if kind == "eps":
        return EpsGreedy(cfg, seed)
    if kind == "sweep":
        return Sweep(cfg)
    if kind == "b0":
        return FixedPerNode("b0_full_offload", {n.node_id: cfg.cuts[0].name for n in cfg.nodes})
    if kind == "b2":
        return FixedPerNode("b2_always_deepest", {n.node_id: cfg.arms_for_tier(n.tier)[-1] for n in cfg.nodes})
    if kind == "b3":
        assert b3_table, "B3 needs the sweep-derived best-static table"
        return FixedPerNode("b3_best_static", b3_table)
    raise ValueError(f"unknown policy {kind}")
