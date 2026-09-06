"""Split-point policies."""

from .config import Config


class Policy:
    name = "base"

    def select(self, node_id: int) -> str:
        raise NotImplementedError

    def wants(self, node_id: int) -> bool:
        """Should the scheduler dispatch to this node right now?"""
        return not self.done(node_id)

    def done(self, node_id: int) -> bool:
        """True once this node needs no further requests."""
        return False


class FixedCut(Policy):
    """One fixed cut per node - the four policies under comparison."""

    def __init__(self, name: str, cut_by_node: dict[int, str]):
        self.name = name
        self.cut_by_node = cut_by_node

    def select(self, node_id: int) -> str:
        return self.cut_by_node[node_id]


class Sweep(Policy):
    """Every feasible cut per node"""

    name = "sweep"

    def __init__(self, cfg: Config):
        self.reqs_per_cut = cfg.experiment.sweep_reqs_per_cut
        self.cuts_by_node = {n.node_id: cfg.cuts_for_tier(n.tier) for n in cfg.nodes}
        self.order = [n.node_id for n in cfg.nodes]
        self.count: dict[int, int] = {}

    def active(self) -> int | None:
        """The one node currently being measured; None when the sweep is over."""
        return next((n for n in self.order if not self.done(n)), None)

    def wants(self, node_id: int) -> bool:
        return node_id == self.active()

    def select(self, node_id: int) -> str:
        i = self.count.get(node_id, 0)
        self.count[node_id] = i + 1
        cuts = self.cuts_by_node[node_id]
        return cuts[min(i // self.reqs_per_cut, len(cuts) - 1)]

    def done(self, node_id: int) -> bool:
        return self.count.get(node_id, 0) >= self.reqs_per_cut * len(self.cuts_by_node[node_id])


def clamp(cfg: Config, tier: str, cut: str) -> str:
    """Deepest cut this tier can actually run, at or before `cut`."""
    feasible = cfg.cuts_for_tier(tier)
    if cut in feasible:
        return cut
    wanted = cfg.cut_index(cut)
    return max((c for c in feasible if cfg.cut_index(c) <= wanted),
               key=cfg.cut_index, default=feasible[0])


def make_policy(kind: str, cfg: Config, best_table: dict[int, str] | None = None) -> Policy:
    if kind == "sweep":
        return Sweep(cfg)
    if kind == "best":
        assert best_table, "policy 'best' needs the sweep-derived table (run --policy sweep first)"
        return FixedCut("best", best_table)
    if kind in cfg.cut_names():
        return FixedCut(kind, {n.node_id: clamp(cfg, n.tier, kind) for n in cfg.nodes})
    raise ValueError(f"unknown policy {kind}")
