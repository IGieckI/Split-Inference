"""Split-point policies: per-tier feasibility clamping and sweep blocking."""

import pytest

from orchestrator.config import load_config
from orchestrator.policy import Sweep, clamp, make_policy

CFG = load_config()


def test_fixed_cut_policies_per_node():
    assert make_policy("k0", CFG).select(11) == "k0"
    assert make_policy("k_shallow", CFG).select(11) == "k_shallow"
    assert make_policy("k_deep", CFG).select(11) == "k_deep"


def test_infeasible_cut_clamps_to_deepest_the_tier_can_run():
    # A tier that cannot host a cut must not be handed it.
    for tier in CFG.tiers:
        feasible = CFG.cuts_for_tier(tier)
        clamped = clamp(CFG, tier, "k_deep")
        assert clamped in feasible
        assert CFG.cut_index(clamped) == max(CFG.cut_index(c) for c in feasible)
    assert clamp(CFG, "A", "k_deep") == "k_deep"
    assert make_policy("k_deep", CFG).select(31) in CFG.cuts_for_tier("C")


def test_best_policy_replays_the_sweep_table():
    p = make_policy("best", CFG, best_table={11: "k_deep", 21: "k0", 31: "k0"})
    assert (p.select(11), p.select(21), p.select(31)) == ("k_deep", "k0", "k0")


def test_best_policy_requires_a_table():
    with pytest.raises(AssertionError):
        make_policy("best", CFG)


def test_fixed_cut_policies_never_report_done():
    # Only the sweep bounds itself
    assert not make_policy("k0", CFG).done(11)


def test_sweep_runs_equal_blocks_of_every_feasible_cut():
    p = Sweep(CFG)
    n = CFG.experiment.sweep_reqs_per_cut
    cuts = CFG.cuts_for_tier("A")  # the tier with the most feasible cuts
    seq = [p.select(11) for _ in range(n * len(cuts))]
    for i, cut in enumerate(cuts):
        assert seq[i * n:(i + 1) * n] == [cut] * n
    assert p.done(11) and not p.done(31)


def test_sweep_stops_dispatching_once_a_node_is_finished():
    # A fast node must not keep drawing requests while a slow one catches up
    p = Sweep(CFG)
    for _ in range(CFG.experiment.sweep_reqs_per_cut * len(CFG.cuts_for_tier("A"))):
        p.select(11)
    assert p.done(11) and not p.wants(11)


def test_sweep_measures_one_node_at_a_time():
    # Blocks on different nodes must not overlap
    p = Sweep(CFG)
    first = p.order[0]
    assert p.wants(first)
    assert not any(p.wants(n) for n in p.order[1:])

    for _ in range(CFG.experiment.sweep_reqs_per_cut * len(CFG.cuts_for_tier(
            next(n.tier for n in CFG.nodes if n.node_id == first)))):
        p.select(first)

    assert p.active() == p.order[1]        # the next node takes over
    assert not p.wants(first)
