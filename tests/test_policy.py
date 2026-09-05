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
    # Tier C (node 31) has no k_deep head
    assert clamp(CFG, "C", "k_deep") == "k_shallow"
    assert clamp(CFG, "A", "k_deep") == "k_deep"
    assert make_policy("k_deep", CFG).select(31) == "k_shallow"


def test_best_policy_replays_the_sweep_table():
    p = make_policy("best", CFG, best_table={11: "k_deep", 21: "k0", 31: "k0"})
    assert (p.select(11), p.select(21), p.select(31)) == ("k_deep", "k0", "k0")


def test_best_policy_requires_a_table():
    with pytest.raises(AssertionError):
        make_policy("best", CFG)


def test_sweep_runs_equal_blocks_of_every_feasible_cut():
    p = Sweep(CFG)
    n = CFG.experiment.sweep_reqs_per_cut
    cuts = CFG.cuts_for_tier("C")  # node 31: 2 feasible cuts
    seq = [p.select(31) for _ in range(n * len(cuts))]
    assert seq[:n] == [cuts[0]] * n and seq[n:] == [cuts[1]] * n
    assert p.done(31) and not p.done(11)
