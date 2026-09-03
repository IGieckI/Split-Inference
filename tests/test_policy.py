"""epsilon-greedy bandit math + baseline policies."""

from orchestrator.config import load_config
from orchestrator.policy import Context, EpsGreedy, Sweep, make_policy

CFG = load_config()
CTX_A = Context(tier="A", rssi_bin=0)


def test_epsilon_schedule():
    p = EpsGreedy(CFG)
    eps = CFG.policy.epsilon
    assert p._eps(11) == eps.start  # t=0
    p.t[11] = 100
    assert p._eps(11) == max(eps.floor, eps.start * eps.decay**100)
    p.t[11] = 100000
    assert p._eps(11) == eps.floor


def test_untried_arms_first():
    p = EpsGreedy(CFG, seed=1)
    arms = set(CFG.arms_for_tier("A"))
    picked = set()
    for _ in range(3):
        a = p.select(11, CTX_A)
        picked.add(a)
        p.update(11, CTX_A, a, -1.0)
    assert picked == arms  # all 3 arms tried before any repeat


def test_incremental_mean():
    p = EpsGreedy(CFG)
    p.update(11, CTX_A, "k0", -1.0)
    p.update(11, CTX_A, "k0", -2.0)
    assert p.q[CTX_A.cell()]["k0"] == (-1.5, 2)


def test_pure_greedy_exploits_best_arm():
    cfg = CFG.model_copy(deep=True)
    cfg.policy.epsilon.start = 0.0
    cfg.policy.epsilon.floor = 0.0
    p = EpsGreedy(cfg)
    for arm, r in (("k0", -2.0), ("k_shallow", -1.0), ("k_deep", -3.0)):
        p.update(11, CTX_A, arm, r)
    assert all(p.select(11, CTX_A) == "k_shallow" for _ in range(10))


def test_context_cells_are_independent():
    p = EpsGreedy(CFG)
    ctx_bad = Context(tier="A", rssi_bin=2)
    p.update(11, CTX_A, "k0", -1.0)
    assert CTX_A.cell() in p.q and ctx_bad.cell() not in p.q


def test_fixed_policies():
    b0 = make_policy("b0", CFG)
    b2 = make_policy("b2", CFG)
    assert b0.select(31, CTX_A) == "k0"
    assert b2.select(11, CTX_A) == "k_deep"      # tier A deepest
    assert b2.select(31, CTX_A) == "k_shallow"   # tier C deepest feasible
    b3 = make_policy("b3", CFG, b3_table={11: "k_deep", 21: "k0", 31: "k0"})
    assert b3.select(21, CTX_A) == "k0"


def test_sweep_blocks_and_done():
    p = Sweep(CFG)
    n = CFG.experiment.sweep_reqs_per_arm
    arms = CFG.arms_for_tier("C")  # node 31: 2 arms
    seq = [p.select(31, CTX_A) for _ in range(n * len(arms))]
    assert seq[:n] == [arms[0]] * n and seq[n:] == [arms[1]] * n
    assert p.done(31) and not p.done(11)
