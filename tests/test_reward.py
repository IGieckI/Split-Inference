"""Reward normalization, warm-up T_ref, failure penalty, T_max bound."""

from orchestrator.config import load_config
from orchestrator.reward import WARMUP_T_MAX_MS, RewardComputer

CFG = load_config()


def test_warmup_median_and_persistence(tmp_path):
    rc = RewardComputer(CFG, tmp_path / "t_ref.json")
    assert rc.t_max_ms(11) == WARMUP_T_MAX_MS  # no T_ref yet
    for v in [100, 200, 300, 400, 1000]:  # median 300
        rc.observe_warmup(11, v)
    assert not rc.warmup_complete([11])  # needs 20
    for _ in range(15):
        rc.observe_warmup(11, 300)
    assert rc.warmup_complete([11])
    rc.finalize_warmup()
    assert rc.t_ref[11] == 300

    rc2 = RewardComputer(CFG, tmp_path / "t_ref.json")  # reload from disk
    assert rc2.t_ref[11] == 300
    assert rc2.t_max_ms(11) == CFG.protocol.t_max_multiplier * 300


def test_reward_normalization_and_penalty(tmp_path):
    rc = RewardComputer(CFG, tmp_path / "t_ref.json")
    rc.t_ref = {11: 200.0}
    assert rc.reward(11, 200.0, ok=True) == -1.0   # k0-parity latency -> -1
    assert rc.reward(11, 100.0, ok=True) == -0.5   # 2x faster -> better
    assert rc.reward(11, None, ok=False) == CFG.reward.failure_penalty
    assert CFG.reward.failure_penalty == -3.0      # section 11.10-B
