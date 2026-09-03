"""Registry: heartbeat lifecycle, LOST threshold, boot-count events, RSSI bins."""

from orchestrator.config import load_config
from orchestrator.protocol import Heartbeat
from orchestrator.registry import Registry

CFG = load_config()
ADDR = ("192.168.4.11", 5001)


def hb(rssi=-50, boot=0, busy=0):
    return Heartbeat(rssi=rssi, free_heap=100000, temp=35, fw_hash=0xABC, busy=busy, boot_count=boot)


def make_registry():
    clock = [1000.0]
    reg = Registry(CFG, now_fn=lambda: clock[0])
    return reg, clock


def test_lifecycle_and_lost():
    reg, clock = make_registry()
    st = reg.nodes[11]
    assert reg.is_lost(st) and not reg.ready()

    events = reg.update_heartbeat(11, ADDR, hb())
    assert any("up" in e for e in events)
    assert not reg.is_lost(st) and st in reg.ready()

    clock[0] += CFG.registry.lost_after_s + 1  # silent > 5 s -> LOST
    assert reg.is_lost(st) and st not in reg.ready()

    reg.update_heartbeat(11, ADDR, hb())
    assert not reg.is_lost(st)


def test_boot_count_change_flagged():
    reg, _ = make_registry()
    reg.update_heartbeat(11, ADDR, hb(boot=1))
    events = reg.update_heartbeat(11, ADDR, hb(boot=2))
    assert any("BOOT COUNT" in e for e in events)  # section 11.10-E data-invalidation signal


def test_busy_and_inflight_gate_ready():
    reg, _ = make_registry()
    reg.update_heartbeat(11, ADDR, hb(busy=1))
    assert reg.nodes[11] not in reg.ready()
    reg.update_heartbeat(11, ADDR, hb(busy=0))
    reg.nodes[11].in_flight = CFG.experiment.node_inflight_cap
    assert reg.nodes[11] not in reg.ready()


def test_unknown_node_ignored():
    reg, _ = make_registry()
    events = reg.update_heartbeat(99, ADDR, hb())
    assert any("unknown" in e for e in events)
    assert 99 not in reg.nodes


def test_rssi_bins():
    assert CFG.rssi_bin(-50) == 0  # good
    assert CFG.rssi_bin(-55) == 0
    assert CFG.rssi_bin(-60) == 1  # mid
    assert CFG.rssi_bin(-70) == 1
    assert CFG.rssi_bin(-80) == 2  # bad
