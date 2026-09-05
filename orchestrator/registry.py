"""Node table built from heartbeats."""

import time
from dataclasses import dataclass, field

from .config import Config
from .protocol import Heartbeat


@dataclass
class NodeState:
    node_id: int
    tier: str
    ctrl_addr: tuple | None = None
    rssi: int = -127
    free_heap: int = 0
    temp: int = 0
    fw_hash: int = 0
    busy: bool = False
    boot_count: int = -1
    last_hb: float = field(default=-1.0)  # monotonic; -1 = never seen
    in_flight: int = 0
    was_lost: bool = True  # starts unseen


class Registry:
    def __init__(self, cfg: Config, now_fn=time.monotonic):
        self.cfg = cfg
        self.now = now_fn
        self.nodes = {n.node_id: NodeState(n.node_id, n.tier) for n in cfg.nodes}

    def update_heartbeat(self, node_id: int, addr: tuple, hb: Heartbeat) -> list[str]:
        """Returns human-readable events: node_up / boot_count_changed."""
        st = self.nodes.get(node_id)
        if st is None:
            return [f"heartbeat from unknown node_id {node_id} at {addr} - ignored"]
        events = []
        if st.was_lost:
            events.append(f"node {node_id} (tier {st.tier}) up, rssi={hb.rssi}")
            st.was_lost = False
        if st.boot_count >= 0 and hb.boot_count != st.boot_count:
            events.append(f"node {node_id} BOOT COUNT {st.boot_count} -> {hb.boot_count} "
                          "(rebooted mid-run: its data for this run is invalid, section 11.10-E)")
        st.ctrl_addr = addr
        st.rssi, st.free_heap, st.temp = hb.rssi, hb.free_heap, hb.temp
        st.fw_hash, st.busy, st.boot_count = hb.fw_hash, bool(hb.busy), hb.boot_count
        st.last_hb = self.now()
        return events

    def is_lost(self, st: NodeState) -> bool:
        return st.last_hb < 0 or (self.now() - st.last_hb) > self.cfg.registry.lost_after_s

    def mark_lost(self, node_id: int):
        """Force LOST until the next heartbeat (e.g., 3 unacked ASSIGNs, section 11.3)."""
        st = self.nodes[node_id]
        st.last_hb = -1.0
        st.was_lost = True

    def ready(self) -> list[NodeState]:
        cap = self.cfg.experiment.node_inflight_cap
        return [st for st in self.nodes.values()
                if not self.is_lost(st) and not st.busy and st.in_flight < cap]

    def alive(self) -> list[NodeState]:
        return [st for st in self.nodes.values() if not self.is_lost(st)]

    def fleet_in_flight(self) -> int:
        return sum(st.in_flight for st in self.nodes.values())
