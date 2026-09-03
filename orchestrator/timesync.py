"""NTP-style two-way sync"""

import asyncio
import time
from collections import defaultdict, deque

import numpy as np

from . import protocol as P
from .config import Config


def mono_us() -> int:
    return time.monotonic_ns() // 1000


class TimeSync:
    def __init__(self, cfg: Config, send_fn):
        """send_fn(addr, datagram) on the timesync socket."""
        self.cfg = cfg
        self.send = send_fn
        self.samples = defaultdict(lambda: deque(maxlen=cfg.timesync.fit_window))
        self._task = None

    async def _loop(self, registry):
        while True:
            for st in registry.alive():
                if st.ctrl_addr is None:
                    continue
                addr = (st.ctrl_addr[0], self.cfg.network.ports.timesync)
                self.send(addr, P.pack(P.SYNC_REQ, st.node_id, 0, P.SYNC_REQ_S.pack(mono_us())))
            await asyncio.sleep(self.cfg.timesync.period_s)

    def start(self, registry):
        self._task = asyncio.ensure_future(self._loop(registry))

    def stop(self):
        if self._task:
            self._task.cancel()

    def on_resp(self, node_id: int, payload: bytes):
        t4 = mono_us()
        t1, t2, t3 = P.SYNC_RESP_S.unpack_from(payload)
        offset = ((t2 - t1) + (t3 - t4)) / 2  # device_clock - pi_clock, us
        self.samples[node_id].append((t4, offset))

    def offset_us(self, node_id: int) -> float | None:
        """Linear-fit offset at 'now' (drift-corrected once >=2 samples)."""
        s = self.samples.get(node_id)
        if not s:
            return None
        if len(s) == 1:
            return s[0][1]
        t = np.array([x[0] for x in s], dtype=float)
        o = np.array([x[1] for x in s], dtype=float)
        b, a = np.polyfit(t, o, 1)
        return float(a + b * mono_us())
