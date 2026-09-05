"""Closed-loop request generator."""

import asyncio
import time

from . import protocol as P
from .reassembly import ReassemblyError


class Scheduler:
    def __init__(self, cfg, registry, policy, reassembler, tail, server, logger):
        self.cfg = cfg
        self.registry = registry
        self.policy = policy
        self.reassembler = reassembler
        self.tail = tail
        self.server = server
        self.logger = logger
        self.req_counter = 0
        self.acks: dict[tuple, asyncio.Future] = {}
        self.ok_count: dict[int, int] = {}
        self.attempts: dict[int, int] = {}
        self.next_ok_time: dict[int, float] = {}

    # called by server on ASSIGN_ACK
    def on_assign_ack(self, node_id: int, req_id: int):
        fut = self.acks.get((node_id, req_id))
        if fut and not fut.done():
            fut.set_result(True)

    # main loop
    async def run(self, stop_when, max_duration_s: float | None = None):
        t0 = time.monotonic()
        tasks: set[asyncio.Task] = set()
        while not stop_when():
            if max_duration_s and time.monotonic() - t0 > max_duration_s:
                print("[scheduler] max duration reached")
                break
            now = time.monotonic()
            for st in self.registry.ready():
                if self.registry.fleet_in_flight() >= self.cfg.experiment.fleet_inflight_cap:
                    break
                if now < self.next_ok_time.get(st.node_id, 0.0):
                    continue
                st.in_flight += 1
                task = asyncio.create_task(self._handle(st))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            await asyncio.sleep(0.005)
        if tasks:
            await asyncio.wait(tasks, timeout=15)

    async def _handle(self, st):
        cfg = self.cfg
        self.req_counter += 1
        req_id = self.req_counter
        node_id = st.node_id
        arm = self.policy.select(node_id)
        action_idx = cfg.cut_index(arm)
        self.attempts[node_id] = self.attempts.get(node_id, 0) + 1

        row = dict(req_id=req_id, node_id=node_id, tier=st.tier, action=arm,
                   rssi_dbm=st.rssi)
        status, t_total_ms, res, tail_res = "LOST", None, None, None
        tensor_fut = self.reassembler.start(node_id, req_id)
        t_assign = time.monotonic()
        row["t_assign"] = t_assign
        try:
            if await self._assign(st, req_id, action_idx):
                try:
                    res = await asyncio.wait_for(
                        tensor_fut, cfg.protocol.t_max_ms / 1000)
                    tail_res = await self.tail.infer(arm, res.data)
                    t_total_ms = (tail_res.t_done - t_assign) * 1000
                    status = "OK"
                    self.server.send_ctrl(st.ctrl_addr, P.pack(
                        P.RESULT, node_id, req_id, P.RESULT_S.pack(tail_res.pred_class)))
                except ReassemblyError as e:
                    status = e.status  # TIMEOUT (abort already sent) or CRC
                    res = None
                except asyncio.TimeoutError:
                    status = "TIMEOUT"
                    self.reassembler.abort(req_id)  # ABORT via data path if addr known
                    # All-fragments-lost case
                    self.server.send_ctrl(st.ctrl_addr, P.pack(P.ABORT, node_id, req_id))
            else:
                tensor_fut.cancel()  # nobody awaits it on this path
                self.registry.mark_lost(node_id)
                self.reassembler.abort(req_id, notify_device=False)
                print(f"[scheduler] node {node_id}: {cfg.protocol.assign_max_retries} "
                      f"unacked ASSIGNs -> LOST")
        finally:
            st.in_flight -= 1
            self.next_ok_time[node_id] = time.monotonic() + cfg.experiment.request_gap_ms / 1000

        ok = status == "OK"
        if ok:
            self.ok_count[node_id] = self.ok_count.get(node_id, 0) + 1
        if res is not None:
            row.update(t_capture_us=res.t_capture_us, t_edge_us=res.t_edge_us,
                       t_first_frag=res.t_first, t_last_frag=res.t_last,
                       n_frags=res.n_frags, n_retx=res.n_retx,
                       bytes_on_air=res.bytes_on_air)
        if tail_res is not None:
            row.update(t_queue_in=tail_res.t_queue_in, t_queue_out=tail_res.t_queue_out,
                       t_tail_us=tail_res.t_tail_us, pred_class=tail_res.pred_class)
        row.update(t_total_ms=t_total_ms, status=status)
        self.logger.log_request(**row)

    async def _assign(self, st, req_id: int, action_idx: int) -> bool:
        cfg = self.cfg
        key = (st.node_id, req_id)
        for _ in range(cfg.protocol.assign_max_retries):
            fut = asyncio.get_running_loop().create_future()
            self.acks[key] = fut
            try:
                self.server.send_ctrl(st.ctrl_addr, P.pack(
                    P.ASSIGN, st.node_id, req_id, P.ASSIGN_S.pack(action_idx)))
                await asyncio.wait_for(fut, cfg.protocol.assign_ack_timeout_ms / 1000)
                return True
            except asyncio.TimeoutError:
                continue
            finally:
                self.acks.pop(key, None)
        return False
