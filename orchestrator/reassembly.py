"""Per-request fragment reassembly with NACK-based selective repeat."""

import asyncio
import time
import zlib
from dataclasses import dataclass, field

from . import protocol as P
from .config import Config


class ReassemblyError(Exception):
    def __init__(self, status: str, partial: "ReqBuf | None" = None):
        super().__init__(status)
        self.status = status  # TIMEOUT | CRC
        self.partial = partial


@dataclass
class Reassembled:
    data: bytes
    t_capture_us: int
    t_edge_us: int
    t_first: float          # Pi monotonic, first fragment arrival
    t_last: float           # Pi monotonic, completing fragment arrival
    n_frags: int
    n_retx: int             # duplicate fragments received (retransmissions)
    bytes_on_air: int       # all DATA_FRAG datagram bytes seen + NACK bytes sent


@dataclass
class ReqBuf:
    node_id: int
    addr: tuple | None = None
    chunks: dict = field(default_factory=dict)
    total: int | None = None
    t_first: float | None = None
    t_last: float | None = None
    n_rx: int = 0
    n_retx: int = 0
    bytes_on_air: int = 0
    nack_rounds: int = 0
    timer: object = None
    future: asyncio.Future = None


class Reassembler:
    def __init__(self, cfg: Config, send_data_fn, now_fn=time.monotonic):
        """send_data_fn(addr, datagram) on the data socket (NACK/ABORT)."""
        self.cfg = cfg
        self.send = send_data_fn
        self.now = now_fn
        self.reqs: dict[int, ReqBuf] = {}

    def start(self, node_id: int, req_id: int) -> asyncio.Future:
        buf = ReqBuf(node_id=node_id, future=asyncio.get_running_loop().create_future())
        self.reqs[req_id] = buf
        return buf.future

    def abort(self, req_id: int, notify_device: bool = True):
        buf = self.reqs.pop(req_id, None)
        if buf is None:
            return
        if buf.timer:
            buf.timer.cancel()
        if notify_device and buf.addr:
            self.send(buf.addr, P.pack(P.ABORT, buf.node_id, req_id))
        if not buf.future.done():
            buf.future.set_exception(ReassemblyError("TIMEOUT", buf))

    def on_frag(self, pkt: P.Packet, addr: tuple, datagram_len: int):
        buf = self.reqs.get(pkt.req_id)
        if buf is None or pkt.node_id != buf.node_id:
            return  # stale/aborted request or spoofed node - drop
        frag_idx, frag_total, chunk = P.parse_frag(pkt.payload)
        now = self.now()
        buf.addr = addr
        buf.total = frag_total
        buf.t_first = buf.t_first if buf.t_first is not None else now
        buf.t_last = now
        buf.n_rx += 1
        buf.bytes_on_air += datagram_len
        if frag_idx in buf.chunks:
            buf.n_retx += 1
        buf.chunks[frag_idx] = chunk
        if len(buf.chunks) == frag_total:
            self._complete(pkt.req_id, buf)
        else:
            self._arm_timer(pkt.req_id, buf)

    # internals
    def _arm_timer(self, req_id: int, buf: ReqBuf):
        if buf.timer:
            buf.timer.cancel()
        delay = self.cfg.protocol.nack_delay_ms / 1000
        buf.timer = asyncio.get_running_loop().call_later(delay, self._on_timer, req_id)

    def _on_timer(self, req_id: int):
        buf = self.reqs.get(req_id)
        if buf is None or buf.future.done():
            return
        if buf.nack_rounds >= self.cfg.protocol.max_nack_rounds:
            self.abort(req_id)  # sends ABORT, raises TIMEOUT on the future
            return
        missing = set(range(buf.total)) - set(buf.chunks)
        nack = P.pack_nack(buf.node_id, req_id, missing, buf.total)
        buf.bytes_on_air += len(nack)
        buf.nack_rounds += 1
        if buf.addr:
            self.send(buf.addr, nack)
        self._arm_timer(req_id, buf)

    def _complete(self, req_id: int, buf: ReqBuf):
        if buf.timer:
            buf.timer.cancel()
        self.reqs.pop(req_id, None)
        last = buf.chunks[buf.total - 1]
        if len(last) < P.TRAILER_S.size:
            buf.future.set_exception(ReassemblyError("CRC", buf))
            return
        crc, t_capture_us, t_edge_us = P.TRAILER_S.unpack(last[-P.TRAILER_S.size:])
        body = b"".join(buf.chunks[i] for i in range(buf.total - 1)) + last[:-P.TRAILER_S.size]
        if zlib.crc32(body) != crc:
            buf.future.set_exception(ReassemblyError("CRC", buf))
            return
        buf.future.set_result(Reassembled(
            data=body, t_capture_us=t_capture_us, t_edge_us=t_edge_us,
            t_first=buf.t_first, t_last=buf.t_last, n_frags=buf.total,
            n_retx=buf.n_retx, bytes_on_air=buf.bytes_on_air))
