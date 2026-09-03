"""ARQ/reassembly scenarios (highest-defect-density code)"""

import asyncio
import zlib

import pytest

from orchestrator import protocol as P
from orchestrator.config import load_config
from orchestrator.reassembly import Reassembler, ReassemblyError

NODE, REQ, ADDR = 11, 1, ("127.0.0.1", 40000)


def make_frags(tensor: bytes, chunk=1400, t_cap=100, t_edge=200):
    trailer = P.TRAILER_S.pack(zlib.crc32(tensor), t_cap, t_edge)
    return P.fragment_tensor(NODE, REQ, tensor, trailer, chunk)


def feed(r: Reassembler, datagram: bytes):
    r.on_frag(P.unpack(datagram), ADDR, len(datagram))


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def rig():
    cfg = load_config()
    sent = []
    r = Reassembler(cfg, lambda addr, data: sent.append((addr, P.unpack(data))))
    return cfg, r, sent


def test_in_order_completion(rig):
    cfg, r, sent = rig

    async def go():
        fut = r.start(NODE, REQ)
        tensor = bytes(3000)
        for f in make_frags(tensor):
            feed(r, f)
        res = await asyncio.wait_for(fut, 1)
        assert res.data == tensor
        assert res.t_capture_us == 100 and res.t_edge_us == 200
        assert res.n_frags == 3 and res.n_retx == 0
        assert not sent  # no NACK when nothing was lost
        assert res.t_first <= res.t_last

    run(go())


def test_out_of_order_completion(rig):
    cfg, r, sent = rig

    async def go():
        fut = r.start(NODE, REQ)
        tensor = bytes(range(256)) * 20
        frags = make_frags(tensor, chunk=1000)
        for f in reversed(frags):
            feed(r, f)
        res = await asyncio.wait_for(fut, 1)
        assert res.data == tensor and not sent

    run(go())


def test_loss_triggers_nack_then_completes(rig):
    cfg, r, sent = rig

    async def go():
        fut = r.start(NODE, REQ)
        tensor = bytes(5000)
        frags = make_frags(tensor)  # 4 frags
        for i, f in enumerate(frags):
            if i != 2:
                feed(r, f)
        await asyncio.sleep(cfg.protocol.nack_delay_ms / 1000 + 0.03)
        # timer re-arms after each NACK (re-NACK if the retransmission is lost too)
        assert 1 <= len(sent) <= 2
        for addr, nack in sent:
            assert nack.ptype == P.NACK and addr == ADDR
            assert P.parse_nack(nack.payload) == {2}
        feed(r, frags[2])  # retransmission arrives
        res = await asyncio.wait_for(fut, 1)
        assert res.data == tensor and res.n_retx == 0

    run(go())


def test_duplicate_frags_counted_as_retx(rig):
    cfg, r, sent = rig

    async def go():
        fut = r.start(NODE, REQ)
        tensor = bytes(3000)
        frags = make_frags(tensor)
        feed(r, frags[0])
        feed(r, frags[0])  # duplicate
        for f in frags[1:]:
            feed(r, f)
        res = await asyncio.wait_for(fut, 1)
        assert res.n_retx == 1
        assert res.bytes_on_air == sum(len(f) for f in frags) + len(frags[0])

    run(go())


def test_crc_failure(rig):
    cfg, r, sent = rig

    async def go():
        fut = r.start(NODE, REQ)
        tensor = bytes(3000)
        frags = make_frags(tensor)
        bad = bytearray(frags[1])
        bad[20] ^= 0xFF  # corrupt payload byte
        feed(r, frags[0])
        feed(r, bytes(bad))
        feed(r, frags[2])
        with pytest.raises(ReassemblyError) as ei:
            await asyncio.wait_for(fut, 1)
        assert ei.value.status == "CRC"

    run(go())


def test_three_nack_rounds_then_abort(rig):
    cfg, r, sent = rig

    async def go():
        fut = r.start(NODE, REQ)
        frags = make_frags(bytes(5000))
        feed(r, frags[0])  # rest never arrive
        with pytest.raises(ReassemblyError) as ei:
            await asyncio.wait_for(fut, 2)
        assert ei.value.status == "TIMEOUT"
        nacks = [p for _, p in sent if p.ptype == P.NACK]
        aborts = [p for _, p in sent if p.ptype == P.ABORT]
        assert len(nacks) == cfg.protocol.max_nack_rounds
        assert len(aborts) == 1
        assert P.parse_nack(nacks[0].payload) == {1, 2, 3}

    run(go())


def test_stale_frags_ignored(rig):
    cfg, r, sent = rig

    async def go():
        for f in make_frags(bytes(2000)):
            feed(r, f)  # no start() -> unknown req, dropped silently
        assert not r.reqs and not sent

    run(go())
