"""section 11.10-D ARQ acceptance harness (CI-able version of the netem test)."""

import asyncio
import pathlib
import random
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator import protocol as P          # noqa: E402
from orchestrator.config import load_config     # noqa: E402
from orchestrator.reassembly import Reassembler, ReassemblyError  # noqa: E402

NODE = 11
SIZES = [876, 4620, 15000]  # k_deep, k_shallow (+trailer they're our real sizes), stress
T_MAX_S = 0.8               # >> 4 NACK rounds of 20 ms; bounds the all-frags-lost case


class _Proto(asyncio.DatagramProtocol):
    def __init__(self, handler):
        self.handler = handler

    def datagram_received(self, data, addr):
        self.handler(P.unpack(data), addr, len(data))

    def error_received(self, exc):
        pass


async def run_level(loss: float, n_tensors: int, seed: int = 7):
    cfg = load_config()
    loop = asyncio.get_running_loop()
    rng = random.Random(seed)
    stats = {"delivered": 0, "corrupted": 0, "aborted": 0, "retx_frags": 0}

    # receiver side: real Reassembler behind a real UDP socket
    reassembler = None
    recv_transport, _ = await loop.create_datagram_endpoint(
        lambda: _Proto(lambda pkt, addr, size: reassembler.on_frag(pkt, addr, size)),
        local_addr=("127.0.0.1", 0))
    recv_addr = recv_transport.get_extra_info("sockname")

    # sender side: fragments with loss injection + NACK-driven retransmission
    frag_store: dict[int, dict[int, bytes]] = {}

    def on_sender_rx(pkt: P.Packet, addr, size):
        if pkt.ptype == P.NACK and pkt.req_id in frag_store:
            for i in sorted(P.parse_nack(pkt.payload)):
                stats["retx_frags"] += 1
                if rng.random() >= loss:
                    send_transport.sendto(frag_store[pkt.req_id][i], recv_addr)

    send_transport, _ = await loop.create_datagram_endpoint(
        lambda: _Proto(on_sender_rx), local_addr=("127.0.0.1", 0))
    # NACK/ABORT go to the fragments' observed source address = sender's socket.
    reassembler = Reassembler(cfg, lambda addr, data: recv_transport.sendto(data, addr))

    for req_id in range(1, n_tensors + 1):
        tensor = rng.randbytes(SIZES[req_id % len(SIZES)])
        trailer = P.TRAILER_S.pack(zlib.crc32(tensor), 1000, 2000)
        frags = P.fragment_tensor(NODE, req_id, tensor, trailer,
                                  cfg.network.fragment_payload_bytes)
        frag_store[req_id] = dict(enumerate(frags))
        fut = reassembler.start(NODE, req_id)
        for f in frags:
            if rng.random() >= loss:
                send_transport.sendto(f, recv_addr)
        try:
            res = await asyncio.wait_for(fut, T_MAX_S)
            if res.data == tensor:
                stats["delivered"] += 1
            else:
                stats["corrupted"] += 1
        except ReassemblyError:
            stats["aborted"] += 1
        except asyncio.TimeoutError:
            reassembler.abort(req_id, notify_device=False)
            stats["aborted"] += 1
        frag_store.pop(req_id, None)

    recv_transport.close()
    send_transport.close()
    return stats


def expected_aborts(loss: float, n: int, max_chunk: int) -> float:
    """3-round-NACK abort model"""
    total = 0.0
    for req_id in range(1, n + 1):
        frags = -(-(SIZES[req_id % len(SIZES)] + P.TRAILER_S.size) // max_chunk)
        total += (1 - (1 - loss**4) ** frags) + loss**frags
    return total


async def amain():
    cfg = load_config()
    chunk = cfg.network.fragment_payload_bytes
    print(f"{'loss':>6} {'tensors':>8} {'delivered':>10} {'aborted':>8} "
          f"{'exp.abort':>10} {'corrupted':>10} {'retx frags':>10}")
    failures = []
    for loss, n in [(0.0, 200), (0.02, 200), (0.05, 200), (0.10, 200), (0.40, 60)]:
        s = await run_level(loss, n)
        exp = expected_aborts(loss, n, chunk)
        print(f"{loss:>6.0%} {n:>8} {s['delivered']:>10} {s['aborted']:>8} "
              f"{exp:>10.1f} {s['corrupted']:>10} {s['retx_frags']:>10}")
        if s["corrupted"]:
            failures.append(f"{loss:.0%}: {s['corrupted']} corrupted deliveries")
        if s["aborted"] > 3 * exp + 3:  # generous stochastic slack around the model
            failures.append(f"{loss:.0%}: {s['aborted']} aborts vs expected {exp:.1f} "
                            "- above the 3-round NACK bound")
        if loss == 0.40 and s["aborted"] == 0:
            failures.append("40%: abort path never fired")
        if s["delivered"] + s["aborted"] != n:
            failures.append(f"{loss:.0%}: request accounting mismatch")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nPASS: zero corrupted deliveries; abort rate within the NACK-bound model; "
          "abort path fires at 40% (section 11.10-D)")


if __name__ == "__main__":
    asyncio.run(amain())
