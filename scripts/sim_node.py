"""Protocol-faithful simulated fleet nodes (dev machine; NOT hardware)."""

import argparse
import asyncio
import json
import pathlib
import random
import sys
import time
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator import protocol as P          # noqa: E402
from orchestrator.config import load_config     # noqa: E402

JITTER = 0.08  # +/-8 % multiplicative on all synthetic times


class _Proto(asyncio.DatagramProtocol):
    def __init__(self, handler):
        self.handler = handler

    def datagram_received(self, data, addr):
        try:
            self.handler(P.unpack(data))
        except ValueError:
            pass

    def error_received(self, exc):  # ICMP refusals between experiment runs
        pass


class SimNode:
    def __init__(self, cfg, node_id: int, server: str, loss: float, seed: int, assets_dir: pathlib.Path):
        self.cfg = cfg
        self.node = cfg.node(node_id)
        self.tier = self.node.tier
        self.rng = random.Random(seed * 1000 + node_id)
        self.loss = loss
        self.server = server
        self.rssi = cfg.sim.rssi_dbm[self.tier]
        self.busy = False
        self.current_req = None
        self.serve_task = None
        self.result_evt = asyncio.Event()
        self.assets_dir = assets_dir
        self.interp = {}
        self.ctrl = self.data = None

    # setup
    async def start(self):
        loop = asyncio.get_running_loop()
        ports = self.cfg.network.ports
        self.ctrl, _ = await loop.create_datagram_endpoint(
            lambda: _Proto(self._on_ctrl), remote_addr=(self.server, ports.control))
        self.data, _ = await loop.create_datagram_endpoint(
            lambda: _Proto(self._on_data), remote_addr=(self.server, ports.data))
        self._load_assets()
        self._load_heads()
        asyncio.ensure_future(self._heartbeat_loop())
        print(f"[sim {self.node.node_id}] tier {self.tier} up (rssi {self.rssi}, loss {self.loss})")

    def _load_assets(self):
        m = json.loads((self.assets_dir / "manifest.json").read_text())
        self.jpegs = [(self.assets_dir / im["jpg"]).read_bytes() for im in m["images"]]
        self.raws = [(self.assets_dir / im["raw"]).read_bytes() for im in m["images"]]

    def _load_heads(self):
        from tensorflow.lite.python.interpreter import Interpreter

        art = ROOT / self.cfg.model.artifacts_dir
        for arm in self.cfg.arms_for_tier(self.tier):
            if arm == self.cfg.cuts[0].name:
                continue
            it = Interpreter(model_path=str(art / f"head_{arm}.tflite"))
            it.allocate_tensors()
            self.interp[arm] = it

    # helpers
    def _jit(self, ms: float) -> float:
        return ms * self.rng.uniform(1 - JITTER, 1 + JITTER) / 1000

    def _goodput(self) -> float:
        gp = self.cfg.sim.goodput_bytes_per_s
        return gp[("good", "mid", "bad")[self.cfg.rssi_bin(self.rssi)]]

    async def _send_frag(self, datagram: bytes):
        await asyncio.sleep(len(datagram) / self._goodput())  # airtime (spent even if dropped)
        if self.rng.random() >= self.loss:
            self.data.sendto(datagram)

    # rx
    def _on_ctrl(self, pkt: P.Packet):
        nid = self.node.node_id
        if pkt.ptype == P.ASSIGN:
            if self.busy:
                if pkt.req_id == self.current_req:  # duplicate ASSIGN -> re-ACK
                    self.ctrl.sendto(P.pack(P.ASSIGN_ACK, nid, pkt.req_id))
                return
            self.ctrl.sendto(P.pack(P.ASSIGN_ACK, nid, pkt.req_id))
            (action_idx,) = P.ASSIGN_S.unpack_from(pkt.payload)
            self.serve_task = asyncio.ensure_future(self._serve(pkt.req_id, action_idx))
        elif pkt.ptype == P.RESULT and pkt.req_id == self.current_req:
            self.result_evt.set()
        elif pkt.ptype == P.ABORT and pkt.req_id == self.current_req:
            if self.serve_task:  # control-channel ABORT (all-frags-lost fallback)
                self.serve_task.cancel()

    def _on_data(self, pkt: P.Packet):
        if pkt.req_id != self.current_req:
            return
        if pkt.ptype == P.NACK:
            asyncio.ensure_future(self._retransmit(P.parse_nack(pkt.payload)))
        elif pkt.ptype == P.ABORT:
            if self.serve_task:
                self.serve_task.cancel()

    # request pipeline
    async def _serve(self, req_id: int, action_idx: int):
        cfg, nid = self.cfg, self.node.node_id
        self.busy, self.current_req = True, req_id
        self.result_evt.clear()
        self.frags = {}
        try:
            arm = cfg.cuts[action_idx].name
            idx = req_id % len(self.jpegs)
            cap_s = self._jit(cfg.sim.capture_ms[self.node.input])
            await asyncio.sleep(cap_s)
            if action_idx == 0:  # k0: JPEG on the air
                edge_s = self._jit(cfg.sim.jpeg_encode_ms[self.node.input])
                await asyncio.sleep(edge_s)
                payload = self.jpegs[idx]
            else:  # run the real head, pace by the simulated edge time
                edge_s = self._jit(cfg.sim.t_edge_ms[self.tier][arm])
                it = self.interp[arm]
                d_in = it.get_input_details()[0]
                x = np.frombuffer(self.raws[idx], dtype=np.int8).reshape(d_in["shape"])
                it.set_tensor(d_in["index"], x)
                it.invoke()
                payload = it.get_tensor(it.get_output_details()[0]["index"]).tobytes()
                await asyncio.sleep(edge_s)
            trailer = P.TRAILER_S.pack(zlib.crc32(payload), int(cap_s * 1e6), int(edge_s * 1e6))
            datagrams = P.fragment_tensor(nid, req_id, payload, trailer,
                                          cfg.network.fragment_payload_bytes)
            self.frags = dict(enumerate(datagrams))
            for d in datagrams:
                await self._send_frag(d)
            try:  # wait for RESULT (or ABORT cancels us); then idle
                await asyncio.wait_for(self.result_evt.wait(), 3)
            except asyncio.TimeoutError:
                pass
        except asyncio.CancelledError:
            pass  # ABORT: drop straight back to idle (soft-reset semantics, section 4.7)
        finally:
            self.busy, self.current_req = False, None

    async def _retransmit(self, missing: set[int]):
        for i in sorted(missing):
            if i in self.frags and self.current_req is not None:
                await self._send_frag(self.frags[i])

    # heartbeat
    async def _heartbeat_loop(self):
        nid = self.node.node_id
        while True:
            hb = P.Heartbeat(rssi=self.rssi, free_heap=150_000, temp=30, fw_hash=0x51D0DE,
                             busy=int(self.busy), boot_count=0)
            self.ctrl.sendto(P.pack_heartbeat(nid, hb))
            await asyncio.sleep(self.cfg.registry.heartbeat_period_s)


async def amain(args):
    cfg = load_config()
    assets = ROOT / "model" / ("assets_dev" if (ROOT / "model/assets_dev").exists() else "assets")
    node_ids = [n.node_id for n in cfg.nodes] if args.nodes == "all" \
        else [int(x) for x in args.nodes.split(",")]
    nodes = [SimNode(cfg, nid, args.server, args.loss, args.seed, assets) for nid in node_ids]
    for n in nodes:
        await n.start()
    await asyncio.Event().wait()  # run until killed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1")
    ap.add_argument("--nodes", default="all", help="'all' or comma list of node_ids")
    ap.add_argument("--loss", type=float, default=0.0, help="uplink fragment loss probability")
    ap.add_argument("--seed", type=int, default=1)
    asyncio.run(amain(ap.parse_args()))
