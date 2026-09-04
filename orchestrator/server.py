"""UDP endpoints + packet dispatch (control 5001 / data 5002)."""

import asyncio

from . import protocol as P


class _Proto(asyncio.DatagramProtocol):
    def __init__(self, handler):
        self.handler = handler

    def datagram_received(self, data, addr):
        try:
            self.handler(P.unpack(data), addr, len(data))
        except ValueError as e:
            print(f"[server] dropped packet from {addr}: {e}")


class FleetServer:
    """Owns both sockets; routes packets to registry/scheduler/reassembler."""

    def __init__(self, cfg, registry, logger):
        self.cfg = cfg
        self.registry = registry
        self.logger = logger
        self.scheduler = None   # wired after construction (mutual reference)
        self.reassembler = None
        self._transports = []

    async def start(self, bind: str):
        loop = asyncio.get_running_loop()
        ports = self.cfg.network.ports
        for port, handler in ((ports.control, self._on_control),
                              (ports.data, self._on_data)):
            transport, _ = await loop.create_datagram_endpoint(
                lambda h=handler: _Proto(h), local_addr=(bind, port))
            self._transports.append(transport)

    def close(self):
        for t in self._transports:
            t.close()

    # tx
    def send_ctrl(self, addr, datagram: bytes):
        if addr:
            self._transports[0].sendto(datagram, addr)

    def send_data(self, addr, datagram: bytes):
        if addr:
            self._transports[1].sendto(datagram, addr)

    # rx
    def _on_control(self, pkt: P.Packet, addr, _size):
        if pkt.ptype == P.HEARTBEAT:
            hb = P.parse_heartbeat(pkt.payload)
            for event in self.registry.update_heartbeat(pkt.node_id, addr, hb):
                print(f"[registry] {event}")
            self.logger.log_heartbeat(pkt.node_id, hb.rssi, hb.free_heap,
                                      hb.temp, hb.boot_count, hb.fw_hash)
        elif pkt.ptype == P.ASSIGN_ACK and self.scheduler:
            self.scheduler.on_assign_ack(pkt.node_id, pkt.req_id)

    def _on_data(self, pkt: P.Packet, addr, size):
        if pkt.ptype == P.DATA_FRAG and self.reassembler:
            self.reassembler.on_frag(pkt, addr, size)
