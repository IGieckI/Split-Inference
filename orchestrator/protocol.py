"""Wire protocol"""

import struct
from dataclasses import dataclass

MAGIC = 0xF5

HEARTBEAT = 1
ASSIGN = 2
ASSIGN_ACK = 3
DATA_FRAG = 4
NACK = 5
ABORT = 6
RESULT = 7

HDR = struct.Struct("<BBHI")                 # magic, type, node_id, req_id
HEARTBEAT_S = struct.Struct("<bIbIBH")       # rssi, free_heap, temp, fw_hash, busy, boot_count
ASSIGN_S = struct.Struct("<B")               # action = cut index
FRAG_S = struct.Struct("<HH")                # frag_idx, frag_total (payload chunk follows)
TRAILER_S = struct.Struct("<III")            # crc32, t_capture_us, t_edge_us (end of LAST frag)
RESULT_S = struct.Struct("<B")               # class id


@dataclass
class Packet:
    ptype: int
    node_id: int
    req_id: int
    payload: bytes


@dataclass
class Heartbeat:
    rssi: int
    free_heap: int
    temp: int
    fw_hash: int
    busy: int
    boot_count: int


def pack(ptype: int, node_id: int, req_id: int, payload: bytes = b"") -> bytes:
    return HDR.pack(MAGIC, ptype, node_id, req_id) + payload


def unpack(datagram: bytes) -> Packet:
    if len(datagram) < HDR.size:
        raise ValueError(f"short packet ({len(datagram)} B)")
    magic, ptype, node_id, req_id = HDR.unpack_from(datagram)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:02x}")
    return Packet(ptype, node_id, req_id, datagram[HDR.size:])


def pack_heartbeat(node_id: int, hb: Heartbeat) -> bytes:
    return pack(HEARTBEAT, node_id, 0,
                HEARTBEAT_S.pack(hb.rssi, hb.free_heap, hb.temp, hb.fw_hash, hb.busy, hb.boot_count))


def parse_heartbeat(payload: bytes) -> Heartbeat:
    return Heartbeat(*HEARTBEAT_S.unpack_from(payload))


def pack_frag(node_id: int, req_id: int, frag_idx: int, frag_total: int, chunk: bytes) -> bytes:
    return pack(DATA_FRAG, node_id, req_id, FRAG_S.pack(frag_idx, frag_total) + chunk)


def parse_frag(payload: bytes) -> tuple[int, int, bytes]:
    frag_idx, frag_total = FRAG_S.unpack_from(payload)
    return frag_idx, frag_total, payload[FRAG_S.size:]


def fragment_tensor(node_id: int, req_id: int, tensor: bytes, trailer: bytes, max_chunk: int) -> list[bytes]:
    """Split tensor||trailer into DATA_FRAG datagrams of <= max_chunk payload."""
    blob = tensor + trailer
    chunks = [blob[i: i + max_chunk] for i in range(0, len(blob), max_chunk)] or [b""]
    total = len(chunks)
    return [pack_frag(node_id, req_id, i, total, c) for i, c in enumerate(chunks)]


def pack_nack(node_id: int, req_id: int, missing: set[int], frag_total: int) -> bytes:
    """Bitmap of missing fragment indices (bit i set = please resend frag i)."""
    bitmap = bytearray((frag_total + 7) // 8)
    for i in missing:
        bitmap[i // 8] |= 1 << (i % 8)
    return pack(NACK, node_id, req_id, struct.pack("<H", frag_total) + bytes(bitmap))


def parse_nack(payload: bytes) -> set[int]:
    (frag_total,) = struct.unpack_from("<H", payload)
    bitmap = payload[2:]
    return {i for i in range(frag_total) if bitmap[i // 8] & (1 << (i % 8))}
