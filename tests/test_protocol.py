"""Wire-protocol golden bytes + roundtrips."""

import zlib

import pytest

from orchestrator import protocol as P


def test_header_golden_bytes():
    pkt = P.pack(P.ASSIGN, 11, 7, P.ASSIGN_S.pack(2))
    assert pkt == bytes([0xF5, 2, 11, 0, 7, 0, 0, 0, 2])  # LE u16 node, LE u32 req


def test_heartbeat_roundtrip_and_size():
    hb = P.Heartbeat(rssi=-61, free_heap=123456, temp=42, fw_hash=0xDEADBEEF, busy=1, boot_count=3)
    pkt = P.unpack(P.pack_heartbeat(21, hb))
    assert pkt.ptype == P.HEARTBEAT and pkt.node_id == 21
    assert P.parse_heartbeat(pkt.payload) == hb
    assert P.HEARTBEAT_S.size == 13  # pinned: firmware struct must pack to this


def test_bad_magic_and_short_packet():
    with pytest.raises(ValueError):
        P.unpack(bytes([0x00]) + bytes(8))
    with pytest.raises(ValueError):
        P.unpack(b"\xf5\x01")


def test_fragmentation_roundtrip():
    tensor = bytes(range(256)) * 12  # 3072 B
    trailer = P.TRAILER_S.pack(zlib.crc32(tensor), 1111, 2222)
    frags = P.fragment_tensor(11, 9, tensor, trailer, max_chunk=1400)
    assert len(frags) == 3  # 3084 B -> 1400+1400+284
    chunks = {}
    for f in frags:
        pkt = P.unpack(f)
        assert pkt.ptype == P.DATA_FRAG and pkt.req_id == 9
        idx, total, chunk = P.parse_frag(pkt.payload)
        assert total == 3
        chunks[idx] = chunk
    blob = b"".join(chunks[i] for i in range(3))
    assert blob[:-12] == tensor
    assert P.TRAILER_S.unpack(blob[-12:])[0] == zlib.crc32(tensor)


def test_nack_bitmap_roundtrip():
    missing = {1, 5, 10}
    pkt = P.unpack(P.pack_nack(11, 9, missing, frag_total=12))
    assert pkt.ptype == P.NACK
    assert P.parse_nack(pkt.payload) == missing


def test_nack_bitmap_512_frags_fits_64_bytes():
    pkt = P.unpack(P.pack_nack(11, 9, {0, 511}, frag_total=512))
    assert len(pkt.payload) == 2 + 64  # u16 total + 64 B bitmap
    assert P.parse_nack(pkt.payload) == {0, 511}


def test_crc32_vector_matches_esp_rom_crc32_le():
    # CRC-32/ISO-HDLC check value
    assert zlib.crc32(b"123456789") == 0xCBF43926
