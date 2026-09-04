/* Wire protocol */
#pragma once
#include <stdint.h>

#define FLEET_MAGIC 0xF5

enum {
    PKT_HEARTBEAT = 1,
    PKT_ASSIGN = 2,
    PKT_ASSIGN_ACK = 3,
    PKT_DATA_FRAG = 4,
    PKT_NACK = 5,
    PKT_ABORT = 6,
    PKT_RESULT = 7,
};

typedef struct __attribute__((packed)) {
    uint8_t magic;
    uint8_t type;
    uint16_t node_id;
    uint32_t req_id;
} pkt_hdr_t;

typedef struct __attribute__((packed)) {
    int8_t rssi;
    uint32_t free_heap;
    int8_t temp;          /* INT8_MIN = no sensor (plain ESP32) */
    uint32_t fw_hash;
    uint8_t busy;
    uint16_t boot_count;
} pkt_heartbeat_t;

typedef struct __attribute__((packed)) { uint8_t action; } pkt_assign_t;
typedef struct __attribute__((packed)) { uint16_t frag_idx; uint16_t frag_total; } pkt_frag_hdr_t;
typedef struct __attribute__((packed)) { uint32_t crc32; uint32_t t_capture_us; uint32_t t_edge_us; } pkt_trailer_t;
typedef struct __attribute__((packed)) { uint16_t frag_total; /* bitmap[] follows */ } pkt_nack_hdr_t;
typedef struct __attribute__((packed)) { uint8_t class_id; } pkt_result_t;

_Static_assert(sizeof(pkt_hdr_t) == 8, "hdr must pack to 8 B");
_Static_assert(sizeof(pkt_heartbeat_t) == 13, "heartbeat must pack to 13 B");
_Static_assert(sizeof(pkt_trailer_t) == 12, "trailer must pack to 12 B");
