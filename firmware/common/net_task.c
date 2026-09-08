#include "net_task.h"

#include <string.h>
#include <sys/param.h>

#include "esp_log.h"
#include "esp_rom_crc.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "fleet_config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "input.h"
#include "lwip/sockets.h"
#include "ml_task.h"
#include "proto.h"
#include "sdkconfig.h"
#include "soc/soc_caps.h"
#include "wifi.h"

#if SOC_TEMPERATURE_SENSOR_SUPPORTED
#include "driver/temperature_sensor.h"
static temperature_sensor_handle_t s_temp_sensor;
#endif

static const char *TAG = "net";

#define NODE_ID CONFIG_FLEETSPLIT_NODE_ID
#ifndef FLEET_FW_HASH
#define FLEET_FW_HASH 0
#endif

#ifdef CONFIG_FLEETSPLIT_TIER_C
#define BLOB_CAP (12 * 1024)
#else
#define BLOB_CAP (16 * 1024)
#endif

#define RESULT_BIT BIT0
#define ABORT_BIT BIT1

typedef struct {
    uint32_t req_id;
    uint8_t action;
} assign_t;

static int s_sock_ctrl = -1, s_sock_data = -1;
static struct sockaddr_in s_dst_ctrl, s_dst_data;
static QueueHandle_t s_assign_q;
static EventGroupHandle_t s_evt;
static uint16_t s_boot_count;

/* Current request; blob = tensor||trailer */
static volatile bool s_busy;
static volatile uint32_t s_cur_req;
static uint8_t s_blob[BLOB_CAP];
static size_t s_blob_len;
static uint16_t s_frag_total;
static int8_t s_raw[FLEET_INPUT_BYTES];

/* tx */
static void fill_hdr(uint8_t *p, uint8_t type, uint32_t req_id)
{
    pkt_hdr_t h = {FLEET_MAGIC, type, NODE_ID, req_id};
    memcpy(p, &h, sizeof(h));
}

static void ctrl_send(uint8_t type, uint32_t req_id, const void *payload, size_t n)
{
    uint8_t pkt[sizeof(pkt_hdr_t) + 16];
    fill_hdr(pkt, type, req_id);
    if (n) memcpy(pkt + sizeof(pkt_hdr_t), payload, n);
    sendto(s_sock_ctrl, pkt, sizeof(pkt_hdr_t) + n, 0,
           (struct sockaddr *)&s_dst_ctrl, sizeof(s_dst_ctrl));
}

static void send_frag(uint16_t idx)
{
    uint8_t pkt[sizeof(pkt_hdr_t) + sizeof(pkt_frag_hdr_t) + FLEET_FRAG_PAYLOAD];
    size_t off = (size_t)idx * FLEET_FRAG_PAYLOAD;
    if (off >= s_blob_len) return;
    size_t n = MIN((size_t)FLEET_FRAG_PAYLOAD, s_blob_len - off);
    fill_hdr(pkt, PKT_DATA_FRAG, s_cur_req);
    pkt_frag_hdr_t fh = {idx, s_frag_total};
    memcpy(pkt + sizeof(pkt_hdr_t), &fh, sizeof(fh));
    memcpy(pkt + sizeof(pkt_hdr_t) + sizeof(fh), s_blob + off, n);
    sendto(s_sock_data, pkt, sizeof(pkt_hdr_t) + sizeof(fh) + n, 0,
           (struct sockaddr *)&s_dst_data, sizeof(s_dst_data));
}

static void send_heartbeat(void)
{
    pkt_heartbeat_t hb = {
        .rssi = fleet_wifi_rssi(),
        .free_heap = esp_get_free_heap_size(),
        .temp = INT8_MIN,
        .fw_hash = FLEET_FW_HASH,
        .busy = s_busy ? 1 : 0,
        .boot_count = s_boot_count,
    };
#if SOC_TEMPERATURE_SENSOR_SUPPORTED
    float c;
    if (s_temp_sensor && temperature_sensor_get_celsius(s_temp_sensor, &c) == ESP_OK) {
        hb.temp = (int8_t)c;
    }
#endif
    ctrl_send(PKT_HEARTBEAT, 0, &hb, sizeof(hb));
}

/* ml pipeline */
/* section 11.5 ml_task (input_task folded in: the section 4 pipeline is sequential). */
static void ml_pipeline_task(void *arg)
{
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));
    for (;;) {
        esp_task_wdt_reset();
        assign_t a;
        if (xQueueReceive(s_assign_q, &a, pdMS_TO_TICKS(1000)) != pdTRUE) continue;

        uint32_t cap_us = 0, edge_us = 0;
        size_t body_len = 0;
        bool ok;
        if (a.action == FLEET_CUT_K0) {
            size_t len = BLOB_CAP - sizeof(pkt_trailer_t);
            ok = input_get_jpeg(a.req_id, s_blob, &len, &cap_us) == ESP_OK;
            body_len = len;
        } else {
            ok = input_get_raw(a.req_id, s_raw, &cap_us) == ESP_OK;
            if (ok) {
                const int8_t *out;
                size_t out_len;
                ok = ml_run(a.action, s_raw, &out, &out_len, &edge_us) == ESP_OK
                     && out_len <= BLOB_CAP - sizeof(pkt_trailer_t);
                if (ok) {
                    memcpy(s_blob, out, out_len);
                    body_len = out_len;
                }
            }
        }
        esp_task_wdt_reset();
        if (!ok) {
            ESP_LOGE(TAG, "req %lu action %u: input/inference failed",
                     (unsigned long)a.req_id, a.action);
            s_busy = false; /* orchestrator's T_max handles it (penalty) */
            continue;
        }

        pkt_trailer_t tr = {
            .crc32 = esp_rom_crc32_le(0, s_blob, body_len),  /* == zlib.crc32 */
            .t_capture_us = cap_us,
            .t_edge_us = edge_us,
        };
        memcpy(s_blob + body_len, &tr, sizeof(tr));
        s_blob_len = body_len + sizeof(tr);
        s_frag_total = (s_blob_len + FLEET_FRAG_PAYLOAD - 1) / FLEET_FRAG_PAYLOAD;

        for (uint16_t i = 0; i < s_frag_total; i++) send_frag(i);

        /* SENDING -> wait for RESULT (or ABORT / idle timeout) -> IDLE */
        xEventGroupWaitBits(s_evt, RESULT_BIT | ABORT_BIT, pdTRUE, pdFALSE,
                            pdMS_TO_TICKS(10000));
        s_busy = false;
    }
}

/* rx */
static void on_ctrl(const uint8_t *buf, int n)
{
    const pkt_hdr_t *h = (const pkt_hdr_t *)buf;
    if (h->type == PKT_ASSIGN && n >= (int)(sizeof(*h) + sizeof(pkt_assign_t))) {
        const pkt_assign_t *as = (const pkt_assign_t *)(buf + sizeof(*h));
        if (s_busy) {
            if (h->req_id == s_cur_req) ctrl_send(PKT_ASSIGN_ACK, h->req_id, NULL, 0);
            return; /* new request while busy: ignore, orchestrator caps 1/node */
        }
        s_busy = true;
        s_cur_req = h->req_id;
        xEventGroupClearBits(s_evt, RESULT_BIT | ABORT_BIT);
        ctrl_send(PKT_ASSIGN_ACK, h->req_id, NULL, 0);
        assign_t a = {h->req_id, as->action};
        xQueueSend(s_assign_q, &a, 0);
    } else if (h->type == PKT_RESULT && h->req_id == s_cur_req) {
        xEventGroupSetBits(s_evt, RESULT_BIT);
    } else if (h->type == PKT_ABORT && h->req_id == s_cur_req) {
        xEventGroupSetBits(s_evt, ABORT_BIT); /* control-channel fallback */
    }
}

static void on_data(const uint8_t *buf, int n)
{
    const pkt_hdr_t *h = (const pkt_hdr_t *)buf;
    if (h->req_id != s_cur_req || !s_busy) return;
    if (h->type == PKT_NACK && n > (int)(sizeof(*h) + sizeof(pkt_nack_hdr_t))) {
        const pkt_nack_hdr_t *nh = (const pkt_nack_hdr_t *)(buf + sizeof(*h));
        const uint8_t *bitmap = buf + sizeof(*h) + sizeof(*nh);
        int bitmap_bytes = n - sizeof(*h) - sizeof(*nh);
        uint16_t total = MIN(nh->frag_total, s_frag_total);
        for (uint16_t i = 0; i < total && i / 8 < bitmap_bytes; i++) {
            if (bitmap[i / 8] & (1 << (i % 8))) send_frag(i);
        }
    } else if (h->type == PKT_ABORT) {
        xEventGroupSetBits(s_evt, ABORT_BIT);
    }
}

/* task */
static int bind_udp(uint16_t port)
{
    int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    struct sockaddr_in addr = {
        .sin_family = AF_INET, .sin_port = htons(port),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    bind(s, (struct sockaddr *)&addr, sizeof(addr));
    return s;
}

static void net_task(void *arg)
{
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));
    s_sock_ctrl = bind_udp(FLEET_PORT_CTRL);
    s_sock_data = bind_udp(FLEET_PORT_DATA);

    int64_t next_hb = 0;
    uint8_t buf[sizeof(pkt_hdr_t) + sizeof(pkt_frag_hdr_t) + FLEET_FRAG_PAYLOAD + 64];
    for (;;) {
        esp_task_wdt_reset();
        if (esp_timer_get_time() >= next_hb) {
            send_heartbeat();
            next_hb = esp_timer_get_time() + (int64_t)FLEET_HEARTBEAT_MS * 1000;
        }
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(s_sock_ctrl, &rfds);
        FD_SET(s_sock_data, &rfds);
        int maxfd = MAX(s_sock_ctrl, s_sock_data);
        struct timeval tv = {.tv_sec = 0, .tv_usec = 100 * 1000};
        if (select(maxfd + 1, &rfds, NULL, NULL, &tv) <= 0) continue;

        struct sockaddr_in from;
        socklen_t flen = sizeof(from);
        for (int i = 0; i < 2; i++) {
            int s = (int[]){s_sock_ctrl, s_sock_data}[i];
            if (!FD_ISSET(s, &rfds)) continue;
            int n = recvfrom(s, buf, sizeof(buf), 0, (struct sockaddr *)&from, &flen);
            if (n < (int)sizeof(pkt_hdr_t)) continue;
            const pkt_hdr_t *h = (const pkt_hdr_t *)buf;
            if (h->magic != FLEET_MAGIC || h->node_id != NODE_ID) continue;
            if (s == s_sock_ctrl) on_ctrl(buf, n);
            else on_data(buf, n);
        }
    }
}

void net_start(uint16_t boot_count)
{
    s_boot_count = boot_count;
    s_assign_q = xQueueCreate(1, sizeof(assign_t));
    s_evt = xEventGroupCreate();

#if SOC_TEMPERATURE_SENSOR_SUPPORTED
    temperature_sensor_config_t tcfg = TEMPERATURE_SENSOR_CONFIG_DEFAULT(-10, 80);
    if (temperature_sensor_install(&tcfg, &s_temp_sensor) == ESP_OK) {
        temperature_sensor_enable(s_temp_sensor);
    }
#endif

    struct sockaddr_in dst = {.sin_family = AF_INET};
    inet_pton(AF_INET, FLEET_SERVER_IP, &dst.sin_addr);
    s_dst_ctrl = dst;
    s_dst_ctrl.sin_port = htons(FLEET_PORT_CTRL);
    s_dst_data = dst;
    s_dst_data.sin_port = htons(FLEET_PORT_DATA);

    xTaskCreate(ml_pipeline_task, "ml", 8192, NULL, 4, NULL);
    xTaskCreate(net_task, "net", 6144, NULL, 5, NULL);
    ESP_LOGI(TAG, "node %d up, fw 0x%08x", NODE_ID, (unsigned)FLEET_FW_HASH);
}
