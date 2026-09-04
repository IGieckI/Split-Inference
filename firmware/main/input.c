#include "input.h"

#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "fleet_config.h"
#include "sdkconfig.h"

static const char *TAG = "input";

#ifdef CONFIG_FLEETSPLIT_TIER_B
/* Tier B */
#include "esp_camera.h"
#include "img_converters.h"

esp_err_t input_init(void)
{
    camera_config_t cfg = {
        .pin_pwdn = 32, .pin_reset = -1, .pin_xclk = 0,
        .pin_sccb_sda = 26, .pin_sccb_scl = 27,
        .pin_d7 = 35, .pin_d6 = 34, .pin_d5 = 39, .pin_d4 = 36,
        .pin_d3 = 21, .pin_d2 = 19, .pin_d1 = 18, .pin_d0 = 5,
        .pin_vsync = 25, .pin_href = 23, .pin_pclk = 22,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0, .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_RGB565,
        .frame_size = FRAMESIZE_96X96,
        .fb_count = 1,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    };
    esp_err_t err = esp_camera_init(&cfg);
    ESP_LOGI(TAG, "camera init: %s", esp_err_to_name(err));
    return err;
}

static void rgb565_to_int8(const uint8_t *src, int8_t *dst, int n_px)
{
    for (int i = 0; i < n_px; i++) {
        uint16_t px = ((uint16_t)src[2 * i] << 8) | src[2 * i + 1];
        uint8_t r = (px >> 11) << 3, g = ((px >> 5) & 0x3F) << 2, b = (px & 0x1F) << 3;
        dst[3 * i + 0] = (int8_t)(r - 128);
        dst[3 * i + 1] = (int8_t)(g - 128);
        dst[3 * i + 2] = (int8_t)(b - 128);
    }
}

esp_err_t input_get_raw(uint32_t idx, int8_t *buf, uint32_t *cap_us)
{
    (void)idx;
    int64_t t0 = esp_timer_get_time();
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;
    rgb565_to_int8(fb->buf, buf, FLEET_INPUT_SIZE * FLEET_INPUT_SIZE);
    esp_camera_fb_return(fb);
    *cap_us = (uint32_t)(esp_timer_get_time() - t0);
    return ESP_OK;
}

/* Phase-1 deviation from section 11.10-A () */
esp_err_t input_get_jpeg(uint32_t idx, uint8_t *buf, size_t *len,
                         uint32_t *cap_us, uint32_t *enc_us)
{
    (void)idx;
    int64_t t0 = esp_timer_get_time();
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;
    int64_t t1 = esp_timer_get_time();
    uint8_t *jpg = NULL;
    size_t jpg_len = 0;
    bool ok = frame2jpg(fb, FLEET_JPEG_QUALITY, &jpg, &jpg_len);
    esp_camera_fb_return(fb);
    if (!ok || jpg_len > *len) {
        free(jpg);
        return ESP_FAIL;
    }
    memcpy(buf, jpg, jpg_len);
    free(jpg);
    *len = jpg_len;
    *cap_us = (uint32_t)(t1 - t0);
    *enc_us = (uint32_t)(esp_timer_get_time() - t1);
    return ESP_OK;
}

#else
/* Tiers A/C */
#include "esp_spiffs.h"

esp_err_t input_init(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/assets",
        .partition_label = "assets",
        .max_files = 4,
        .format_if_mount_failed = false,
    };
    esp_err_t err = esp_vfs_spiffs_register(&conf);
    size_t total = 0, used = 0;
    if (err == ESP_OK) esp_spiffs_info("assets", &total, &used);
    ESP_LOGI(TAG, "spiffs: %s (%u/%u B)", esp_err_to_name(err), used, total);
    return err;
}

static esp_err_t read_file(const char *fmt, uint32_t idx, uint8_t *buf,
                           size_t cap, size_t *out_len)
{
    char path[48];
    snprintf(path, sizeof(path), fmt, (unsigned)(idx % FLEET_FLASH_IMAGES));
    FILE *f = fopen(path, "rb");
    if (!f) {
        ESP_LOGE(TAG, "missing %s", path);
        return ESP_FAIL;
    }
    size_t n = fread(buf, 1, cap, f);
    fclose(f);
    *out_len = n;
    return n > 0 ? ESP_OK : ESP_FAIL;
}

esp_err_t input_get_raw(uint32_t idx, int8_t *buf, uint32_t *cap_us)
{
    int64_t t0 = esp_timer_get_time();
    size_t n = 0;
    esp_err_t err = read_file("/assets/img%02u.raw", idx, (uint8_t *)buf,
                              FLEET_INPUT_BYTES, &n);
    *cap_us = (uint32_t)(esp_timer_get_time() - t0);
    return (err == ESP_OK && n == FLEET_INPUT_BYTES) ? ESP_OK : ESP_FAIL;
}

esp_err_t input_get_jpeg(uint32_t idx, uint8_t *buf, size_t *len,
                         uint32_t *cap_us, uint32_t *enc_us)
{
    int64_t t0 = esp_timer_get_time();
    esp_err_t err = read_file("/assets/img%02u.jpg", idx, buf, *len, len);
    *cap_us = (uint32_t)(esp_timer_get_time() - t0);
    *enc_us = 0; /* stored pre-encoded (section 11.10-A) */
    return err;
}
#endif
