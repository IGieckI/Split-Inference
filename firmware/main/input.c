#include "input.h"

#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_timer.h"
#include "fleet_config.h"

static const char *TAG = "input";

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

esp_err_t input_get_jpeg(uint32_t idx, uint8_t *buf, size_t *len, uint32_t *cap_us)
{
    int64_t t0 = esp_timer_get_time();
    esp_err_t err = read_file("/assets/img%02u.jpg", idx, buf, *len, len);
    *cap_us = (uint32_t)(esp_timer_get_time() - t0);
    return err;
}
