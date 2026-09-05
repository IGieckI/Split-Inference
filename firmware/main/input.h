#pragma once
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

/* Input source: every node cycles the same 50 stored test images from its SPIFFS assets partition */

esp_err_t input_init(void);

/* Raw int8 96x96x3 into buf; *cap_us = read time. */
esp_err_t input_get_raw(uint32_t idx, int8_t *buf, uint32_t *cap_us);

/* JPEG into buf (capacity *len, out actual) */
esp_err_t input_get_jpeg(uint32_t idx, uint8_t *buf, size_t *len, uint32_t *cap_us);
