#pragma once
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

/* Input source (input_task role, folded into the ml pipeline) */

esp_err_t input_init(void);

/* Raw int8 96x96x3 into buf */
esp_err_t input_get_raw(uint32_t idx, int8_t *buf, uint32_t *cap_us);

/* JPEG into buf (capacity *len, out actual) */
esp_err_t input_get_jpeg(uint32_t idx, uint8_t *buf, size_t *len,
                         uint32_t *cap_us, uint32_t *enc_us);
