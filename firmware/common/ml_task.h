#pragma once
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TFLM runner */
esp_err_t ml_init(void);

/* Run head for global cut index on int8 input (FLEET_INPUT_BYTES). */
esp_err_t ml_run(uint8_t cut_idx, const int8_t *input,
                 const int8_t **out, size_t *out_len, uint32_t *t_us);

#ifdef __cplusplus
}
#endif
