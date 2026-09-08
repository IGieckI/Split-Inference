#include "ml_task.h"

#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "fleet_config.h"
#include "models.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "ml";

struct Head {
    uint8_t cut_idx;
    const unsigned char *data;
    tflite::MicroInterpreter *interp;
};

static Head s_heads[] = {
    {FLEET_CUT_K_SHALLOW, g_head_k_shallow_tflite, nullptr},
#if defined(CONFIG_FLEETSPLIT_TIER_A) || defined(CONFIG_FLEETSPLIT_TIER_B)
    {FLEET_CUT_K_DEEP, g_head_k_deep_tflite, nullptr},
#endif
};
static constexpr int N_HEADS = sizeof(s_heads) / sizeof(s_heads[0]);
static constexpr size_t ARENA_BYTES = CONFIG_FLEETSPLIT_ARENA_KB * 1024;

#ifndef CONFIG_SPIRAM
/* Tier C: internal static arena, one head only */
static uint8_t s_static_arena[ARENA_BYTES];
#endif

static tflite::MicroMutableOpResolver<12> s_resolver;

esp_err_t ml_init(void)
{
    s_resolver.AddMul();
    s_resolver.AddAdd();
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddMean();
    s_resolver.AddFullyConnected();
    s_resolver.AddSoftmax();
    s_resolver.AddPad();
    s_resolver.AddAveragePool2D();
    s_resolver.AddReshape();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();

    for (int i = 0; i < N_HEADS; i++) {
        const tflite::Model *model = tflite::GetModel(s_heads[i].data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE(TAG, "head %d schema %lu != %d", i,
                     (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
            return ESP_FAIL;
        }
#ifdef CONFIG_SPIRAM
        uint8_t *arena = (uint8_t *)heap_caps_malloc(
            ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!arena) {
            ESP_LOGE(TAG, "PSRAM arena alloc failed (gate G1!)");
            return ESP_ERR_NO_MEM;
        }
#else
        uint8_t *arena = s_static_arena;  /* single head on tier C */
#endif
        auto *interp = new tflite::MicroInterpreter(model, s_resolver, arena, ARENA_BYTES);
        if (interp->AllocateTensors() != kTfLiteOk) {
            ESP_LOGE(TAG, "AllocateTensors failed for cut %u - arena %u KB too small "
                     "(gate G2: this cut leaves this tier's feasibility set)",
                     s_heads[i].cut_idx, (unsigned)(ARENA_BYTES / 1024));
            return ESP_ERR_NO_MEM;
        }
        s_heads[i].interp = interp;
        /* The gate G1/G2 number - goes into cuts.json arena_bytes */
        ESP_LOGI(TAG, "ARENA cut=%u used=%u of %u B", s_heads[i].cut_idx,
                 (unsigned)interp->arena_used_bytes(), (unsigned)ARENA_BYTES);
    }
    return ESP_OK;
}

esp_err_t ml_run(uint8_t cut_idx, const int8_t *input,
                 const int8_t **out, size_t *out_len, uint32_t *t_us)
{
    for (int i = 0; i < N_HEADS; i++) {
        if (s_heads[i].cut_idx != cut_idx) continue;
        tflite::MicroInterpreter *it = s_heads[i].interp;
        std::memcpy(it->input(0)->data.int8, input, FLEET_INPUT_BYTES);
        int64_t t0 = esp_timer_get_time();
        if (it->Invoke() != kTfLiteOk) return ESP_FAIL;
        *t_us = (uint32_t)(esp_timer_get_time() - t0);
        *out = it->output(0)->data.int8;
        *out_len = it->output(0)->bytes;
        return ESP_OK;
    }
    ESP_LOGE(TAG, "cut %u not linked on this tier", cut_idx);
    return ESP_ERR_NOT_FOUND;
}
