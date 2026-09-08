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

/* Heads linked per tier. */
#if defined(CONFIG_FLEETSPLIT_TIER_C)
static Head *const s_heads = nullptr;
static constexpr int N_HEADS = 0;
#else
static Head s_heads[] = {
    {FLEET_CUT_K_SHALLOW, g_head_k_shallow_tflite, nullptr},
    {FLEET_CUT_K_DEEP, g_head_k_deep_tflite, nullptr},
};
static constexpr int N_HEADS = sizeof(s_heads) / sizeof(s_heads[0]);
#endif

static constexpr size_t ARENA_BYTES = CONFIG_FLEETSPLIT_ARENA_KB * 1024;

/* One arena per head, allocated at boot and never freed. */
#ifdef CONFIG_SPIRAM
static constexpr uint32_t ARENA_CAPS = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
#else
static constexpr uint32_t ARENA_CAPS = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
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

    int usable = 0;
    for (int i = 0; i < N_HEADS; i++) {
        const tflite::Model *model = tflite::GetModel(s_heads[i].data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE(TAG, "head %d schema %lu != %d", i,
                     (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
            return ESP_FAIL;  /* a build error, not a property of the board */
        }
        uint8_t *arena = (uint8_t *)heap_caps_malloc(ARENA_BYTES, ARENA_CAPS);
        if (!arena) {
            ESP_LOGE(TAG, "cut %u UNAVAILABLE: arena alloc failed (%u KB)",
                     s_heads[i].cut_idx, (unsigned)(ARENA_BYTES / 1024));
            continue;
        }
        auto *interp = new tflite::MicroInterpreter(model, s_resolver, arena, ARENA_BYTES);
        if (interp->AllocateTensors() != kTfLiteOk) {
            /* Gate G3: a cut this board cannot host is a RESULT */
            ESP_LOGE(TAG, "cut %u UNAVAILABLE: arena %u KB too small "
                     "(gate G3: drop it from tiers.<T>.cuts in config.yaml)",
                     s_heads[i].cut_idx, (unsigned)(ARENA_BYTES / 1024));
            delete interp;
            heap_caps_free(arena);
            continue;
        }
        s_heads[i].interp = interp;
        usable++;
        /* The gate G3 number - goes into cuts.json arena_bytes */
        ESP_LOGI(TAG, "ARENA cut=%u used=%u of %u B", s_heads[i].cut_idx,
                 (unsigned)interp->arena_used_bytes(), (unsigned)ARENA_BYTES);
    }
    ESP_LOGI(TAG, "%d of %d heads usable - this node serves %s", usable, N_HEADS,
             usable ? "k0 plus the cuts above" : "k0 only (offload-only node)");
    return ESP_OK;
}

esp_err_t ml_run(uint8_t cut_idx, const int8_t *input,
                 const int8_t **out, size_t *out_len, uint32_t *t_us)
{
    for (int i = 0; i < N_HEADS; i++) {
        if (s_heads[i].cut_idx != cut_idx) continue;
        tflite::MicroInterpreter *it = s_heads[i].interp;
        if (!it) break;  /* linked, but this board could not allocate its arena */
        std::memcpy(it->input(0)->data.int8, input, FLEET_INPUT_BYTES);
        int64_t t0 = esp_timer_get_time();
        if (it->Invoke() != kTfLiteOk) return ESP_FAIL;
        *t_us = (uint32_t)(esp_timer_get_time() - t0);
        *out = it->output(0)->data.int8;
        *out_len = it->output(0)->bytes;
        return ESP_OK;
    }
    ESP_LOGE(TAG, "cut %u not available on this node (not linked, or arena too small)",
             cut_idx);
    return ESP_ERR_NOT_FOUND;
}
