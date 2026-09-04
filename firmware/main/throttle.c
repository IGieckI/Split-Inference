#include "throttle.h"

#include "esp_log.h"
#include "esp_pm.h"

void throttle_set_mhz(uint16_t mhz)
{
    int f = (mhz <= 80) ? 80 : (mhz <= 160) ? 160 : 240;
    esp_pm_config_t cfg = {
        .max_freq_mhz = f,
        .min_freq_mhz = f,
        .light_sleep_enable = false,
    };
    esp_err_t err = esp_pm_configure(&cfg);
    ESP_LOGI("throttle", "cpu -> %d MHz (%s)", f, esp_err_to_name(err));
}
