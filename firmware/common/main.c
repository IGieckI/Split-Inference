#include "esp_log.h"
#include "fleet_config.h"
#include "input.h"
#include "ml_task.h"
#include "net_task.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "sdkconfig.h"
#include "wifi.h"

#if defined(CONFIG_FLEETSPLIT_TIER_A)
#define TIER "A"
#elif defined(CONFIG_FLEETSPLIT_TIER_B)
#define TIER "B"
#else
#define TIER "C"
#endif

/* Increments and returns the NVS boot counter */
static uint16_t boot_count_init(void)
{
    nvs_handle_t h;
    uint16_t count = 0;
    if (nvs_open("fleet", NVS_READWRITE, &h) == ESP_OK) {
        nvs_get_u16(h, "boot", &count);
        count++;
        nvs_set_u16(h, "boot", count);
        nvs_commit(h);
        nvs_close(h);
    }
    ESP_LOGI("boot", "boot_count=%u", count);
    return count;
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
    uint16_t boots = boot_count_init();
    ESP_LOGI("fleet", "FleetSplit node %d tier " TIER ", boot #%u",
             CONFIG_FLEETSPLIT_NODE_ID, boots);

    ESP_ERROR_CHECK(input_init());
    ESP_ERROR_CHECK(ml_init());   /* arenas + interpreters at boot (section 11.5) */
    fleet_wifi_start();           /* blocks until associated + IP */
    net_start(boots);
}
