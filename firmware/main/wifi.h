#pragma once
#include <stdint.h>

void fleet_wifi_start(void);   /* blocks until STA has an IP; retries forever */
int8_t fleet_wifi_rssi(void);  /* device-side view of the link (section 11.10-B) */
