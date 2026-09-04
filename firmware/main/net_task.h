#pragma once
#include <stdint.h>

/* Starts net_task (prio 5: sockets, heartbeat, protocol state machine) and ml_task */
void net_start(uint16_t boot_count);
