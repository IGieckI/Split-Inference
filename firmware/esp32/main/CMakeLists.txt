# Shared sources (../../common) + the heads this tier can host.
set(SHARED ../../common)
set(MODELS ../../models/generated)

idf_component_register(
    SRCS "${SHARED}/main.c" "${SHARED}/wifi.c" "${SHARED}/net_task.c"
         "${SHARED}/input.c" "${SHARED}/ml_task.cc"
    INCLUDE_DIRS "${SHARED}" "${MODELS}"
    REQUIRES nvs_flash spiffs esp_wifi esp_netif esp_timer driver esp_hw_support)

# Firmware git hash compiled into every heartbeat
execute_process(COMMAND git rev-parse --short=8 HEAD
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE FLEET_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT FLEET_GIT_HASH)
    set(FLEET_GIT_HASH "0")
endif()
target_compile_definitions(${COMPONENT_LIB} PRIVATE FLEET_FW_HASH=0x${FLEET_GIT_HASH})

# Every node carries the same 50 test images in a SPIFFS partition.
# Real VWW assets (model/assets) win over dev synthetic ones when present.
set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../model/assets_dev)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../model/assets/manifest.json)
    set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../model/assets)
endif()
spiffs_create_partition_image(assets ${ASSETS_DIR} FLASH_IN_PROJECT)
