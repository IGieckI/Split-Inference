# plain ESP32, no PSRAM - tier C, node 31.
# A normal ESP-IDF project: `idf.py build` / `flash` / `monitor` from this folder.
# Sources live one level up in ../common and are shared with the other two
# boards, so a firmware fix is made once, not three times.
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(fleetsplit_esp32)
