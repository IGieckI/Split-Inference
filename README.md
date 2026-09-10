Split inference on three heterogeneous ESP32: each board runs the first *k*
layers of a quantized MobileNetV2 and ships the intermediate activation tensor
over Wi-Fi to a Linux host, which finishes the inference. The project measures
what the choice of *k* costs, across split points and across hardware tiers.

## Layout

```
config.yaml      nodes, cuts, ports, timers, caps
model/           training, quantization, flash assets
slicer/          tflite splitting tool + identity test
firmware/        one ESP-IDF project per board, sources in firmware/common/
orchestrator/    asyncio UDP server: registry, policy, scheduler, reassembly
scripts/         codegen, flashing, host AP setup, sim nodes, preflight
analysis/        figures and tables, queried from the run DBs
report/          LaTeX report + PDF
```

## Firmware

| project | board | node | tier | heads |
|---|---|---|---|---|
| `firmware/esp32s3/` | ESP32-S3 devkit | 11 | A | `k_shallow`, `k_deep` |
| `firmware/esp32cam/` | ESP32-CAM | 21 | B | `k_shallow`, `k_deep` |
| `firmware/esp32/` | plain ESP32 | 31 | C | none, offload only |

```bash
make header       # config.yaml -> firmware/common/fleet_config.h
make fw-all
scripts/flash_all.sh esp32s3=/dev/ttyACM0 esp32cam=/dev/ttyUSB0 esp32=/dev/ttyUSB1
```

Each board prints `ARENA cut=... used=...` at boot for every head it links.
Those numbers decide which split points the tier can host.

## Measuring

The server is this same machine, The Wi-Fi NIC must be able to run as an AP (`iw list`, look for "AP") and its name goes in `config.yaml` as `network.ap_interface`. The AP must be 2.4 GHz since ESP32 radios have no 5 GHz band, so pick the emptiest of channels 1/6/11.

```bash
uv run python scripts/host/gen_ap_configs.py
sudo scripts/host/ap_up.sh
```

Wait for `AP-ENABLED` and a `DHCPACK` per board. If boards associate but never
get an address, suspect the host firewall, firewalld's default `public` zone
permits only `ssh` and `dhcpv6-client` and drops DHCP on UDP 67, which looks
exactly like a working AP that hands out no addresses. `ap_up.sh` moves the
interface into the `trusted` zone while it runs.

Cap the server first, and pin every run in a session identically; runs pinned
differently are not comparable to each other.

```bash
sudo cpupower frequency-set -u 400MHz
make bench-tail
for p in sweep k0 k_shallow k_deep best; do
  taskset -c 0-3 uv run python -m orchestrator.experiment --policy $p
done
make figures
```