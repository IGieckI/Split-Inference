# FleetSplit build/test glue. Python side runs through uv.

UV := uv run
IDF_ENV := scripts/idf_env.sh

.PHONY: env model-dev slice-list slice assets-dev header test test-slow arq-test \
        preflight bench-tail figures fw-A fw-B fw-C clean

env:                ## create venv + install pinned deps
	uv sync

model-dev:          ## placeholder int8 model (dev machine; real training: model/train.py on workstation)
	$(UV) python model/make_dev_model.py

slice-list:         ## print candidate cut boundaries (pin chosen ops in config.yaml)
	$(UV) python slicer/slice.py --list

slice:              ## cut model at config.yaml boundaries -> heads/tails/cuts.json/.cc arrays
	$(UV) python slicer/slice.py

assets-dev:         ## 50 synthetic dev images -> JPEG + int8 raw flash assets
	$(UV) python model/make_flash_assets.py --dev

header:             ## config.yaml -> firmware/main/fleet_config.h
	$(UV) python scripts/gen_config_header.py

test:               ## fast unit tests
	$(UV) pytest -q -m "not slow"

test-slow:          ## slice identity test (200 imgs, bit-exact) + ARQ loss sweep
	$(UV) pytest -q -m slow

arq-test:           ## ARQ acceptance: 200 tensors per loss level {0,2,5,10,40}%
	$(UV) python scripts/test_arq_loss.py

preflight:          ## full measurement sequence against simulated nodes + verification
	$(UV) python scripts/preflight.py

bench-tail:         ## what the server tail costs on this host, per cut
	taskset -c 0-3 $(UV) python scripts/bench_tail.py


figures:            ## regenerate all figures + results.md from the newest run DBs
	$(UV) python analysis/figures.py

fw-A fw-B fw-C:     ## compile firmware per tier (uses ~/.espressif v5.5.3)
	bash $(IDF_ENV) $(subst fw-,,$@)

clean:
	rm -rf firmware/build* model/artifacts model/assets_dev firmware/models/generated analysis/out
