PYTHON ?= python
RUFF ?= ruff
CONFIG ?= configs/dvslip_e0.yaml
DVSLIP_TRAIN_ROOT ?= data/DVS-Lip/DVS-Lip/train
DVSLIP_SPLIT_MANIFEST ?= data/DVS-Lip/dvslip_development_split.json
DVSLIP_PREFLIGHT_OUTPUT ?= artifacts/dvslip_preflight.json
DVSLIP_PROFILE_OUTPUT ?= artifacts/dvslip_dataset_profile.json
DVSLIP_SHORTCUT_OUTPUT ?= artifacts/dvslip_shortcut_control.json
DVSLIP_CONFIG ?= configs/dvslip_e0.yaml
DVSLIP_HASH_SAMPLES ?= 0
DVSGESTURE_SOURCE_ROOT ?= data/DvsGesture/DvsGesture
DVSGESTURE_TRAIN_ROOT ?= data/DvsGesture/events/train
DVSGESTURE_PREPARATION_OUTPUT ?= artifacts/dvsgesture_preparation.json
DVSGESTURE_PROFILE_OUTPUT ?= artifacts/dvsgesture_dataset_profile.json
DATASET ?=
SMILIES_CONFIG ?=
SMILIES_SESSION ?=
SMILIES_TRAIN_ARGS ?=

.PHONY: install install-dev test lint check-scripts clean
.PHONY: prepare-dvslip-split preflight-dvslip profile-dvslip shortcut-dvslip
.PHONY: prepare-dvsgesture profile-dvsgesture
.PHONY: train
.PHONY: smilies-build smilies-prepare smilies-gate
.PHONY: smilies-train

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

prepare-dvslip-split:
	$(PYTHON) -m etsr.cli prepare-dvslip-split --train-root "$(DVSLIP_TRAIN_ROOT)" --output "$(DVSLIP_SPLIT_MANIFEST)"

preflight-dvslip:
	$(PYTHON) -m etsr.cli preflight-dvslip --train-root "$(DVSLIP_TRAIN_ROOT)" --output "$(DVSLIP_PREFLIGHT_OUTPUT)" $(if $(wildcard $(DVSLIP_SPLIT_MANIFEST)),--split-manifest "$(DVSLIP_SPLIT_MANIFEST)") $(if $(filter 1 true yes,$(DVSLIP_HASH_SAMPLES)),--hash-samples)

profile-dvslip:
	@test -f "$(DVSLIP_SPLIT_MANIFEST)" || (echo "Manifest split mancante: $(DVSLIP_SPLIT_MANIFEST)" && exit 1)
	$(PYTHON) -m etsr.cli profile-dvslip --train-root "$(DVSLIP_TRAIN_ROOT)" --split-manifest "$(DVSLIP_SPLIT_MANIFEST)" --output "$(DVSLIP_PROFILE_OUTPUT)"

shortcut-dvslip:
	$(PYTHON) -m etsr.cli shortcut-dvslip --config "$(DVSLIP_CONFIG)" --output "$(DVSLIP_SHORTCUT_OUTPUT)"

prepare-dvsgesture:
	$(PYTHON) -m etsr.cli prepare-dvsgesture --source-root "$(DVSGESTURE_SOURCE_ROOT)" --output-root "$(DVSGESTURE_TRAIN_ROOT)" --report "$(DVSGESTURE_PREPARATION_OUTPUT)"

profile-dvsgesture:
	@test -d "$(DVSGESTURE_TRAIN_ROOT)" || (echo "Dataset preparato mancante: $(DVSGESTURE_TRAIN_ROOT)" && exit 1)
	$(PYTHON) -m etsr.cli profile-dvsgesture --train-root "$(DVSGESTURE_TRAIN_ROOT)" --output "$(DVSGESTURE_PROFILE_OUTPUT)"

train:
	$(PYTHON) -m etsr.cli train --config $(CONFIG)

lint:
	$(RUFF) check src tests

check-scripts:
	find scripts -type f -name '*.sh' -print0 | xargs -0 -r -n1 bash -n

smilies-build:
	bash scripts/smilies/build_container.sh

smilies-prepare:
	@test -n "$(DATASET)" || (echo "Uso: make smilies-prepare DATASET={dvslip|dvsgesture}" && exit 1)
	DVSLIP_TRAIN_ROOT="$(DVSLIP_TRAIN_ROOT)" DVSLIP_SPLIT_MANIFEST="$(DVSLIP_SPLIT_MANIFEST)" DVSGESTURE_SOURCE_ROOT="$(DVSGESTURE_SOURCE_ROOT)" DVSGESTURE_TRAIN_ROOT="$(DVSGESTURE_TRAIN_ROOT)" bash scripts/smilies/dataset_workflow.sh "$(DATASET)" prepare

smilies-gate:
	@test -n "$(DATASET)" || (echo "Uso: make smilies-gate DATASET={dvslip|dvsgesture}" && exit 1)
	DVSLIP_TRAIN_ROOT="$(DVSLIP_TRAIN_ROOT)" DVSLIP_SPLIT_MANIFEST="$(DVSLIP_SPLIT_MANIFEST)" DVSGESTURE_SOURCE_ROOT="$(DVSGESTURE_SOURCE_ROOT)" DVSGESTURE_TRAIN_ROOT="$(DVSGESTURE_TRAIN_ROOT)" bash scripts/smilies/dataset_workflow.sh "$(DATASET)" gate

smilies-train:
	@test -n "$(SMILIES_CONFIG)" || (echo "Uso: make smilies-train SMILIES_CONFIG=configs/<dataset>.yaml [SMILIES_SESSION=nome] [SMILIES_TRAIN_ARGS='...']" && exit 1)
	@if [ -n "$(SMILIES_SESSION)" ]; then \
		bash scripts/smilies/run_training.sh "$(SMILIES_CONFIG)" "$(SMILIES_SESSION)" -- $(SMILIES_TRAIN_ARGS); \
	else \
		bash scripts/smilies/run_training.sh "$(SMILIES_CONFIG)" -- $(SMILIES_TRAIN_ARGS); \
	fi

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
