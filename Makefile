PYTHON ?= python
RUFF ?= ruff
CONFIG ?= configs/temporal_audit_dvsgc_order2.yaml
SMOKE_CONFIG ?= configs/smoke.yaml
AUDIT_CONFIG ?= configs/mechanistic_audit_dvsgc_order2.yaml
CHECKPOINT ?=
SEED ?=
CHECKPOINTS ?=
DVSLIP_TRAIN_ROOT ?=
DVSLIP_SPLIT_MANIFEST ?= data/dvslip_development_split.json
DVSLIP_PREFLIGHT_OUTPUT ?= artifacts/dvslip_preflight.json
DVSLIP_PROFILE_OUTPUT ?= artifacts/dvslip_dataset_profile.json
DVSLIP_SHORTCUT_OUTPUT ?= artifacts/dvslip_shortcut_control.json
DVSLIP_CONFIG ?= configs/dvslip_e0.yaml
DVSLIP_HASH_SAMPLES ?= 0
SMILIES_CONFIG ?=
SMILIES_SESSION ?=
SMILIES_TRAIN_ARGS ?=

.PHONY: install install-dev test smoke lint check-scripts clean
.PHONY: prepare-dvslip-split preflight-dvslip profile-dvslip shortcut-dvslip
.PHONY: train temporal-audit prepare-matched-dvsgc train-audit-seed mechanistic-audit
.PHONY: smilies-build smilies-dvslip-prepare smilies-dvslip-gate
.PHONY: smilies-train

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install spikingjelly==0.0.0.0.14
	$(PYTHON) -m pip install --no-deps dvsgc==0.1.2

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'
	$(PYTHON) -m pip install spikingjelly==0.0.0.0.14
	$(PYTHON) -m pip install --no-deps dvsgc==0.1.2

test:
	$(PYTHON) -m pytest -q

smoke:
	PYTHON="$(PYTHON)" bash scripts/checks/smoke_test.sh "$(SMOKE_CONFIG)"

prepare-dvslip-split:
	@test -n "$(DVSLIP_TRAIN_ROOT)" || (echo "Uso: make prepare-dvslip-split DVSLIP_TRAIN_ROOT=/path/to/DVS-Lip/train" && exit 1)
	$(PYTHON) -m etsr.cli prepare-dvslip-split --train-root "$(DVSLIP_TRAIN_ROOT)" --output "$(DVSLIP_SPLIT_MANIFEST)"

preflight-dvslip:
	@test -n "$(DVSLIP_TRAIN_ROOT)" || (echo "Uso: make preflight-dvslip DVSLIP_TRAIN_ROOT=/path/to/DVS-Lip/train" && exit 1)
	$(PYTHON) -m etsr.cli preflight-dvslip --train-root "$(DVSLIP_TRAIN_ROOT)" --output "$(DVSLIP_PREFLIGHT_OUTPUT)" $(if $(wildcard $(DVSLIP_SPLIT_MANIFEST)),--split-manifest "$(DVSLIP_SPLIT_MANIFEST)") $(if $(filter 1 true yes,$(DVSLIP_HASH_SAMPLES)),--hash-samples)

profile-dvslip:
	@test -n "$(DVSLIP_TRAIN_ROOT)" || (echo "Uso: make profile-dvslip DVSLIP_TRAIN_ROOT=/path/to/DVS-Lip/train" && exit 1)
	@test -f "$(DVSLIP_SPLIT_MANIFEST)" || (echo "Manifest split mancante: $(DVSLIP_SPLIT_MANIFEST)" && exit 1)
	$(PYTHON) -m etsr.cli profile-dvslip --train-root "$(DVSLIP_TRAIN_ROOT)" --split-manifest "$(DVSLIP_SPLIT_MANIFEST)" --output "$(DVSLIP_PROFILE_OUTPUT)"

shortcut-dvslip:
	$(PYTHON) -m etsr.cli shortcut-dvslip --config "$(DVSLIP_CONFIG)" --output "$(DVSLIP_SHORTCUT_OUTPUT)"

train:
	$(PYTHON) -m etsr.cli train --config $(CONFIG)

temporal-audit:
	@test -n "$(CHECKPOINT)" || (echo "Uso: make temporal-audit CHECKPOINT=checkpoints/<RUN_ID>/best.pt" && exit 1)
	$(PYTHON) -m etsr.cli temporal-audit --config $(CONFIG) --checkpoint $(CHECKPOINT)

prepare-matched-dvsgc:
	$(PYTHON) -m etsr.cli prepare-matched-dvsgc --config $(AUDIT_CONFIG)

train-audit-seed:
	@test -n "$(SEED)" || (echo "Uso: make train-audit-seed SEED=42" && exit 1)
	$(PYTHON) -m etsr.cli train --config $(AUDIT_CONFIG) --seed $(SEED)

mechanistic-audit:
	@test -n "$(CHECKPOINTS)" || (echo "Uso: make mechanistic-audit CHECKPOINTS='42=... 123=... 2026=...'" && exit 1)
	bash scripts/legacy/dvsgc/run_mechanistic_audit.sh $(AUDIT_CONFIG) $(CHECKPOINTS)

lint:
	$(RUFF) check src tests

check-scripts:
	find scripts -type f -name '*.sh' -print0 | xargs -0 -r -n1 bash -n

smilies-build:
	bash scripts/smilies/build_container.sh

smilies-dvslip-prepare:
	bash scripts/smilies/dvslip_workflow.sh prepare

smilies-dvslip-gate:
	bash scripts/smilies/dvslip_workflow.sh gate

smilies-train:
	@test -n "$(SMILIES_CONFIG)" || (echo "Uso: make smilies-train SMILIES_CONFIG=configs/<dataset>.yaml [SMILIES_SESSION=nome] [SMILIES_TRAIN_ARGS='...']" && exit 1)
	@if [ -n "$(SMILIES_SESSION)" ]; then \
		bash scripts/smilies/run_training.sh "$(SMILIES_CONFIG)" "$(SMILIES_SESSION)" -- $(SMILIES_TRAIN_ARGS); \
	else \
		bash scripts/smilies/run_training.sh "$(SMILIES_CONFIG)" -- $(SMILIES_TRAIN_ARGS); \
	fi

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
