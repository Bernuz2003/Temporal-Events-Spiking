PYTHON ?= python
RUFF ?= ruff
CONFIG ?= configs/temporal_audit_dvsgc_order2.yaml
SMOKE_CONFIG ?= configs/smoke.yaml
AUDIT_CONFIG ?= configs/mechanistic_audit_dvsgc_order2.yaml
CHECKPOINT ?=
SEED ?=
CHECKPOINTS ?=
DVSLIP_TRAIN_ROOT ?=
DVSLIP_SPEAKER_MANIFEST ?=
DVSLIP_SPLIT_MANIFEST ?=
DVSLIP_TERMS ?=
DVSLIP_PREFLIGHT_OUTPUT ?= artifacts/dvslip_preflight.json
DVSLIP_HASH_SAMPLES ?= 0

.PHONY: install install-dev test smoke preflight-dvslip train temporal-audit prepare-matched-dvsgc train-audit-seed mechanistic-audit lint clean

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
	PYTHON="$(PYTHON)" bash scripts/smoke_test.sh "$(SMOKE_CONFIG)"

preflight-dvslip:
	@test -n "$(DVSLIP_TRAIN_ROOT)" || (echo "Uso: make preflight-dvslip DVSLIP_TRAIN_ROOT=/path/to/DVS-Lip/train" && exit 1)
	$(PYTHON) -m etsr.cli preflight-dvslip --train-root "$(DVSLIP_TRAIN_ROOT)" --output "$(DVSLIP_PREFLIGHT_OUTPUT)" $(if $(DVSLIP_SPEAKER_MANIFEST),--speaker-manifest "$(DVSLIP_SPEAKER_MANIFEST)") $(if $(DVSLIP_SPLIT_MANIFEST),--split-manifest "$(DVSLIP_SPLIT_MANIFEST)") $(if $(DVSLIP_TERMS),--terms "$(DVSLIP_TERMS)") $(if $(filter 1 true yes,$(DVSLIP_HASH_SAMPLES)),--hash-samples)

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
	bash scripts/run_mechanistic_audit.sh $(AUDIT_CONFIG) $(CHECKPOINTS)

lint:
	$(RUFF) check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
