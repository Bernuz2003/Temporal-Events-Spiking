PYTHON ?= python
CONFIG ?= configs/temporal_audit_dvsgc_order2.yaml
AUDIT_CONFIG ?= configs/mechanistic_audit_dvsgc_order2.yaml
CHECKPOINT ?=
SEED ?=
CHECKPOINTS ?=

.PHONY: install install-dev test smoke train temporal-audit prepare-matched-dvsgc train-audit-seed mechanistic-audit lint clean

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install spikingjelly==0.0.0.0.14
	$(PYTHON) -m pip install --no-deps dvsgc==0.1.2

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'
	$(PYTHON) -m pip install spikingjelly==0.0.0.0.14
	$(PYTHON) -m pip install --no-deps dvsgc==0.1.2

test:
	pytest -q

smoke:
	bash scripts/smoke_test.sh

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
	ruff check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
