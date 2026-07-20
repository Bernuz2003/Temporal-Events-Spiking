PYTHON ?= python
CONFIG ?= configs/phase1_dvsgc_order2.yaml
PHASE2_CONFIG ?= configs/phase2_dvsgc_order2.yaml
CHECKPOINT ?=
SEED ?=
CHECKPOINTS ?=

.PHONY: install install-dev test smoke train audit phase2-prepare phase2-train phase2-audit lint clean

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

audit:
	@test -n "$(CHECKPOINT)" || (echo "Uso: make audit CHECKPOINT=checkpoints/<RUN_ID>/best.pt" && exit 1)
	$(PYTHON) -m etsr.cli audit --config $(CONFIG) --checkpoint $(CHECKPOINT)

phase2-prepare:
	$(PYTHON) -m etsr.cli phase2-prepare --config $(PHASE2_CONFIG)

phase2-train:
	@test -n "$(SEED)" || (echo "Uso: make phase2-train SEED=42" && exit 1)
	$(PYTHON) -m etsr.cli phase2-train --config $(PHASE2_CONFIG) --seed $(SEED)

phase2-audit:
	@test -n "$(CHECKPOINTS)" || (echo "Uso: make phase2-audit CHECKPOINTS='42=... 123=... 2026=...'" && exit 1)
	bash scripts/run_phase2_audit.sh $(PHASE2_CONFIG) $(CHECKPOINTS)

lint:
	ruff check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
