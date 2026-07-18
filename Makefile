PYTHON ?= python
CONFIG ?= configs/phase1_dvsgc_order2.yaml
CHECKPOINT ?=

.PHONY: install install-dev test smoke train audit lint clean

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

lint:
	ruff check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
