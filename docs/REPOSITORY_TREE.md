# Repository tree

**Updated:** 2026-08-21
**Scope:** transition state before DVS-Lip implementation

```text
temporal-event-spiking-research/
├── configs/
│   ├── smoke.yaml                   # bounded synthetic integration config
│   └── *dvsgc*.yaml                # frozen DVS-GC experiment configs
├── containers/
│   └── temporal_event_spiking.def   # current legacy-capable Singularity image
├── data/
│   └── README.md                    # data policy; datasets ignored
├── docs/
│   ├── README.md                    # documentation map and authority
│   ├── PROJECT_CHARTER.md           # full authoritative charter v2
│   ├── AI_PROJECT_CHARTER_DVSLIP.md # compatibility pointer only
│   ├── AGENT_RULES.md
│   ├── ACTIVE_PLAN.md
│   ├── ROADMAP.md
│   ├── TASKS.md
│   ├── DECISIONS.md
│   ├── REPOSITORY_STATE.md
│   ├── REPOSITORY_TREE.md
│   ├── LEGACY_BOUNDARY.md
│   ├── EXPERIMENT_LEDGER.md
│   ├── SOURCES.md
│   ├── TRAINING_RECIPE.md
│   ├── HARDWARE_NOTES.md
│   ├── novelty_matrix.md
│   ├── dvslip_pareto.md
│   ├── OPERATIONS_SMILIES.md
│   └── archive/
│       ├── README.md
│       └── dvsgc/                   # frozen documents from 806c0aa
├── notebooks/
│   ├── README.md
│   └── archive/dvsgc/               # frozen analysis notebook
├── scripts/
│   ├── smoke_test.sh                # bounded end-to-end launcher
│   └── remaining scripts            # frozen DVS-GC launch/preparation paths
├── src/etsr/
│   ├── data/                        # frame-first dataset path
│   ├── evaluation/                  # behavioral/mechanistic audit tooling
│   ├── models/                      # Mini-QKFormer and local MultiStepLIF
│   ├── profiling/                   # firing, MAC/AC and Horowitz proxy
│   ├── training/                    # dense-frame train/evaluate engine
│   ├── utils/
│   ├── cli.py
│   ├── config.py
│   ├── reproducibility.py
│   └── runner.py
├── tests/                           # unit/regression tests; runtime count pending pytest
├── Makefile                         # quality, bounded smoke and legacy regression targets
├── pyproject.toml
├── requirements.txt
└── README.md
```

Directories proposed by the charter but intentionally absent until implementation needs them:

```text
src/etsr/encoders/
src/etsr/temporal/
```

Their absence is deliberate under the “no abstraction theater” rule. The first real encoder/core
task must create only the files justified by its contract and tests.
