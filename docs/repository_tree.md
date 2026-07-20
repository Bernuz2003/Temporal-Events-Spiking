# Alberatura del repository

```text
temporal-event-spiking-research/
├── configs/
│   ├── phase1_dvsgc_order2.yaml
│   ├── phase1_dvsgc_chain4.yaml
│   ├── phase2_dvsgc_order2.yaml
│   └── smoke.yaml
├── containers/
│   └── temporal_event_spiking.def
├── docs/
│   ├── datasets.md
│   ├── PHASE2_MECHANISTIC_TEMPORAL_AUDIT_IMPLEMENTATION_BRIEF.md
│   ├── experiment_protocol.md
│   ├── metrics.md
│   ├── phase1_scope.md
│   ├── research_directions.md
│   ├── repository_tree.md
│   ├── roadmap.md
│   ├── smilies_setup.md
│   ├── sources_and_design_notes.md
│   └── vision.md
├── scripts/
│   ├── prepare_dvsgc.sh
│   ├── run_phase1.sh
│   ├── run_phase1_screen.sh
│   ├── prepare_phase2.sh
│   ├── train_phase2_seed.sh
│   ├── run_phase2_audit.sh
│   └── smoke_test.sh
├── src/etsr/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── phase2/
│   ├── profiling/
│   ├── training/
│   ├── utils/
│   ├── cli.py
│   ├── config.py
│   ├── reproducibility.py
│   └── runner.py
├── tests/
├── artifacts/      # metriche e figure; ignorati da Git
├── checkpoints/    # pesi; ignorati da Git e separati dagli artifact
├── data/           # dataset locali; ignorati da Git
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

La struttura è intenzionalmente piccola: ogni nuova direzione viene aggiunta soltanto dopo una decisione sperimentale, evitando cartelle o astrazioni speculative.
