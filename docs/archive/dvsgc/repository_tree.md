# Alberatura del repository

```text
temporal-event-spiking-research/
├── configs/
│   ├── temporal_audit_dvsgc_order2.yaml
│   ├── temporal_audit_dvsgc_chain4.yaml
│   ├── mechanistic_audit_dvsgc_order2.yaml
│   └── smoke.yaml
├── containers/
│   └── temporal_event_spiking.def
├── docs/
│   ├── datasets.md
│   ├── experiment_protocol.md
│   ├── mechanistic_temporal_audit.md
│   ├── metrics.md
│   ├── research_directions.md
│   ├── repository_tree.md
│   ├── roadmap.md
│   ├── smilies_setup.md
│   ├── sources_and_design_notes.md
│   ├── temporal_audit_scope.md
│   ├── validation.md
│   └── vision.md
├── scripts/
│   ├── prepare_dvsgc.sh
│   ├── prepare_matched_dvsgc.sh
│   ├── run_mechanistic_audit.sh
│   ├── smoke_test.sh
│   ├── train_audit_seed.sh
│   ├── train_temporal_baseline.sh
│   └── train_temporal_baseline_screen.sh
├── src/etsr/
│   ├── data/
│   ├── evaluation/
│   ├── models/
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
