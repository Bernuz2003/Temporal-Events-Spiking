# Notebooks

Notebooks are secondary analysis records, not authoritative protocols or machine-readable experiment
artifacts. A notebook must identify its phase, source artifacts and lifecycle.

## Frozen archive

`archive/dvsgc/mechanistic_audit_analysis.ipynb`

- phase: closed DVS-Gesture-Chain behavioral/mechanistic audit;
- lifecycle: frozen by D007 on 2026-08-21;
- cells: 43 total (23 code, 20 markdown);
- embedded outputs: 39 across 22 code cells;
- size: 1,461,740 bytes;
- SHA-256: `4cbe751ad1c83c7cfa7634fea855428b38d219d0ca4871a44bd85a4d1961630b`;
- credential-pattern scan at archive time: no match for common API-key/password/token labels.

The notebook was preserved unchanged. Its embedded values are historical observations and do not
replace resolved configs, manifests or the experiment ledger. Do not use it to select a DVS-Lip
architecture.

## Future notebook policy

- prefer scripts/modules for reusable computation;
- keep large tensors and raw datasets outside Git;
- state artifact paths/hashes and commit in the first cells;
- remove secrets and machine-specific credentials;
- decide explicitly whether outputs are evidence worth versioning;
- archive notebooks when their phase closes.
