# Registro esperimenti

**Aggiornato:** 2026-09-08

Tutti i valori sono development validation e usano soltanto l'official-train DVS-Lip. I profili
sono validi solo quando derivano dal best checkpoint dello stesso run. Le due celle «sì» indicano
profili storici v1 senza energia Horowitz: nessun profilo v4 è ancora disponibile localmente.
La riga DVS-Gesture usa la development validation del suo official-train, separata da DVS-Lip.

| Run artifact | Seed | Parametri | Best epoch | Accuracy % | Macro-F1 % | Profilo | Esito |
|---|---:|---:|---:|---:|---:|---|---|
| `dvslip_e0__20260825_211710__seed42` | 42 | 500,708 | 112 | 44.81 | 44.15 | sì | baseline canonica |
| `dvslip_e0_capacity_1m__20260826_100031__seed42` | 42 | 1,113,508 | 116 | 49.58 | 49.38 | sì | controllo capacità |
| `dvslip_e0_capacity_2m__20260902_120335__seed42` | 42 | 1,967,972 | 114 | 51.79 | 51.81 | no | controllo capacità |
| `dvslip_e0_no_cross_time__20260902_143346__seed42` | 42 | 500,708 | 126 | 15.13 | 13.01 | no | dipendenza temporale necessaria |
| `dvslip_e0_readout_time_last_event__20260902_143353__seed42` | 42 | 500,708 | 106 | 42.50 | 42.02 | no | mean, ultimo bin occupato |
| `dvslip_e0_readout_last_readout_time_last_event__20260902_143408__seed42` | 42 | 500,708 | 116 | 32.32 | 31.83 | no | last, ultimo bin occupato |
| `dvslip_e0_readout_diagonal_gated__20260903_030141__seed42` | 42 | 501,476 | 125 | 19.80 | 17.60 | no | non conclusivo: init memoria errata |
| `dvsgesture_e0__20260825_213021__seed42` | 42 | 489,227 | 117 | 84.47 | 83.62 | no | baseline transfer, speaker-disjoint |

## Regole per nuovi record

Un full run entra nella tabella solo se contiene `summary.json`, config risolta, ambiente, curve e
predizioni. Se shortlisted, deve contenere anche `hardware_profile_v4.json`. Overfit, smoke e profili
senza training restano nei rispettivi artifact di gate e non vengono presentati come risultati.

Per i nuovi candidati aggiungere: variante (`F`, `T`, `F+T`, `gated-v2`), stato del bounded overfit, hash del
checkpoint profilato, numero di campioni del profilo e delta rispetto alla baseline su Macro-F1,
stato e operazioni.
