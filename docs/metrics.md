# Metriche

La Fase 1 usa metriche di alto livello, riproducibili e interpretabili. Non vengono stimati accessi BRAM, dimensioni dei buffer o bandwidth.

## Prestazione

- **Top-1 accuracy**
- **Macro-F1**
- **Confusion matrix**
- media e deviazione standard nelle repliche decisive

Il checkpoint viene selezionato esclusivamente sulla validation. Il test viene valutato una sola volta sul checkpoint migliore.

## Temporalità

### Perturbation degradation

Differenza di accuracy e Macro-F1 tra sequenza originale e sequenza temporalmente perturbata.

Perturbazioni iniziali:

- inversione temporale completa;
- shuffle deterministico dei timestep;
- sostituzione con la catena di azioni inversa accoppiata per file sorgente.

Una degradazione elevata indica sensibilità all'ordine, ma non prova da sola una comprensione corretta. Per questo viene salvata anche la predizione campione-per-campione.

L'audit salva un solo `perturbation_summary.csv`, contenente i drop di accuracy e Macro-F1 rispetto alla condizione originale.

### Analisi accoppiata

Le predizioni originali e perturbate vengono allineate tramite l'indice stabile del campione. Per ogni condizione sono riportati:

- cambi di predizione;
- transizioni da corretto a errato e da errato a corretto;
- tassi condizionati alla correttezza originale;
- cambi del target, rilevanti per `reverse_class`;
- matrice di transizione tra predizione originale e perturbata.

I tassi non condizionati usano tutti i campioni come denominatore; i tassi `given_original_*` usano rispettivamente i soli campioni originariamente corretti o errati.

### Prefix curve

Accuracy e Macro-F1 usando soltanto una frazione iniziale della sequenza.

### Prefix AUC

Area trapezoidale sotto la curva accuracy–frazione osservata. Vengono salvate entrambe le forme:

```text
prefix_accuracy_auc_raw = ∫[f_min, f_max] accuracy(f) df
prefix_accuracy_auc_normalized = prefix_accuracy_auc_raw / (f_max - f_min)
```

Con la griglia `[0.25, 0.5, 0.75, 1.0]`, una curva perfetta ha quindi AUC raw `0.75` e AUC
normalizzata `1.0`. La normalizzazione rende esplicita la prestazione media sull'intervallo osservato;
non estrapola la curva tra `0` e `f_min`. `prefix_accuracy_auc` resta un alias della forma raw per
compatibilità con artifact precedenti. Se è disponibile un solo punto entrambe le AUC sono `null`,
perché un'area non è definibile.

## Complessità

### Numero di parametri

Conteggio dei parametri trainabili totali e per modulo principale.

### Firing rate

Media degli spike prodotti dai layer LIF, sia globale sia layer-wise. Il firing rate non è energia: descrive soltanto l'attività.

### SOP stimate

Per convoluzioni e lineari spike-driven:

```text
SOP ≈ operazioni dense equivalenti × non-zero rate dell'ingresso
```

Le operazioni MAC del primo layer e del classificatore vengono mantenute separate.

L'attenzione viene stimata con formule specifiche e activity-weighted. Le SOP sono una proxy comparativa, non una misura di latenza o consumo reale.

## Energia teorica: modello di Horowitz

La configurazione predefinita usa i valori comunemente adottati nella letteratura Spiking Transformer a 45 nm:

- MAC: 4.6 pJ;
- AC: 0.9 pJ.

```text
E = N_MAC × E_MAC + N_AC × E_AC
```

### Limitazioni da riportare sempre

- non include memoria, data movement, routing e controllo;
- non è una misura sul dispositivo FPGA;
- dipende da tecnologia, precisione e implementazione;
- presume che gli zeri possano essere realmente saltati;
- confronta classi di operazioni, non il tempo GPU;
- può sottostimare modelli con molto stato o accessi irregolari;
- BatchNorm viene considerata fondibile in inference e quindi esclusa.

Ogni report deve usare l'espressione **energia teorica stimata**, mai semplicemente energia consumata.
