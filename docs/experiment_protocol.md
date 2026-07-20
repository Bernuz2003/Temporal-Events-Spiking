# Protocollo sperimentale

## Principio

Ogni esperimento deve rispondere a una sola domanda. Un run non è giustificato se non è chiaro quale ipotesi possa supportare o smentire.

## Procedura

1. eseguire i test automatici;
2. usare la configurazione smoke dopo modifiche strutturali;
3. lanciare un solo seed di sviluppo;
4. ispezionare validation, perturbazioni e profili;
5. replicare con almeno tre seed soltanto configurazioni decisive;
6. usare il test set una volta per seed, sul miglior checkpoint di validation;
7. documentare il commit Git e la configurazione risolta.

## Run ID

Il run ID include nome, timestamp e seed. Esempio:

```text
phase1_dvsgc_order2_miniqkformer__20260717_210000__seed42
```

## Logging

Vengono salvati:

- una riga per epoca in `history.csv`;
- messaggi essenziali in `run.log`;
- un JSON riepilogativo;
- una confusion matrix;
- un profilo aggregato;
- predizioni compresse soltanto nell'audit.

Non vengono salvate attivazioni complete o log per batch.

## Interpretazione delle perturbazioni

- `reverse_time`: distrugge ordine e direzione interna di ogni gesto;
- `shuffle_time`: distrugge la continuità mantenendo il contenuto dei frame;
- `reverse_actions`: sostituisce `AB` con la catena valida `BA` costruita dallo stesso file sorgente,
  mantenendo il target `AB`; misura quanto la decisione dipende dall'ordine;
- `reverse_actions + reverse_class`: usa lo stesso input `BA` e rimappa anche il target a `BA`;
  consente l'analisi accoppiata delle transizioni, ma sull'intero test set è una permutazione dei
  campioni originali e le sue metriche aggregate sono quindi ridondanti.

Il metodo risolto (`paired_reversed_action_sample`) viene salvato negli artifact. Non va interpretato
come una permutazione frame-esatta: le due catene sono campioni DVS-GC validi generati separatamente.

Nessuna perturbazione isolata dimostra comprensione causale. I risultati devono essere interpretati congiuntamente.

## Protocollo specifico della Fase 2

La Fase 2 deroga intenzionalmente alla procedura esplorativa a singolo seed: le anomalie della Fase 1
devono essere replicate sui tre seed `42`, `123` e `2026` prima di orientare l’architettura.

Usa esclusivamente `train_core`, `checkpoint_validation` e `development_audit`, ottenuti dalla
partizione ufficiale di training e raggruppati per `source_filename`. L’official test è sotto embargo.
Il medesimo split manifest è obbligatorio per tutti i checkpoint e viene verificato tramite hash.

Il risultato primario è un profilo, non uno scalare: fattorizzazione contenuto/ordine, consistenza
inversa e alle durate, utilità dell’evidenza, probe con controllo shuffled-label, shortcut di input e
activation patching. `development_audit` non deve essere usato per selezionare checkpoint,
regolarizzazione dei probe o configurazioni alternative.
