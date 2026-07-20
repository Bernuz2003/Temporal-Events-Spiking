# Fonti software analizzate e scelte implementative

## QKFormer ufficiale

Repository: `zhouchenlin2096/QKFormer`.

Osservazioni trasferite nel nuovo repository:

- separazione tra patch embedding gerarchico, Q-K token attention e SSA;
- input DVS in forma `[B,T,C,H,W]`;
- LIF multi-step con `tau=2` e reset distaccato;
- primo layer non-spiking trattato come MAC, layer successivi come spike-driven;
- attenzione alla forma del tensore prima dell'applicazione LIF.

Il repository ufficiale ha segnalato successivamente un problema nell'ordine reshape/LIF della proiezione SSA. La baseline qui inclusa ripristina sempre `[T,B,...]` prima del LIF.

Il codice di questa release è una riscrittura compatta, non una copia verbatim dell'implementazione ufficiale. Serve come baseline sperimentale modulare, non come replica numerica certificata del paper.

## DVS-Gesture-Chain ufficiale

Repository: `VicenteAlex/DVS-Gesture-Chain`.

Scelte trasferite:

- uso dell'API `DVSGestureChain`;
- split train/validation/test;
- parametri `frames_number`, `split_by`, `seq_len`, `class_num` e `repeat`;
- download manuale del dataset DVS128 Gesture originale.

Il package PyPI `dvsgc==0.1.2` dichiara rigidamente `spikingjelly==0.0.0.0.8`. Questa release usa invece le API dataset, ancora disponibili in `spikingjelly==0.0.0.0.14`, e installa `dvsgc` con `--no-deps`. La deviazione evita di trascinare dipendenze obsolete della release 0.0.0.0.8 e deve essere verificata sul dataset reale prima delle run definitive.

## LMUFormer ufficiale

Repository: `zeyuliu1037/LMUFormer`.

Spunti organizzativi:

- blocchi temporali modulari;
- configurazioni YAML;
- valutazione sui prefissi della sequenza;
- separazione tra embedding locale, memoria temporale e channel mixing.

Nessun modulo LMU viene incluso negli audit iniziali.

## SpikingJelly

Viene usato indirettamente dal package DVS-GC per preparare il dataset. Il modello del repository usa un LIF PyTorch minimale per ridurre il coupling con versioni specifiche di SpikingJelly e facilitare modifiche future.
