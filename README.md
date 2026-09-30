# Renderer 2D di cerchi parallelo con Numba

Progetto mid-term del corso di **Parallel Programming** (Università degli Studi di Firenze).

Il programma disegna *N* cerchi semitrasparenti su un'immagine. I cerchi sono ordinati
lungo l'asse Z (painter's algorithm) e fusi con l'operatore alpha "over"; un pixel appartiene a
un cerchio se il suo centro cade dentro il cerchio. Lo stesso algoritmo è implementato in
quattro versioni per confrontare strategie di parallelizzazione e vettorizzazione su CPU:

| Versione | Parallelismo | Descrizione |
|----------|--------------|-------------|
| `seq`    | nessuno (1 thread) | per ogni cerchio visita solo il suo bounding box. È la baseline. |
| `pixels` | `prange` sulle righe | ogni pixel controlla tutti i cerchi: nessuna selezione, quindi lento. |
| `tiles`  | `prange` sulle tile | l'immagine è divisa in tile; ogni tile considera solo i cerchi che la toccano. |
| `simd`   | `prange` sulle tile + SIMD | come `tiles`, con il ciclo interno senza salti, vettorizzato da LLVM. |

Tutte le versioni producono la stessa immagine e non usano lock: ogni thread scrive su righe o
tile diverse.

## Struttura

```
renderer_seq.py      dataset (Structure of Arrays) e le quattro versioni del renderer
benchmark_plots.py   verifica di correttezza, 7 esperimenti e grafici
figures/             grafici prodotti dal benchmark + immagine di esempio
relazione/           relazione tecnica (LaTeX e PDF)
```

## Requisiti

Python 3.11 con:

| Pacchetto  | Versione usata |
|------------|----------------|
| numpy      | 2.4.6  |
| numba      | 0.67.0 |
| llvmlite   | 0.49.0 |
| matplotlib | 3.11.1 |

```bash
python3.11 -m venv .venv
.venv/bin/pip install numpy numba matplotlib
```

## Esecuzione

```bash
.venv/bin/python benchmark_plots.py
```

`renderer_seq.py` è solo una libreria: non va lanciato, lo importa `benchmark_plots.py`.

Il programma prima confronta l'output di ogni versione parallela con quello sequenziale e si
ferma se differiscono oltre la tolleranza. Poi esegue i 7 esperimenti, stampa tutti i tempi
sul terminale e salva i grafici in `figures/`. La run completa dura circa 15–25 minuti.

La relazione si compila dall'interno di `relazione/`, dove i grafici vengono letti da
`../figures/`:

```bash
cd relazione
pdflatex Relazione_RendererMidTerm.tex
pdflatex Relazione_RendererMidTerm.tex
```

## Esperimenti

| # | Esperimento | Parametri | Grafico |
|---|-------------|-----------|---------|
| 0 | Verifica di correttezza | 512², 1000×700; 2000 cerchi; tolleranza 1e-3 | — |
| 1 | Strong scaling | 2048², 100.000 cerchi; 1–16 thread | `bench_scaling.png` |
| 2 | Tempo vs risoluzione | 512², 1024², 2048²; 3000 cerchi | `bench_resolution.png` |
| 3 | Tempo vs numero di cerchi | 1024²; 500–8000 cerchi | `bench_circles.png` |
| 4 | Tempo vs dimensione tile | 1024², 4000 cerchi; tile 16–256 px | `bench_tile.png` |
| 5 | Scalare vs SIMD | 1024², 4000 cerchi; 1–16 thread | `bench_simd.png` |
| 6 | Layout di memoria SoA vs AoS | 512²–2048²; 3000 cerchi | `bench_memory_layout.png` |
| 7 | Weak scaling | 1024²; 500 cerchi per thread | `bench_weak_scaling.png` |

Nell'esperimento 1 `pixels` viene misurato su un dataset più piccolo (1024², 8000 cerchi) e
solo da 4 thread in su, perché senza selezione dei cerchi sarebbe troppo lento.

## Parametri di misura

| Parametro | Valore | Dove si cambia |
|-----------|--------|----------------|
| Esecuzioni di riscaldamento (escluse dalla misura) | 2 | `WARMUP` in `benchmark_plots.py` |
| Esecuzioni misurate per configurazione | 5 | `ITERATIONS` in `benchmark_plots.py` |
| Numero massimo di thread | 2 × core logici (16 su Apple M2) | variabile d'ambiente `NUMBA_NUM_THREADS` |
| Dimensione tile di default | 64 px | argomento `tile` delle funzioni di render |

Per ogni configurazione vengono misurati tempo reale (media, minimo, massimo, deviazione
standard, intervallo di confidenza al 95%) e tempo CPU.

## Dataset

Il dataset è sintetico, generato da `generate_dataset()` con un seed fisso, quindi
riproducibile. Ogni cerchio ha posizione e profondità Z uniformi, raggio da 3 px fino al 12%
del lato dell'immagine, colore RGB casuale e alpha tra 0,15 e 1. Per un rasterizzatore la
geometria generata proceduralmente è il carico di lavoro realistico: non esiste un dataset
reale di cerchi con profondità e trasparenza da usare al suo posto.

## Ambiente di test

MacBook Air Apple M2 (4 core performance + 4 core efficiency), macOS, Python 3.11.9.
I risultati completi e la loro analisi sono nella relazione in `relazione/`.
