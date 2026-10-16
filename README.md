# Renderer 2D di cerchi parallelo con Numba

Progetto mid-term del corso di **Parallel Programming** (Università degli Studi di Firenze).

Il programma disegna *N* cerchi semitrasparenti su un'immagine. I cerchi sono ordinati
lungo l'asse Z (painter's algorithm) e fusi con l'operatore alpha "over"; un pixel appartiene a
un cerchio se il suo centro cade dentro il cerchio. Lo stesso algoritmo è implementato in
quattro versioni per confrontare strategie di parallelizzazione e vettorizzazione su CPU:

| Versione | Parallelismo | Descrizione |
|----------|--------------|-------------|
| `seq`    | nessuno (1 thread) | per ogni cerchio visita solo il suo bounding box. È la baseline. |
| `pixels` | thread sulle righe | ogni pixel controlla tutti i cerchi: nessuna selezione, quindi lento. |
| `tiles`  | thread sulle tile | l'immagine è divisa in tile; ogni tile considera solo i cerchi che la toccano. |
| `simd`   | thread sulle tile + SIMD | come `tiles`, con il ciclo interno senza salti, vettorizzato da LLVM. |

Il parallelismo usa normali thread Python (`ThreadPoolExecutor`) su **Python 3.14
free-threaded** (`3.14t`), la versione dell'interprete senza GIL (PEP 703): i thread girano
davvero in parallelo su core diversi. Il thread *k* di *p* elabora le righe/tile *k*, *k+p*,
*k+2p*, … (come `schedule(static,1)` di OpenMP; l'esperimento 6 confronta anche `static`,
`static,c` e `dynamic,c`). L'esperimento 10 riesegue lo stesso codice con il GIL riattivato
(`-X gil=1`) per mostrare la differenza.

Tutte le versioni producono la stessa immagine e non usano lock: ogni thread scrive su righe o
tile diverse.

## Struttura

```
renderer_seq.py      dataset (Structure of Arrays) e le quattro versioni del renderer
benchmark_plots.py   verifica di correttezza, 10 esperimenti e grafici
pyproject.toml       dipendenze del progetto (gestite con uv)
.python-version      interprete richiesto: 3.14t (free-threaded)
figures/             grafici prodotti dal benchmark + immagine di esempio
relazione/           relazione tecnica (LaTeX e PDF)
```

## Requisiti

[uv](https://docs.astral.sh/uv/) e Python 3.14 free-threaded. `uv` installa l'interprete e
crea l'ambiente `.venv` leggendo `.python-version` e `pyproject.toml` (versioni esatte in
`uv.lock`):

```bash
uv python install 3.14+freethreaded
uv sync
```

| Pacchetto  | Versione usata |
|------------|----------------|
| Python     | 3.14.8 free-threaded |
| numpy      | 2.5.3  |
| numba      | 0.68.0 (Python 3.14t supportato da 0.65) |
| llvmlite   | 0.50.0 |
| matplotlib | 3.11.2 |

## Esecuzione

```bash
uv run python benchmark_plots.py
```

All'avvio il programma controlla l'interprete e si ferma se il GIL è attivo.

`renderer_seq.py` è solo una libreria: non va lanciato, lo importa `benchmark_plots.py`.

Il programma prima confronta l'output di ogni versione parallela con quello sequenziale e si
ferma se differiscono oltre la tolleranza. Poi esegue i 10 esperimenti, stampa tutti i tempi
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
| 0 | Verifica di correttezza | 512², 1000×700; 2000 cerchi; tolleranza 1e-3; anche `simd` con altri scheduling | — |
| 1 | Strong scaling | 2048², 60.000 cerchi; 1–16 thread | `bench_scaling.png` |
| 2 | Speedup per dimensione dell'input | `simd`, 512², 1024², 2048²; 3000 cerchi; 1–16 thread | `bench_speedup_sizes.png` |
| 3 | Tempo vs risoluzione | 512², 1024², 2048²; 3000 cerchi | `bench_resolution.png` |
| 4 | Tempo vs numero di cerchi | 1024²; 500–8000 cerchi | `bench_circles.png` |
| 5 | Tempo vs dimensione tile | 1024², 4000 cerchi; tile 16–256 px | `bench_tile.png` |
| 6 | Strategie di scheduling | `simd`, 1024², 4000 cerchi uniformi e concentrati al centro; `static`, `static,1/4/16`, `dynamic,1/4/16`; 8 e 16 thread | `bench_scheduling.png` |
| 7 | Scalare vs SIMD | 1024², 4000 cerchi; 1–16 thread | `bench_simd.png` |
| 8 | Layout di memoria SoA vs AoS | 512²–2048²; 3000 cerchi | `bench_memory_layout.png` |
| 9 | Weak scaling | 1024²; 500 cerchi per thread | `bench_weak_scaling.png` |
| 10 | GIL vs free-threading | `simd`, 1024², 4000 cerchi; stesso interprete con `-X gil=0` e `-X gil=1` | `bench_gil.png` |

Nell'esperimento 1 `pixels` viene misurato su un dataset più piccolo (1024², 8000 cerchi) e
solo da 4 thread in su, perché senza selezione dei cerchi sarebbe troppo lento; il suo
speedup è calcolato rispetto a un sequenziale misurato sullo stesso dataset ridotto.

## Parametri di misura

| Parametro | Valore | Dove si cambia |
|-----------|--------|----------------|
| Esecuzioni di riscaldamento (escluse dalla misura) | 2 | `WARMUP` in `benchmark_plots.py` |
| Esecuzioni misurate per configurazione | 5 | `ITERATIONS` in `benchmark_plots.py` |
| Numero massimo di thread | 2 × core logici (16 sul Ryzen 5 2400G) | `MAX_THREADS` in `benchmark_plots.py` |
| Dimensione tile di default | 32 px | costante `TILE` in `renderer_seq.py` |
| Scheduling di default | `static,1` (a salti) | parametro `schedule` di `render_parallel_tiles` / `render_tiles_simd` |

Per ogni configurazione vengono misurati e stampati tempo reale (media, minimo, massimo,
deviazione standard campionaria, intervallo di confidenza al 95%) e tempo CPU.

## Dataset

Il dataset è sintetico, generato da `generate_dataset()` con un seed fisso, quindi
riproducibile. Ogni cerchio ha posizione e profondità Z uniformi, raggio da 3 px fino al 12%
del lato dell'immagine, colore RGB casuale e alpha tra 0,15 e 1. Per un rasterizzatore la
geometria generata proceduralmente è il carico di lavoro realistico: non esiste un dataset
reale di cerchi con profondità e trasparenza da usare al suo posto.

## Ambiente di test

Server Linux (Debian 13, macchina virtuale Proxmox) con AMD Ryzen 5 2400G: 4 core fisici con
SMT (8 core logici), AVX2/FMA, cache L1 dati da 32 KB per core; Python 3.14.8 free-threaded.
I parametri dipendenti dalla macchina (tile da 32 px, 60.000 cerchi nell'esperimento 1) sono
stati scelti misurando su questo server. Il codice gira anche su macOS (sviluppato su un
MacBook Air M2).
I risultati completi e la loro analisi sono nella relazione in `relazione/`.
