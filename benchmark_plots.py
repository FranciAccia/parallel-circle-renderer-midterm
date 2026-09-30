from __future__ import annotations

import functools
import os
import time

# Numero massimo di thread Numba (2x i core logici). Va impostato qui, prima
# di importare numba: questo file importa numba prima di renderer_seq.
if "NUMBA_NUM_THREADS" not in os.environ:
    os.environ["NUMBA_NUM_THREADS"] = str(2 * (os.cpu_count() or 1))

import numpy as np
import numba
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from renderer_seq import (
    RENDERERS, generate_dataset, render, render_parallel_tiles_aos,
)


MAX_THREADS = numba.config.NUMBA_NUM_THREADS
WARMUP = 2       # esecuzioni di riscaldamento, escluse dalla misura
ITERATIONS = 5   # esecuzioni misurate per ogni configurazione


def time_render(render_fn, circles, resolution, threads, warmup, iterations,
                budget_s: float = 20.0):
    """Misura tempo reale e tempo CPU di render_fn.

    Si ferma prima se le misure superano `budget_s` secondi.
    Ritorna (media, min, max, std, media_cpu, ci95) in ms, dove ci95 e'
    l'intervallo di confidenza al 95%: 1.96 * std / sqrt(n).
    """
    numba.set_num_threads(max(1, min(threads, MAX_THREADS)))

    for _ in range(warmup):
        render_fn(circles, resolution)

    ts, cpu_ts = [], []
    spent = 0.0
    for _ in range(iterations):
        t0 = time.perf_counter()
        c0 = time.process_time()
        render_fn(circles, resolution)
        dt = time.perf_counter() - t0
        cdt = time.process_time() - c0
        ts.append(dt)
        cpu_ts.append(cdt)
        spent += dt
        if spent > budget_s:
            break

    ms = np.asarray(ts) * 1e3
    cpu_ms = np.asarray(cpu_ts) * 1e3
    n = len(ms)
    ci95 = 1.96 * ms.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return (float(ms.mean()), float(ms.min()), float(ms.max()), float(ms.std()),
            float(cpu_ms.mean()), float(ci95))


def make_fn(mode, tile):
    fn = RENDERERS[mode]
    if mode in ("tiles", "simd"):
        return functools.partial(fn, tile=tile)
    return fn


def check_correctness(tol: float = 1e-3) -> None:
    """Confronta ogni versione parallela con quella sequenziale.

    Si usa una tolleranza e non l'uguaglianza esatta: con fastmath=True i
    risultati possono differire negli ultimi bit. Oltre all'immagine quadrata
    si prova un'immagine rettangolare con lati non multipli della tile.
    """
    print("[0/7] verifica correttezza ...")
    for res in (512, (1000, 700)):
        circ = generate_dataset(resolution=res, n=2000, seed=7)
        numba.set_num_threads(MAX_THREADS)
        ref = render(circ, res)
        fns = {mode: make_fn(mode, tile=64) for mode in ("pixels", "tiles", "simd")}
        fns["tiles_aos"] = functools.partial(render_parallel_tiles_aos, tile=64)
        for name, fn in fns.items():
            err = float(np.abs(fn(circ, res) - ref).max())
            label = res if isinstance(res, int) else f"{res[0]}x{res[1]}"
            print(f"    {name:9s} {label} -> errore max {err:.1e}")
            if err > tol:
                raise SystemExit(f"ERRORE: '{name}' diverso dal sequenziale ({err:.1e} > {tol})")


def run_all(warmup: int, iterations: int) -> dict:
    # Thread da testare, fino a 2x i core logici.
    thread_list = sorted(
        {1, 2, 4, 8, 16, 32, MAX_THREADS} & set(range(1, MAX_THREADS + 1)) | {1, MAX_THREADS}
    )
    resolutions = [512, 1024, 2048]
    circle_counts = [500, 1000, 2000, 4000, 8000]
    tile_sizes = [16, 32, 64, 128, 256]

    results: dict = {
        "meta": {
            "max_threads": MAX_THREADS,
            "warmup": warmup,
            "iterations": iterations,
        }
    }

    # --- 1. scaling vs thread ----------------------------------------------
    # Dataset grande (100.000 cerchi) perché il sequenziale duri almeno 10 s.
    print("[1/7] scaling vs thread ...")
    # Se il tempo sequenziale stampato scende sotto 10 s, aumentare n e n_max.
    res, n, n_max = 2048, 100_000, 100_000
    circ = generate_dataset(resolution=res, n=n, n_max=n_max, seed=42)
    base_ms, _, _, _, base_cpu_ms, base_ci = time_render(
        render, circ, res, 1, warmup, iterations, budget_s=90.0
    )
    print(f"    seq (baseline) -> {base_ms / 1000:8.2f} s  (CI95 +/-{base_ci:.1f} ms, "
          f"cpu {base_cpu_ms / 1000:.2f} s)")
    scaling = {"resolution": res, "circles": n, "seq_ms": base_ms,
               "seq_cpu_ms": base_cpu_ms, "seq_ci95_ms": base_ci,
               "threads": thread_list, "modes": {}, "ci95_ms": {}}

    # 'pixels' non ha culling (selezione) e sul dataset grande sarebbe troppo lento:
    # lo misuriamo su un dataset piu' piccolo (1024x1024, 8000 cerchi).
    circ_pixels = generate_dataset(resolution=1024, n=8000, seed=42)
    res_pixels = 1024

    for mode in ("pixels", "tiles", "simd"):
        fn = make_fn(mode, tile=64)
        m_threads = [t for t in thread_list if t >= 4] if mode == "pixels" else thread_list
        row, ci_row = [], []
        for t in thread_list:
            if t not in m_threads:
                row.append(None)
                ci_row.append(None)
                continue
            # Qui basta 1 warm-up: ogni esecuzione dura secondi.
            if mode == "pixels":
                mean, _, _, _, _, ci = time_render(
                    fn, circ_pixels, res_pixels, t, 1, iterations, budget_s=15.0)
            else:
                mean, _, _, _, _, ci = time_render(
                    fn, circ, res, t, 1, iterations, budget_s=70.0)
            row.append(mean)
            ci_row.append(ci)
            print(f"    {mode:7s} t={t:<2d} -> {mean:8.1f} ms")
        scaling["modes"][mode] = row
        scaling["ci95_ms"][mode] = ci_row
    results["scaling"] = scaling

    # --- 2. tempo vs risoluzione: 3000 cerchi, max thread ----------------
    print("[2/7] tempo vs risoluzione ...")
    n = 3000
    rres = {"circles": n, "threads": MAX_THREADS, "resolutions": resolutions, "modes": {}}
    for mode in ("seq", "tiles", "simd"):
        fn = make_fn(mode, tile=64)
        row = []
        for r in resolutions:
            circ_r = generate_dataset(resolution=r, n=n, seed=42)
            t = 1 if mode == "seq" else MAX_THREADS
            mean, *_ = time_render(fn, circ_r, r, t, warmup, iterations)
            row.append(mean)
            print(f"    {mode:7s} {r}^2 -> {mean:8.1f} ms")
        rres["modes"][mode] = row
    results["resolution"] = rres

    # --- 3. tempo vs numero di cerchi: 1024x1024, max thread ------------
    print("[3/7] tempo vs numero cerchi ...")
    res = 1024
    cres = {"resolution": res, "threads": MAX_THREADS,
            "counts": circle_counts, "modes": {}}
    for mode in ("seq", "tiles", "simd"):
        fn = make_fn(mode, tile=64)
        row = []
        for n in circle_counts:
            circ_n = generate_dataset(resolution=res, n=n, seed=42)
            t = 1 if mode == "seq" else MAX_THREADS
            mean, *_ = time_render(fn, circ_n, res, t, warmup, iterations)
            row.append(mean)
            print(f"    {mode:7s} n={n:<5d} -> {mean:8.1f} ms")
        cres["modes"][mode] = row
    results["circles"] = cres

    # --- 4. tempo vs dimensione tile: 1024x1024, 4000 cerchi -----------
    print("[4/7] tempo vs dimensione tile ...")
    res, n = 1024, 4000
    circ = generate_dataset(resolution=res, n=n, seed=42)
    tres = {"resolution": res, "circles": n, "threads": MAX_THREADS,
            "tiles": tile_sizes, "modes": {}}
    for mode in ("tiles", "simd"):
        row = []
        for ts_ in tile_sizes:
            fn = make_fn(mode, tile=ts_)
            mean, *_ = time_render(fn, circ, res, MAX_THREADS, warmup, iterations)
            row.append(mean)
            print(f"    {mode:7s} tile={ts_:<3d} -> {mean:8.1f} ms")
        tres["modes"][mode] = row
    results["tile"] = tres

    # --- 5. effetto SIMD: tiles vs simd a vari thread -----------------
    print("[5/7] effetto SIMD ...")
    res, n = 1024, 4000
    circ = generate_dataset(resolution=res, n=n, seed=42)
    sres = {"resolution": res, "circles": n, "threads": thread_list, "modes": {}}
    for mode in ("tiles", "simd"):
        fn = make_fn(mode, tile=64)
        row = []
        for t in thread_list:
            mean, *_ = time_render(fn, circ, res, t, warmup, iterations)
            row.append(mean)
            print(f"    {mode:7s} t={t:<2d} -> {mean:8.1f} ms")
        sres["modes"][mode] = row
    for t, a, b in zip(thread_list, sres["modes"]["tiles"], sres["modes"]["simd"]):
        print(f"    speedup SIMD t={t:<2d} -> {a / b:5.2f}x")
    results["simd"] = sres

    # --- 6. memory layout: SoA vs AoS -----------------------------------
    print("[6/7] memory layout: SoA vs AoS ...")
    res_list = [512, 1024, 2048]
    n_fixed = 3000
    mres = {"circles": n_fixed, "threads": MAX_THREADS, "resolutions": res_list,
            "modes": {"soa": [], "aos": []}}
    for r in res_list:
        circ_r = generate_dataset(resolution=r, n=n_fixed, seed=42)
        fn_soa = make_fn("tiles", tile=64)
        mean_soa, *_ = time_render(fn_soa, circ_r, r, MAX_THREADS, warmup, iterations)
        fn_aos = functools.partial(render_parallel_tiles_aos, tile=64)
        mean_aos, *_ = time_render(fn_aos, circ_r, r, MAX_THREADS, warmup, iterations)
        mres["modes"]["soa"].append(mean_soa)
        mres["modes"]["aos"].append(mean_aos)
        print(f"    {r}^2  SoA={mean_soa:7.1f} ms  AoS={mean_aos:7.1f} ms")
    results["memory_layout"] = mres

    # --- 7. weak scaling: lavoro per thread costante ---------------------
    print("[7/7] weak scaling ...")
    res = 1024
    base_n = 500
    fn = make_fn("simd", tile=64)
    base_circ = generate_dataset(resolution=res, n=base_n, seed=99)
    base_t, *_ = time_render(fn, base_circ, res, 1, warmup, iterations)
    wres = {"resolution": res, "base_circles": base_n, "base_ms": base_t,
            "threads": thread_list, "times_ms": [], "efficiency": []}
    for t in thread_list:
        n_t = min(base_n * t, 10_000)
        circ_t = generate_dataset(resolution=res, n=n_t, seed=99)
        mean, *_ = time_render(fn, circ_t, res, t, warmup, iterations)
        eff = base_t / mean if mean > 0 else 0.0
        wres["times_ms"].append(mean)
        wres["efficiency"].append(eff)
        print(f"    t={t:<2d} n={n_t:<6d} -> {mean:8.1f} ms  (eff={eff:.2f})")
    results["weak_scaling"] = wres

    return results


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------
MODE_STYLE = {
    "seq":    dict(color="#555555", marker="o", label="seq (1 thread)"),
    "pixels": dict(color="#d1495b", marker="s", label="pixels"),
    "tiles":  dict(color="#edae49", marker="^", label="tiles"),
    "simd":   dict(color="#00798c", marker="D", label="simd"),
}


def _clean(xs, ys):
    xs2, ys2 = [], []
    for a, b in zip(xs, ys):
        if b is not None:
            xs2.append(a)
            ys2.append(b)
    return xs2, ys2


def plot_scaling(d, path):
    s = d["scaling"]
    threads = s["threads"]
    seq_ms = s["seq_ms"]
    ci_by_mode = s.get("ci95_ms", {})

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) tempo assoluto
    ax[0].axhline(seq_ms, color=MODE_STYLE["seq"]["color"], ls="--",
                  label=f"seq 1 thread ({seq_ms:.0f} ms)")
    for mode, row in s["modes"].items():
        x, y = _clean(threads, row)
        ax[0].plot(x, y, **{k: MODE_STYLE[mode][k] for k in ("color", "marker")},
                   label=MODE_STYLE[mode]["label"])
    ax[0].set(xlabel="thread", ylabel="tempo [ms]",
              title=f"Tempo di rendering  ({s['resolution']}², {s['circles']} cerchi)")
    ax[0].set_yscale("log")
    ax[0].grid(True, alpha=.3)
    ax[0].legend()

    # (b) speedup rispetto al sequenziale, con barre di errore CI95%
    for mode, row in s["modes"].items():
        x, y = _clean(threads, row)
        ci_row = ci_by_mode.get(mode, [None] * len(row))
        _, ci = _clean(threads, ci_row)
        speedup = [seq_ms / v for v in y]
        yerr = [(seq_ms / v) * (c / v) if c else 0.0 for v, c in zip(y, ci)]
        ax[1].errorbar(x, speedup, yerr=yerr, capsize=3,
                        **{k: MODE_STYLE[mode][k] for k in ("color", "marker")},
                        label=MODE_STYLE[mode]["label"])
    ax[1].plot(threads, threads, "k:", alpha=.5, label="ideale (lineare)")
    ax[1].set(xlabel="thread", ylabel="speedup  (T_seq / T)",
              title="Speedup rispetto al sequenziale  (barre: CI 95%)")
    ax[1].grid(True, alpha=.3)
    ax[1].legend()

    # (c) efficienza: ogni kernel e' confrontato con se stesso a 1 thread
    for mode, row in s["modes"].items():
        x, y = _clean(threads, row)
        if not y:
            continue
        t0, base = x[0], y[0]
        ax[2].plot(x, [(base / v) / (t / t0) for v, t in zip(y, x)],
                   **{k: MODE_STYLE[mode][k] for k in ("color", "marker")},
                   label=MODE_STYLE[mode]["label"])
    ax[2].axhline(1.0, color="k", ls=":", alpha=.5, label="ideale")
    ax[2].set(xlabel="thread",
              ylabel="efficienza  (T(1) / T(p)) / p",
              title="Efficienza parallela  (baseline: kernel a 1 thread)",
              ylim=(0, None))
    ax[2].grid(True, alpha=.3)
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_resolution(d, path):
    r = d["resolution"]
    res = r["resolutions"]
    x = np.arange(len(res))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (mode, row) in enumerate(r["modes"].items()):
        ax.bar(x + (i - 1) * w, row, w, color=MODE_STYLE[mode]["color"],
               label=MODE_STYLE[mode]["label"])
        for xi, v in zip(x + (i - 1) * w, row):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set(xticks=x, xticklabels=[f"{s}×{s}" for s in res],
           ylabel="tempo [ms]",
           title=f"Tempo vs risoluzione  ({r['circles']} cerchi, "
                 f"paralleli su {r['threads']} thread)")
    ax.grid(True, axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_circles(d, path):
    c = d["circles"]
    counts = c["counts"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, row in c["modes"].items():
        ax.plot(counts, row, **{k: MODE_STYLE[mode][k] for k in ("color", "marker")},
                label=MODE_STYLE[mode]["label"])
    ax.set(xlabel="numero di cerchi", ylabel="tempo [ms]",
           title=f"Tempo vs carico  ({c['resolution']}², "
                 f"paralleli su {c['threads']} thread)")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_tile(d, path):
    t = d["tile"]
    tiles = t["tiles"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, row in t["modes"].items():
        ax.plot(tiles, row, **{k: MODE_STYLE[mode][k] for k in ("color", "marker")},
                label=MODE_STYLE[mode]["label"])
        best = int(np.argmin(row))
        ax.annotate(f"min {row[best]:.0f} ms\n@ {tiles[best]}px",
                    (tiles[best], row[best]), textcoords="offset points",
                    xytext=(6, 10), fontsize=8,
                    color=MODE_STYLE[mode]["color"])
    ax.set(xlabel="lato del tile [px]", ylabel="tempo [ms]")
    ax.set_xscale("log", base=2)
    ax.set_xticks(tiles)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_title(f"Sensibilità alla dimensione del tile  "
                 f"({t['resolution']}², {t['circles']} cerchi, {t['threads']} thread)")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_simd(d, path):
    s = d["simd"]
    threads = s["threads"]
    x = np.arange(len(threads))
    w = 0.38

    tiles = s["modes"]["tiles"]
    simd = s["modes"]["simd"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    ax[0].bar(x - w / 2, tiles, w, color=MODE_STYLE["tiles"]["color"], label="tiles (scalare)")
    ax[0].bar(x + w / 2, simd, w, color=MODE_STYLE["simd"]["color"], label="simd (vettorizzato)")
    ax[0].set(xticks=x, xticklabels=threads, xlabel="thread", ylabel="tempo [ms]",
              title=f"tiles vs simd  ({s['resolution']}², {s['circles']} cerchi)")
    ax[0].grid(True, axis="y", alpha=.3)
    ax[0].legend()

    ax[1].plot(threads, [a / b for a, b in zip(tiles, simd)],
               color=MODE_STYLE["simd"]["color"], marker="D")
    ax[1].axhline(1.0, color="k", ls=":", alpha=.5)
    ax[1].set(xlabel="thread", ylabel="speedup SIMD  (T_tiles / T_simd)",
              title="Guadagno della vettorizzazione", ylim=(0, None))
    ax[1].grid(True, alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_aos_vs_soa(d, path):
    if "memory_layout" not in d:
        return
    m = d["memory_layout"]
    res = m["resolutions"]
    x = np.arange(len(res))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, m["modes"]["soa"], w, label="SoA", color=MODE_STYLE["tiles"]["color"])
    ax.bar(x + w / 2, m["modes"]["aos"], w, label="AoS", color=MODE_STYLE["pixels"]["color"])
    ax.set(xticks=x, xticklabels=[f"{r}×{r}" for r in res],
           ylabel="tempo [ms]",
           title=f"Memory layout: SoA vs AoS  ({m['circles']} cerchi, {m['threads']} thread)")
    ax.grid(True, axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_weak_scaling(d, path):
    if "weak_scaling" not in d:
        return
    w = d["weak_scaling"]
    threads = w["threads"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(threads, w["times_ms"], color=MODE_STYLE["simd"]["color"], marker="D")
    ax[0].axhline(w["base_ms"], color="k", ls="--", alpha=.5,
                  label=f"T(1) = {w['base_ms']:.1f} ms")
    ax[0].set(xlabel="thread (lavoro per thread costante)", ylabel="tempo [ms]",
              title=f"Weak scaling  ({w['resolution']}², base {w['base_circles']} cerchi/thread)")
    ax[0].grid(True, alpha=.3)
    ax[0].legend()

    ax[1].plot(threads, w["efficiency"], color=MODE_STYLE["simd"]["color"], marker="D")
    ax[1].axhline(1.0, color="k", ls=":", alpha=.5, label="ideale")
    ax[1].set(xlabel="thread", ylabel="efficienza  (T(1) / T(p))",
              title="Efficienza weak scaling", ylim=(0, None))
    ax[1].grid(True, alpha=.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    check_correctness()
    results = run_all(WARMUP, ITERATIONS)
    print(f"\nBenchmark completato in {time.perf_counter() - t0:.1f} s")

    # I grafici vanno in figures/, la cartella usata dalla relazione.
    os.makedirs("figures", exist_ok=True)
    plot_scaling(results, "figures/bench_scaling.png")
    plot_resolution(results, "figures/bench_resolution.png")
    plot_circles(results, "figures/bench_circles.png")
    plot_tile(results, "figures/bench_tile.png")
    plot_simd(results, "figures/bench_simd.png")
    plot_aos_vs_soa(results, "figures/bench_memory_layout.png")
    plot_weak_scaling(results, "figures/bench_weak_scaling.png")


if __name__ == "__main__":
    main()
