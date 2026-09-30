"""
Renderer 2D di cerchi con profondita' Z, in quattro versioni:
seq (sequenziale), pixels, tiles e simd (parallele con Numba).

I cerchi sono salvati come Structure of Arrays (un array per attributo).

Regole di rendering (uguali per tutti i kernel):
  * i cerchi si disegnano per Z crescente: quelli con Z maggiore stanno sopra;
  * un pixel e' dentro un cerchio se il suo centro (i+0.5, j+0.5) rispetta
    dx^2 + dy^2 <= r^2;
  * alpha blending "over": out = src * a + dst * (1 - a).
"""

from __future__ import annotations

import os

# Numero massimo di thread Numba: va impostato prima di importare numba.
# Usiamo 2x i core logici per poter misurare anche l'oversubscription.
if "NUMBA_NUM_THREADS" not in os.environ:
    os.environ["NUMBA_NUM_THREADS"] = str(2 * (os.cpu_count() or 1))

import numpy as np
import numba
from numba import njit, prange


# ---------------------------------------------------------------------------
# Struttura dati SoA
# ---------------------------------------------------------------------------
class CircleSoA:
    """Cerchi in formato Structure of Arrays: 8 array float32 lunghi N.

      x, y, z  : centro (x, y in pixel) e profondita'
      radius   : raggio in pixel
      r, g, b  : colore in [0, 1]
      a        : opacita' in [0, 1]
    """

    __slots__ = ("x", "y", "z", "radius", "r", "g", "b", "a")

    def __init__(self, x, y, z, radius, r, g, b, a):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.z = np.ascontiguousarray(z, dtype=np.float32)
        self.radius = np.ascontiguousarray(radius, dtype=np.float32)
        self.r = np.ascontiguousarray(r, dtype=np.float32)
        self.g = np.ascontiguousarray(g, dtype=np.float32)
        self.b = np.ascontiguousarray(b, dtype=np.float32)
        self.a = np.ascontiguousarray(a, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def sorted_by_z(self) -> "CircleSoA":
        """Nuova CircleSoA ordinata per Z crescente (mergesort: stabile)."""
        order = np.argsort(self.z, kind="mergesort")
        return CircleSoA(
            self.x[order], self.y[order], self.z[order], self.radius[order],
            self.r[order], self.g[order], self.b[order], self.a[order],
        )

    def to_aos(self) -> np.ndarray:
        """Converte in Array of Structures: array (N, 8), un cerchio per riga.
        Serve solo al confronto SoA vs AoS in benchmark_plots.py."""
        return np.stack(
            [self.x, self.y, self.z, self.radius, self.r, self.g, self.b, self.a],
            axis=1,
        ).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Utility risoluzione
# ---------------------------------------------------------------------------
def _as_wh(resolution) -> tuple[int, int]:
    """Ritorna (width, height) da un int (immagine quadrata) o da una coppia."""
    if isinstance(resolution, (int, np.integer)):
        return int(resolution), int(resolution)
    w, h = resolution
    return int(w), int(h)


# ---------------------------------------------------------------------------
# Generazione dataset casuale
# ---------------------------------------------------------------------------
def generate_dataset(
    resolution=512,
    n: int | None = None,
    n_min: int = 100,
    n_max: int = 10_000,
    seed: int | None = None,
) -> CircleSoA:
    """Genera cerchi casuali per un'immagine di data risoluzione.

    n: numero di cerchi, limitato a [n_min, n_max]; se None e' casuale.
    seed: seme per rendere il dataset riproducibile.
    """
    rng = np.random.default_rng(seed)
    w, h = _as_wh(resolution)

    if n is None:
        n = int(rng.integers(n_min, n_max + 1))
    else:
        n = int(min(max(n, n_min), n_max))

    x = rng.uniform(0.0, w, size=n)
    y = rng.uniform(0.0, h, size=n)
    z = rng.uniform(0.0, 1.0, size=n)

    # Raggio da 3 px fino al 12% del lato dell'immagine.
    r_max = max(4.0, 0.12 * min(w, h))
    radius = rng.uniform(3.0, r_max, size=n)

    rgb = rng.uniform(0.0, 1.0, size=(n, 3))
    alpha = rng.uniform(0.15, 1.0, size=n)

    return CircleSoA(
        x, y, z, radius,
        rgb[:, 0], rgb[:, 1], rgb[:, 2], alpha,
    )


# ---------------------------------------------------------------------------
# Kernel sequenziale
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _render_kernel(img, x, y, z, radius, cr, cg, cb, ca):
    """Disegna i cerchi su `img` (H, W, 3), uno alla volta.

    I cerchi arrivano gia' ordinati per Z. `z` non e' usato, resta nella
    firma solo per coerenza con gli altri kernel.
    """
    h = img.shape[0]
    w = img.shape[1]
    n = x.shape[0]

    for k in range(n):
        cx = x[k]
        cy = y[k]
        rad = radius[k]
        if rad <= 0.0:
            continue

        sa = ca[k]
        if sa <= 0.0:
            continue  # trasparente: non cambia nulla
        sr = cr[k]
        sg = cg[k]
        sb = cb[k]
        inv = 1.0 - sa
        r2 = rad * rad

        # Bounding box del cerchio, limitata ai bordi dell'immagine.
        x0 = int(np.floor(cx - rad))
        x1 = int(np.ceil(cx + rad))
        y0 = int(np.floor(cy - rad))
        y1 = int(np.ceil(cy + rad))
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > w:
            x1 = w
        if y1 > h:
            y1 = h

        for j in range(y0, y1):
            py = j + 0.5
            dy = py - cy
            dy2 = dy * dy
            if dy2 > r2:
                continue
            for i in range(x0, x1):
                px = i + 0.5
                dx = px - cx
                if dx * dx + dy2 <= r2:
                    img[j, i, 0] = sr * sa + img[j, i, 0] * inv
                    img[j, i, 1] = sg * sa + img[j, i, 1] * inv
                    img[j, i, 2] = sb * sa + img[j, i, 2] * inv


# ---------------------------------------------------------------------------
# Versione sequenziale
# ---------------------------------------------------------------------------
def render(circles: CircleSoA, resolution=512, background=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Versione sequenziale. Ritorna un'immagine float32 (H, W, 3) in [0, 1]."""
    w, h = _as_wh(resolution)

    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :, 0] = np.float32(background[0])
    img[:, :, 1] = np.float32(background[1])
    img[:, :, 2] = np.float32(background[2])

    s = circles.sorted_by_z()
    _render_kernel(img, s.x, s.y, s.z, s.radius, s.r, s.g, s.b, s.a)
    return img


# ---------------------------------------------------------------------------
# Kernel parallelo sulle righe (pixels)
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def _render_kernel_parallel_pixels(
    img, x, y, z, radius, cr, cg, cb, ca, bg0, bg1, bg2
):
    """Parallelo sulle righe dell'immagine.

    Ogni thread scrive righe diverse, quindi non servono lock. Per ogni pixel
    si controllano TUTTI i cerchi (nessun culling): per questo e' lento.
    """
    h = img.shape[0]
    w = img.shape[1]
    n = x.shape[0]

    for j in prange(h):                 # righe divise tra i thread
        py = j + 0.5
        for i in range(w):
            px = i + 0.5

            # colore del pixel, parte dallo sfondo
            acc_r = bg0
            acc_g = bg1
            acc_b = bg2

            for k in range(n):
                rad = radius[k]
                dx = px - x[k]
                if dx < -rad or dx > rad:      # fuori dal bounding box
                    continue
                dy = py - y[k]
                if dy < -rad or dy > rad:
                    continue
                if dx * dx + dy * dy <= rad * rad:
                    sa = ca[k]
                    inv = 1.0 - sa
                    acc_r = cr[k] * sa + acc_r * inv
                    acc_g = cg[k] * sa + acc_g * inv
                    acc_b = cb[k] * sa + acc_b * inv

            img[j, i, 0] = acc_r
            img[j, i, 1] = acc_g
            img[j, i, 2] = acc_b


def render_parallel_pixels(
    circles: CircleSoA,
    resolution=512,
    background=(1.0, 1.0, 1.0),
    num_threads: int | None = None,
) -> np.ndarray:
    """Come `render`, con il kernel parallelo sulle righe.
    num_threads: thread da usare (limitato a [1, NUMBA_NUM_THREADS])."""
    if num_threads is not None:
        n = max(1, min(int(num_threads), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n)

    w, h = _as_wh(resolution)
    img = np.empty((h, w, 3), dtype=np.float32)

    s = circles.sorted_by_z()
    _render_kernel_parallel_pixels(
        img, s.x, s.y, s.z, s.radius, s.r, s.g, s.b, s.a,
        np.float32(background[0]), np.float32(background[1]), np.float32(background[2]),
    )
    return img


# ---------------------------------------------------------------------------
# Kernel parallelo a tile (tiles)
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def _render_kernel_parallel_tiles(
    img, x, y, z, radius, cr, cg, cb, ca, bg0, bg1, bg2, tile
):
    """Parallelo su blocchi quadrati (tile) di lato `tile`.

    Ogni thread lavora su tile diversi, quindi non servono lock. Per ogni tile:
      1. si selezionano i cerchi che toccano il tile (culling);
      2. si disegnano i pixel del tile usando solo quei cerchi.
    """
    h = img.shape[0]
    w = img.shape[1]
    n = x.shape[0]

    n_tx = (w + tile - 1) // tile
    n_ty = (h + tile - 1) // tile
    n_tiles = n_tx * n_ty

    for t in prange(n_tiles):              # tile divisi tra i thread
        tx = t % n_tx
        ty = t // n_tx

        x0 = tx * tile
        y0 = ty * tile
        x1 = min(x0 + tile, w)
        y1 = min(y0 + tile, h)

        # 1. culling: cerchi che toccano il tile
        cand = np.empty(n, dtype=np.int64)
        cnt = 0
        for k in range(n):
            rad = radius[k]
            if x[k] + rad < x0 or x[k] - rad > x1:
                continue
            if y[k] + rad < y0 or y[k] - rad > y1:
                continue
            cand[cnt] = k
            cnt += 1

        # 2. disegno dei pixel del tile con i soli cerchi candidati
        for j in range(y0, y1):
            py = j + 0.5
            for i in range(x0, x1):
                px = i + 0.5

                acc_r = bg0
                acc_g = bg1
                acc_b = bg2

                for c in range(cnt):
                    k = cand[c]
                    rad = radius[k]
                    dx = px - x[k]
                    if dx < -rad or dx > rad:
                        continue
                    dy = py - y[k]
                    if dy < -rad or dy > rad:
                        continue
                    if dx * dx + dy * dy <= rad * rad:
                        sa = ca[k]
                        inv = 1.0 - sa
                        acc_r = cr[k] * sa + acc_r * inv
                        acc_g = cg[k] * sa + acc_g * inv
                        acc_b = cb[k] * sa + acc_b * inv

                img[j, i, 0] = acc_r
                img[j, i, 1] = acc_g
                img[j, i, 2] = acc_b


def render_parallel_tiles(
    circles: CircleSoA,
    resolution=512,
    background=(1.0, 1.0, 1.0),
    tile: int = 64,
    num_threads: int | None = None,
) -> np.ndarray:
    """Come `render`, con il kernel a tile.
    tile: lato del tile in pixel; num_threads: thread da usare."""
    if num_threads is not None:
        nt = max(1, min(int(num_threads), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(nt)

    tile = max(1, int(tile))
    w, h = _as_wh(resolution)
    img = np.empty((h, w, 3), dtype=np.float32)

    s = circles.sorted_by_z()
    _render_kernel_parallel_tiles(
        img, s.x, s.y, s.z, s.radius, s.r, s.g, s.b, s.a,
        np.float32(background[0]), np.float32(background[1]), np.float32(background[2]),
        np.int64(tile),
    )
    return img


# ---------------------------------------------------------------------------
# Kernel a tile vettorizzato (SIMD)
# ---------------------------------------------------------------------------
# Stesso schema di `_render_kernel_parallel_tiles`, ma il ciclo interno sui
# pixel e' scritto perche' il compilatore lo vettorizzi:
#   * nessun if/continue nel ciclo: "dentro o fuori" diventa una maschera
#     m (1.0 o 0.0) e il colore si aggiorna con m*nuovo + (1-m)*vecchio;
#   * si disegna un cerchio alla volta su tutto il tile, cosi' ogni pixel e'
#     indipendente dagli altri;
#   * il tile usa tre piccoli buffer contigui (R, G, B) che restano in cache;
#   * fastmath=True permette al compilatore di usare FMA e riordinare i calcoli.
@njit(parallel=True, cache=True, fastmath=True)
def _render_kernel_tiles_simd(
    img, x, y, z, radius, cr, cg, cb, ca, bg0, bg1, bg2, tile
):
    h = img.shape[0]
    w = img.shape[1]
    n = x.shape[0]

    n_tx = (w + tile - 1) // tile
    n_ty = (h + tile - 1) // tile
    n_tiles = n_tx * n_ty
    cap = tile * tile

    one = np.float32(1.0)

    for t in prange(n_tiles):
        tx = t % n_tx
        ty = t // n_tx
        x0 = tx * tile
        y0 = ty * tile
        x1 = min(x0 + tile, w)
        y1 = min(y0 + tile, h)
        tw = x1 - x0
        th = y1 - y0

        # buffer R, G, B del tile, inizializzati con lo sfondo
        rr = np.empty(cap, dtype=np.float32)
        gg = np.empty(cap, dtype=np.float32)
        bb = np.empty(cap, dtype=np.float32)
        for p in range(tw * th):
            rr[p] = bg0
            gg[p] = bg1
            bb[p] = bg2

        # culling: cerchi che toccano il tile
        cand = np.empty(n, dtype=np.int64)
        cnt = 0
        for k in range(n):
            rad = radius[k]
            if x[k] + rad < x0 or x[k] - rad > x1:
                continue
            if y[k] + rad < y0 or y[k] - rad > y1:
                continue
            cand[cnt] = k
            cnt += 1

        # un cerchio alla volta (in ordine di Z) su tutti i pixel del tile
        for c in range(cnt):
            k = cand[c]
            cx = x[k]
            cy = y[k]
            r2 = radius[k] * radius[k]
            sa = ca[k]
            inv = one - sa
            psr = cr[k] * sa        # colore gia' moltiplicato per alpha
            psg = cg[k] * sa
            psb = cb[k] * sa

            for jj in range(th):
                dy = (y0 + jj + np.float32(0.5)) - cy
                dy2 = dy * dy
                row = jj * tw
                for ii in range(tw):          # ciclo vettorizzato
                    dx = (x0 + ii + np.float32(0.5)) - cx
                    d2 = dx * dx + dy2
                    # 1.0 se il pixel e' dentro il cerchio, 0.0 se fuori
                    m = one if d2 <= r2 else np.float32(0.0)
                    idx = row + ii
                    ov = rr[idx]
                    rr[idx] = m * (psr + ov * inv) + (one - m) * ov
                    ov = gg[idx]
                    gg[idx] = m * (psg + ov * inv) + (one - m) * ov
                    ov = bb[idx]
                    bb[idx] = m * (psb + ov * inv) + (one - m) * ov

        # copia del tile nell'immagine
        for jj in range(th):
            row = jj * tw
            for ii in range(tw):
                idx = row + ii
                img[y0 + jj, x0 + ii, 0] = rr[idx]
                img[y0 + jj, x0 + ii, 1] = gg[idx]
                img[y0 + jj, x0 + ii, 2] = bb[idx]


def render_tiles_simd(
    circles: CircleSoA,
    resolution=512,
    background=(1.0, 1.0, 1.0),
    tile: int = 64,
    num_threads: int | None = None,
) -> np.ndarray:
    """Come `render_parallel_tiles`, con il kernel vettorizzato (SIMD)."""
    if num_threads is not None:
        nt = max(1, min(int(num_threads), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(nt)

    tile = max(1, int(tile))
    w, h = _as_wh(resolution)
    img = np.empty((h, w, 3), dtype=np.float32)

    s = circles.sorted_by_z()
    _render_kernel_tiles_simd(
        img, s.x, s.y, s.z, s.radius, s.r, s.g, s.b, s.a,
        np.float32(background[0]), np.float32(background[1]), np.float32(background[2]),
        np.int64(tile),
    )
    return img


# ---------------------------------------------------------------------------
# Kernel a tile con layout AoS - solo per il confronto SoA vs AoS
# ---------------------------------------------------------------------------
# Identico a `_render_kernel_parallel_tiles`, ma legge i cerchi da un unico
# array (N, 8) invece che da 8 array separati. Cambia solo il layout in memoria.
_AOS_X, _AOS_Y, _AOS_Z, _AOS_R, _AOS_CR, _AOS_CG, _AOS_CB, _AOS_CA = range(8)


@njit(parallel=True, cache=True, fastmath=True)
def _render_kernel_tiles_aos(img, flat, bg0, bg1, bg2, tile):
    h = img.shape[0]
    w = img.shape[1]
    n = flat.shape[0]

    n_tx = (w + tile - 1) // tile
    n_ty = (h + tile - 1) // tile
    n_tiles = n_tx * n_ty

    for t in prange(n_tiles):
        tx = t % n_tx
        ty = t // n_tx

        x0 = tx * tile
        y0 = ty * tile
        x1 = min(x0 + tile, w)
        y1 = min(y0 + tile, h)

        cand = np.empty(n, dtype=np.int64)
        cnt = 0
        for k in range(n):
            rad = flat[k, _AOS_R]
            if flat[k, _AOS_X] + rad < x0 or flat[k, _AOS_X] - rad > x1:
                continue
            if flat[k, _AOS_Y] + rad < y0 or flat[k, _AOS_Y] - rad > y1:
                continue
            cand[cnt] = k
            cnt += 1

        for j in range(y0, y1):
            py = j + 0.5
            for i in range(x0, x1):
                px = i + 0.5

                acc_r = bg0
                acc_g = bg1
                acc_b = bg2

                for c in range(cnt):
                    k = cand[c]
                    rad = flat[k, _AOS_R]
                    dx = px - flat[k, _AOS_X]
                    if dx < -rad or dx > rad:
                        continue
                    dy = py - flat[k, _AOS_Y]
                    if dy < -rad or dy > rad:
                        continue
                    if dx * dx + dy * dy <= rad * rad:
                        sa = flat[k, _AOS_CA]
                        inv = 1.0 - sa
                        acc_r = flat[k, _AOS_CR] * sa + acc_r * inv
                        acc_g = flat[k, _AOS_CG] * sa + acc_g * inv
                        acc_b = flat[k, _AOS_CB] * sa + acc_b * inv

                img[j, i, 0] = acc_r
                img[j, i, 1] = acc_g
                img[j, i, 2] = acc_b


def render_parallel_tiles_aos(
    circles: CircleSoA,
    resolution=512,
    background=(1.0, 1.0, 1.0),
    tile: int = 64,
    num_threads: int | None = None,
) -> np.ndarray:
    """Come `render_parallel_tiles`, ma con layout AoS.
    Usata solo da benchmark_plots.py (non e' in RENDERERS)."""
    if num_threads is not None:
        nt = max(1, min(int(num_threads), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(nt)

    tile = max(1, int(tile))
    w, h = _as_wh(resolution)
    img = np.empty((h, w, 3), dtype=np.float32)

    s = circles.sorted_by_z()
    flat = s.to_aos()
    _render_kernel_tiles_aos(
        img, flat,
        np.float32(background[0]), np.float32(background[1]), np.float32(background[2]),
        np.int64(tile),
    )
    return img


# ---------------------------------------------------------------------------
# Le quattro versioni, per nome
# ---------------------------------------------------------------------------
RENDERERS = {
    "seq": render,
    "pixels": render_parallel_pixels,
    "tiles": render_parallel_tiles,
    "simd": render_tiles_simd,
}
