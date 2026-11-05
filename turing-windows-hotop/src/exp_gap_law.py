"""
Unified criterion: window contrast is controlled by the number of in-band eigenvalues,
not by the disorder.

Motivation

Three scans give apparently conflicting results:

    exp_disorder        contrast decays monotonically with disorder q, but the
                        trivial effect of the band leaving the spectrum is included
    exp_band_position   with the band position controlled, q = 1 keeps a contrast of
                        0.16, but on only two samples
    exp_band_matched    with the band pinned to the spectral median, the contrast of
                        every q becomes 0

All three reduce to one mechanism. The window phenomenon originates from the in-band
eigenvalue count N_in being an O(1) random quantity that jumps with n. Once the band
holds many eigenvalues (N_in much greater than 1), every n has an eigenvalue in the
band, the verdict is always unstable and the contrast must vanish, independently of
whether the structure is regular or random.

Hence the criterion: the contrast is a function of mean(N_in), and mean(N_in) is set
by the local spectral density times the band width.

This script tests the criterion directly: it computes mean(N_in) and the contrast on a
(q, rho) grid and checks whether the two fall on a single relation across all q.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complex import SimplicialComplex  # noqa: E402
from rd import Schnakenberg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def ws_edges(n: int, k: int, q: float, rng) -> list[tuple[int, int]]:
    half = k // 2
    edges = set()
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            edges.add((min(i, j), max(i, j)))
    for (i, j) in list(edges):
        if rng.random() < q:
            edges.discard((i, j))
            for _ in range(30):
                w = int(rng.integers(0, n))
                if w == i:
                    continue
                e = (min(i, w), max(i, w))
                if e not in edges:
                    edges.add(e)
                    break
    return sorted(edges)


def ws_complex(n: int, k: int, q: float, rng) -> SimplicialComplex:
    edges = ws_edges(n, k, q, rng)
    sc = SimplicialComplex.from_edges(n, edges)
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        for j in sorted(adj[i]):
            if j <= i:
                continue
            for kk in sorted(adj[i] & adj[j]):
                if kk > j:
                    sc.add_triangle(i, j, kk)
    return sc


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    width = hi - lo
    k_deg = 6
    n_list = list(range(12, 121, 4))
    qs = [0.0, 0.05, 0.1, 0.5, 1.0]
    rhos = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_seeds = 3

    # Precompute spectra
    print("precomputing spectra ...", flush=True)
    spec = {}
    for q in qs:
        for n in n_list:
            for s in range(n_seeds):
                rng = np.random.default_rng(50021 * s + 7919 * n + int(q * 1000))
                sc = ws_complex(n, k_deg, q, rng)
                lam = np.linalg.eigvalsh(sc.laplacian(0.0).toarray())
                spec[(q, n, s)] = lam[lam > 1e-9]
    print("done\n", flush=True)

    rows = []
    print(f"{'q':>5} {'ρ':>6} {'mean N_in':>10} {'P(unstable)':>8} {'contrast':>8} "
          f"{'band density':>13}")
    print("-" * 62)
    for q in qs:
        for rho in rhos:
            n_in_all, p_n = [], []
            for n in n_list:
                cnt_in, cnt_unst = 0.0, 0
                for s in range(n_seeds):
                    lam = spec[(q, n, s)]
                    if lam.size == 0:
                        continue
                    sc_lam = rho * lam
                    k_in = int(((sc_lam > lo) & (sc_lam < hi)).sum())
                    cnt_in += k_in
                    if k_in > 0:
                        cnt_unst += 1
                n_in_all.append(cnt_in / n_seeds)
                p_n.append(cnt_unst / n_seeds)
            p_n = np.array(p_n)
            mean_in = float(np.mean(n_in_all))
            contrast = float(p_n.std())
            # in-band spectral density: eigenvalues per unit width after mapping the band back to the original spectral scale
            dens = mean_in / max(width / rho, 1e-12)
            rows.append({"q": q, "rho": float(rho), "mean_n_in": mean_in,
                         "p_unstable": float(p_n.mean()), "contrast": contrast,
                         "band_density": float(dens)})
            print(f"{q:>5.2f} {rho:>6.2f} {mean_in:>10.2f} {p_n.mean():>8.2f} "
                  f"{contrast:>8.3f} {dens:>13.4f}")
        print("-" * 62)

    p = RESULTS / "exp_gap_law.json"
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Criterion test: is the contrast only a function of mean(N_in)? ===")
    # Bin by mean N_in and inspect the contrast
    bins = [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 8.0), (8.0, 1e9)]
    print(f"{'mean N_in bin':>16} {'samples':>7} {'contrast mean':>11} {'contrast max':>11} "
          f"{'q involved':>18}")
    for a, b in bins:
        sub = [r for r in rows if a <= r["mean_n_in"] < b]
        if sub:
            c = np.array([r["contrast"] for r in sub])
            qset = sorted({r["q"] for r in sub})
            print(f"[{a:.1f}, {b if b < 1e8 else float('inf'):.1f})".rjust(16) +
                  f" {len(sub):>7} {c.mean():>11.3f} {c.max():>11.3f} "
                  f"{str(qset):>18}")

    lo_g = [r for r in rows if r["mean_n_in"] < 1.0]
    hi_g = [r for r in rows if r["mean_n_in"] > 8.0]
    if lo_g and hi_g:
        print(f"\n  mean N_in < 1 : contrast {np.mean([r['contrast'] for r in lo_g]):.3f}")
        print(f"  mean N_in > 8 : contrast {np.mean([r['contrast'] for r in hi_g]):.3f}")
    print(f"\n[out] {p}")


if __name__ == "__main__":
    main()
