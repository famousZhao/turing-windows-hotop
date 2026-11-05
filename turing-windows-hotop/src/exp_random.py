"""Universality check: does the mechanism hold only on the special "ring + consecutive triangles" structure?

Mechanism under test (from the analytic window theory):
    Higher-order (2-simplex) coupling multiplies the low end of the spectrum by roughly a
    factor (1 + c·σ), which raises the "minimum pattern-forming size" and widens the dead
    zone where instability is lost.

If the mechanism is universal, random simplicial complexes should also show:
    P1  σ ↑ -> the low end of the spectrum is raised (smallest nonzero eigenvalue grows)
    P2  σ ↑ -> the probability of Turing instability drops (dead zone widens)
    P3  on random structures the on/off windows should be washed out (denser spectrum),
        but the "minimum size threshold" should survive

Structure families
    flag    : take all 3-cliques of a G(n, k) random graph as 2-simplices (flag complex)
    rand    : random edges plus random triangles (no clique closure required), more general
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


def er_edges(n: int, k_avg: float, rng) -> list[tuple[int, int]]:
    """Erdős–Rényi random graph with target mean degree k_avg."""
    p = k_avg / (n - 1)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    return edges


def flag_complex(n: int, k_avg: float, rng) -> SimplicialComplex:
    """Flag complex: all 3-cliques of G(n,p) as 2-simplices."""
    edges = er_edges(n, k_avg, rng)
    sc = SimplicialComplex.from_edges(n, edges)
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k > j:
                    sc.add_triangle(i, j, k)
    return sc


def rand_complex(n: int, k_avg: float, tri_ratio: float, rng) -> SimplicialComplex:
    """Random edges plus random triangles (triangle count = tri_ratio × node count)."""
    edges = er_edges(n, k_avg, rng)
    sc = SimplicialComplex.from_edges(n, edges)
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    # add triangles only on existing edges, no extra edges (1-skeleton stays fixed)
    cand = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k > j:
                    cand.append((i, j, k))
    n_tri = int(tri_ratio * n)
    if cand:
        rng.shuffle(cand)
        for t in cand[:n_tri]:
            sc.add_triangle(*t)
    return sc


def low_spectrum(L, k: int = 3) -> np.ndarray:
    """The k smallest nonzero eigenvalues."""
    lam = np.linalg.eigvalsh(L.toarray())
    nz = lam[lam > 1e-9]
    return nz[:k] if nz.size else np.array([np.nan])


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    rng_master = np.random.default_rng(20240607)

    ns = [20, 40, 60, 80, 120]
    sigmas = [0.0, 0.25, 0.5, 1.0, 2.0]
    n_seeds = 10
    k_avg = 4.0

    print(f"Turing band Λ ∈ ({lo:.3f}, {hi:.3f})   random structures: mean degree {k_avg}, "
          f"{n_seeds} seeds\n")

    rows = []
    print(f"{'kind':>6} {'n':>4} {'σ':>5} {'P(unst)':>8} {'rand':>5} "
          f"{'<λ1>':>8} {'<λ2>':>8} {'<λmax>':>8} {'<#tris>':>8}")
    print("-" * 74)
    for kind in ("flag", "rand"):
        for n in ns:
            for sigma in sigmas:
                unstable_cnt = 0
                l1, l2, lmax, ntri = [], [], [], []
                for s in range(n_seeds):
                    rng = np.random.default_rng(rng_master.integers(1 << 31))
                    sc = flag_complex(n, k_avg, rng) if kind == "flag" \
                        else rand_complex(n, k_avg, 1.0, rng)
                    L = sc.laplacian(sigma)
                    lam = np.linalg.eigvalsh(L.toarray())
                    nz = lam[lam > 1e-9]
                    if nz.size:
                        l1.append(nz[0])
                        l2.append(nz[1] if nz.size > 1 else np.nan)
                    lmax.append(lam.max())
                    ntri.append(len(sc.triangles))
                    if ((lam > lo) & (lam < hi)).any():
                        unstable_cnt += 1
                p_unst = unstable_cnt / n_seeds
                n_rand = int(ntri and (unstable_cnt and unstable_cnt < n_seeds))
                row = {
                    "kind": kind, "n": n, "sigma": sigma,
                    "p_unstable": p_unst,
                    "mean_lam1": float(np.nanmean(l1)) if l1 else np.nan,
                    "mean_lam2": float(np.nanmean(l2)) if l2 else np.nan,
                    "mean_lammax": float(np.mean(lmax)),
                    "mean_ntri": float(np.mean(ntri)),
                    "purely_stochastic": bool(0 < p_unst < 1),
                }
                rows.append(row)
                print(f"{kind:>6} {n:>4} {sigma:>5.2f} {p_unst:>8.2f} "
                      f"{'yes' if row['purely_stochastic'] else 'no':>5} "
                      f"{row['mean_lam1']:>8.4f} {row['mean_lam2']:>8.4f} "
                      f"{row['mean_lammax']:>8.4f} {row['mean_ntri']:>8.1f}")
        print("-" * 74)

    out = RESULTS / "exp_random.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")

    # mechanism readout
    print("\n=== Mechanism readout ===")
    for kind in ("flag", "rand"):
        print(f"\n{kind}:")
        for sigma in sigmas:
            sub = [r for r in rows if r["kind"] == kind and r["sigma"] == sigma and r["n"] >= 40]
            if not sub:
                continue
            p = np.mean([r["p_unstable"] for r in sub])
            l1 = np.nanmean([r["mean_lam1"] for r in sub])
            print(f"  σ={sigma:>4.2f}  mean instability probability={p:.2f}   mean λ1={l1:.4f}")


if __name__ == "__main__":
    main()
