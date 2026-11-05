"""Disorder scan: how much disorder does the "size window" structure of Turing instability survive?

Background
----------
On a regular ring, instability depends on whether an integer k places f(2πk/n) inside the
unstable band, so the outcome shows exact arithmetic on/off windows in n,
analytically predictable and verified at all 298 sizes.

Real materials and biological tissues are not perfectly regular. The key question:
    Is this window structure a pathological special case of regular lattices, or a
    physical effect with some robustness?

Method
------
Watts–Strogatz small-world model: k nearest neighbors on a ring, each edge rewired with
probability q.
    q=0     regular lattice (O(n) triangles, strongest window structure)
    q=0.1   weak disorder
    q=1.0   ER random graph (strongest disorder)
q serves as a continuous scale for disorder and the decay of the window structure is recorded.

Quantifying the window structure
-------------------------------
For fixed (q, σ), scan n to obtain P(unstable; n) ∈ {0,1} (averaged over seeds).
    window contrast = std_n P(unstable; n), near 0.5 for regular structures,
                      tending to 0 under disorder
    transition count = number of 0/1 flips along n
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
    """Watts–Strogatz: k nearest neighbors on a ring, each edge rewired with probability q."""
    half = k // 2
    edges = set()
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            edges.add((min(i, j), max(i, j)))
    for (i, j) in list(edges):
        if rng.random() < q:
            edges.discard((i, j))
            for _ in range(20):
                w = int(rng.integers(0, n))
                if w == i:
                    continue
                e = (min(i, w), max(i, w))
                if e not in edges:
                    edges.add(e)
                    break
    return sorted(edges)


def ws_complex(n: int, k: int, q: float, rng) -> SimplicialComplex:
    """Take all 3-cliques of the WS graph 1-skeleton as 2-simplices (flag complex)."""
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


def unstable_in(L, lo: float, hi: float) -> bool:
    lam = np.linalg.eigvalsh(L.toarray())
    return bool(((lam > lo) & (lam < hi)).any())


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n_seeds = 5
    k = 6          # mean degree
    qs = [0.0, 0.05, 0.2, 0.5, 1.0]
    sigmas = [0.0, 0.5]
    ns = list(range(12, 121, 2))

    print(f"Turing band Λ ∈ ({lo:.3f}, {hi:.3f})   mean degree k={k}   seeds={n_seeds}\n")
    rows = []
    for sigma in sigmas:
        print(f"##########  σ = {sigma}  ##########")
        print(f"{'q':>6} {'P(unst)':>9} {'contrast':>11} {'flips':>9} {'<#tri/n>':>9}")
        for q in qs:
            p_n = []
            tri_ratio = []
            for n in ns:
                cnt = 0
                for s in range(n_seeds):
                    rng = np.random.default_rng(1000 * s + 7919 * n + int(q * 1000))
                    sc = ws_complex(n, k, q, rng)
                    tri_ratio.append(len(sc.triangles) / n)
                    if unstable_in(sc.laplacian(sigma), lo, hi):
                        cnt += 1
                p_n.append(cnt / n_seeds)
            p_n = np.array(p_n)
            contrast = float(p_n.std())
            flips = int(np.abs(np.diff((p_n > 0.5).astype(int))).sum())
            print(f"{q:>6.2f} {p_n.mean():>9.2f} {contrast:>11.3f} {flips:>9d} "
                  f"{np.mean(tri_ratio):>9.2f}")
            rows.append({
                "sigma": sigma, "q": q, "k": k, "n_range": [ns[0], ns[-1]],
                "p_unstable_mean": float(p_n.mean()),
                "window_contrast": contrast, "n_transitions": flips,
                "tri_per_node": float(np.mean(tri_ratio)),
                "p_unstable_by_n": p_n.tolist(),
            })
        print()

    out = RESULTS / "exp_disorder.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out}")

    print("\n=== Reading ===")
    for sigma in sigmas:
        sub = [r for r in rows if r["sigma"] == sigma]
        c = [r["window_contrast"] for r in sub]
        print(f"σ={sigma}: window contrast vs q: " +
              " -> ".join(f"q={r['q']}:{r['window_contrast']:.3f}" for r in sub) +
              f"   (monotonic decay? {all(c[i] >= c[i+1] - 0.02 for i in range(len(c)-1))})")


if __name__ == "__main__":
    main()
