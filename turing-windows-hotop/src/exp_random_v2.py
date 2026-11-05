"""
Universality check on random simplicial complexes.

Why the sparse graph is not enough

A sparse random graph with mean degree 4 produces only O(1) triangles (8.6 at n = 120),
so the triangle term does nothing and the spectrum barely changes as sigma goes from 0
to 2. The test has insufficient power.

Design

1. Raise the graph density (mean degree 10) so that the triangle count reaches O(n).
2. Paired design: for the same random graph, compute the spectrum at sigma = 0 (graph
   only) and at sigma > 0 (triangles added). The 1-skeleton is identical, so the only
   difference is the 2-simplices, which isolates the causal effect.
3. Report the paired difference in the lowest nonzero eigenvalue, per seed, with a sign
   test.

Expectation: the triangle term raises the low end of the spectrum roughly on the scale
of theta^2, so the paired difference is positive and part of the modes that would fall
inside the unstable band are pushed out of it.
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


def random_graph(n: int, k_avg: float, rng) -> list[tuple[int, int]]:
    p = k_avg / (n - 1)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def flag_complex(n: int, k_avg: float, rng) -> SimplicialComplex:
    edges = random_graph(n, k_avg, rng)
    sc = SimplicialComplex.from_edges(n, edges)
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        for j in sorted(adj[i]):
            if j <= i:
                continue
            for k in sorted(adj[i] & adj[j]):
                if k > j:
                    sc.add_triangle(i, j, k)
    return sc


def spectrum_stats(L, lo: float, hi: float) -> dict:
    lam = np.linalg.eigvalsh(L.toarray())
    nz = lam[lam > 1e-9]
    return {
        "lam1": float(nz[0]) if nz.size else np.nan,
        "lam2": float(nz[1]) if nz.size > 1 else np.nan,
        "lammax": float(lam.max()),
        "unstable": bool(((lam > lo) & (lam < hi)).any()),
        "n_in_band": int(((lam > lo) & (lam < hi)).sum()),
    }


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    ns = [30, 60, 120, 240]
    k_avg = 10.0
    n_seeds = 15
    sigmas = [0.25, 0.5, 1.0, 2.0]
    master = np.random.default_rng(424242)

    print(f"Turing band Λ ∈ ({lo:.3f}, {hi:.3f})   mean degree {k_avg}   {n_seeds} random graphs\n")
    rows = []

    for n in ns:
        print(f"=== n={n} ===")
        # fix a batch of random graphs first so every σ uses the same 1-skeleton
        graphs = []
        for _ in range(n_seeds):
            rng = np.random.default_rng(int(master.integers(1 << 31)))
            graphs.append(flag_complex(n, k_avg, rng))
        n_tri = np.mean([len(g.triangles) for g in graphs])
        print(f"  mean triangle count = {n_tri:.1f}  ({n_tri/n:.2f} per node)")

        base = [spectrum_stats(g.laplacian(0.0), lo, hi) for g in graphs]
        p_base = np.mean([b["unstable"] for b in base])
        l1_base = np.nanmean([b["lam1"] for b in base])
        print(f"  σ=0     P(unstable)={p_base:.2f}  <λ1>={l1_base:.4f}  "
              f"<λmax>={np.mean([b['lammax'] for b in base]):.3f}")

        for sigma in sigmas:
            cur = [spectrum_stats(g.laplacian(sigma), lo, hi) for g in graphs]
            p = np.mean([c["unstable"] for c in cur])
            l1 = np.nanmean([c["lam1"] for c in cur])
            d = [c["lam1"] - b["lam1"] for c, b in zip(cur, base)
                 if np.isfinite(c["lam1"]) and np.isfinite(b["lam1"])]
            d = np.array(d)
            n_pos = int((d > 0).sum())
            print(f"  σ={sigma:<5} P(unstable)={p:.2f}  <λ1>={l1:.4f}  "
                  f"Δλ1={d.mean():+.4f}  (increased {n_pos}/{len(d)})  "
                  f"<λmax>={np.mean([c['lammax'] for c in cur]):.3f}")
            rows.append({
                "n": n, "sigma": sigma, "n_tri_per_node": float(n_tri / n),
                "p_unstable": float(p), "p_unstable_base": float(p_base),
                "mean_lam1": float(l1), "mean_lam1_base": float(l1_base),
                "delta_lam1_mean": float(d.mean()),
                "delta_lam1_positive": n_pos, "n_pairs": int(len(d)),
                "mean_lammax": float(np.mean([c["lammax"] for c in cur])),
            })
        print()

    out = RESULTS / "exp_random_v2.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out}")

    print("\n=== Mechanism interpretation ===")
    print(f"{'n':>5} {'σ':>6} {'Δλ1':>10} {'sign test':>12} {'P unst(σ)':>9} {'P unst(0)':>9}")
    for r in rows:
        frac = r["delta_lam1_positive"] / max(r["n_pairs"], 1)
        print(f"{r['n']:>5} {r['sigma']:>6.2f} {r['delta_lam1_mean']:>+10.4f} "
              f"{r['delta_lam1_positive']:>5}/{r['n_pairs']:<6} "
              f"{r['p_unstable']:>9.2f} {r['p_unstable_base']:>9.2f}")


if __name__ == "__main__":
    main()
