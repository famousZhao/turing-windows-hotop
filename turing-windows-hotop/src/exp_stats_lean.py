"""Lean statistics: distribution and significance of growth end state vs static end
state (efficient version).

Basis for the optimization (measured in exp_control.py):
    Λ_dom of the static system is fully unchanged for T>=400, so the static control
    only needs the integration to T=400, which saves 87% of the time compared with
    "as long as growth" (T=3200) and gives the same conclusion.

Conditions
    growth: σ=1, n=40->120, dτ=40, η=0     (range where the pattern forms stably)
    static: σ=1, n=120, T=400, η=0
    12 seeds per cell, individual values reported so that a test can be done.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complex import SimplicialComplex  # noqa: E402
from rd import Schnakenberg, rk4_step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def build(n: int, simplicial: bool) -> SimplicialComplex:
    edges = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 2) % n) for i in range(n)]
    sc = SimplicialComplex.from_edges(n, edges)
    if simplicial:
        for i in range(n):
            sc.add_triangle(i, (i + 1) % n, (i + 2) % n)
    return sc


def lam_dom(L, u):
    pat = u - u.mean()
    return float((pat @ (L @ pat)) / (pat @ pat + 1e-300)), float(np.abs(pat).max())


def main() -> None:
    model = Schnakenberg()
    sigma, n0, n1 = 1.0, 40, 120
    seeds = list(range(101, 113))
    sc = build(n1, True)
    L = sc.laplacian(sigma)
    lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
    an = model.spectrum_analysis(L)
    print(f"σ={sigma} dt={dt:.4f} static Λ*={an['lam_star']:.4f} seeds={seeds[0]}..{seeds[-1]}\n")

    # static
    T = 400.0
    stat_vals, stat_amp = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        Ls = sc.laplacian(sigma)
        u = model.ustar + 1e-2 * rng.standard_normal(n1)
        v = model.vstar + 1e-2 * rng.standard_normal(n1)
        for _ in range(int(T / dt)):
            u, v = rk4_step(Ls, model.reaction, u, v, dt, model.Du, model.Dv)
        l, a = lam_dom(Ls, u)
        stat_vals.append(l)
        stat_amp.append(a)
    stat_vals = np.array(stat_vals)
    print(f"static (n=120, T={T:.0f}): Λ_dom = {stat_vals.mean():.4f} ± "
          f"{stat_vals.std(ddof=1):.4f}  [{stat_vals.min():.3f}, {stat_vals.max():.3f}]  "
          f"amp={np.mean(stat_amp):.3f}", flush=True)

    # growth
    dtau = 40.0
    grow_vals, grow_amp = [], []
    steps = max(1, int(round(dtau / dt)))
    for s in seeds:
        rng = np.random.default_rng(s)
        n = n0
        Lg = build(n, True).laplacian(sigma)
        u = model.ustar + 1e-2 * rng.standard_normal(n)
        v = model.vstar + 1e-2 * rng.standard_normal(n)
        for tgt in range(n0 + 1, n1 + 1):
            for _ in range(steps):
                u, v = rk4_step(Lg, model.reaction, u, v, dt, model.Du, model.Dv)
            Lg = build(tgt, True).laplacian(sigma)
            u = np.append(u, model.ustar)
            v = np.append(v, model.vstar)
        l, a = lam_dom(Lg, u)
        grow_vals.append(l)
        grow_amp.append(a)
        print(f"  seed {s}: Λ_dom={l:.4f} amp={a:.3f}", flush=True)
    grow_vals = np.array(grow_vals)
    print(f"\ngrowth (n=40→120, dτ={dtau:.0f}): Λ_dom = {grow_vals.mean():.4f} ± "
          f"{grow_vals.std(ddof=1):.4f}  [{grow_vals.min():.3f}, {grow_vals.max():.3f}]  "
          f"amp={np.mean(grow_amp):.3f}")

    # tests
    ustat, p = st.mannwhitneyu(grow_vals, stat_vals, alternative="two-sided")
    gt = sum((x > y) for x in grow_vals for y in stat_vals)
    lt = sum((x < y) for x in grow_vals for y in stat_vals)
    delta = (gt - lt) / (grow_vals.size * stat_vals.size)
    # relative deviation
    rel = (grow_vals.mean() - an["lam_star"]) / an["lam_star"] * 100
    rel_s = (stat_vals.mean() - an["lam_star"]) / an["lam_star"] * 100
    print(f"\n=== tests ===")
    print(f"  static vs linear prediction Λ*={an['lam_star']:.4f}: deviation {rel_s:+.2f}%  (linear theory should be accurate)")
    print(f"  growth vs linear prediction:             deviation {rel:+.2f}%")
    print(f"  Mann–Whitney U = {ustat:.1f},  p = {p:.3e},  Cliff's δ = {delta:+.3f}")

    out = RESULTS / "exp_stats.json"
    out.write_text(json.dumps({
        "lambda_star": an["lam_star"],
        "grown": {"dtau": dtau, "n0": n0, "n1": n1, "seeds": seeds,
                  "values": grow_vals.tolist(), "amp": grow_amp,
                  "mean": float(grow_vals.mean()),
                  "std": float(grow_vals.std(ddof=1))},
        "static": {"n": n1, "T": T, "seeds": seeds,
                   "values": stat_vals.tolist(), "amp": stat_amp,
                   "mean": float(stat_vals.mean()),
                   "std": float(stat_vals.std(ddof=1))},
        "test": {"mannwhitney_u": float(ustat), "p": float(p),
                 "cliffs_delta": float(delta)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
