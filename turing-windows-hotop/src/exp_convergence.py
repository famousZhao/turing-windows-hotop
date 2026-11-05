"""Convergence and robustness checks.

Two questions:
  R1  Does slower growth make Λ_dom of the higher-order case converge to the static value?
      If it does not converge, the "frozen" effect is not "growth is not slow enough"
      but a structural history dependence.
  R2  Are the conclusions stable under a change of random seed? (rules out one-off chance)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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


def grow(n0, n1, dtau, sigma, model, dt, seed, amp=1e-2):
    rng = np.random.default_rng(seed)
    n = n0
    sc = build(n, sigma > 0)
    L = sc.laplacian(sigma)
    u = model.ustar + amp * rng.standard_normal(n)
    v = model.vstar + amp * rng.standard_normal(n)
    steps = max(1, int(round(dtau / dt)))
    for tgt in range(n0 + 1, n1 + 1):
        for _ in range(steps):
            u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        sc = build(tgt, sigma > 0)
        L = sc.laplacian(sigma)
        u = np.append(u, model.ustar)
        v = np.append(v, model.vstar)
    pat = u - u.mean()
    lam_dom = float((pat @ (L @ pat)) / (pat @ pat + 1e-300))
    return lam_dom, float(np.abs(pat).max())


def main() -> None:
    model = Schnakenberg()
    n0, n1 = 40, 120
    rows = []
    for sigma in (0.0, 1.0):
        sc = build(n1, sigma > 0)
        L = sc.laplacian(sigma)
        lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
        dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
        print(f"\n=== sigma={sigma}  dt={dt:.4f}", flush=True)
        print(f"{'dτ':>7} {'seed':>5} {'Λ_dom':>9} {'amp':>8}", flush=True)
        for dtau in (40.0, 80.0, 160.0):
            for seed in (7, 11):
                lam, amp = grow(n0, n1, dtau, sigma, model, dt, seed)
                rows.append({"sigma": sigma, "dtau": dtau, "seed": seed,
                             "lambda_dom": lam, "amplitude": amp})
                print(f"{dtau:>7.1f} {seed:>5} {lam:>9.4f} {amp:>8.3f}", flush=True)

    out = RESULTS / "exp_convergence.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}", flush=True)


if __name__ == "__main__":
    main()
