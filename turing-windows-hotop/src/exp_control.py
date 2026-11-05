"""
Control experiment: separate the growth effect from long-time nonlinear evolution.

A deviation measured in the growth experiment has one serious confound: slower growth
also means a longer total integration time. The deviation may therefore be nothing but
long-time nonlinear evolution (mode competition, coarsening) with no relation to
growth.

This experiment does two things.

C1  Static control. Hold n fixed at 120 with no growth and integrate for the same total
    time as the growth experiment, then check whether the dominant eigenvalue drifts
    in the same way. If it does, growth is not the cause.

C2  Structure control. Separate more edges from a genuine triangle term:
        ring1 : +-1 edges only (pure ring)
        ring2 : +-1 and +-2 edges (same 1-skeleton as the triangle structure,
                no 2-simplices)
        simpl : ring2 plus triangles (sigma = 1)
    ring2 and simpl share exactly the same 1-skeleton and differ only in 2-simplices,
    which is the clean control for the higher-order term.
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
RESULTS.mkdir(exist_ok=True)


def build(n: int, kind: str, sigma: float) -> SimplicialComplex:
    if kind == "ring1":
        sc = SimplicialComplex.from_edges(n, [(i, (i + 1) % n) for i in range(n)])
    elif kind == "ring2":
        edges = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 2) % n) for i in range(n)]
        sc = SimplicialComplex.from_edges(n, edges)
    elif kind == "simpl":
        edges = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 2) % n) for i in range(n)]
        sc = SimplicialComplex.from_edges(n, edges)
        for i in range(n):
            sc.add_triangle(i, (i + 1) % n, (i + 2) % n)
    else:
        raise ValueError(kind)
    return sc


def static(n: int, kind: str, sigma: float, model, T: float, dt: float, seed: int = 0) -> dict:
    sc = build(n, kind, sigma)
    L = sc.laplacian(sigma)
    rng = np.random.default_rng(seed)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    for _ in range(int(round(T / dt))):
        u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
    pat = u - u.mean()
    lam_dom = float((pat @ (L @ pat)) / (pat @ pat + 1e-300))
    an = model.spectrum_analysis(L)
    return {
        "kind": kind, "sigma": sigma, "n": n, "T": T,
        "lambda_dom": lam_dom,
        "lambda_star": an["lam_star"],
        "mu_star": an["mu_star"],
        "n_unstable": an["n_unstable"],
        "amplitude": float(np.abs(pat).max()),
    }


def main() -> None:
    model = Schnakenberg()
    n = 120
    Ts = [40.0, 80.0, 160.0, 400.0, 800.0, 1600.0, 3200.0]
    rows = []

    print("=" * 78)
    print("C1  static control (n=120, no growth): does Λ_dom drift in time?")
    print("=" * 78)
    for kind, sigma in (("ring1", 0.0), ("ring2", 0.0), ("simpl", 1.0)):
        sc = build(n, kind, sigma)
        L = sc.laplacian(sigma)
        lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
        dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
        an = model.spectrum_analysis(L)
        print(f"\n--- {kind} (σ={sigma}) | triangles={len(sc.triangles)} "
              f"| unstable modes={an['n_unstable']} | Λ*={an['lam_star']:.4f} | dt={dt:.4f}")
        print(f"{'T':>8} {'Λ_dom':>9} {'dev%':>8} {'amp':>8}")
        for T in Ts:
            r = static(n, kind, sigma, model, T, dt)
            r["dtau"] = None
            rows.append(r)
            dev = (r["lambda_dom"] - r["lambda_star"]) / r["lambda_star"] * 100
            print(f"{T:>8.0f} {r['lambda_dom']:>9.4f} {dev:>8.2f} {r['amplitude']:>8.3f}")

    print("\n" + "=" * 78)
    print("C2  structure control: identical 1-skeleton, differing only in 2-simplices (T=800 fixed)")
    print("=" * 78)
    print(f"{'kind':>8} {'edges':>5} {'tri':>7} {'Λ*':>8} {'Λ_dom':>8} {'dev%':>8} {'unstable':>8}")
    for kind, sigma in (("ring1", 0.0), ("ring2", 0.0), ("simpl", 1.0)):
        sc = build(n, kind, sigma)
        L = sc.laplacian(sigma)
        lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
        dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
        r = static(n, kind, sigma, model, 800.0, dt)
        rows.append({**r, "dtau": None, "experiment": "C2"})
        dev = (r["lambda_dom"] - r["lambda_star"]) / r["lambda_star"] * 100
        print(f"{kind:>8} {len(sc.edges):>5} {len(sc.triangles):>7} "
              f"{r['lambda_star']:>8.4f} {r['lambda_dom']:>8.4f} {dev:>8.2f} {r['n_unstable']:>8}")

    out = RESULTS / "exp_control.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
