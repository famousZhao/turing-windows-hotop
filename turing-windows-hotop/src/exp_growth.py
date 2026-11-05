"""
Main experiment: a Turing pattern on a growing domain, adiabatic versus non-adiabatic.

Design

Reaction-diffusion on a growing ring. The ring grows from n0 to n1, one node is added
every dtau of time, and a new node starts in the uniform state. Two kinds of support:

    graph      edges only             classical network Turing problem
    simplicial edges + triangles      higher-order interaction, L = L0 + sigma L_tri

Observable

    wavenumber = pattern^T L pattern / pattern^T pattern,   pattern = u - mean(u)

evaluated with the final Laplacian: the physical wavenumber the pattern actually
occupies.

Adiabatic prediction

Quasi-static theory says the final pattern should occupy the eigenvalue where mu is
largest. If growth is slow enough the measured wavenumber approaches it; if growth is
fast the pattern cannot track the evolving spectrum and lags behind.

The experiment therefore tests directly:

    H1  whether the measured wavenumber approaches the adiabatic value monotonically
        with growth rate
    H2  whether the higher-order term sigma changes this convergence
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


def ring_complex(n: int, simplicial: bool) -> SimplicialComplex:
    """Ring of n nodes; if simplicial=True also add triangles from consecutive triples."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    sc = SimplicialComplex.from_edges(n, edges)
    if simplicial and n >= 3:
        for i in range(n):
            sc.add_triangle(i, (i + 1) % n, (i + 2) % n)
    return sc


def stability_limit(sc: SimplicialComplex, model: Schnakenberg, sigma: float) -> float:
    """Return the largest allowed dt (RK4 real-axis stability margin ~2.78)."""
    L = sc.laplacian(sigma)
    lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
    return 2.5 / ((model.Du + model.Dv) * max(lam_max, 1e-9))


def grow(n0: int, n1: int, dtau: float, sigma: float, simplicial: bool,
         model: Schnakenberg, dt: float, amp: float = 1e-2,
         seed: int = 0) -> dict:
    """Grow from n0 to n1 and return the final state and the history."""
    rng = np.random.default_rng(seed)
    sc = ring_complex(n0, simplicial)
    L = sc.laplacian(sigma)
    u = model.ustar + amp * rng.standard_normal(n0)
    v = model.vstar + amp * rng.standard_normal(n0)

    steps_per_add = max(1, int(round(dtau / dt)))
    hist = []
    for n_target in range(n0 + 1, n1 + 1):
        for _ in range(steps_per_add):
            u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        sc = ring_complex(n_target, simplicial)
        L = sc.laplacian(sigma)
        u = np.append(u, model.ustar)
        v = np.append(v, model.vstar)
        if n_target % 10 == 0:
            hist.append({"n": n_target, "amp": float(np.abs(u - model.ustar).max())})
    return {"sc": sc, "L": L, "u": u, "v": v, "n": n1, "hist": hist}


def analyze(res: dict, model: Schnakenberg) -> dict:
    """Compute the physical wavenumber Λ_dom of the final pattern and compare it with the adiabatic prediction Λ*."""
    L, u = res["L"], res["u"]
    pat = u - u.mean()
    denom = float(pat @ pat)
    if denom < 1e-20:
        return {"lambda_dom": np.nan, "amplitude": 0.0, "note": "no_pattern"}
    lam_dom = float((pat @ (L @ pat)) / denom)
    an = model.spectrum_analysis(L)
    return {
        "lambda_dom": lam_dom,
        "lambda_star": an["lam_star"],
        "mu_star": an["mu_star"],
        "n_unstable": an["n_unstable"],
        "amplitude": float(np.abs(pat).max()),
        "turing_band": an["turing_band"],
        "n": res["n"],
    }


def main() -> None:
    model = Schnakenberg()
    n0, n1 = 40, 120
    dtaos = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    sigmas = [0.0, 1.0]  # 0 = graph only; 1 = add the triangle higher-order term

    print(f"Schnakenberg a={model.a} b={model.b} Du={model.Du} Dv={model.Dv}")
    print(f"Turing analytic band: {model.turing_band()}")
    print(f"Ring grows from n={n0} to n={n1}\n")

    rows = []
    for sigma in sigmas:
        simplicial = sigma > 0
        sc_full = ring_complex(n1, simplicial)
        an_full = model.spectrum_analysis(sc_full.laplacian(sigma))
        dt_lim = stability_limit(sc_full, model, sigma)
        dt = min(0.02, dt_lim * 0.4)
        tag = "simplicial(σ=%.1f)" % sigma if simplicial else "graph(σ=0)"
        print(f"=== {tag} | final structure: edges={len(sc_full.edges)} triangles={len(sc_full.triangles)} "
              f"| unstable modes={an_full['n_unstable']} | Λ*={an_full['lam_star']:.4f} "
              f"μ*={an_full['mu_star']:.5f} | dt={dt:.4f}")
        for dtau in dtaos:
            total_t = dtau * (n1 - n0)
            if total_t / dt > 4_000_000:
                print(f"  dτ={dtau:>6} skipped (too many steps)")
                continue
            res = grow(n0, n1, dtau, sigma, simplicial, model, dt, seed=7)
            a = analyze(res, model)
            a["dtau"] = dtau
            a["sigma"] = sigma
            rows.append(a)
            dev = (a["lambda_dom"] - a["lambda_star"]) / a["lambda_star"] * 100
            print(f"  dτ={dtau:>6.1f}  T={total_t:>8.0f}  Λ_dom={a['lambda_dom']:>7.4f}  "
                  f"Λ*={a['lambda_star']:.4f}  deviation={dev:>7.2f}%  amplitude={a['amplitude']:.3f}")
        print()

    out = RESULTS / "exp_growth.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
