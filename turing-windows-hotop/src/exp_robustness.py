"""
Robustness hardening: a second set of dynamical parameters plus different growth
granularity.

Goals

1. The dynamical results of the main text used a single parameter set (0.1, 0.9, 1, 10).
   This script repeats the core conclusions with a second set (0.05, 1.00, 1, 10),
   which has a different Turing band:
       mode selection is adiabatic (the in-band wavenumber equals the band maximum)
       growth creates defects, and they heal after growth stops
2. Above, only single-node sequential insertion was tested. This script tests batch
   insertion, adding k nodes at once, and checks whether the defect level depends on
   growth granularity.

Criteria, consistent with the main text

    adiabatic  |mode_lambda - band maximum| < 1e-3
    defect     grown end state defect_frac > static end state
    healing    after holding, defect_frac returns to the static level
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


def build(n: int) -> SimplicialComplex:
    edges = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 2) % n) for i in range(n)]
    sc = SimplicialComplex.from_edges(n, edges)
    for i in range(n):
        sc.add_triangle(i, (i + 1) % n, (i + 2) % n)
    return sc


def decompose(L, u, lo, hi):
    lam, vecs = np.linalg.eigh(L.toarray())
    p = u - u.mean()
    c = vecs.T @ p
    e = c**2
    tot = e.sum()
    if tot < 1e-24:
        return {"mode_lambda": np.nan, "defect_frac": np.nan,
                "rayleigh": np.nan, "amp": float(np.abs(p).max())}
    e = e / tot
    inb = (lam > lo) & (lam < hi)
    E_in = e[inb].sum()
    return {
        "mode_lambda": float((e[inb] * lam[inb]).sum() / max(E_in, 1e-30)),
        "defect_frac": float(e[~inb].sum()),
        "rayleigh": float((e * lam).sum()),
        "amp": float(np.abs(p).max()),
    }


def grow_burst(n0, n1, dtau, model, dt, seed, burst=1, rule="uniform"):
    """Add burst nodes at once at each step."""
    rng = np.random.default_rng(seed)
    n = n0
    L = build(n).laplacian(1.0)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    steps = max(1, int(round(dtau / dt)))
    tgt = n0
    while tgt < n1:
        for _ in range(steps):
            u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        tgt = min(n1, tgt + burst)
        L = build(tgt).laplacian(1.0)
        add = tgt - len(u)
        if rule == "neighbor_mean":
            fill_u = np.full(add, 0.5 * (u[-1] + u[0]))
            fill_v = np.full(add, 0.5 * (v[-1] + v[0]))
        else:
            fill_u = np.full(add, model.ustar)
            fill_v = np.full(add, model.vstar)
        u = np.append(u, fill_u)
        v = np.append(v, fill_v)
    return L, u, v


def main() -> None:
    n0, n1 = 40, 120
    rows = []

    # Part 1: second parameter set
    print("=" * 70)
    print("Part 1: second dynamical parameter set (a,b,Du,Dv) = (0.05, 1.00, 1, 10)")
    print("=" * 70)
    model2 = Schnakenberg(a=0.05, b=1.00, Du=1.0, Dv=10.0)
    lo2, hi2 = model2.turing_band()
    sc = build(n1)
    L = sc.laplacian(1.0)
    an = model2.spectrum_analysis(L)
    lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
    dt = min(0.012, 2.5 / ((model2.Du + model2.Dv) * lam_max) * 0.8)
    print(f"band ({lo2:.3f},{hi2:.3f})   Λ*(n=120)={an['lam_star']:.4f}   "
          f"unstable modes={an['n_unstable']}   dt={dt:.4f}")
    if an["n_unstable"] == 0:
        print("  [skip] at this parameter set n=120 is outside the unstable window")
    else:
        # static reference
        rng = np.random.default_rng(101)
        us = model2.ustar + 1e-2 * rng.standard_normal(n1)
        vs = model2.vstar + 1e-2 * rng.standard_normal(n1)
        for _ in range(int(400.0 / dt)):
            us, vs = rk4_step(L, model2.reaction, us, vs, dt, model2.Du, model2.Dv)
        st = decompose(L, us, lo2, hi2)
        print(f"  static: modeΛ={st['mode_lambda']:.4f}  defect={st['defect_frac']*100:.3f}%  "
              f"Rayleigh={st['rayleigh']:.4f}")

        # growth
        Lg, u, v = grow_burst(n0, n1, 40.0, model2, dt, 101, burst=1)
        g = decompose(Lg, u, lo2, hi2)
        print(f"  grown: modeΛ={g['mode_lambda']:.4f}  defect={g['defect_frac']*100:.3f}%  "
              f"Rayleigh={g['rayleigh']:.4f}")
        # healing
        for _ in range(int(400.0 / dt)):
            u, v = rk4_step(Lg, model2.reaction, u, v, dt, model2.Du, model2.Dv)
        h = decompose(Lg, u, lo2, hi2)
        print(f"  healed: modeΛ={h['mode_lambda']:.4f}  defect={h['defect_frac']*100:.3f}%  "
              f"Rayleigh={h['rayleigh']:.4f}")

        checks = {
            "adiabatic": abs(g["mode_lambda"] - an["lam_star"]) < 1e-3,
            "defect_present": g["defect_frac"] > st["defect_frac"],
            "heals": abs(h["defect_frac"] - st["defect_frac"]) < 5e-4,
            "mode_stable": abs(h["mode_lambda"] - an["lam_star"]) < 1e-3,
        }
        for k, val in checks.items():
            print(f"    [{'PASS' if val else 'FAIL'}] {k}")
        rows.append({"part": 1, "lambda_star": float(an["lam_star"]),
                     "static": st, "grown": g, "healed": h, "checks": checks})

    # Part 2: growth granularity
    print("\n" + "=" * 70)
    print("Part 2: growth granularity (burst nodes inserted at once)")
    print("=" * 70)
    model1 = Schnakenberg()
    lo1, hi1 = model1.turing_band()
    an1 = model1.spectrum_analysis(build(n1).laplacian(1.0))
    print(f"baseline parameters Λ*={an1['lam_star']:.4f}")
    print(f"{'burst':>6} {'modeΛ':>8} {'defect%':>10} {'Rayleigh':>10} {'amp':>7}")
    for burst in (1, 2, 4, 10):
        Lg, u, v = grow_burst(n0, n1, 40.0, model1, dt, 101, burst=burst)
        d = decompose(Lg, u, lo1, hi1)
        print(f"{burst:>6} {d['mode_lambda']:>8.4f} {d['defect_frac']:>10.5f} "
              f"{d['rayleigh']:>10.4f} {d['amp']:>7.3f}")
        rows.append({"part": 2, "burst": burst,
                     "mode_lambda": d["mode_lambda"],
                     "defect_frac": d["defect_frac"],
                     "rayleigh": d["rayleigh"], "amp": d["amp"]})

    out = RESULTS / "exp_robustness.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
