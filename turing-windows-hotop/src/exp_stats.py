"""
Thicker statistics: distribution comparison of the growth end state against the static
end state, with error bars and a significance test.

Purpose

A two-seed comparison of the Rayleigh quotient suggested a large difference between the
grown and static end states. Two seeds are not enough for a journal claim, and that
quotient is defect-dominated, so a distributional measurement is needed. This script
reports:

    S1  Rayleigh quotient of the growth end state over 12 seeds,
        sigma = 1, dtau in {40, 160}, eta in {0, 1e-3}
    S2  Rayleigh quotient of the static end state (n = 120, no growth) over 12 seeds
    S3  Mann-Whitney U test, growth against static, plus Cliff's delta effect size
    S4  effect of noise on the within-group standard deviation of the mode
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


def em_step(L, model, u, v, dt, eta, rng):
    u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
    if eta > 0:
        sq = np.sqrt(dt) * eta
        u = u + sq * rng.standard_normal(u.shape)
        v = v + sq * rng.standard_normal(v.shape)
    return u, v


def lam_dom(L, u):
    pat = u - u.mean()
    return float((pat @ (L @ pat)) / (pat @ pat + 1e-300)), float(np.abs(pat).max())


def grown(n0, n1, dtau, sigma, model, dt, eta, seed):
    rng = np.random.default_rng(seed)
    n = n0
    sc = build(n, sigma > 0)
    L = sc.laplacian(sigma)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    steps = max(1, int(round(dtau / dt)))
    for tgt in range(n0 + 1, n1 + 1):
        for _ in range(steps):
            u, v = em_step(L, model, u, v, dt, eta, rng)
        sc = build(tgt, sigma > 0)
        L = sc.laplacian(sigma)
        u = np.append(u, model.ustar)
        v = np.append(v, model.vstar)
    return lam_dom(L, u)


def static(n, sigma, model, dt, T, eta, seed):
    rng = np.random.default_rng(seed)
    sc = build(n, sigma > 0)
    L = sc.laplacian(sigma)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    for _ in range(int(round(T / dt))):
        u, v = em_step(L, model, u, v, dt, eta, rng)
    return lam_dom(L, u)


def describe(name, vals):
    v = np.array(vals)
    v = v[np.isfinite(v)]
    return {
        "name": name, "n": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "median": float(np.median(v)) if v.size else float("nan"),
        "min": float(v.min()) if v.size else float("nan"),
        "max": float(v.max()) if v.size else float("nan"),
        "values": v.tolist(),
    }


def main() -> None:
    model = Schnakenberg()
    sigma = 1.0
    n0, n1 = 40, 120
    n_seeds = 12
    seeds = list(range(101, 101 + n_seeds))
    sc = build(n1, True)
    L = sc.laplacian(sigma)
    lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
    an = model.spectrum_analysis(L)
    print(f"σ={sigma}  dt={dt:.4f}  static Λ*={an['lam_star']:.4f}  seeds={n_seeds}\n")

    groups = {}
    for eta in (0.0, 1e-3):
        for dtau in (40.0, 160.0):
            key = f"grown_dtau{dtau:.0f}_eta{eta:.0e}"
            vals = []
            for s in seeds:
                l, a = grown(n0, n1, dtau, sigma, model, dt, eta, s)
                vals.append(l)
            groups[key] = describe(key, vals)
            g = groups[key]
            print(f"{key:<26} n={g['n']:>2}  Λ_dom = {g['mean']:.4f} ± {g['std']:.4f}  "
                  f"[{g['min']:.3f}, {g['max']:.3f}]", flush=True)

    # static control: use the same total duration as growth
    for eta in (0.0, 1e-3):
        T = 40.0 * (n1 - n0)
        key = f"static_T{T:.0f}_eta{eta:.0e}"
        vals = [static(n1, sigma, model, dt, T, eta, s)[0] for s in seeds]
        groups[key] = describe(key, vals)
        g = groups[key]
        print(f"{key:<26} n={g['n']:>2}  Λ_dom = {g['mean']:.4f} ± {g['std']:.4f}  "
              f"[{g['min']:.3f}, {g['max']:.3f}]", flush=True)

    print("\n=== significance test (growth dτ=40, η=0 vs static) ===")
    a = np.array(groups["grown_dtau40_eta0e+00"]["values"])
    b = np.array(groups[f"static_T{40.0*(n1-n0):.0f}_eta0e+00"]["values"])
    u, p = st.mannwhitneyu(a, b, alternative="two-sided")
    # Cliff's delta
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    delta = (gt - lt) / (len(a) * len(b))
    print(f"  grown: {a.mean():.4f} ± {a.std(ddof=1):.4f}")
    print(f"  static: {b.mean():.4f} ± {b.std(ddof=1):.4f}")
    print(f"  Mann–Whitney U={u:.1f}, p={p:.3e}, Cliff's δ={delta:.3f}")
    print(f"  Λ* (linear prediction) = {an['lam_star']:.4f}")

    # exp_stats_lean.py writes exp_stats.json with a different schema; keep this
    # full version on its own file so either can be run without clobbering the other.
    out = RESULTS / "exp_stats_full.json"
    out.write_text(json.dumps({"groups": groups,
                               "test": {"mannwhitney_u": float(u), "p": float(p),
                                        "cliffs_delta": float(delta),
                                        "lambda_star": an["lam_star"]}},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
