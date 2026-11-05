"""
Mode selection under noise, with mode decomposition.

Background

The seed-to-seed spread of the Rayleigh quotient cannot be read as a change of mode:
that quotient is dominated by defects, so a statement about noise changing the final
mode inferred from it is unreliable. This script redoes the measurement with genuine
observables:

    mode_lambda  in-band energy-weighted wavenumber  tests whether noise changes mode
                                                     selection
    defect_frac  out-of-band energy fraction         tests whether noise changes the
                                                     defect content

Expectation

If noise affects only the defects and not the mode, mode_lambda should stay at the
adiabatic value while defect_frac rises with eta.
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
    e = (c**2)
    tot = e.sum()
    out = {"amp": float(np.abs(p).max())}
    if tot < 1e-24:
        out.update({"mode_lambda": np.nan, "defect_frac": np.nan,
                    "rayleigh": np.nan})
        return out
    e = e / tot
    inb = (lam > lo) & (lam < hi)
    E_in = e[inb].sum()
    out.update({
        "mode_lambda": float((e[inb] * lam[inb]).sum() / max(E_in, 1e-30)),
        "defect_frac": float(e[~inb].sum()),
        "rayleigh": float((e * lam).sum()),
    })
    return out


def grow(n0, n1, dtau, model, dt, eta, seed):
    rng = np.random.default_rng(seed)
    n = n0
    L = build(n).laplacian(1.0)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    steps = max(1, int(round(dtau / dt)))
    for tgt in range(n0 + 1, n1 + 1):
        for _ in range(steps):
            u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
            if eta > 0:
                sq = np.sqrt(dt) * eta
                u = u + sq * rng.standard_normal(u.shape)
                v = v + sq * rng.standard_normal(v.shape)
        L = build(tgt).laplacian(1.0)
        u = np.append(u, model.ustar)
        v = np.append(v, model.vstar)
    return decompose(L, u, *model.turing_band())


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n0, n1 = 40, 120
    sc = build(n1)
    lam_max = float(np.linalg.eigvalsh(sc.laplacian(1.0).toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
    an = model.spectrum_analysis(sc.laplacian(1.0))
    print(f"Λ*={an['lam_star']:.4f}  dt={dt:.4f}  dτ=40\n")
    print(f"{'η':>8} {'seed':>5} {'mode Λ':>9} {'defect frac':>10} {'Rayleigh':>10} {'amp':>7}")
    rows = []
    for eta in (0.0, 1e-3, 1e-2):
        for seed in (101, 102, 103, 104):
            r = grow(n0, n1, 40.0, model, dt, eta, seed)
            rows.append({"eta": eta, "seed": seed, **r})
            print(f"{eta:>8.0e} {seed:>5} {r['mode_lambda']:>9.4f} "
                  f"{r['defect_frac']:>10.5f} {r['rayleigh']:>10.4f} {r['amp']:>7.3f}",
                  flush=True)
        print()

    out = RESULTS / "exp_noise_modes.json"
    out.write_text(json.dumps({"lambda_star": an["lam_star"], "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== interpretation ===")
    print(f"Λ* = {an['lam_star']:.4f}")
    for eta in (0.0, 1e-3, 1e-2):
        sub = [r for r in rows if r["eta"] == eta]
        ml = np.array([r["mode_lambda"] for r in sub])
        df = np.array([r["defect_frac"] for r in sub])
        ry = np.array([r["rayleigh"] for r in sub])
        print(f"η={eta:.0e}: mode Λ = {ml.mean():.4f} ± {ml.std(ddof=1):.4f} | "
              f"defect = {df.mean()*100:.2f}% ± {df.std(ddof=1)*100:.2f}% | "
              f"Rayleigh = {ry.mean():.4f} ± {ry.std(ddof=1):.4f}")
    ml_all = np.array([r["mode_lambda"] for r in rows if np.isfinite(r["mode_lambda"])])
    print(f"\nmax deviation of mode Λ from Λ* over all {ml_all.size} runs: "
          f"{np.abs(ml_all - an['lam_star']).max():.5f}")
    print("[out] results/exp_noise_modes.json")


if __name__ == "__main__":
    main()
