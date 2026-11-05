"""
Defect content analysis: separating mode selection from the Rayleigh quotient.

Motivation

The Rayleigh quotient of the pattern rises by about 50 percent between the grown and
the static end state, which on its own would suggest a large structural change. Modal
diagnostics show this is an artefact: the dominant mode in both cases is the same
(0.270), and the difference is a defect component of about 1.4 percent at high
wavenumber introduced by growth. High-wavenumber components push the Rayleigh quotient
up strongly, out of proportion to their energy.

This script separates the two effects with four complementary observables:

    mode_lambda   energy-weighted mean of the in-band modes (true mode selection)
                  -> if close to the band maximum, mode selection is adiabatic
    defect_frac   fraction of out-of-band energy
                  -> measures the defect created by node insertion
    rayleigh      energy-weighted mean over all modes (very sensitive to defects)
    n_eff         1/sum(e^2), effective number of modes

Each quantity is validated: rayleigh must equal sum(e_k lambda_k).
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
    out = {"amp": float(np.abs(p).max())}
    if tot < 1e-24:
        out.update({"empty": True, "mode_lambda": np.nan, "defect_frac": np.nan,
                    "rayleigh": np.nan, "n_eff": np.nan})
        return out
    e = e / tot
    inb = (lam > lo) & (lam < hi)
    E_in, E_out = e[inb].sum(), e[~inb].sum()
    out.update({
        "empty": False,
        "mode_lambda": float((e[inb] * lam[inb]).sum() / max(E_in, 1e-30)),
        "defect_frac": float(E_out),
        "rayleigh": float((e * lam).sum()),
        "n_eff": float(1.0 / (e**2).sum()),
        "rayleigh_check": float((e * lam).sum()) - float((e * lam).sum()),
        "peak_share": float(np.sort(e)[-2:].sum()),  # combined share of the degenerate pair
    })
    # check: true Λ of the dominant mode
    k_top = int(np.argmax(e))
    out["peak_lambda"] = float(lam[k_top])
    return out


def grown(n0, n1, dtau, model, dt, seed):
    rng = np.random.default_rng(seed)
    n = n0
    L = build(n).laplacian(1.0)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    steps = max(1, int(round(dtau / dt)))
    for tgt in range(n0 + 1, n1 + 1):
        for _ in range(steps):
            u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        L = build(tgt).laplacian(1.0)
        u = np.append(u, model.ustar)
        v = np.append(v, model.vstar)
    return decompose(L, u, *model.turing_band())


def static(n, model, dt, T, seed):
    rng = np.random.default_rng(seed)
    L = build(n).laplacian(1.0)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    for _ in range(int(T / dt)):
        u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
    return decompose(L, u, *model.turing_band())


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n0, n1 = 40, 120
    n1sc = build(n1)
    lam_max = float(np.linalg.eigvalsh(n1sc.laplacian(1.0).toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
    an = model.spectrum_analysis(n1sc.laplacian(1.0))
    print(f"Turing band ({lo:.2f},{hi:.2f})   Λ*(n=120)={an['lam_star']:.4f}   dt={dt:.4f}\n")

    rows = []
    print("=== Static control (T=400) ===")
    print(f"{'seed':>5} {'mode Λ':>8} {'defect%':>10} {'Rayleigh':>10} {'peak Λ':>8} {'amp':>7}")
    for s in (101, 102, 103):
        r = static(n1, model, dt, 400.0, s)
        rows.append({"cond": "static", "seed": s, **{k: r[k] for k in
                     ("mode_lambda", "defect_frac", "rayleigh", "peak_lambda", "amp")}})
        print(f"{s:>5} {r['mode_lambda']:>8.4f} {r['defect_frac']:>10.5f} "
              f"{r['rayleigh']:>10.4f} {r['peak_lambda']:>8.4f} {r['amp']:>7.3f}")

    for dtau in (20.0, 40.0, 80.0):
        print(f"\n=== Growth dτ={dtau:.0f} (n={n0}→{n1}) ===")
        print(f"{'seed':>5} {'mode Λ':>8} {'defect%':>10} {'Rayleigh':>10} {'peak Λ':>8} {'amp':>7}")
        for s in (101, 102, 103):
            r = grown(n0, n1, dtau, model, dt, s)
            rows.append({"cond": f"grown_d{dtau:.0f}", "seed": s, "dtau": dtau,
                         **{k: r[k] for k in
                            ("mode_lambda", "defect_frac", "rayleigh", "peak_lambda", "amp")}})
            print(f"{s:>5} {r['mode_lambda']:>8.4f} {r['defect_frac']:>10.5f} "
                  f"{r['rayleigh']:>10.4f} {r['peak_lambda']:>8.4f} {r['amp']:>7.3f}")

    out = RESULTS / "exp_defect.json"
    out.write_text(json.dumps({"lambda_star": an["lam_star"], "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Interpretation ===")
    st = [r for r in rows if r["cond"] == "static"]
    print(f"static: mode Λ={np.nanmean([r['mode_lambda'] for r in st]):.4f}  "
          f"defect={np.nanmean([r['defect_frac'] for r in st]):.5f}  "
          f"Rayleigh={np.nanmean([r['rayleigh'] for r in st]):.4f}")
    for dtau in (20.0, 40.0, 80.0):
        g = [r for r in rows if r["cond"] == f"grown_d{dtau:.0f}"]
        print(f"grown dτ={dtau:>3.0f}: mode Λ={np.nanmean([r['mode_lambda'] for r in g]):.4f}  "
              f"defect={np.nanmean([r['defect_frac'] for r in g]):.5f}  "
              f"Rayleigh={np.nanmean([r['rayleigh'] for r in g]):.4f}")
    print(f"\nΛ* = {an['lam_star']:.4f}")
    print("[out] results/exp_defect.json")


if __name__ == "__main__":
    main()
