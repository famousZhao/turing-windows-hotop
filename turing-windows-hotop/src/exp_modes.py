"""
Modal diagnostics: spectral content of the grown end state against the static one.

Hypothesis under test

Growth drives the Rayleigh quotient of the pattern up from the linear prediction 0.270
to about 0.42. Candidate mechanism: node-by-node insertion creates defects in the
pattern and excites short-wavelength components, which raise the quotient.

Without this mechanism the modal distribution of the grown end state should be
qualitatively the same as for the static end state, with energy concentrated near the
band.

Observables

    e_k        (v_k^T p)^2 / sum((v_k^T p)^2),   p = u - mean(u)
    wavenumber sum(e_k lambda_k)        Rayleigh quotient
    high_frac  sum of e_k over lambda_k above the band   out-of-band fraction
    n_eff      1 / sum(e_k^2)           effective mode count
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


def modal_report(L, u, lo: float, hi: float, topn: int = 8) -> dict:
    lam, vecs = np.linalg.eigh(L.toarray())
    p = u - u.mean()
    c = vecs.T @ p
    e = c**2
    tot = e.sum()
    if tot < 1e-30:
        return {"empty": True, "lambda_dom": float("nan")}
    e = e / tot
    order = np.argsort(-e)
    lam_dom = float((e * lam).sum())
    return {
        "empty": False,
        "lambda_dom": lam_dom,
        "high_frac": float(e[lam > hi].sum()),
        "low_frac": float(e[lam < lo].sum()),
        "in_band_frac": float(e[(lam > lo) & (lam < hi)].sum()),
        "n_eff": float(1.0 / (e**2).sum()),
        "top": [{"k": int(k), "lam": float(lam[k]), "share": float(e[k])}
                for k in order[:topn]],
        "amp": float(np.abs(p).max()),
    }


def run_grown(n0, n1, dtau, model, dt, seed, record_at=()):
    sigma = 1.0
    rng = np.random.default_rng(seed)
    n = n0
    L = build(n).laplacian(sigma)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    steps = max(1, int(round(dtau / dt)))
    snaps = {}
    for tgt in range(n0 + 1, n1 + 1):
        for _ in range(steps):
            u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        L = build(tgt).laplacian(sigma)
        u = np.append(u, model.ustar)
        v = np.append(v, model.vstar)
        if tgt in record_at:
            snaps[tgt] = modal_report(L, u, *model.turing_band())
            snaps[tgt]["n"] = tgt
    snaps[n1] = modal_report(L, u, *model.turing_band())
    snaps[n1]["n"] = n1
    return snaps


def run_static(n, model, dt, T, seed):
    sigma = 1.0
    rng = np.random.default_rng(seed)
    L = build(n).laplacian(sigma)
    u = model.ustar + 1e-2 * rng.standard_normal(n)
    v = model.vstar + 1e-2 * rng.standard_normal(n)
    for _ in range(int(T / dt)):
        u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
    return modal_report(L, u, *model.turing_band())


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n0, n1 = 40, 120
    sc = build(n1)
    lam_max = float(np.linalg.eigvalsh(sc.laplacian(1.0).toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
    print(f"Turing band ({lo:.2f}, {hi:.2f})   dt={dt:.4f}\n")

    print("=== Static end state (n=120, T=400, seed 101) ===")
    st = run_static(n1, model, dt, 400.0, 101)
    print(f"  Λ_dom={st['lambda_dom']:.4f}  in-band={st['in_band_frac']:.3f}  "
          f"out-of-band high={st['high_frac']:.4f}  n_eff={st['n_eff']:.2f}")
    print(f"  top {len(st['top'])} modes: " +
          ", ".join(f"(Λ={t['lam']:.3f}:{t['share']*100:.1f}%)" for t in st["top"]))

    print("\n=== Grown end state (n=40→120, dτ=40, seed 101) ===")
    record_at = (45, 50, 60, 80, 100)
    snaps = run_grown(n0, n1, 40.0, model, dt, 101, record_at=record_at)
    for n in sorted(snaps):
        s = snaps[n]
        if s.get("empty"):
            print(f"  n={n:>4}: empty (no pattern)")
            continue
        print(f"  n={n:>4}: Λ_dom={s['lambda_dom']:.4f}  "
              f"in-band={s['in_band_frac']:.3f}  out-of-band high={s['high_frac']:.4f}  "
              f"n_eff={s['n_eff']:>6.2f}  amp={s['amp']:.3f}")
    fin = snaps[n1]
    print(f"\n  top {len(fin['top'])} modes of the grown end state: " +
          ", ".join(f"(Λ={t['lam']:.3f}:{t['share']*100:.1f}%)" for t in fin["top"]))

    print("\n=== Interpretation ===")
    print(f"  static: Λ_dom={st['lambda_dom']:.4f}, n_eff={st['n_eff']:.2f}")
    print(f"  grown: Λ_dom={fin['lambda_dom']:.4f}, n_eff={fin['n_eff']:.2f}")
    print(f"  high_frac: static {st['high_frac']:.4f} vs grown {fin['high_frac']:.4f}")
    verdict = ("defect hypothesis supported" if fin["high_frac"] > st["high_frac"] * 1.5
               else "defect hypothesis not supported")
    print(f"  -> {verdict}")

    out = RESULTS / "exp_modes.json"
    out.write_text(json.dumps({"static": st, "grown": snaps},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
