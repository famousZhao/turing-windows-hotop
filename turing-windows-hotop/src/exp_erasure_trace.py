"""
Trace the mechanism by which the pattern is erased.

The puzzle

    dtau = 80    amplitude 0.431 (survives)
    dtau = 160   amplitude 0.000 (erased)

The dead zone n = 47..58 spans only 12 nodes. If the decay rate inside the dead zone
were 0.043, the accumulated decay across the zone at dtau = 80 would be e^-41 and the
pattern would long since be gone. It is not, which means a saturated pattern is
nonlinearly stable inside the dead zone; the erasure mechanism is not linear decay.

This script records the amplitude node by node along the growth path to locate the
segment where erasure actually happens.

Criterion: the segment where the amplitude starts to decrease monotonically is where
erasure occurs.
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


def grow_trace(n0, n1, dtau, sigma, model, dt, seed, amp=1e-2):
    """Grow and record amplitude and linear stability node by node."""
    rng = np.random.default_rng(seed)
    n = n0
    sc = build(n, sigma > 0)
    L = sc.laplacian(sigma)
    u = model.ustar + amp * rng.standard_normal(n)
    v = model.vstar + amp * rng.standard_normal(n)
    steps = max(1, int(round(dtau / dt)))
    trace = []
    for tgt in range(n0, n1 + 1):
        if tgt > n0:
            for _ in range(steps):
                u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
            sc = build(tgt, sigma > 0)
            L = sc.laplacian(sigma)
            u = np.append(u, model.ustar)
            v = np.append(v, model.vstar)
        an = model.spectrum_analysis(L)
        pat = u - u.mean()
        trace.append({
            "n": tgt,
            "amp": float(np.abs(pat).max()),
            "mu_star": an["mu_star"],
            "lambda_star": an["lam_star"],
            "n_unstable": an["n_unstable"],
        })
    return trace


def main() -> None:
    model = Schnakenberg()
    n0, n1 = 40, 120
    sigma = 1.0
    sc = build(n1, True)
    L = sc.laplacian(sigma)
    lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)

    out_all = {}
    for dtau in (80.0, 120.0, 160.0):
        tr = grow_trace(n0, n1, dtau, sigma, model, dt, seed=7)
        out_all[str(dtau)] = tr
        amps = np.array([t["amp"] for t in tr])
        print(f"\n=== dτ={dtau}  dt={dt:.4f}  final amplitude={amps[-1]:.4f}")
        # find where the amplitude first drops below half the peak
        peak_i = int(np.argmax(amps))
        half = amps[peak_i] * 0.5
        drop_i = next((i for i in range(peak_i, len(amps)) if amps[i] < half), None)
        print(f"    peak amplitude={amps[peak_i]:.3f} @ n={tr[peak_i]['n']}")
        if drop_i is not None:
            print(f"    first drop below half @ n={tr[drop_i]['n']}  (decay starts at n={tr[peak_i]['n']})")
        else:
            print("    never drops below half")
        print(f"    {'n':>5} {'amp':>9} {'mu*':>9} {'#unst':>6} {'Λ*':>8}")
        for t in tr:
            if t["n"] % 5 == 0 or t is tr[0]:
                print(f"    {t['n']:>5} {t['amp']:>9.4f} {t['mu_star']:>9.4f} "
                      f"{t['n_unstable']:>6} {t['lambda_star']:>8.4f}")

    out = RESULTS / "exp_erasure_trace.json"
    out.write_text(json.dumps(out_all, indent=1), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
