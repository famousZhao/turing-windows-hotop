"""
Hysteresis scan: does higher-order coupling make the Turing bifurcation subcritical.

Observation

At Dv = 9.0, just past threshold, the final amplitude of the simplicial structure
depends on the initial condition (0.0975 from a small perturbation against 0.1652 from
a large one, 70 percent apart), while the graph structure gives the same value under
both conditions (0.2560). That suggests bistability on the simplicial structure, the
signature of a subcritical bifurcation.

Bistability is not the only explanation, though: two initial conditions giving two
results can also mean the integration time is too short. Adiabatic continuation is
needed:

    upward    raise Dv slowly from 6 to 12, each step starting from the previous
              final state
    downward  lower Dv slowly from 12 back to 6 by the same continuation

If upward and downward give different amplitudes at the same Dv there is a real
hysteresis loop and the bifurcation is subcritical.

Each Dv is also repeated from an independent initial condition (a small random
perturbation), as a cross-check that the same attractor is reached.
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


def build(n: int, kind: str) -> SimplicialComplex:
    ring1 = [(i, (i + 1) % n) for i in range(n)]
    ring2 = ring1 + [(i, (i + 2) % n) for i in range(n)]
    if kind == "ring1":
        return SimplicialComplex.from_edges(n, ring1)
    sc = SimplicialComplex.from_edges(n, ring2)
    for i in range(n):
        sc.add_triangle(i, (i + 1) % n, (i + 2) % n)
    return sc


def integrate(L, model, u, v, dt, T):
    for _ in range(int(round(T / dt))):
        u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
    return u, v


def amp_of(u):
    p = u - u.mean()
    return float(np.abs(p).max())


def main() -> None:
    n = 120
    Dvs = np.round(np.arange(6.0, 12.01, 0.25), 2)
    T = 600.0          # long enough to saturate (slowest case 1/μ* ~ 170)
    rows = []

    for kind in ("ring1", "simpl"):
        sc = build(n, kind)
        L = sc.laplacian(1.0 if kind == "simpl" else 0.0)
        lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
        print(f"\n=== {kind} ===")
        print(f"{'Dv':>6} {'μ*':>8} {'A up':>9} {'A down':>9} {'diff':>9} {'A indep':>9}")

        # upward: start from a small perturbation at Dv=6 and continue step by step
        rng = np.random.default_rng(11)
        model0 = Schnakenberg(Dv=Dvs[0])
        dt0 = min(0.02, 2.5 / ((model0.Du + model0.Dv) * lam_max) * 0.8)
        u = model0.ustar + 1e-6 * rng.standard_normal(n)
        v = model0.vstar + 1e-6 * rng.standard_normal(n)
        up = {}
        for Dv in Dvs:
            model = Schnakenberg(Dv=Dv)
            dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
            u, v = integrate(L, model, u, v, dt, min(T, 200.0))
            up[Dv] = amp_of(u)

        # downward: continue from the end state of the upward sweep
        down = {}
        for Dv in Dvs[::-1]:
            model = Schnakenberg(Dv=Dv)
            dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
            u, v = integrate(L, model, u, v, dt, min(T, 200.0))
            down[Dv] = amp_of(u)

        max_hyst = 0.0
        for Dv in Dvs:
            model = Schnakenberg(Dv=Dv)
            dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
            an = model.spectrum_analysis(L)
            # cross-check from an independent initial condition
            rng2 = np.random.default_rng(77)
            ui = model.ustar + 1e-3 * rng2.standard_normal(n)
            vi = model.vstar + 1e-3 * rng2.standard_normal(n)
            ui, vi = integrate(L, model, ui, vi, dt, T)
            a_ind = amp_of(ui)
            d = abs(up[Dv] - down[Dv])
            max_hyst = max(max_hyst, d)
            print(f"{Dv:>6.2f} {an['mu_star']:>8.4f} {up[Dv]:>9.4f} "
                  f"{down[Dv]:>9.4f} {d:>9.4f} {a_ind:>9.4f}", flush=True)
            rows.append({"kind": kind, "Dv": float(Dv),
                         "mu_star": float(an["mu_star"]),
                         "amp_up": up[Dv], "amp_down": down[Dv],
                         "hysteresis": d, "amp_independent": a_ind,
                         "n_unstable": an["n_unstable"]})

        print(f"  max hysteresis difference = {max_hyst:.4f}  "
              f"-> {'hysteresis loop present (subcritical)' if max_hyst > 0.05 else 'no hysteresis (supercritical)'}")

    out = RESULTS / "exp_hysteresis.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
