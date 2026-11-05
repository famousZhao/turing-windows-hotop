"""
Bifurcation type: saturation time and hysteresis protocol.

Requirements for a correct measurement

1. Saturation time. At Dv = 9 (mu* = 0.006) an integration of T = 1200 covers about
   seven e-folds, so the amplitude is far from saturation and the fitted scaling
   exponent is polluted. The integration time must scale as T ~ 1/mu*.
2. Hysteresis protocol. Starting below the pole from a zero perturbation and sweeping
   upward, the perturbation decays exponentially and never grows back, so both sweep
   directions read zero. The pattern must be carried in both directions: start from
   an existing pattern at high Dv, sweep down, then back up.
3. Noise floor. A purely deterministic system underflows to exact zero; a small noise
   term prevents this unphysical total disappearance.

Protocol

A. Scaling: each point is integrated for T = 200 + 120/mu* to ensure saturation, and
   the exponent of A ~ mu*^beta is fitted.
B. Hysteresis: Dv goes from 12 down to 6 and back to 12 by continuation, recording
   the amplitude difference between the two directions.
C. Noise floor eta = 1e-6 prevents underflow.
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
N = 120


def build(kind: str) -> SimplicialComplex:
    ring1 = [(i, (i + 1) % N) for i in range(N)]
    ring2 = ring1 + [(i, (i + 2) % N) for i in range(N)]
    if kind == "ring1":
        return SimplicialComplex.from_edges(N, ring1)
    sc = SimplicialComplex.from_edges(N, ring2)
    for i in range(N):
        sc.add_triangle(i, (i + 1) % N, (i + 2) % N)
    return sc


def amp_of(u):
    p = u - u.mean()
    return float(np.abs(p).max())


def run(L, model, u, v, dt, T, eta=1e-6, rng=None):
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        if eta > 0 and rng is not None:
            u = u + eta * np.sqrt(dt) * rng.standard_normal(u.shape)
    return u, v


def main() -> None:
    rows = []
    for kind in ("ring1", "simpl"):
        sc = build(kind)
        L = sc.laplacian(1.0 if kind == "simpl" else 0.0)
        lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
        print(f"\n{'='*66}\n=== {kind}  (n={N}, lam_max={lam_max:.2f}) ===\n{'='*66}")

        # A. scaling
        print("\n[A] saturated amplitude vs distance past threshold (T = 200 + 120/μ*)")
        print(f"{'Dv':>6} {'μ*':>9} {'Λ*':>8} {'T':>10} {'A':>9}")
        pts = []
        for Dv in (9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 14.0):
            model = Schnakenberg(Dv=Dv)
            an = model.spectrum_analysis(L)
            mu = an["mu_star"]
            if mu <= 1e-4:
                continue
            dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
            T = 200.0 + 120.0 / mu
            if T / dt > 3_000_000:
                print(f"{Dv:>6.1f} {mu:>9.4f} skipped (step count {T/dt:.0f} too large)")
                continue
            rng = np.random.default_rng(3)
            u = model.ustar + 1e-3 * rng.standard_normal(N)
            v = model.vstar + 1e-3 * rng.standard_normal(N)
            u, v = run(L, model, u, v, dt, T, eta=0.0)
            a = amp_of(u)
            print(f"{Dv:>6.1f} {mu:>9.4f} {an['lam_star']:>8.4f} {T:>10.0f} {a:>9.4f}",
                  flush=True)
            rows.append({"kind": kind, "part": "A", "Dv": Dv, "mu_star": float(mu),
                         "amp": a, "T": T})
            pts.append((mu, a))
        if len(pts) >= 3:
            x = np.log([p[0] for p in pts])
            y = np.log([p[1] for p in pts])
            beta, ic = np.polyfit(x, y, 1)
            r2 = 1 - ((y - (beta * x + ic)) ** 2).sum() / (((y - y.mean()) ** 2).sum())
            print(f"    scaling: A ∝ μ*^{beta:.3f}   R²={r2:.4f}   (supercritical=0.500)")
            rows.append({"kind": kind, "part": "A_fit", "beta": float(beta),
                         "r2": float(r2), "n": len(pts)})

        # B. hysteresis
        print("\n[B] hysteresis: Dv 12→6→12, continued throughout (pattern carried along)")
        Dvs = np.round(np.arange(12.0, 5.99, -0.5), 2)
        rng = np.random.default_rng(5)
        m0 = Schnakenberg(Dv=12.0)
        dt0 = min(0.02, 2.5 / ((m0.Du + m0.Dv) * lam_max) * 0.8)
        u = m0.ustar + 1e-3 * rng.standard_normal(N)
        v = m0.vstar + 1e-3 * rng.standard_normal(N)
        u, v = run(L, m0, u, v, dt0, 400.0, eta=1e-6, rng=rng)

        down = {}
        for Dv in Dvs:
            model = Schnakenberg(Dv=Dv)
            dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
            u, v = run(L, model, u, v, dt, 300.0, eta=1e-6, rng=rng)
            down[Dv] = amp_of(u)
        up = {}
        for Dv in Dvs[::-1]:
            model = Schnakenberg(Dv=Dv)
            dt = min(0.02, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
            u, v = run(L, model, u, v, dt, 300.0, eta=1e-6, rng=rng)
            up[Dv] = amp_of(u)

        print(f"{'Dv':>6} {'μ*':>9} {'A down':>9} {'A up':>9} {'diff':>9}")
        mx = 0.0
        for Dv in Dvs:
            model = Schnakenberg(Dv=Dv)
            an = model.spectrum_analysis(L)
            d = abs(up[Dv] - down[Dv])
            mx = max(mx, d)
            print(f"{Dv:>6.1f} {an['mu_star']:>9.4f} {down[Dv]:>9.4f} "
                  f"{up[Dv]:>9.4f} {d:>9.4f}")
            rows.append({"kind": kind, "part": "B", "Dv": float(Dv),
                         "mu_star": float(an["mu_star"]),
                         "amp_down": down[Dv], "amp_up": up[Dv], "hyst": d})
        print(f"    max hysteresis difference = {mx:.4f}  "
              f"-> {'hysteresis loop present (subcritical)' if mx > 0.05 else 'no hysteresis (supercritical)'}")
        rows.append({"kind": kind, "part": "B_summary", "max_hyst": mx})

    out = RESULTS / "exp_bifurcation.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
