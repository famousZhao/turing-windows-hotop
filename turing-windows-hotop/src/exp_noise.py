"""
Noise test: recast permanent erasure as a competition between deterministic memory and
noise reseeding.

Question

The deterministic system, left to itself, either keeps or loses the pattern according
to the composition of a residual of order 1e-13, which is an artefact of floating-point
arithmetic. A real system carries thermal noise, and the noise continually reseeds
small perturbations inside the dead zone. Three questions follow:

    Q1  with noise added, is the pattern still erased at dtau = 160?
    Q2  how does the erase/survive boundary move with noise strength and growth rate?
    Q3  what should permanent erasure be recast as?

Theoretical estimate

Inside the dead zone the pattern decays at rate kappa equal to the absolute value of
the maximum growth rate, while noise drives a steady-state amplitude (Ornstein-Uhlenbeck
balance):

    A_noise ~ eta / sqrt(2 kappa)

The time for an initial amplitude A0 to decay to the noise floor is

    T_star = ln(A0 / A_noise) / kappa

The dwell time inside the dead zone is T_dead = dtau * L_dead, with L_dead = 12 for the
simplicial structure. Hence the criterion

    dtau < T_star / L_dead   the residual stays above the noise floor,
                             deterministic memory
    dtau > T_star / L_dead   the residual is taken over by the noise, noise reseeding

that is, a critical growth rate dtau_c = ln(A0 sqrt(2 kappa) / eta) / (kappa L_dead).

Stochastic integration (Euler-Maruyama for the noise, RK4 for the deterministic part)

    du = [f - Du L u] dt + eta dW
    dv = [g - Dv L v] dt + eta dW'
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


def em_step(L, model, u, v, dt, eta, rng):
    """One deterministic RK4 step plus additive white noise (Euler–Maruyama)."""
    u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
    if eta > 0:
        sq = np.sqrt(dt) * eta
        u = u + sq * rng.standard_normal(u.shape)
        v = v + sq * rng.standard_normal(v.shape)
    return u, v


def grow_noisy(n0, n1, dtau, sigma, model, dt, eta, seed, amp=1e-2):
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
                u, v = em_step(L, model, u, v, dt, eta, rng)
            sc = build(tgt, sigma > 0)
            L = sc.laplacian(sigma)
            u = np.append(u, model.ustar)
            v = np.append(v, model.vstar)
        trace.append(float(np.abs(u - u.mean()).max()))
    pat = u - u.mean()
    lam_dom = float((pat @ (L @ pat)) / (pat @ pat + 1e-300))
    an = model.spectrum_analysis(L)
    return {
        "lambda_dom": lam_dom,
        "amp_final": float(np.abs(pat).max()),
        "amp_min_deadzone": float(np.min(trace[7:19])) if len(trace) > 19 else float(np.min(trace)),
        "lambda_star": an["lam_star"],
        "trace": trace,
    }


def main() -> None:
    model = Schnakenberg()
    sigma = 1.0
    n0, n1 = 40, 120
    sc = build(n1, True)
    L = sc.laplacian(sigma)
    lam_max = float(np.linalg.eigvalsh(L.toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)

    # slowest decay rate inside the dead zone (at n=50)
    sc50 = build(50, True)
    an50 = model.spectrum_analysis(sc50.laplacian(sigma))
    gamma = abs(min(an50["mu_star"], 0.0))
    A0, L_dead = 0.43, 12
    print(f"σ={sigma}  dt={dt:.4f}  dead zone [47,58], {L_dead} nodes")
    print(f"slowest decay rate in the dead zone γ = {gamma:.4f}  (μ* at n=50 = {an50['mu_star']:.4f})")
    print(f"theoretical critical growth rate dτ_c = ln(A0·√(2γ)/η) / (γ·L_dead),  A0={A0}\n")
    print(f"{'η':>8} {'A_noise':>9} {'T*':>8} {'dτ_c':>8}")
    for eta in (1e-4, 1e-3, 1e-2):
        a_n = eta / np.sqrt(2 * gamma)
        Tstar = np.log(A0 / a_n) / gamma
        print(f"{eta:>8.0e} {a_n:>9.4f} {Tstar:>8.1f} {Tstar/L_dead:>8.2f}")
    print()

    rows = []
    for eta in (0.0, 1e-4, 1e-3, 1e-2):
        print(f"##########  η = {eta:.0e}  ##########")
        print(f"{'dτ':>6} {'seed':>5} {'Λ_dom':>9} {'final amp':>9} {'min amp in dead zone':>11}")
        for dtau in (40.0, 160.0):
            for seed in (7, 11):
                r = grow_noisy(n0, n1, dtau, sigma, model, dt, eta, seed)
                rows.append({"eta": eta, "dtau": dtau, "seed": seed,
                             "lambda_dom": r["lambda_dom"],
                             "amp_final": r["amp_final"],
                             "amp_min_deadzone": r["amp_min_deadzone"],
                             "lambda_star": r["lambda_star"]})
                print(f"{dtau:>6.0f} {seed:>5} {r['lambda_dom']:>9.4f} "
                      f"{r['amp_final']:>9.4f} {r['amp_min_deadzone']:>11.3e}", flush=True)
        print()

    out = RESULTS / "exp_noise.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out}")

    print("\n=== Reading ===")
    print(f"{'η':>8} {'amp at dτ=40':>12} {'amp at dτ=160':>13} {'survives?':>7}")
    for eta in (0.0, 1e-4, 1e-3, 1e-2):
        a40 = np.mean([r["amp_final"] for r in rows if r["eta"] == eta and r["dtau"] == 40])
        a160 = np.mean([r["amp_final"] for r in rows if r["eta"] == eta and r["dtau"] == 160])
        print(f"{eta:>8.0e} {a40:>12.4f} {a160:>13.4f} {('yes' if a160 > 0.05 else 'no'):>7}")


if __name__ == "__main__":
    main()
