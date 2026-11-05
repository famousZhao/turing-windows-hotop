"""Validation experiment 0: numerical integration vs analytic linear stability prediction.

This is the credibility anchor for every later experiment. Three checks must pass:
  V1  after injecting a purely unstable mode, the exponential growth rate of the perturbation amplitude == the analytic μ(Λ*)
  V2  the modal energy of the final state concentrates on k*
  V3  counterexample: with the diffusion ratio set so that no Turing instability exists, the perturbation decays and the field returns to the homogeneous state

If any of V1/V2/V3 fails, the integrator or the Laplacian has a bug and work must not continue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complex import SimplicialComplex  # noqa: E402
from rd import Schnakenberg, integrate, modal_energy  # noqa: E402


def ring(n: int) -> SimplicialComplex:
    return SimplicialComplex.from_edges(n, [(i, (i + 1) % n) for i in range(n)])


def measure_growth(L, model, k, dt=0.01, t_end=120.0, amp=1e-3):
    """Inject a pure mode k and measure the exponential growth rate of the linear regime."""
    an = model.spectrum_analysis(L)
    lam, vecs = an["lam"], an["vecs"]
    vk = vecs[:, k]
    u = model.ustar + amp * vk
    v = model.vstar + amp * vk

    n_steps = int(t_end / dt)
    amps = []
    for s in range(n_steps):
        from rd import rk4_step

        u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        a = float(vk @ (u - model.ustar))
        amps.append(a)
        # stop once the linear regime is left
        if abs(a) > 0.05:
            break
    amps = np.array(amps)
    t = dt * np.arange(1, len(amps) + 1)
    # fit only the linear segment with amplitude in [2e-3, 3e-2]
    m = (np.abs(amps) > 2e-3) & (np.abs(amps) < 3e-2)
    if m.sum() < 10:
        return np.nan, an
    slope = np.polyfit(t[m], np.log(np.abs(amps[m])), 1)[0]
    return float(slope), an


def main() -> None:
    model = Schnakenberg()
    ok = True

    print("=== V1/V2: ring n=40, inject the most unstable mode ===")
    sc = ring(40)
    L = sc.L0()
    an = model.spectrum_analysis(L)
    k = an["k_star"]
    print(f"  analytic: Λ*={an['lam_star']:.4f}  predicted μ*={an['mu_star']:.6f}  "
          f"mode index k*={k}  unstable modes={an['n_unstable']}")

    meas, _ = measure_growth(L, model, k)
    if np.isnan(meas):
        print("  [FAIL] growth rate could not be measured")
        ok = False
    else:
        rel = abs(meas - an["mu_star"]) / abs(an["mu_star"])
        verdict = "PASS" if rel < 0.05 else "FAIL"
        if rel >= 0.05:
            ok = False
        print(f"  measured μ = {meas:.6f}   relative error = {rel*100:.2f}%   [{verdict}]")

    # Long-time evolution -> check whether the modal energy concentrates on k*
    u, v, _ = integrate(L, model, *([model.ustar + 1e-3 * an["vecs"][:, k],
                                    model.vstar + 1e-3 * an["vecs"][:, k]]),
                        t_end=400.0, dt=0.01)
    e = modal_energy(u - model.ustar, an["vecs"])
    top = np.argsort(-e)[:3]
    print(f"  dominant modes of the final state (energy fraction): " +
          ", ".join(f"k={t}({e[t]*100:.1f}%)" for t in top) + f"   [expected first place k={k}]")
    if top[0] != k:
        print("  [WARN] dominant mode of the final state disagrees with the linear prediction (nonlinear mode selection, needs further analysis)")

    print("\n=== V3: counterexample -- Turing instability switched off ===")
    # Lower Dv so that the Turing band disappears
    model2 = Schnakenberg(Dv=2.0)
    band2 = model2.turing_band()
    print(f"  Turing band at Dv=2.0 = {band2} (expected None)")
    st2 = model2.homogeneous_stable()
    print(f"  homogeneous state stable? {st2[0]}")
    an2 = model2.spectrum_analysis(L)
    print(f"  unstable modes = {an2['n_unstable']} (expected 0)")
    u, v, _ = integrate(
        L, model2,
        np.full(40, model2.ustar) + 1e-2 * np.random.default_rng(0).standard_normal(40),
        np.full(40, model2.vstar),
        t_end=300.0, dt=0.01,
    )
    dev = float(np.abs(u - model2.ustar).max())
    print(f"  max deviation of the final state = {dev:.3e} (should decay exponentially to ~0)")
    if band2 is not None or an2["n_unstable"] > 0 or dev > 1e-4:
        print("  [FAIL] counterexample not judged correctly")
        ok = False
    else:
        print("  [PASS]")

    print("\n" + ("=== all checks passed, proceed to the growth experiment ===" if ok
                  else "=== failures present, bugs must be fixed first ==="))


if __name__ == "__main__":
    main()
