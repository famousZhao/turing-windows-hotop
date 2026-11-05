"""
Analytic prediction of the size windows of Turing instability on higher-order structures.

For a circulant structure the Laplacian eigenvalues have an exact symbolic form

    lambda_k = f(2 pi k / n),   k = 0, 1, ..., n-1

where f(theta) is the dispersion symbol of the structure. Let the Turing unstable band
be (lam_lo, lam_hi). An unstable mode exists if and only if some integer k gives
2 pi k / n in (theta_lo, theta_hi), where f(theta_lo) = lam_lo and f(theta_hi) = lam_hi
on the monotone branch of f. Hence n must fall in the union of

    2 pi k / theta_hi < n < 2 pi k / theta_lo,   k = 1, 2, 3, ...

which is an exact and analytically computable size window.

Symbols of the three structures

    ring1   only +-1 edges                  f = 2 - 2 cos(theta)
    ring2   +-1 and +-2 edges               f = 4 - 2 cos(theta) - 2 cos(2 theta)
    simpl   ring2 plus contiguous triangles f = 10 - 6 cos(theta) - 4 cos(2 theta)
            (symbol of L0 plus symbol of L_tri, the latter being
             6 - 4 cos(theta) - 2 cos(2 theta))

The small-angle expansion exposes the structural difference:

    f_ring1 ~ 1 theta^2     f_ring2 ~ 5 theta^2     f_simpl ~ 11 theta^2

Higher-order coupling amplifies the low end of the spectrum by an order of magnitude,
so at fixed n the largest nonzero eigenvalue of the higher-order structure can jump
over the unstable band. That is the origin of the on/off switching in n.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complex import SimplicialComplex  # noqa: E402
from rd import Schnakenberg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def f_sym(kind: str):
    if kind == "ring1":
        return lambda t: 2.0 - 2.0 * np.cos(t)
    if kind == "ring2":
        return lambda t: 4.0 - 2.0 * np.cos(t) - 2.0 * np.cos(2 * t)
    if kind == "simpl":
        return lambda t: 10.0 - 6.0 * np.cos(t) - 4.0 * np.cos(2 * t)
    raise ValueError(kind)


def build(n: int, kind: str) -> SimplicialComplex:
    ring1 = [(i, (i + 1) % n) for i in range(n)]
    ring2 = ring1 + [(i, (i + 2) % n) for i in range(n)]
    if kind == "ring1":
        return SimplicialComplex.from_edges(n, ring1)
    if kind == "ring2":
        return SimplicialComplex.from_edges(n, ring2)
    sc = SimplicialComplex.from_edges(n, ring2)
    for i in range(n):
        sc.add_triangle(i, (i + 1) % n, (i + 2) % n)
    return sc


def sigma_of(kind: str) -> float:
    return 1.0 if kind == "simpl" else 0.0


def theta_roots(kind: str, lo: float, hi: float) -> tuple[float, float]:
    """Solve f(θ)=lo and f(θ)=hi on the monotonically increasing branch of f."""
    f = f_sym(kind)
    from scipy.optimize import brentq

    grid = np.linspace(1e-9, np.pi, 200001)
    vals = f(grid)
    # upper bound of the monotone interval
    imax = int(np.argmax(vals))
    g = lambda t: f(t) - lo  # noqa: E731
    hl = lambda t: f(t) - hi  # noqa: E731
    t_lo = brentq(g, grid[0] + 1e-12, grid[imax])
    t_hi = brentq(hl, grid[0] + 1e-12, grid[imax])
    return float(t_lo), float(t_hi)


def predict_windows(kind: str, lo: float, hi: float, n_max: int):
    """Return (instability mask, θ_lo, θ_hi, list of windows)."""
    t_lo, t_hi = theta_roots(kind, lo, hi)
    unstable = np.zeros(n_max + 1, dtype=bool)
    windows = []
    k = 1
    while 2 * np.pi * k / t_hi <= n_max:
        n_lo = int(np.floor(2 * np.pi * k / t_hi)) + 1
        n_hi = int(np.ceil(2 * np.pi * k / t_lo)) - 1
        n_lo = max(n_lo, 3)
        if n_hi >= n_lo:
            unstable[n_lo : n_hi + 1] = True
            windows.append((n_lo, n_hi, k))
        k += 1
    return unstable, t_lo, t_hi, windows


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n_max = 300
    print(f"Schnakenberg Turing band: Λ ∈ ({lo:.4f}, {hi:.4f})")
    print(f"scan n = 3..{n_max}\n")

    summary = {}
    for kind in ("ring1", "ring2", "simpl"):
        print("=" * 74)
        f = f_sym(kind)
        # coefficient of the θ² term as θ→0
        t = 1e-6
        c2 = f(t) / t**2
        pred, t_lo, t_hi, windows = predict_windows(kind, lo, hi, n_max)
        print(f"--- {kind} ---   f(θ→0) ≈ {c2:.3f} θ²   "
              f"θ_lo={t_lo:.5f}  θ_hi={t_hi:.5f}")
        print(f"    analytic windows (k, n_lo, n_hi): " +
              ", ".join(f"k={k}:{a}-{b}" for a, b, k in windows[:8]) +
              (f" ... {len(windows)} in total" if len(windows) > 8 else ""))

        # numerical check
        fails = []
        for n in range(3, n_max + 1):
            sc = build(n, kind)
            L = sc.laplacian(sigma_of(kind))
            lam = np.linalg.eigvalsh(L.toarray())
            num_unstable = bool(((lam > lo) & (lam < hi)).any())
            if num_unstable != bool(pred[n]):
                fails.append((n, bool(pred[n]), num_unstable))
        n_unstable_pred = int(pred[3:].sum())
        n_unstable_num = n_max - 2 - len(fails)  # rough
        print(f"    numerical check, n in 3..{n_max}: {'exact match' if not fails else f'{len(fails)} mismatches'}")
        if fails:
            print(f"      mismatches (first 10): {fails[:10]}")
        summary[kind] = {
            "quad_coeff": c2, "theta_lo": t_lo, "theta_hi": t_hi,
            "n_windows": len(windows), "n_unstable_predicted": n_unstable_pred,
            "mismatches": len(fails), "coverage": float(pred[3:].mean()),
        }
        print(f"    predicted unstable n fraction: {pred[3:].mean()*100:.1f}%\n")

    out = RESULTS / "analytic_windows.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out}")

    # key comparison: do the three structures agree on instability at equal size
    print("\n" + "=" * 74)
    print("Comparison: instability of the three structures for n=10..60  (○=pattern possible  ×=none)")
    print("=" * 74)
    header = "    n | " + "".join(f"{n%10}" for n in range(10, 61))
    print(header)
    for kind in ("ring1", "ring2", "simpl"):
        pred, *_ = predict_windows(kind, lo, hi, 60)
        row = "".join("○" if pred[n] else "×" for n in range(10, 61))
        print(f"{kind:>5} | {row}")


if __name__ == "__main__":
    main()
