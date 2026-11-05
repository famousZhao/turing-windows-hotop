"""
Does the optimal wavenumber change with system size n?

The frozen-wavenumber hypothesis

During growth the pattern forms and saturates early, at small n, with the optimal
wavenumber of that moment. A pattern is hard to restructure once formed, so it stays
frozen at that wavenumber. If the optimal wavenumber depends strongly on n, the
measured wavenumber of the end state locks onto the early value and therefore deviates
systematically from the end-state linear prediction.

This would explain why the effect is weak on graph structures, where the optimal
wavenumber varies little with n, and strong on simplicial complexes, where it varies a
lot.

This script measures the optimal wavenumber as a function of n directly and compares
the measured wavenumber of the growth runs against it.
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


def main() -> None:
    model = Schnakenberg()
    band = model.turing_band()
    print(f"Turing analytic interval Λ ∈ ({band[0]:.3f}, {band[1]:.3f})\n")

    ns = list(range(20, 161, 10))
    rows = []
    hdr = f"{'n':>4} | " + " | ".join(f"{k:>22}" for k in ("ring1", "ring2", "simpl"))
    print(hdr)
    print(f"{'':>4} | " + " | ".join(f"{'Λ*':>8}{'μ*':>7}{'#unst':>7}" for _ in range(3)))
    print("-" * len(hdr))
    for n in ns:
        cells = []
        for kind, sigma in (("ring1", 0.0), ("ring2", 0.0), ("simpl", 1.0)):
            sc = build(n, kind)
            L = sc.laplacian(sigma)
            an = model.spectrum_analysis(L)
            # degeneracy: number of eigenvalues within 1e-6 of Λ*
            deg = int((np.abs(an["lam"] - an["lam_star"]) < 1e-6).sum())
            rows.append({
                "n": n, "kind": kind, "sigma": sigma,
                "lambda_star": an["lam_star"],
                "mu_star": an["mu_star"],
                "n_unstable": an["n_unstable"],
                "degeneracy": deg,
            })
            cells.append(f"{an['lam_star']:>8.4f}{an['mu_star']:>7.4f}{an['n_unstable']:>7d}")
        print(f"{n:>4} | " + " | ".join(cells))

    # compare against the end-state Λ_dom of the growth runs
    gp = RESULTS / "exp_growth.json"
    if gp.exists():
        grown = json.loads(gp.read_text(encoding="utf-8"))
        print("\n=== Grown end-state Λ_dom vs initial/final linear prediction ===")
        print(f"{'σ':>4} {'dτ':>6} {'Λ_dom(grown)':>12} {'Λ*(n=40)':>10} {'Λ*(n=120)':>10} "
              f"{'closer':>8}")
        for g in grown:
            kind = "simpl" if g["sigma"] > 0 else "ring1"
            lam40 = next(r["lambda_star"] for r in rows if r["n"] == 40 and r["kind"] == kind)
            lam120 = next(r["lambda_star"] for r in rows if r["n"] == 120 and r["kind"] == kind)
            if not np.isfinite(g["lambda_dom"]):
                continue
            closer = "start" if abs(g["lambda_dom"] - lam40) < abs(g["lambda_dom"] - lam120) else "end"
            print(f"{g['sigma']:>4.0f} {g['dtau']:>6.1f} {g['lambda_dom']:>12.4f} "
                  f"{lam40:>10.4f} {lam120:>10.4f} {closer:>8}")

    out = RESULTS / "exp_spectrum_vs_n.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
