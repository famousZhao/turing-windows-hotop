"""
Parameter set scan: does the analytic window theory hold for several (a, b, Du, Dv)?

Motivation

The limitations state that all quantitative results used a single parameter set, which
invites the question whether they hold only for that set. This script repeats the
exhaustive verification of the main text (n = 3..200, three structures) for several
parameter sets and reports the minimal size.

The window condition itself does not depend on the kinetics beyond the band edges, so
this tests the generality of the theory rather than the validity of a fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_windows import build, f_sym, predict_windows, sigma_of  # noqa: E402
from rd import Schnakenberg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# (a, b, Du, Dv)
PARAMS = [
    (0.10, 0.90, 1.0, 10.0),   # baseline
    (0.10, 0.90, 1.0, 5.0),
    (0.10, 0.90, 1.0, 20.0),
    (0.05, 1.00, 1.0, 10.0),
    (0.20, 0.80, 1.0, 10.0),
    (0.10, 0.90, 0.5, 10.0),
    (0.10, 0.90, 2.0, 20.0),
]


def main() -> None:
    n_max = 200
    out = []
    print(f"{'a':>5} {'b':>5} {'Du':>5} {'Dv':>5} | {'band':>16} | "
          f"{'n_min':>6} {'n_min':>6} {'n_min':>6} | {'mism':>5} {'checks':>6}")
    print(f"{'':>5} {'':>5} {'':>5} {'':>5} | {'':>16} | "
          f"{'ring1':>6} {'ring2':>6} {'simpl':>6} |")
    print("-" * 92)
    total_dec = 0
    total_mism = 0
    for (a, b, Du, Dv) in PARAMS:
        model = Schnakenberg(a=a, b=b, Du=Du, Dv=Dv)
        st, tr, det = model.homogeneous_stable()
        band = model.turing_band()

        # Unified instability criterion: some eigenvalue gives det J_λ < 0.
        # This criterion holds both with a band and without one (negative
        # discriminant), so all seven parameter sets can be scanned the same way
        # without special-casing the bandless ones.
        J = model.jacobian()
        Aq = Du * Dv
        Bq = -(J[0, 0] * Dv + J[1, 1] * Du)
        C0q = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        mins, mism, decisions = [], 0, 0
        for kind in ("ring1", "ring2", "simpl"):
            if band is not None:
                pred, *_ = predict_windows(kind, band[0], band[1], n_max)
                nmin = int(np.argmax(pred)) if pred.any() else -1
            else:
                # no band: no λ satisfies det<0, so the system is stable everywhere
                pred = np.zeros(n_max + 1, dtype=bool)
                nmin = -1
            mins.append(nmin)
            for n in range(3, n_max + 1):
                sc = build(n, kind)
                lam = np.linalg.eigvalsh(sc.laplacian(sigma_of(kind)).toarray())
                num = bool((Aq * lam**2 + Bq * lam + C0q < 0).any())
                decisions += 1
                if num != bool(pred[n]):
                    mism += 1
        total_dec += decisions
        total_mism += mism
        bs = f"({band[0]:.3f},{band[1]:.3f})" if band is not None else "none"
        print(f"{a:>5.2f} {b:>5.2f} {Du:>5.1f} {Dv:>5.1f} | {bs:>16} | "
              f"{mins[0]:>6} {mins[1]:>6} {mins[2]:>6} | {mism:>5} {decisions:>6}")
        out.append({"a": a, "b": b, "Du": Du, "Dv": Dv,
                    "hom_stable": bool(st), "band": list(band) if band else None,
                    "n_min": dict(zip(("ring1", "ring2", "simpl"), mins)),
                    "mismatches": mism, "decisions": decisions,
                    "n_range": [3, n_max]})

    p = RESULTS / "exp_params.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    banded = [r for r in out if r["band"]]
    bandless = [r for r in out if not r["band"]]
    print(f"\nwith band, {len(banded)} sets: checks {sum(r['decisions'] for r in banded)}, "
          f"mismatches {sum(r['mismatches'] for r in banded)}")
    print(f"without band, {len(bandless)} sets: checks {sum(r['decisions'] for r in bandless)}, "
          f"mismatches {sum(r['mismatches'] for r in bandless)}")
    print(f"total: checks {total_dec}, mismatches {total_mism}")
    print(f"[out] {p}")


if __name__ == "__main__":
    main()
