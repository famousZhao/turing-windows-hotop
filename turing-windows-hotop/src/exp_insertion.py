"""
Experiment to locate the source of the defects, plus a prescription for the insertion
rule.

Mechanism under test

The defects come from new nodes being placed at the homogeneous state, which forms a
kink on the existing pattern profile.

Decisive test

If the hypothesis holds, an insertion rule that inherits the neighbour mean should
greatly reduce the defect content; if it does not hold, for instance if the defects
actually come from the pattern re-adapting to the new spectrum, changing the rule has
no effect.

Insertion rules

    uniform        new node = homogeneous state     (produces kinks)
    neighbor_mean  new node = mean of the two neighbours   (smooth continuation)
    global_mean    new node = current global mean          (between the two)

The additivity of the defects is tested as well: fix the final size, vary the insertion
count N, and check whether the defect fraction scales linearly in N.
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
    if tot < 1e-24:
        return {"mode_lambda": np.nan, "defect_frac": np.nan,
                "rayleigh": np.nan, "amp": float(np.abs(p).max())}
    e = e / tot
    inb = (lam > lo) & (lam < hi)
    E_in = e[inb].sum()
    return {
        "mode_lambda": float((e[inb] * lam[inb]).sum() / max(E_in, 1e-30)),
        "defect_frac": float(e[~inb].sum()),
        "rayleigh": float((e * lam).sum()),
        "amp": float(np.abs(p).max()),
    }


def new_value(rule, u, v, model):
    if rule == "uniform":
        return model.ustar, model.vstar
    if rule == "neighbor_mean":
        # the new node is inserted between n-2 and 0
        return 0.5 * (u[-1] + u[0]), 0.5 * (v[-1] + v[0])
    if rule == "global_mean":
        return float(u.mean()), float(v.mean())
    raise ValueError(rule)


def grow(n0, n1, dtau, model, dt, seed, rule):
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
        un, vn = new_value(rule, u, v, model)
        u = np.append(u, un)
        v = np.append(v, vn)
    return decompose(L, u, *model.turing_band())


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n1 = 120
    sc = build(n1)
    lam_max = float(np.linalg.eigvalsh(sc.laplacian(1.0).toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
    an = model.spectrum_analysis(sc.laplacian(1.0))
    print(f"Λ*={an['lam_star']:.4f}  dt={dt:.4f}\n")

    rows = []
    print("=== Experiment A: effect of the insertion rule (n=40->120, dτ=40, 3 seeds) ===")
    print(f"{'rule':>15} {'mode Λ':>8} {'defect frac':>10} {'Rayleigh':>10} {'amp':>7}")
    for rule in ("uniform", "neighbor_mean", "global_mean"):
        res = [grow(40, n1, 40.0, model, dt, s, rule) for s in (101, 102, 103)]
        ml = np.mean([r["mode_lambda"] for r in res])
        df = np.mean([r["defect_frac"] for r in res])
        ry = np.mean([r["rayleigh"] for r in res])
        am = np.mean([r["amp"] for r in res])
        print(f"{rule:>15} {ml:>8.4f} {df:>10.5f} {ry:>10.4f} {am:>7.3f}")
        rows.append({"exp": "A", "rule": rule, "mode_lambda": float(ml),
                     "defect_frac": float(df), "rayleigh": float(ry), "amp": float(am),
                     "n_insertions": 80})

    print("\n=== Experiment B: additivity of the defects (final n=120 fixed, insertion count N varied, uniform rule) ===")
    print(f"{'n0':>5} {'N':>5} {'mode Λ':>8} {'defect frac':>10} {'Rayleigh':>10} {'amp':>7}")
    for n0 in (100, 80, 60, 40):
        res = [grow(n0, n1, 40.0, model, dt, s, "uniform") for s in (101, 102, 103)]
        ml = np.mean([r["mode_lambda"] for r in res])
        df = np.mean([r["defect_frac"] for r in res])
        ry = np.mean([r["rayleigh"] for r in res])
        am = np.mean([r["amp"] for r in res])
        print(f"{n0:>5} {n1-n0:>5} {ml:>8.4f} {df:>10.5f} {ry:>10.4f} {am:>7.3f}")
        rows.append({"exp": "B", "n0": n0, "n_insertions": n1 - n0,
                     "mode_lambda": float(ml), "defect_frac": float(df),
                     "rayleigh": float(ry), "amp": float(am)})

    out = RESULTS / "exp_insertion.json"
    out.write_text(json.dumps({"lambda_star": an["lam_star"], "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Reading ===")
    A = {r["rule"]: r for r in rows if r["exp"] == "A"}
    du, dn = A["uniform"]["defect_frac"], A["neighbor_mean"]["defect_frac"]
    ratio = dn / du
    print(f"uniform defect = {du*100:.2f}%   neighbor_mean defect = {dn*100:.2f}%   "
          f"ratio = {ratio:.3f}")
    if ratio < 0.5:
        print("  -> supports the kink mechanism: smooth continuation greatly reduces defects")
    else:
        print("  -> does not support the kink mechanism; the defects have another source")
    B = [r for r in rows if r["exp"] == "B"]
    Ns = np.array([r["n_insertions"] for r in B], float)
    dfs = np.array([r["defect_frac"] for r in B], float)
    print(f"\ndefect vs insertion count: N={Ns.tolist()}")
    print(f"                  defect={np.round(dfs*100, 2).tolist()} %")
    if Ns.std() > 0:
        slope, icpt = np.polyfit(Ns, dfs, 1)
        pred = slope * Ns + icpt
        ss = 1 - ((dfs - pred) ** 2).sum() / (((dfs - dfs.mean()) ** 2).sum() + 1e-30)
        print(f"  linear fit: defect = {slope*100:.4f}% × N + {icpt*100:.3f}%   R²={ss:.4f}")
        print(f"  extrapolated intercept at zero insertions (N=0) = {icpt*100:.3f}%  "
              f"(static measurement about 0.051%)")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
