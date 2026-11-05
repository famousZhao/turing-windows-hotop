"""
Can defects heal?

Question

The defect fraction at the growth end state is about 1 to 1.6 percent, depending on
the insertion rate, and is independent of the number of insertions. The question is
whether this is a reversible transient or a frozen structure.

    defect decays while the structure is held fixed -> transient, the growth memory
        fades by itself
    defect stays constant                           -> frozen, growth leaves a
        permanent imprint

The smooth insertion rule is run in parallel to see whether the two protocols differ
in healing behaviour.
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


def grow_with_snapshots(n0, n1, dtau, model, dt, seed, rule, hold_times):
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
        if rule == "neighbor_mean":
            un, vn = 0.5 * (u[-1] + u[0]), 0.5 * (v[-1] + v[0])
        else:
            un, vn = model.ustar, model.vstar
        u = np.append(u, un)
        v = np.append(v, vn)

    snaps = [{"hold": 0.0, **decompose(L, u, *model.turing_band())}]
    for T in hold_times:
        for _ in range(int(round(T / dt))):
            u, v = rk4_step(L, model.reaction, u, v, dt, model.Du, model.Dv)
        snaps.append({"hold": float(T), **decompose(L, u, *model.turing_band())})
    return snaps


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n1 = 120
    sc = build(n1)
    lam_max = float(np.linalg.eigvalsh(sc.laplacian(1.0).toarray()).max())
    dt = min(0.012, 2.5 / ((model.Du + model.Dv) * lam_max) * 0.8)
    hold_times = [20.0, 100.0, 500.0]

    rows = []
    for rule in ("uniform", "neighbor_mean"):
        print(f"\n=== rule {rule}  (n=40→120, dτ=40, mean of 3 seeds) ===")
        print(f"{'hold':>9} {'mode Λ':>8} {'defect frac':>10} {'Rayleigh':>10} {'amp':>7}")
        all_snaps = []
        for s in (101, 102, 103):
            all_snaps.append(grow_with_snapshots(40, n1, 40.0, model, dt, s,
                                                 rule, hold_times))
        for i, T in enumerate([0.0] + hold_times):
            ml = np.mean([sn[i]["mode_lambda"] for sn in all_snaps])
            df = np.mean([sn[i]["defect_frac"] for sn in all_snaps])
            ry = np.mean([sn[i]["rayleigh"] for sn in all_snaps])
            am = np.mean([sn[i]["amp"] for sn in all_snaps])
            print(f"{T:>9.0f} {ml:>8.4f} {df:>10.5f} {ry:>10.4f} {am:>7.3f}")
            rows.append({"rule": rule, "hold": T, "mode_lambda": float(ml),
                         "defect_frac": float(df), "rayleigh": float(ry),
                         "amp": float(am)})

    out = RESULTS / "exp_healing.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== interpretation ===")
    for rule in ("uniform", "neighbor_mean"):
        sub = [r for r in rows if r["rule"] == rule]
        d0 = sub[0]["defect_frac"]
        dlast = sub[-1]["defect_frac"]
        red = (1 - dlast / d0) * 100 if d0 > 0 else 0
        tag = ("heals (transient)" if red > 50 else
               "partial healing" if red > 15 else "mostly frozen (permanent imprint)")
        print(f"{rule:>15}: defect {d0*100:.3f}% -> {dlast*100:.3f}% "
              f"(held {sub[-1]['hold']:.0f} time units)  drop {red:.1f}%  -> {tag}")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
