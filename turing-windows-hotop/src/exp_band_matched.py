"""
Adaptive band matching: control the spectral position exactly, then compare window
contrast across disorder strengths.

Why not a fixed rho grid

exp_band_position.py sweeps a fixed grid of the spectral scale factor rho, and for
most rho the unstable band falls outside the spectrum (the trivial region). At q=1
that leaves only two usable configurations, which is too thin for a comparison.

Method

Solve for rho separately for each (q, seed) so that the centre of the unstable band
sits exactly on the spectral median of that structure:

    reference size n_ref = 60, take the spectral median m,
    set rho m = (lam_lo + lam_hi)/2  =>  rho = (lam_lo + lam_hi)/(2 m)

Every configuration is then in band at n_ref, and n is swept to compare the window
contrast. The difference between q values then reduces to the single factor of
discreteness.

Statistics: 10 random seeds per q, reporting mean and standard deviation together
with a significance test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complex import SimplicialComplex  # noqa: E402
from rd import Schnakenberg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def ws_edges(n: int, k: int, q: float, rng) -> list[tuple[int, int]]:
    half = k // 2
    edges = set()
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            edges.add((min(i, j), max(i, j)))
    for (i, j) in list(edges):
        if rng.random() < q:
            edges.discard((i, j))
            for _ in range(30):
                w = int(rng.integers(0, n))
                if w == i:
                    continue
                e = (min(i, w), max(i, w))
                if e not in edges:
                    edges.add(e)
                    break
    return sorted(edges)


def ws_complex(n: int, k: int, q: float, rng) -> SimplicialComplex:
    edges = ws_edges(n, k, q, rng)
    sc = SimplicialComplex.from_edges(n, edges)
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        for j in sorted(adj[i]):
            if j <= i:
                continue
            for kk in sorted(adj[i] & adj[j]):
                if kk > j:
                    sc.add_triangle(i, j, kk)
    return sc


def spectrum_by_n(q: float, k: int, seed: int, n_list) -> dict[int, np.ndarray]:
    out = {}
    for n in n_list:
        rng = np.random.default_rng(50021 * seed + 7919 * n + int(q * 1000))
        sc = ws_complex(n, k, q, rng)
        lam = np.linalg.eigvalsh(sc.laplacian(0.0).toarray())
        out[n] = lam[lam > 1e-9]
    return out


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    centre = 0.5 * (lo + hi)
    k_deg = 6
    n_ref = 60
    n_list = list(range(12, 121, 4))
    qs = [0.0, 0.05, 0.1, 0.5, 1.0]
    n_seeds = 10
    print(f"band ({lo:.2f},{hi:.2f}) centre {centre:.3f}   mean degree {k_deg}   "
          f"reference size {n_ref}   seeds per q {n_seeds}\n")

    rows = []
    print(f"{'q':>5} {'<ρ>':>8} {'P(unstable)':>8} {'contrast mean':>12} {'std':>8} "
          f"{'max':>7} {'in-band frac':>9}")
    print("-" * 70)
    for q in qs:
        contrasts, p_means, rhos, inside = [], [], [], []
        for seed in range(n_seeds):
            spec = spectrum_by_n(q, k_deg, seed, n_list)
            lam_ref = spec[n_ref]
            if lam_ref.size == 0:
                continue
            m = float(np.median(lam_ref))
            rho = centre / m
            rhos.append(rho)
            p_n = []
            ins = 0
            for n in n_list:
                lam = spec[n]
                if lam.size == 0:
                    p_n.append(0.0)
                    continue
                sc_lam = rho * lam
                p_n.append(1.0 if ((sc_lam > lo) & (sc_lam < hi)).any() else 0.0)
                if sc_lam.max() >= lo and sc_lam.min() <= hi:
                    ins += 1
            p_n = np.array(p_n)
            contrasts.append(float(p_n.std()))
            p_means.append(float(p_n.mean()))
            inside.append(ins / len(n_list))
        contrasts = np.array(contrasts)
        p_means = np.array(p_means)
        print(f"{q:>5.2f} {np.mean(rhos):>8.3f} {p_means.mean():>8.2f} "
              f"{contrasts.mean():>12.3f} {contrasts.std(ddof=1):>8.3f} "
              f"{contrasts.max():>7.3f} {np.mean(inside):>9.2f}")
        rows.append({"q": q, "rho_mean": float(np.mean(rhos)),
                     "p_unstable_mean": float(p_means.mean()),
                     "contrast_mean": float(contrasts.mean()),
                     "contrast_std": float(contrasts.std(ddof=1)),
                     "contrast_max": float(contrasts.max()),
                     "inside_frac": float(np.mean(inside)),
                     "n_seeds": int(len(contrasts)),
                     "contrasts": contrasts.tolist()})

    out = RESULTS / "exp_band_matched.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Significance test: regular lattice (q=0) vs random graph (q=1) ===")
    a = np.array(next(r for r in rows if r["q"] == 0.0)["contrasts"])
    b = np.array(next(r for r in rows if r["q"] == 1.0)["contrasts"])
    u, p = st.mannwhitneyu(a, b, alternative="two-sided")
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    delta = (gt - lt) / (a.size * b.size)
    print(f"  q=0: {a.mean():.3f} ± {a.std(ddof=1):.3f}  (n={a.size})")
    print(f"  q=1: {b.mean():.3f} ± {b.std(ddof=1):.3f}  (n={b.size})")
    print(f"  Mann–Whitney U={u:.1f}, p={p:.3e}, Cliff's δ={delta:+.3f}")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
