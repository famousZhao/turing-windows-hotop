"""
Separate window flattening from spectral position mismatch.

Why the disorder scan alone is ambiguous

exp_disorder.py shows that as the disorder q grows, the instability probability and
the window contrast both tend to 0, but two effects are mixed there:

    (a) the window structure is flattened by spectral densification, the real claim
    (b) the unstable band falls entirely outside the spectrum, a trivial cause

A random graph with mean degree 6 has lambda_1 about 1.5, while the upper band edge
is 0.5, so the band lies entirely below the spectrum and no n can destabilise the
state. Contrast = 0 is then trivial and cannot be evidence for window flattening.

Method

Introduce a spectral scale factor rho and study the spectrum of the operator rho L
(equivalent to changing the diffusion ratio at the same time). The unstable band
remains (0.2, 0.5) but lands on rho lambda_k, so sweeping rho over a wide enough
range is equivalent to letting the band sweep across the whole spectrum.

At each (q, rho) one can then ask whether the instability verdict oscillates with n:

    band outside the spectrum (rho too small or too large)
        P = 0 or P = 1 constant, contrast 0, the trivial region
    band inside the bulk of the spectrum
        only here is discreteness actually compared:
            regular lattice (q = 0): sparse spectrum, allowed rho lambda jumps
                with n, high contrast
            random graph (q = 1): dense spectrum, almost always some eigenvalue
                in the band, low contrast

If the conclusion holds, window flattening is established independently of spectral
position.
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
            for _ in range(20):
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


def main() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    k_deg, n_seeds = 6, 3
    qs = [0.0, 0.05, 0.1, 1.0]
    n_list = list(range(12, 121, 4))
    rhos = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    print(f"Turing band ({lo:.2f},{hi:.2f})  operator ρL  mean degree {k_deg}  seeds {n_seeds}")
    print(f"n in [{n_list[0]},{n_list[-1]}], {len(n_list)} sizes, {len(rhos)} values of ρ\n")

    # Precompute the spectrum of each (q, n, seed) and reuse it for all rho
    spec: dict[tuple, np.ndarray] = {}
    for q in qs:
        for n in n_list:
            for s in range(n_seeds):
                rng = np.random.default_rng(10007 * s + 7919 * n + int(q * 1000))
                sc = ws_complex(n, k_deg, q, rng)
                lam = np.linalg.eigvalsh(sc.laplacian(0.0).toarray())
                spec[(q, n, s)] = lam[lam > 1e-9]
    print("spectrum precomputation done\n")

    rows = []
    print(f"{'q':>5} {'ρ':>6} | {'P(unstable)':>8} {'window contrast':>11} {'switch':>5} "
          f"{'trivial':>7} | {'<λ1>':>8} {'band in spectrum':>11}")
    print("-" * 84)
    for q in qs:
        for rho in rhos:
            p_n = []
            l1_mean = []
            inside_cnt = 0
            for n in n_list:
                cnt = 0
                for s in range(n_seeds):
                    lam = spec[(q, n, s)]
                    if lam.size:
                        l1_mean.append(rho * lam[0])
                    if lam.size and ((rho * lam > lo) & (rho * lam < hi)).any():
                        cnt += 1
                    # whether the band lies inside the spectrum range
                    if lam.size and (rho * lam.max() >= lo) and (rho * lam.min() <= hi):
                        inside_cnt += 1
                p_n.append(cnt / n_seeds)
            p_n = np.array(p_n)
            contrast = float(p_n.std())
            flips = int(np.abs(np.diff((p_n > 0.5).astype(int))).sum())
            trivial = bool(p_n.std() < 1e-9)  # always 0 or always 1
            inside_frac = inside_cnt / (len(n_list) * n_seeds)
            rows.append({"q": q, "rho": float(rho), "p_unstable_mean": float(p_n.mean()),
                         "window_contrast": contrast, "n_transitions": flips,
                         "trivial": trivial, "band_inside_frac": inside_frac,
                         "mean_lam1": float(np.mean(l1_mean)) if l1_mean else np.nan})
            print(f"{q:>5.2f} {rho:>6.2f} | {p_n.mean():>8.2f} {contrast:>11.3f} "
                  f"{flips:>5d} {('yes' if trivial else 'no'):>7} | "
                  f"{np.mean(l1_mean) if l1_mean else float('nan'):>8.3f} "
                  f"{inside_frac:>11.2f}")
        print("-" * 84)

    p = RESULTS / "exp_band_position.json"
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Key reading: only configurations with the band inside the spectrum (trivial region excluded) ===")
    print(f"{'q':>5} {'contrast (mean over in-band)':>20} {'max contrast':>11} {'samples':>7}")
    for q in qs:
        sub = [r for r in rows if r["q"] == q and not r["trivial"]
               and r["band_inside_frac"] > 0.5]
        if sub:
            c = np.array([r["window_contrast"] for r in sub])
            print(f"{q:>5.2f} {c.mean():>20.3f} {c.max():>11.3f} {len(sub):>7}")
        else:
            print(f"{q:>5.2f} {'(no in-band config)':>20}")
    print(f"\n[out] {p}")


if __name__ == "__main__":
    main()
