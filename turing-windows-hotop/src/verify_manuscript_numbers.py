"""
Consistency check of the manuscript numbers, main text and supplementary material.

The numbers are spread over paper/manuscript.md and paper/supplementary.md, but the
underlying data is the single set of files in results/. This script reads every number
back from the raw JSON, recomputes it and compares it, whether the number ends up in
the main text or in the SI. Any mismatch prints FAIL.

Check items are organised by data file (analysis_windows, exp_defect, exp_noise, ...)
rather than by section, so moving text between the main text and the SI does not change
the pass status of any item.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results"


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


results = []


def chk(label, got, want, tol):
    ok = abs(float(got) - float(want)) <= tol
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<46} got={got:<10.5f} want={want}")
    return ok


print("=== 1. Analytic windows and minimum size ===")
aw = load("analytic_windows.json")
chk("ring1 C", aw["ring1"]["quad_coeff"], 1.0, 1e-3)
chk("ring2 C", aw["ring2"]["quad_coeff"], 5.0, 1e-3)
chk("simpl C", aw["simpl"]["quad_coeff"], 11.0, 1e-3)
chk("ring1 coverage%", aw["ring1"]["coverage"] * 100, 96.6, 0.1)
chk("ring2 coverage%", aw["ring2"]["coverage"] * 100, 91.6, 0.1)
chk("simpl coverage%", aw["simpl"]["coverage"] * 100, 86.9, 0.1)
for k in ("ring1", "ring2", "simpl"):
    chk(f"{k} mismatches", aw[k]["mismatches"], 0, 0)

print("\n=== 2. Λ* and mode decomposition ===")
df = load("exp_defect.json")
chk("Λ*(n=120)", df["lambda_star"], 0.2696, 1e-4)
want = {"static": (0.2696, 0.00051, 0.2703),
        "grown_d20": (0.2700, 0.01110, 0.3725),
        "grown_d40": (0.2699, 0.01403, 0.4024),
        "grown_d80": (0.2699, 0.01575, 0.4202)}
for cond, (ml, fd, ry) in want.items():
    sub = [r for r in df["rows"] if r["cond"] == cond]
    chk(f"{cond} mode Λ", np.mean([r["mode_lambda"] for r in sub]), ml, 1e-4)
    chk(f"{cond} defect frac", np.mean([r["defect_frac"] for r in sub]), fd, 1e-4)
    chk(f"{cond} Rayleigh", np.mean([r["rayleigh"] for r in sub]), ry, 1e-3)

# invert the defect eigenvalue
sub80 = [r for r in df["rows"] if r["cond"] == "grown_d80"]
ml80 = np.mean([r["mode_lambda"] for r in sub80])
fd80 = np.mean([r["defect_frac"] for r in sub80])
ry80 = np.mean([r["rayleigh"] for r in sub80])
ldef = (ry80 - ml80 * (1 - fd80)) / fd80
chk("mean Λ of the defect component", ldef, 9.8, 0.3)
chk("Rayleigh uplift %", (ry80 - ml80) / ml80 * 100, 55.7, 2.0)

print("\n=== 3. Mode selection under noise ===")
nm = load("exp_noise_modes.json")
chk("Λ*(noise_modes)", nm["lambda_star"], 0.2696, 1e-4)
devs = [abs(r["mode_lambda"] - nm["lambda_star"]) for r in nm["rows"]
        if np.isfinite(r["mode_lambda"])]
chk("max deviation of mode Λ", max(devs), 0.00044, 5e-5)
for eta, mld, dfd, ryd in ((0.0, 0.2699, 0.0126, 0.389),
                           (1e-3, 0.2699, 0.0118, 0.377),
                           (1e-2, 0.2699, 0.0123, 0.380)):
    sub = [r for r in nm["rows"] if r["eta"] == eta]
    chk(f"η={eta:.0e} mean defect", np.mean([r["defect_frac"] for r in sub]), dfd, 5e-4)
    chk(f"η={eta:.0e} mean Rayleigh", np.mean([r["rayleigh"] for r in sub]), ryd, 5e-3)

print("\n=== 4. Noise rescues the pattern (amplitude) ===")
nz = load("exp_noise.json")
for eta in (0.0, 1e-4, 1e-3, 1e-2):
    for dt in (40.0, 160.0):
        v = np.mean([r["amp_final"] for r in nz if r["eta"] == eta and r["dtau"] == dt])
        exp = 0.0 if (eta == 0.0 and dt == 160.0) else None
        if exp is not None:
            chk(f"η={eta:.0e} dτ={dt:.0f} amplitude (should be 0)", v, 0.0, 1e-6)
        else:
            results.append(v > 0.4)
            print(f"  [{'PASS' if v > 0.4 else 'FAIL'}] η={eta:.0e} dτ={dt:.0f} amplitude>0.4      got={v:.4f}")

print("\n=== 5. Statistical tests ===")
st = load("exp_stats.json")
chk("static Λ_dom mean", st["static"]["mean"], 0.2703, 1e-4)
chk("static Λ_dom std", st["static"]["std"], 0.0, 1e-6)
chk("grown Λ_dom mean", st["grown"]["mean"], 0.3575, 1e-3)
chk("grown Λ_dom std", st["grown"]["std"], 0.0441, 1e-3)
chk("Cliff's δ", st["test"]["cliffs_delta"], 1.0, 1e-6)
results.append(st["test"]["p"] < 1e-4)
print(f"  [{'PASS' if st['test']['p'] < 1e-4 else 'FAIL'}] p < 1e-4"
      f"                            got={st['test']['p']:.3e}")

print("\n=== 6. Disorder scan ===")
di = load("exp_disorder.json")
wantd = {0.0: (0.462, 0.398, 0.104, 0.0, 0.0), 0.5: (0.499, 0.462, 0.027, 0.0, 0.0)}
for s in (0.0, 0.5):
    sub = sorted([r for r in di if r["sigma"] == s], key=lambda r: r["q"])
    got = tuple(round(r["window_contrast"], 3) for r in sub)
    ok = all(abs(a - b) < 0.002 for a, b in zip(got, wantd[s]))
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] σ={s} contrast series {got}")

print("\n=== 7. Universality (random flag complexes) ===")
rv = load("exp_random_v2.json")
pos = sum(1 for r in rv if r["delta_lam1_positive"] == r["n_pairs"])
results.append(pos == len(rv))
print(f"  [{'PASS' if pos == len(rv) else 'FAIL'}] configurations with Δλ1>0"
      f"  {pos}/{len(rv)}")
mx = max(r["delta_lam1_mean"] for r in rv)
chk("max Δλ1", mx, 11.2359, 0.01)

print("\n=== 8. Band position control (separating window flattening from spectral position mismatch) ===")
bp = load("exp_band_position.json")
wantbp = {0.0: 0.369, 0.05: 0.303, 0.10: 0.336, 1.00: 0.159}
for q, want in wantbp.items():
    sub = [r for r in bp if r["q"] == q and not r["trivial"]
           and r["band_inside_frac"] > 0.5]
    if sub:
        chk(f"q={q} mean contrast inside the band", np.mean([r["window_contrast"] for r in sub]),
            want, 2e-3)
    else:
        results.append(False)
        print(f"  [FAIL] q={q} no in-band configuration")

print("\n=== 9. Insertion rules and defect healing ===")
ins = load("exp_insertion.json")["rows"]
A = {r["rule"]: r for r in ins if r["exp"] == "A"}
chk("uniform defect", A["uniform"]["defect_frac"], 0.01403, 1e-4)
chk("neighbor_mean defect", A["neighbor_mean"]["defect_frac"], 0.00173, 1e-4)
chk("neighbor_mean Rayleigh", A["neighbor_mean"]["rayleigh"], 0.2703, 1e-3)
chk("global_mean defect", A["global_mean"]["defect_frac"], 0.01409, 1e-4)
chk("defect reduction factor", A["uniform"]["defect_frac"] / A["neighbor_mean"]["defect_frac"],
    8.1, 0.2)

B = sorted([r for r in ins if r["exp"] == "B"], key=lambda r: r["n_insertions"])
gotB = [round(r["defect_frac"] * 100, 2) for r in B]
wantB = [1.19, 1.01, 1.02, 1.40]
ok = all(abs(a - b) < 0.01 for a, b in zip(gotB, wantB))
results.append(ok)
print(f"  [{'PASS' if ok else 'FAIL'}] defect vs number of insertions {gotB}")
Ns = np.array([r["n_insertions"] for r in B], float)
dfs = np.array([r["defect_frac"] for r in B], float)
slope, icpt = np.polyfit(Ns, dfs, 1)
r2 = 1 - ((dfs - (slope * Ns + icpt)) ** 2).sum() / (((dfs - dfs.mean()) ** 2).sum())
chk("additivity linear fit R²", r2, 0.21, 0.02)

hea = load("exp_healing.json")
for rule, d0, dlast in (("uniform", 0.01403, 0.00051),
                        ("neighbor_mean", 0.00173, 0.00051)):
    sub = sorted([r for r in hea if r["rule"] == rule], key=lambda r: r["hold"])
    chk(f"{rule} hold=0 defect", sub[0]["defect_frac"], d0, 1e-4)
    chk(f"{rule} hold=500 defect", sub[-1]["defect_frac"], dlast, 1e-4)
    chk(f"{rule} hold=500 Rayleigh", sub[-1]["rayleigh"], 0.2703, 1e-3)
    chk(f"{rule} hold=500 mode Λ", sub[-1]["mode_lambda"], 0.2696, 1e-3)

print("\n=== 10. Spectral density criterion (contrast is a function of N_in) ===")
gl = load("exp_gap_law.json")
bins = [(0.0, 0.5, 0.056), (0.5, 1.5, 0.440), (1.5, 3.0, 0.388),
        (3.0, 8.0, 0.177), (8.0, 1e9, 0.000)]
for a, b, want_c in bins:
    sub = [r for r in gl if a <= r["mean_n_in"] < b]
    got = np.mean([r["contrast"] for r in sub]) if sub else float("nan")
    chk(f"N_in∈[{a},{b if b < 1e8 else float('inf'):.1f}) mean contrast", got, want_c, 2e-3)
sub_peak = [r for r in gl if 0.5 <= r["mean_n_in"] < 1.5]
chk("peak contrast (should reach the theoretical limit 0.5)", max(r["contrast"] for r in sub_peak), 0.500, 1e-3)
sub_hi = [r for r in gl if r["mean_n_in"] > 8.0]
chk("contrast for N_in>8 (should be 0)", max(r["contrast"] for r in sub_hi), 0.0, 1e-9)

print("\n=== 11. Bifurcation type ===")
bf = load("exp_bifurcation.json")
for kind, want_beta, want_r2 in (("ring1", 0.497, 0.9969), ("simpl", 0.458, 0.9977)):
    fit = next(r for r in bf if r["kind"] == kind and r["part"] == "A_fit")
    chk(f"{kind} scaling exponent β", fit["beta"], want_beta, 2e-3)
    chk(f"{kind} fit R²", fit["r2"], want_r2, 1e-3)
    results.append(abs(fit["beta"] - 0.5) < 0.06)
    print(f"  [{'PASS' if abs(fit['beta'] - 0.5) < 0.06 else 'FAIL'}] "
          f"{kind} difference from the supercritical value 0.5 < 0.06   got={abs(fit['beta']-0.5):.3f}")

# hysteresis criterion: up and down sweeps should agree for μ*>0.03; a difference appears only at low μ*
for kind, mu_thresh, want_max_hyst in (("ring1", 0.03, 0.0194), ("simpl", 0.03, 0.0001)):
    B = [r for r in bf if r["kind"] == kind and r["part"] == "B"]
    fast = [r["hyst"] for r in B if r["mu_star"] > mu_thresh]
    slow = [r["hyst"] for r in B if r["mu_star"] <= mu_thresh]
    ok = (max(fast) < 0.03) and (max(slow) > 0.1 if slow else False)
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {kind}: max up/down difference <0.03 for μ*>0.03 "
          f"({max(fast):.4f}), large difference appears at low μ* ({max(slow) if slow else 0:.4f})")

print("\n=== 12. Robustness: second parameter set + growth granularity ===")
rb = load("exp_robustness.json")
p1 = next(r for r in rb if r["part"] == 1)
chk("second parameter set static modeΛ", p1["static"]["mode_lambda"], 0.2696, 1e-3)
chk("second parameter set grown modeΛ", p1["grown"]["mode_lambda"], 0.2699, 1e-3)
chk("second parameter set healed modeΛ", p1["healed"]["mode_lambda"], 0.2696, 1e-3)
chk("second parameter set static defect%", p1["static"]["defect_frac"] * 100, 0.320, 5e-3)
chk("second parameter set grown defect%", p1["grown"]["defect_frac"] * 100, 0.815, 5e-3)
chk("second parameter set healed defect%", p1["healed"]["defect_frac"] * 100, 0.320, 5e-3)
results.append(all(p1["checks"].values()))
print(f"  [{'PASS' if all(p1['checks'].values()) else 'FAIL'}] "
      f"all four checks of the second parameter set pass")

want_burst = {1: (1.433, 0.2700), 2: (2.457, 0.2708),
              4: (4.836, 0.2748), 10: (8.064, 0.3021)}
for r in [x for x in rb if x["part"] == 2]:
    k = r["burst"]
    wd, wm = want_burst[k]
    chk(f"burst={k} defect%", r["defect_frac"] * 100, wd, 2e-3)
    chk(f"burst={k} mode Λ", r["mode_lambda"], wm, 2e-3)

print("\n=== 13. New assertions after the audit revision ===")
import math
# (a) decomposition of 3.33 into 2.22 and 1.50
chk("ring1->ring2 ratio (9->20)", 20 / 9, 2.22, 5e-3)
chk("ring2->simpl ratio (20->30)", 30 / 20, 1.50, 5e-3)
chk("total ratio (9->30)", 30 / 9, 3.33, 5e-3)
chk("sqrt(11)", math.sqrt(11), 3.32, 5e-3)
chk("sqrt(11/5)", math.sqrt(11 / 5), 1.48, 5e-3)

# (b) §3.4 ratio statistics: 3.24 ± 0.15 (only parameter sets with a band; n_min recorded as -1 without a band)
pp = load("exp_params.json")
ratios, decisions, mism = [], 0, 0
for r in pp:
    decisions += r.get("decisions", 0)
    mism += r.get("mismatches", 0)
    nm = r.get("n_min")
    if r.get("band") and nm and nm.get("ring1", -1) > 0:
        ratios.append(nm["simpl"] / nm["ring1"])
ratios = np.array(ratios, dtype=float)
chk("mean n_min ratio of the five parameter sets", ratios.mean(), 3.24, 5e-3)
chk("std of the n_min ratio of the five parameter sets", ratios.std(ddof=1), 0.15, 5e-3)
print(f"       values: {np.round(ratios, 2).tolist()}")
chk("total decisions in the parameter scan", decisions, 4158, 0)
chk("total mismatches in the parameter scan", mism, 0, 0)
_banded = [r for r in pp if r.get("band")]
_bandless = [r for r in pp if not r.get("band")]
chk("decisions of banded sets", sum(r["decisions"] for r in _banded), 2970, 0)
chk("decisions of bandless sets", sum(r["decisions"] for r in _bandless), 1188, 0)

# (c) exact n_min = 33 for the range-3 ring (C=14); the approximation gives 34
from scipy.optimize import brentq
from rd import Schnakenberg as _SK
_m = _SK()
_lo, _hi = _m.turing_band()
_f3 = lambda t: sum(2.0 * (1 - np.cos(d * t)) for d in (1, 2, 3))
_g = np.linspace(1e-9, np.pi, 200001)
_imax = int(np.argmax(_f3(_g)))
_t_hi = brentq(lambda t: _f3(t) - _hi, _g[0] + 1e-12, _g[_imax])
chk("range-3 ring exact n_min", int(np.ceil(2 * np.pi / _t_hi)), 33, 0)
chk("range-3 ring approximation value", int(np.ceil(2 * np.pi * np.sqrt(14 / _hi))), 34, 0)

# (d) discriminant is negative for Dv=2 / Dv=5 (rather than lambda_+ < 0)
for Dv in (2.0, 5.0):
    _mm = _SK(Dv=Dv)
    _J = _mm.jacobian()
    _A = _mm.Du * _mm.Dv
    _B = -(_J[0, 0] * _mm.Dv + _J[1, 1] * _mm.Du)
    _C = _J[0, 0] * _J[1, 1] - _J[0, 1] * _J[1, 0]
    _disc = _B * _B - 4 * _A * _C
    results.append(_disc < 0)
    print(f"  [{'PASS' if _disc < 0 else 'FAIL'}] Dv={Dv} discriminant negative (no real root)"
          f"                        got={_disc:.2f}")

# (e) residual after crossing the dead zone: about 1e-13 for all three dtau values, with a large spread in the in-band fraction (not a physical threshold)
_tr2 = load("exp_erasure_trace.json")
for _dt in ("80.0", "120.0", "160.0"):
    _t = {x["n"]: x["amp"] for x in _tr2[_dt]}
    _res = _t[59]
    results.append(1e-15 < _res < 1e-11)
    print(f"  [{'PASS' if 1e-15 < _res < 1e-11 else 'FAIL'}] "
          f"dtau={_dt} n=59 residual at the 1e-13 level (not underflow)        got={_res:.2e}")

# (f) integer ratio basis for 2.22 / 1.50 (Abstract and §3.3 are now unified)
chk("integer ratio 20/9", 20 / 9, 2.22, 5e-3)
chk("integer ratio 30/20", 30 / 20, 1.50, 5e-3)
chk("integer ratio 30/9", 30 / 9, 3.33, 5e-3)

print("\n=== 14. Operator consistency with Ref.[5] (Lucas–Cencetti–Battiston 2020, Eq.3/8) ===")
from itertools import combinations as _comb
from complex import SimplicialComplex as _SC

_edges = [(i, (i + 1) % 24) for i in range(24)] + [(i, (i + 2) % 24) for i in range(24)]
_sc = _SC.from_edges(24, _edges)
for i in range(24):
    _sc.add_triangle(i, (i + 1) % 24, (i + 2) % 24)


def _ref_L(n, d, cells):
    Kd = np.zeros(n)
    for c in cells:
        for i in c:
            Kd[i] += 1
    Ad = np.zeros((n, n))
    for c in cells:
        for p in _comb(sorted(c), 2):
            Ad[p[0], p[1]] += 1
            Ad[p[1], p[0]] += 1
    return d * np.diag(Kd) - Ad


_err1 = np.abs(_sc.L0().toarray() - _ref_L(24, 1, [tuple(e) for e in _sc.edges])).max()
_err2 = np.abs(_sc.L_tri().toarray()
               - _ref_L(24, 2, [tuple(t) for t in _sc.triangles])).max()
chk("L^(1) vs original Eq.3 (d=1)", _err1, 0.0, 1e-12)
chk("L^(2) vs original Eq.3 (d=2)", _err2, 0.0, 1e-12)
_K1 = np.array([sum(1 for e in _sc.edges if i in e) for i in range(24)], float).mean()
_K2 = np.array([sum(1 for t in _sc.triangles if i in t) for i in range(24)], float).mean()
chk("<K^(1)> = 4", _K1, 4.0, 1e-12)
chk("<K^(2)> = 3", _K2, 3.0, 1e-12)
chk("sigma=1 corresponds to gamma2/gamma1 = 0.75", _K2 / _K1, 0.75, 1e-12)

print("\n=== 15. Closed-form error analysis (new subsection of §3.3) ===")
from scipy.optimize import brentq as _bq
from rd import Schnakenberg as _SK2
_m2 = _SK2()
_lo2, _hi2 = _m2.turing_band()
_structs = {
    "ring1": (lambda t: 2 - 2 * np.cos(t), 1.0, 1.0),
    "ring2": (lambda t: 4 - 2 * np.cos(t) - 2 * np.cos(2 * t), 5.0, 17.0),
    "simpl": (lambda t: 10 - 6 * np.cos(t) - 4 * np.cos(2 * t), 11.0, 35.0),
}
_dn_list = []
for _nme, (_f, _C2, _C4) in _structs.items():
    _g = np.linspace(1e-9, np.pi, 400001)
    _im = int(np.argmax(_f(_g)))
    _th = _bq(lambda x: _f(x) - _hi2, _g[0] + 1e-12, _g[_im])
    _cont = 2 * np.pi / _th
    _ap = 2 * np.pi * np.sqrt(_C2 / _hi2)
    _dn = _ap - _cont
    _dnf = _cont * (_C4 * _hi2) / (24 * _C2**2)
    chk(f"{_nme} Δn measured vs formula", _dn, _dnf, 2e-2)
    _dn_list.append(_dn)
chk("max Δn among the three ≤ 0.29", max(_dn_list), 0.288, 3e-3)

# fitted exponent and critical r
_r = np.arange(1, 41, dtype=float)
_dn = []
for _ri in _r.astype(int):
    _C2 = sum(d * d for d in range(1, _ri + 1))
    _f = lambda t, _ri=_ri: sum(2 * (1 - np.cos(d * t)) for d in range(1, _ri + 1))  # noqa: E731
    _g = np.linspace(1e-9, np.pi, 400001)
    _im = int(np.argmax(_f(_g)))
    _th = _bq(lambda x: _f(x) - _hi2, _g[0] + 1e-12, _g[_im])
    _dn.append(2 * np.pi * np.sqrt(_C2 / _hi2) - 2 * np.pi / _th)
_dn = np.array(_dn)
_p = np.polyfit(np.log(_r[:20]), np.log(_dn[:20]), 1)[0]
chk("Δn growth exponent", _p, 0.492, 1e-2)
_coef = _dn[-1] / _r[-1] ** _p
_r_c = (1.0 / _coef) ** (1.0 / _p)
chk("critical r for Δn=1", _r_c, 26.4, 0.5)

print("\n" + "=" * 60)
n_ok = sum(results)
print(f"total {n_ok}/{len(results)} items pass")
print("all consistent" if n_ok == len(results) else "inconsistency found, the manuscript must be corrected")
