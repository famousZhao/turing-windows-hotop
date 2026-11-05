"""
Generate all paper figures: PDF and PNG at 300 dpi, English labels.

Main text
    fig1  dispersion symbols of the three structures with the Turing band, and the
          spectral weight C = 1, 5, 11
    fig2  instability size windows: analytic prediction against the numerical decision
    fig3  minimal pattern-forming size against the square-root scaling in C
    fig4  growth traces: pattern amplitude along n for three growth rates

Supplementary
    fig5   noise test: final amplitude against growth rate, grouped by noise strength
    fig6   disorder sweep: window contrast against rewiring probability
    fig7   modal decomposition: mode selection is adiabatic, the Rayleigh quotient is
           defect-dominated
    fig8   band-position sweep: window flattening separated from spectral position
    fig9   insertion rules and healing: defects are transient, not frozen imprints
    fig10  unified criterion: contrast as a function of the in-band eigenvalue count
    fig11  bifurcation type: scaling of the saturated amplitude plus two-direction
           sweeps
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_windows import f_sym, predict_windows  # noqa: E402
from rd import Schnakenberg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
RESULTS = ROOT / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "dejavuserif",
})

KINDS = ["ring1", "ring2", "simpl"]
LABELS = {"ring1": r"ring, $\pm1$ edges ($C=1$)",
          "ring2": r"ring, $\pm1,\pm2$ edges ($C=5$)",
          "simpl": r"ring + triangles ($C=11$)"}
COLS = {"ring1": "#4C72B0", "ring2": "#DD8452", "simpl": "#C44E52"}


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}.pdf / .png")


def fig1() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    th = np.linspace(0, np.pi, 800)
    fig, (ax, axin) = plt.subplots(1, 2, figsize=(7.8, 3.0),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    # widened canvas and larger inter-panel gap: at the default wspace the
    # right panel's y-label and title overlapped the left panel
    fig.subplots_adjust(wspace=0.40)
    for k in KINDS:
        ax.plot(th, f_sym(k)(th), color=COLS[k], lw=1.6, label=LABELS[k])
    ax.axhspan(lo, hi, color="0.75", alpha=0.55, zorder=0,
               label=rf"Turing band $(\lambda_-,\lambda_+)=({lo:.1f},{hi:.1f})$")
    ax.axhline(lo, color="0.35", lw=0.6, ls=":")
    ax.axhline(hi, color="0.35", lw=0.6, ls=":")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"Laplacian eigenvalue $f(\theta)$")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 6.5)
    ax.legend(frameon=False, loc="upper left")

    t = np.linspace(1e-3, 0.35, 300)
    for k in KINDS:
        axin.loglog(t, f_sym(k)(t), color=COLS[k], lw=1.6)
        c = f_sym(k)(1e-4) / (1e-4) ** 2
        axin.loglog(t, c * t**2, color=COLS[k], lw=0.9, ls="--", alpha=0.8)
    axin.set_xlabel(r"$\theta$")
    axin.set_ylabel(r"$f(\theta)$")
    axin.set_title(r"low-$\theta$ scaling: $f\simeq C\,\theta^2$", fontsize=9)
    save(fig, "fig1_symbols")


def fig2() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    n_max = 160
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.4), sharex=True)
    for ax, k in zip(axes, KINDS):
        pred, *_ = predict_windows(k, lo, hi, n_max)
        ns = np.arange(3, n_max + 1)
        ax.fill_between(ns, 0, 1, where=pred[3:n_max + 1], step="mid",
                        color=COLS[k], alpha=0.35, lw=0)
        ax.plot(ns, pred[3:n_max + 1].astype(float), color=COLS[k], lw=1.0,
                drawstyle="steps-mid")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["stable", "Turing"])
        ax.set_ylabel(LABELS[k].split(" ($")[0], fontsize=8)
        ax.axvspan(0, 0, color="none")
    axes[-1].set_xlabel(r"system size $n$")
    axes[0].set_title("Analytic size windows for Turing instability "
                      "(verified at all $n\\leq300$)", fontsize=9)
    save(fig, "fig2_windows")


def nmin_from_symbol(f, lo: float, hi: float) -> int:
    """Exact n_min: solve f(theta_+)=lambda_+ and take ceil(2*pi/theta_+).

    Do not use the approximation ceil(2*pi*sqrt(C/lambda_+)): for the range-3 ring
    the two differ by 1 (exact 33, approximate 34), and using the approximation as an
    exact data point would validate a formula with itself.
    """
    from scipy.optimize import brentq

    grid = np.linspace(1e-9, np.pi, 200001)
    imax = int(np.argmax(f(grid)))
    t_hi = brentq(lambda t: f(t) - hi, grid[0] + 1e-12, grid[imax])
    return int(np.ceil(2 * np.pi / t_hi))


def fig3() -> None:
    model = Schnakenberg()
    lo, hi = model.turing_band()
    data = []
    for k in KINDS:
        pred, tlo, thi, _ = predict_windows(k, lo, hi, 300)
        c = f_sym(k)(1e-4) / 1e-8
        nmin = int(np.argmax(pred))
        data.append((c, nmin, k))
    # fourth point: range-3 ring (±1,±2,±3), exact n_min from the true symbol
    C3 = 1 + 4 + 9
    f3 = lambda t: sum(2.0 * (1 - np.cos(d * t)) for d in (1, 2, 3))  # noqa: E731
    nmin3 = nmin_from_symbol(f3, lo, hi)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    Cs = np.linspace(0.5, 20, 200)
    ax.plot(Cs, 2 * np.pi * np.sqrt(Cs / hi), "k--", lw=1.0,
            label=r"$2\pi\sqrt{C/\lambda_+}$")
    for c, nmin, k in data:
        ax.plot(c, nmin, "o", ms=8, color=COLS[k], label=LABELS[k])
    ax.plot(C3, nmin3, "s", ms=7, mfc="none", color="0.3",
            label=rf"ring, $\pm1,\pm2,\pm3$ ($C=14$, $n_{{\min}}={nmin3}$)")
    ax.set_xlabel(r"spectral weight $C=\sum_{d=1}^{r} d^2$")
    ax.set_ylabel(r"minimal size $n_{\min}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig3_nmin")
    print(f"  [fig3] C=14 exact n_min={nmin3} "
          f"(approximation gives {int(np.ceil(2*np.pi*np.sqrt(C3/hi)))})")


def predict_windows2(lo: float, hi: float, C: float) -> tuple:
    """Approximate prediction: n_min = ceil(2π sqrt(C/λ+))."""
    return int(np.ceil(2 * np.pi * np.sqrt(C / hi))), 0.0, 0.0, []


def fig4() -> None:
    p = RESULTS / "exp_erasure_trace.json"
    if not p.exists():
        print("  [skip] fig4: missing exp_erasure_trace.json")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    from matplotlib.patches import Patch

    # wider canvas, larger subplot spacing (at wspace=0.2 the right y labels overlapped the left panel)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.0))
    fig.subplots_adjust(wspace=0.42)

    cmap = plt.cm.viridis
    entries = sorted(d.items(), key=lambda kv: float(kv[0]))
    for i, (dtau, tr) in enumerate(entries):
        ns = [t["n"] for t in tr]
        amps = [t["amp"] for t in tr]
        mu = [t["mu_star"] for t in tr]
        c = cmap(i / max(len(entries) - 1, 1))
        ax1.semilogy(ns, np.maximum(amps, 1e-16), color=c, lw=1.4,
                     label=rf"$\Delta\tau={float(dtau):.0f}$")
        ax2.plot(ns, mu, color=c, lw=1.0, alpha=0.85)

    # dead zone: shaded in both panels, documented by the left legend only, no vertical text in the axes
    for ax in (ax1, ax2):
        ax.axvspan(47, 58, color="0.85", zorder=0)
    dead = Patch(facecolor="0.85", edgecolor="none", label="dead zone")

    ax1.set_xlabel(r"system size $n$")
    ax1.set_ylabel("pattern amplitude")
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + [dead], l1 + ["dead zone"], frameon=False, fontsize=7,
               loc="center right")
    ax1.set_title("pattern erased in the dead zone", fontsize=9)

    ax2.axhline(0, color="k", lw=0.6, ls=":")
    ax2.set_xlabel(r"system size $n$")
    ax2.set_ylabel(r"max growth rate $\mu^*$")
    ax2.set_title(r"linear stability along growth", fontsize=9)
    save(fig, "fig4_traces")


def fig5() -> None:
    p = RESULTS / "exp_noise.json"
    if not p.exists():
        print("  [skip] fig5: missing exp_noise.json")
        return
    rows = json.loads(p.read_text(encoding="utf-8"))
    etas = sorted({r["eta"] for r in rows})
    dts = sorted({r["dtau"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    w = 0.8 / max(len(etas), 1)
    for i, e in enumerate(etas):
        vals = [np.mean([r["amp_final"] for r in rows
                         if r["eta"] == e and r["dtau"] == d]) for d in dts]
        errs = [np.std([r["amp_final"] for r in rows
                        if r["eta"] == e and r["dtau"] == d], ddof=1) for d in dts]
        ax.bar(np.arange(len(dts)) + i * w, vals, w, yerr=errs, capsize=2,
               label=rf"$\eta={e:.0e}$",
               color=plt.cm.plasma(i / max(len(etas) - 1, 1)))
    ax.set_xticks(np.arange(len(dts)) + w * (len(etas) - 1) / 2)
    ax.set_xticklabels([rf"$\Delta\tau={d:.0f}$" for d in dts])
    ax.set_ylabel("final pattern amplitude")
    ax.set_title("noise prevents permanent erasure", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 0.55)
    save(fig, "fig5_noise")


def fig6() -> None:
    p = RESULTS / "exp_disorder.json"
    if not p.exists():
        print("  [skip] fig6: missing exp_disorder.json")
        return
    rows = json.loads(p.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for sigma, mk in ((0.0, "o"), (0.5, "s")):
        sub = sorted([r for r in rows if r["sigma"] == sigma], key=lambda r: r["q"])
        ax.plot([r["q"] for r in sub], [r["window_contrast"] for r in sub],
                mk + "-", ms=5, lw=1.2,
                label=rf"$\sigma={sigma}$" + (" (graph)" if sigma == 0 else " (higher-order)"))
    ax.set_xlabel("rewiring probability $q$")
    ax.set_ylabel("window contrast")
    ax.set_title("size windows vanish under disorder", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(-0.02, 0.55)
    save(fig, "fig6_disorder")


def fig7() -> None:
    """Modal decomposition: mode selection is adiabatic, but the Rayleigh quotient is defect-dominated."""
    p = RESULTS / "exp_defect.json"
    if not p.exists():
        print("  [skip] fig7: missing exp_defect.json")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["rows"]
    lam_star = d["lambda_star"]
    conds = ["static", "grown_d20", "grown_d40", "grown_d80"]
    names = ["static", r"$\Delta\tau$=20", r"$\Delta\tau$=40", r"$\Delta\tau$=80"]
    col = ["0.45", COLS["ring1"], COLS["ring2"], COLS["simpl"]]

    def agg(key):
        return ([np.nanmean([r[key] for r in rows if r["cond"] == c]) for c in conds],
                [np.nanstd([r[key] for r in rows if r["cond"] == c], ddof=1)
                 for c in conds])

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    x = np.arange(len(conds))

    m, m_err = agg("mode_lambda")
    axes[0].bar(x, m, 0.65, yerr=m_err, capsize=2, color=col)
    axes[0].axhline(lam_star, color="k", ls="--", lw=1.0,
                    label=rf"$\Lambda^*={lam_star:.4f}$")
    axes[0].set_ylim(0.24, 0.30)
    axes[0].set_ylabel(r"in-band mode $\Lambda$")
    axes[0].set_title("mode selection is adiabatic", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")

    df, df_err = agg("defect_frac")
    axes[1].bar(x, np.array(df) * 100, 0.65, yerr=np.array(df_err) * 100,
                capsize=2, color=col)
    axes[1].set_ylabel("defect energy (%)")
    axes[1].set_title("defects accumulate with growth", fontsize=8.5)

    ry, ry_err = agg("rayleigh")
    axes[2].bar(x, ry, 0.65, yerr=ry_err, capsize=2, color=col)
    axes[2].axhline(lam_star, color="k", ls="--", lw=1.0)
    axes[2].set_ylabel(r"Rayleigh quotient $=\Lambda_{\rm dom}$")
    axes[2].set_title("Rayleigh is defect-dominated", fontsize=8.5)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    save(fig, "fig7_modes")


def fig8() -> None:
    """Band-position sweep: separates window flattening from spectral position mismatch."""
    p = RESULTS / "exp_band_position.json"
    if not p.exists():
        print("  [skip] fig8: missing exp_band_position.json")
        return
    rows = json.loads(p.read_text(encoding="utf-8"))
    qs = sorted({r["q"] for r in rows})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    cmap = plt.cm.viridis
    for i, q in enumerate(qs):
        sub = sorted([r for r in rows if r["q"] == q], key=lambda r: r["rho"])
        rho = [r["rho"] for r in sub]
        c = [r["window_contrast"] for r in sub]
        triv = [r["trivial"] for r in sub]
        col = cmap(i / max(len(qs) - 1, 1))
        lbl = rf"$q={q}$" + (" (lattice)" if q == 0 else " (random)" if q == 1 else "")
        ax.semilogx(rho, c, "o-", ms=3.5, lw=1.1, color=col, label=lbl)
        # hollow markers for the trivial region
        ax.semilogx([x for x, t in zip(rho, triv) if t],
                    [y for y, t in zip(c, triv) if t], "o", ms=7, mfc="none",
                    mec=col, mew=0.8)
    ax.set_xlabel(r"spectral scale factor $\rho$")
    ax.set_ylabel("window contrast")
    ax.set_title("contrast vs band position (hollow = trivial)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_ylim(-0.02, 0.56)

    # right panel: keep only configurations whose band lies inside the spectrum
    vals, labels, cols = [], [], []
    for i, q in enumerate(qs):
        sub = [r for r in rows if r["q"] == q and not r["trivial"]
               and r["band_inside_frac"] > 0.5]
        if sub:
            vals.append(np.mean([r["window_contrast"] for r in sub]))
            labels.append(rf"$q={q}$")
            cols.append(cmap(i / max(len(qs) - 1, 1)))
    ax2.bar(np.arange(len(vals)), vals, 0.6, color=cols)
    ax2.set_xticks(np.arange(len(vals)))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("mean contrast")
    ax2.set_title("band inside spectrum only", fontsize=8.5)
    ax2.set_ylim(0, 0.5)
    save(fig, "fig8_band_position")


def fig9() -> None:
    """Insertion rules and healing: defects are transient, not frozen imprints."""
    pi = RESULTS / "exp_insertion.json"
    ph = RESULTS / "exp_healing.json"
    if not (pi.exists() and ph.exists()):
        print("  [skip] fig9: missing exp_insertion.json / exp_healing.json")
        return
    ins = json.loads(pi.read_text(encoding="utf-8"))["rows"]
    hea = json.loads(ph.read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    A = [r for r in ins if r["exp"] == "A"]
    names = ["uniform\n$(u^*,v^*)$", "neighbour\nmean", "global\nmean"]
    order = ["uniform", "neighbor_mean", "global_mean"]
    vals = [next(r["defect_frac"] for r in A if r["rule"] == o) * 100 for o in order]
    cols = ["#C44E52", "#55A868", "#C44E52"]
    ax1.bar(np.arange(3), vals, 0.6, color=cols)
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(names, fontsize=7)
    ax1.set_ylabel("defect energy (%)")
    ax1.set_title("interpolating nodes removes the defect", fontsize=8.5)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax1.set_ylim(0, max(vals) * 1.3)

    for rule, mk, col in (("uniform", "o", "#C44E52"),
                          ("neighbor_mean", "s", "#55A868")):
        sub = sorted([r for r in hea if r["rule"] == rule], key=lambda r: r["hold"])
        t = [r["hold"] for r in sub]
        d = [r["defect_frac"] * 100 for r in sub]
        ax2.semilogy(t, d, mk + "-", ms=4.5, lw=1.2, color=col,
                     label=rule.replace("_", " "))
    ax2.set_xlabel("hold time after growth")
    ax2.set_ylabel("defect energy (%)")
    ax2.set_title("defects heal: growth leaves no trace", fontsize=8.5)
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_xlim(-15, 520)
    save(fig, "fig9_defect_transient")


def fig10() -> None:
    """Unified criterion: contrast is a function of the in-band eigenvalue count N_in, independent of disorder."""
    p = RESULTS / "exp_gap_law.json"
    if not p.exists():
        print("  [skip] fig10: missing exp_gap_law.json")
        return
    rows = json.loads(p.read_text(encoding="utf-8"))
    qs = sorted({r["q"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.9, 3.1))
    cmap = plt.cm.viridis
    for i, q in enumerate(qs):
        sub = [r for r in rows if r["q"] == q and r["mean_n_in"] > 0]
        ax.semilogx([r["mean_n_in"] for r in sub],
                    [r["contrast"] for r in sub],
                    "o", ms=4.5, alpha=0.85,
                    color=cmap(i / max(len(qs) - 1, 1)), label=rf"$q={q}$")
    mu = np.logspace(-2, 1.4, 300)
    ax.plot(mu, np.sqrt(np.exp(-mu) * (1 - np.exp(-mu))), "k--", lw=1.1,
            label=r"Poisson $\sqrt{e^{-\mu}(1-e^{-\mu})}$")
    ax.axvline(np.log(2), color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"mean in-band eigenvalues $\overline{N_{\rm in}}$")
    ax.set_ylabel("window contrast")
    ax.set_title("contrast is set by spectral density", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_ylim(-0.02, 0.56)
    save(fig, "fig10_gap_law")


def fig11() -> None:
    """Bifurcation type: scaling of the saturated amplitude plus two-direction sweeps."""
    p = RESULTS / "exp_bifurcation.json"
    if not p.exists():
        print("  [skip] fig11: missing exp_bifurcation.json")
        return
    rows = json.loads(p.read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for kind, col in (("ring1", COLS["ring1"]), ("simpl", COLS["simpl"])):
        A = sorted([r for r in rows if r["kind"] == kind and r["part"] == "A"],
                   key=lambda r: r["mu_star"])
        if not A:
            continue
        mu = np.array([r["mu_star"] for r in A])
        am = np.array([r["amp"] for r in A])
        fit = next((r for r in rows if r["kind"] == kind and r["part"] == "A_fit"), None)
        lbl = rf"{kind}"
        if fit:
            lbl += rf"  ($\beta={fit['beta']:.3f}$)"
        ax1.loglog(mu, am, "o", ms=4.5, color=col, label=lbl)
        ax1.loglog(mu, am[0] * (mu / mu[0]) ** 0.5, "--", lw=0.9, color=col, alpha=0.7)
    ax1.set_xlabel(r"growth rate $\mu^*$")
    ax1.set_ylabel("saturated amplitude $A$")
    ax1.set_title(r"$A\propto\mu^{*1/2}$: supercritical", fontsize=8.5)
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left")

    for kind, col in (("ring1", COLS["ring1"]), ("simpl", COLS["simpl"])):
        B = sorted([r for r in rows if r["kind"] == kind and r["part"] == "B"],
                   key=lambda r: r["Dv"])
        if not B:
            continue
        Dv = [r["Dv"] for r in B]
        ax2.plot(Dv, [r["amp_down"] for r in B], "o-", ms=3.2, lw=1.0, color=col,
                 label=rf"{kind} down")
        ax2.plot(Dv, [r["amp_up"] for r in B], "s--", ms=3.2, lw=1.0, color=col,
                 alpha=0.75, mfc="none", label=rf"{kind} up")
    ax2.axvspan(8.5, 9.6, color="0.9", zorder=0)
    ax2.set_xlabel("$D_v$")
    ax2.set_ylabel("amplitude")
    ax2.set_title("up/down sweeps agree except near onset", fontsize=8.5)
    ax2.legend(frameon=False, fontsize=6, loc="upper left")
    ax2.invert_xaxis()
    save(fig, "fig11_bifurcation")


if __name__ == "__main__":
    print("Generating figures:")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    print(f"\nOutput directory: {FIGS}")
