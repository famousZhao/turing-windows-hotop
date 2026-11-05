# Size windows and a spectral-weight scaling law for Turing instability on higher-order networks

Simulation, analysis and figure code for the paper of the same name, together with the
raw output of every experiment.

## What is here

    src/            model, experiments, analysis, figures
    results/        raw output of every experiment, JSON
    figs/           figures, vector PDF and 300 dpi PNG
    paper/          the two Markdown sources, for reference

This repository holds the scientific code only. Typesetting the Markdown sources into
the manuscript PDF needs a separate pandoc and LaTeX pipeline that is not distributed
here.

## Requirements

Python 3.13 or newer:

    pip install -r requirements.txt

## The model

    src/complex.py             simplicial complexes, boundary operators, Laplacians
    src/rd.py                  Schnakenberg reaction-diffusion, linear stability, RK4
    src/analytic_windows.py    exact size windows and the minimal size, analytic

Three circulant supports are used throughout. Their Laplacian eigenvalues have the
closed form f(2 pi k / n), where f is the dispersion symbol:

    ring1   only +-1 edges                  f = 2 - 2 cos(theta),                  C = 1
    ring2   +-1 and +-2 edges               f = 4 - 2 cos(theta) - 2 cos(2 theta), C = 5
    simpl   ring2 plus contiguous triangles f = 10 - 6 cos(theta) - 4 cos(2 theta), C = 11

The system is Turing unstable at size n if and only if some integer k places
f(2 pi k / n) inside the band of the kinetics. For the default kinetics the band is
(0.2, 0.5), and the minimal unstable size is 9, 20 and 30 for the three supports.

## Experiments

Each `exp_*.py` script runs one experiment and writes `results/<script name>.json`.
Examples:

    python src/exp_growth.py       pattern on a growing support
    python src/exp_params.py       seven kinetic parameter sets, three supports
    python src/exp_defect.py       mode selection against the Rayleigh quotient
    python src/exp_noise.py        noise floor and reseeding in a dead zone

`exp_stats_lean.py` produces the `results/exp_stats.json` used in the paper and runs in
a few minutes. `exp_stats.py` is the full version of the same measurement and writes
`results/exp_stats_full.json`.

## Figures

    python src/make_figures.py

Reads `results/*.json` and writes all eleven figures to `figs/`, as PDF and as 300 dpi
PNG. It does not need the figures to be present beforehand.

## Verification

    python src/verify_manuscript_numbers.py

Reads every number quoted in the manuscript back from `results/*.json`, recomputes it
and compares. Prints one PASS or FAIL line per item and a final count.

    python src/check_closed_form_error.py

Tabulates the exact minimal size, the continuous closed form and the rounded integer
side by side. The closed form has a continuous error below one node up to C about 6e3,
always on the high side, so rounding it up can overshoot the true integer by one.

## License

MIT, see `LICENSE`. The manuscript text and the figures are released under CC BY 4.0.
