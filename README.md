# Size windows and a spectral-weight scaling law for Turing instability on higher-order networks

Code and data for the paper of the same name. The paper is written in Markdown and
typeset with pandoc and LaTeX, so every number in it can be traced back to a script and
a JSON file in this repository.

## What is here

    paper/manuscript.md          main text
    paper/supplementary.md       supplementary material
    paper/latex/*.pdf            typeset output
    src/                         simulation, analysis and figure scripts
    src/tools/                   manuscript quality checks
    figs/                        figures, vector PDF and 300 dpi PNG
    results/                     raw output of every experiment, JSON

`paper/latex/*.tex` is produced by the build and is not tracked.

## Requirements

Python 3.13 or newer:

    pip install -r requirements.txt

The build additionally needs `pandoc` (2.0 or newer) and a LaTeX installation
providing `pdflatex`. Both must be on `PATH`.

## Build the paper

    python src/build_paper.py

Regenerates `paper/latex/manuscript.pdf` and `paper/latex/supplementary.pdf`.

## Regenerate the figures

    python src/make_figures.py

## Rerun the experiments

Each `exp_*.py` script runs one experiment and writes `results/<script name>.json`.
For example:

    python src/exp_growth.py
    python src/exp_params.py
    python src/exp_defect.py

The analysis scripts `complex.py`, `rd.py` and `analytic_windows.py` hold the model,
the linear stability analysis and the closed-form size windows; every `exp_*.py` script
builds on them.

Two scripts write statistical summaries of the same experiment:
`exp_stats_lean.py` produces the `results/exp_stats.json` used in the paper and runs in
a few minutes, while `exp_stats.py` is the full version and writes
`results/exp_stats_full.json`.

## Verification

    python src/verify_manuscript_numbers.py

Reads every number quoted in the main text and the supplementary material back from
`results/*.json`, recomputes it and compares. Prints one PASS or FAIL line per item and
a final count.

    python src/tools/check_paragraph_completeness.py

Checks that every paragraph and caption of the Markdown sources reaches the built PDFs.

    python src/tools/check_closed_form_error.py
    python src/tools/audit_pdf_structure.py
    python src/tools/page_fill.py

Closed-form error table, PDF structural audit, and a per-page pagination report.

## Note on the size-window result

The smallest size that supports a pattern is

    n_min = ceil(2 pi / theta_+) ~ ceil(2 pi sqrt(C / lambda_+)),   C = sum_d d^2

for a circulant support with dispersion symbol f(theta). The closed form is an
approximation: its continuous value differs from the exact 2 pi/theta_+ by less than one
node up to C about 6e3, always on the high side, so rounding it up can overshoot the true
integer by one. `src/tools/check_closed_form_error.py` tabulates exact, continuous and
integer values side by side.

## License

Code is released under the MIT license, see `LICENSE`. The manuscript text and the
figures are released under CC BY 4.0.
