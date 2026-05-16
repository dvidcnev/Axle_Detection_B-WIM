---
description: "Use when writing or editing LaTeX thesis files (.tex, .bib). Covers chapter structure, citation style, figure/table conventions, and FAMNIT formatting rules."
applyTo: "**/*.tex, **/*.bib"
---

# LaTeX Thesis Rules — B-WIM (FAMNIT Template)

## Template & Build
- Template class: `famnit-thesis` (`famnit-thesis.cls`). Do not modify the class file.
- Build with English option: `\documentclass[english]{famnit-thesis}`
- Build order: `pdflatex thesis` → `bibtex thesis` → `pdflatex thesis` → `pdflatex thesis`
- Or use Overleaf (recommended for collaboration with supervisor).

## Chapter Structure (thesis.tex inputs)
```
uvod.tex        → Chapter 1: Introduction
background.tex  → Chapter 2: Background and Related Work
metodologija.tex→ Chapter 3: Methodology
results.tex     → Chapter 4: Results and Discussion
zakljucek.tex   → Chapter 5: Conclusion
priloga.tex     → Appendix A (optional)
```
- The `tehnicni-predpisi.tex` file is a formatting guide — it is **not** included in the final thesis (`\input` it is removed in `thesis.tex`).

## Citations
- All citations use `\cite{key}`. Never use footnotes for references.
- Citation keys follow the pattern: `AuthorYear` (e.g., `Moses1979`, `Soberl2025`, `Bai2018`).
- References must be in `bibliography.bib`. Do not hardcode reference text in chapter files.
- Key references already in the bib: `Moses1979`, `Soberl2025`, `SavitzkyGolay1964`, `Ronneberger2015`, `Bai2018`, `Paszke2019`, `Scipy2020`, `RussellNorvig2021`.

## Figures
- All figures stored in `latex-format/images/` as `.pdf` (vector) or `.png` (raster).
- Generated from Python notebooks using `plt.savefig('../latex-format/images/<name>.pdf', bbox_inches='tight')`.
- Required figures (from notebooks, to be uncommented in results.tex when ready):
  - `cnn_training_curves.pdf` (notebook 03)
  - `tcn_training_curves.pdf` (notebook 04)
  - `tcn_receptive_field.pdf` (notebook 04)
  - `tcn_visual_inspection.pdf` (notebook 04)
  - `comparison_f1_bar.pdf` (notebook 05)
- Always use `\label{fig:name}` and `\ref{fig:name}`. Never refer to figures by number in text.

## Tables
- Use `booktabs` package: `\toprule`, `\midrule`, `\bottomrule`. Never use `\hline` in data tables.
- Caption above tables (`\caption{}` before `\begin{tabular}`), below figures.
- Results table (Table 4.1 in results.tex): fill in Baseline row once notebook 02 is run.

## Math
- Define all equations with `\label{eq:name}` and reference with `\eqref{eq:name}`.
- Key equations already defined:
  - `\eqref{eq:rf}` — TCN receptive field formula
  - `\eqref{eq:bce}` — Weighted BCE loss
- Use `\text{}` for non-italic labels inside equations: `\text{MATE}`, `\text{Precision}`.

## Abbreviations
- Defined in `thesis.tex` via `\abbreviation{KEY}{Definition}`.
- Already defined: B-WIM, BCE, CNN, F1, GVW, HGV, MATE, ReLU, TCN.
- First use in text: spell out + abbreviation in parentheses, e.g., "Temporal Convolutional Network (TCN)".

## Slovene Extended Abstract
- Required for English theses: 4,000–10,000 characters including spaces.
- Goes in a separate file `abstract_sl.tex` as the last numbered chapter before References.
- Uncomment `\input{abstract_sl.tex}` in `thesis.tex` when ready.

## TODO items remaining in thesis files
- `thesis.tex`: Replace `[FirstName]`, `[LastName]`, `[prof. dr. Supervisor Name]` with real values.
- `results.tex`: Fill in Baseline row in Table 4.1.
- `results.tex`: Uncomment all `\includegraphics` lines once figures are exported from notebooks.
- `zakljucek.tex` / `uvod.tex`: Add `abstract_sl.tex` when Slovene abstract is written.
