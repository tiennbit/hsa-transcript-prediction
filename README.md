# Predicting Standardized Competency Performance from High-School Transcripts

Analysis code and aggregate results for the paper:

> Nguyen, B. T., Vuong, T. P. T., Ngo, T. T. T., & Nguyen, H.-N. *Predicting Standardized Competency Performance from High-School Transcripts: A Large-Scale Educational Data Mining Study on Predictive Alignment and Admission Equity in Vietnam.* Submitted to *Education and Information Technologies* (Springer).

The study links three years of high-school transcripts to scores on Vietnam's HSA competency assessment (56,882 students, 1,124 province–school units, 49 provinces; 2024 cohort) and analyzes predictive alignment (cross-validated R² ≈ .43; school-held-out R² ≈ .40), transcript signal structure (mathematics dominance; grade-11 vs grade-12 fidelity), and admission equity (group-conditional signed-bias audit with school-clustered inference).

## Data availability

The de-identified student-level dataset **cannot be redistributed** (administrative records containing student-level educational information). This repository contains the full analysis code and all **aggregate** outputs (metrics, correlations, group-level audit tables, figures). Requests for collaboration on the data should be addressed to the corresponding author (namnh@vnu.edu.vn), subject to the data owner's approval.

## Repository layout

```
analysis/
  01_baseline.py                 De-identification + linear baseline
  02_experiments.py              Main results: model comparison, importance/SHAP,
                                 grade distributions, component targets (Tables 2–5, 7)
  03_revision_analyses.py        OOF equity audit (attribute-blind), registration-time
                                 model, Steiger tests, coverage, variance decomposition
  04_figures.py                  Publication figures (Fig 1–8)
  05_mixed_effects_icc.py        ICC: random-effects variance decomposition
                                 (school/province; unconditional + residual)
  06_school_heldout.py           GroupKFold-by-school generalization (unseen schools)
  07_cluster_bootstrap.py        School-cluster bootstrap CIs vs iid (core quantities)
  08_cluster_bootstrap_full.py   Cluster bootstrap for all audited groups +
                                 Bonferroni recount for the 22-province audit
  09_range_screen_sensitivity.py Grade range screen (4 decimal-shift cells) +
                                 sensitivity of descriptive stats and model metrics
  outputs/                       Aggregate CSV/PNG results produced by the scripts
```

Scripts expect the (non-distributed) `analysis/data_deid.csv` next to them; all outputs in `analysis/outputs/` are reproducible from it with the pinned environment below.

## Reproducibility

- Python 3.12.11 — see `requirements.txt` (pinned versions)
- Global seed: 42 (train/test split 80/20, n_test = 11,377; 5-fold CV; cluster bootstrap 500 school resamples)
- Sample: N = 56,882 students; 1,124 province–school units (987 distinct school-name strings); 49 provinces

## Ethics

Retrospective analysis of de-identified administrative educational records held by the Institute of Digital Education and Testing, VNU Hanoi, under its data-management mandate for the HSA examination; no interaction with students; results reported only at aggregate level (minimum audited group size n = 200).

## License

Code is released under the MIT License (see `LICENSE`). The aggregate result files are provided for verification and reuse with attribution to the paper.
