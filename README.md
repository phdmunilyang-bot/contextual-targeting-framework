# Contextual Optimization for Budget-Constrained Public-Sector Targeting

Replication code for the manuscript

> *Contextual Optimization for Budget-Constrained Public-Sector Targeting:
> Counterfactual Side Information, Robust Allocation, and the Efficiency–Equity
> Trade-off, with an Application to Children's Digital Literacy* (under review).

The framework couples counterfactual explanation (explainable AI) with
budget-constrained optimisation (operations research) into a single
estimate-then-allocate pipeline for deciding whom to reach first under a fixed
public budget.

## Overview

1. Module 1, Prediction. Out-of-sample comparison of a provisioning/resource
   block against a participation/activity block for predicting Computer &
   Information Literacy (CIL). Provisioning carries almost no out-of-sample signal;
   participation does, reframing allocation from spending to reach.
2. Module 2, Counterfactual sensitivity. Per-subgroup (SES × sex) intervention
   sensitivity, estimated by raising positively-associated, policy-deliverable
   engagement levers to the national weighted 75th percentile. Interpreted as a
   relative priority for prioritisation, not an average treatment effect.
3. Module 3, Targeting & frontier. Effect/cost ratios feed a fractional-knapsack
   allocation; the efficiency–equity frontier is traced across budgets and cost
   structures, with a robust (lower-confidence-bound) variant for the estimated
   sensitivities.
4. Cross-national replication over all 34 participating education systems, and a
   policy-context appendix built from the IEA national-contexts survey.

## Repository structure

```
.
├── data/                 # place IEA ICILS 2023 files here (not redistributed, see below)
├── notebooks/
│   │  # --- core pipeline (run in order) ---
│   ├── 01_prediction_signal.ipynb       # Module 1: provisioning vs participation OOS R^2
│   ├── 02_counterfactual_sensitivity.ipynb # Module 2: subgroup sensitivities
│   ├── 03_targeting_frontier.ipynb      # Module 3: targeting order + efficiency–equity frontier
│   ├── 04_cross_country.ipynb           # eight-system cross-national replication
│   ├── 05_policy_appendix.ipynb         # national policy-context appendix
│   ├── 06_inference.ipynb               # cluster-bootstrap CIs (Modules 1–2), reduced levers, 5-PV
│   ├── 07_robustness.ipynb              # learner × benchmark robustness, split-sample sign screen
│   ├── 08_optimization.ipynb            # MKP, submodular (exact PWL-MILP), robust knapsack, exact Pareto
│   ├── 09_binding.ipynb                 # sensitivity: when the optimisation structures bind (Appendix D)
│   ├── 10_cross_country_full.ipynb      # full 34-system replication + sign test
│   ├── 11_replication.ipynb             # optimisation replication in further systems
│   ├── 12_convergent.ipynb              # convergent-validity check vs model-free contrast
│   │  # --- revision analyses (referee response; each is a standalone script) ---
│   ├── 13_rev_signtests.py              # conservative sign tests on the cross-national sign
│   ├── 14_rev_finer_partition.py        # finer SES×sex×migration partition where migrant n permits
│   ├── 15_rev_lever_split.py            # directly-deliverable vs attitudinal lever re-estimation
│   ├── 16_rev_cate_comparison.py        # causal-forest CATE vs counterfactual, same knapsack
│   ├── 17_rev_separation_diagnostic.py  # adjacent-separation diagnostic (Proposition 1)
│   ├── 18_rev_reordering_probability.py # joint-bootstrap reordering probability
│   ├── 19_rev_pof_spread.py             # price-of-fairness spread across the correlation spectrum
│   └── 20_rev_denmark_mechanism.py      # why a weak-correlation system can carry a high price of fairness
├── results/
│   ├── figures/          # grayscale .png + .pdf, 600 dpi
│   └── tables/           # .csv (+ .tex)
├── requirements.txt
└── README.md
```

The core notebooks (01–12) reproduce the main text and appendices. The revision
scripts (13–20) are standalone Python files added in response to peer review; each
reads the IEA data (and, for 13, the `cross_country_35.csv` written by notebook 10)
and writes its own table to `results/tables/`. Run them with `python -I <script>.py`.

## Data access

This study uses the IEA ICILS 2023 International Database (public-use) and its
National Contexts Survey. The data are not redistributed here under the IEA
licence. To reproduce:

1. Download the ICILS 2023 International Database (SAS format) and the National
   Contexts Survey from the IEA Data Repository
   (`https://www.iea.nl/data-tools/repository`). DOI: `10.58150/ICILS_2023_data_2`.
2. Place the student (`bsgi3.sas7bdat`) and school (`bcgi3.sas7bdat`) files for the
   analysed systems, together with `ICILS2023NCS.sav`, directly inside `data/`.

The analysis uses plausible values `PV1CIL–PV5CIL`, final student weights (`TOTWGTS`),
and jackknife-compatible school-clustered resampling.

## Environment

Python 3.13. Install dependencies:

```
pip install -r requirements.txt
```

The revision script `16_rev_cate_comparison.py` additionally requires `econml`
(causal forest); install it with `pip install econml` if it is not already present.

## Reproduce

Run the core notebooks in order:

```
for nb in notebooks/0.ipynb notebooks/1[0-2]_.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Notebooks 03, 05, 07 and 08 read tables written by earlier notebooks, so run them in
sequence. Notebook 08 requires `scipy>=1.9` (HiGHS MILP solver). Then run the revision
analyses:

```
cd notebooks
for s in 13_.py 14_.py 15_.py 16_.py 17_.py 18_.py 19_.py 20_.py; do
  python -I "$s"
done
```

All figures are written to `results/figures/` as 600-dpi grayscale PNG and PDF (no
titles or captions; captions belong in the manuscript); all tables to
`results/tables/`.

## Second-round revision (scripts 21–26)

Scripts `21–26` were added in the second round of revision. Each is standalone
(`python -I <script>.py`), reads the IEA data from `data/`, and writes to
`results/tables/`.

- `21_rev2_cross_country_34.py`: Authoritative full cross-national replication over
  the 34 participating countries. This supersedes `cross_country_35.csv`: the earlier
  full scan (notebook 10) inadvertently (a) included the Denmark Danish-language
  benchmarking participant (`DNW`, IEA code 276001) as if it were a distinct education
  system, and (b) silently dropped Sweden when scikit-learn's histogram binner failed
  on a background scale (`S_ICTSTUD`) that is 100% missing in the Swedish public-use file.
  Script 21 excludes the benchmark participant, includes all 34 countries, and guards
  against all-missing columns, writing `cross_country_34.csv`. The headline result is
  unchanged (31/34 negative, sign-test p = 3.8e-7, median −0.93; positives: AZE, DEU, URY),
  because `DNW` (−0.69) and `SWE` (−0.94) are both negative, so the swap is sign-neutral.
- `22_rev2_corr_separation.py`: Validates the correlation-aware separation condition
  (Proposition 2): the gap standard error σ_Δ predicts the joint-bootstrap swap frequency
  to within 0.002 on average, versus 0.074 for the independence-based marginal condition.
- `23_rev2_robust_box_vs_ellipsoid.py`: Shows the Bertsimas–Sim box/Γ uncertainty set
  is a conservative (≈15.5× larger) cover of the covariance-aware ellipsoidal set; the two
  select the identical funded set at every uncertainty scale, so the box price of robustness
  upper-bounds the correlation-aware one.
- `24_rev2_student_level_interaction.py`: Student-level test of the
  disadvantage→responsiveness relation (school-clustered SE), bypassing the six-group-mean
  correlation: SES slope −2.40 per SD (p ≈ 1e-6), baseline-CIL slope −14.1 per SD.
- `25_rev2_kappa_calibration.py`: Calibrates the robustness stress multiplier κ against
  real school subsamples: κ ≈ 1.6 at half the schools, 2.3 at a quarter, 3.3 at an eighth.
- `26_rev2_cate_multisystem.py`: Extends the causal-forest comparison to eight systems:
  the coarse decision (low-SES group first; negative baseline–response relation) is
  estimation-route-invariant in 7/8; mean fine-order Kendall τ = 0.60.

Run them after the core notebooks:

```
cd notebooks
for s in 2[1-6]_rev2_*.py; do python -I "$s"; done
```

Note on `cross_country_35.csv`: it is retained for provenance but is not the analysis
of record; use `cross_country_34.csv` (script 21). The main-text claim of "34 participating
education systems" refers to the 34 countries, excluding the benchmarking participant.

## License

Code is released under the MIT License (see `LICENSE`). The ICILS data are governed by
the IEA licence and are not covered by this repository's licence.
