# A Counterfactual-Explanation-and-Optimisation Framework for Budget-Constrained Intervention Targeting

Replication code for the manuscript *"A Counterfactual-Explanation-and-Optimisation
Framework for Budget-Constrained Intervention Targeting: An Application to Children's
Digital and AI Literacy"* (under review, **anonymous journal**).

> **Double-blind notice.** This repository is intentionally anonymised. It contains no
> author names, affiliations, or contact details, and should not be de-anonymised while
> the manuscript is under review.

## Overview

The framework couples **counterfactual explanation** (explainable AI) with
**budget-constrained optimisation** (operations research) into a single estimate-then-allocate
pipeline:

1. **Module 1 — Prediction.** Out-of-sample comparison of a *provisioning/resource* block
   against a *participation/activity* block for predicting Computer & Information Literacy
   (CIL). Provisioning carries almost no out-of-sample signal; participation does.
2. **Module 2 — Counterfactual sensitivity.** Per-subgroup (SES × sex) intervention
   sensitivity, estimated by raising positively-associated, policy-deliverable engagement
   levers to the national weighted 75th percentile. Interpreted as *relative sensitivity
   for prioritisation*, not as an average treatment effect.
3. **Module 3 — Targeting & frontier.** Effect/cost ratios feed a fractional-knapsack
   allocation; the efficiency–equity frontier is traced across budgets and cost structures.
4. **Cross-national replication** over eight education systems, and a **policy-context
   appendix** built from the IEA national-contexts survey.

## Repository structure

```
.
├── data/                 # place IEA ICILS 2023 files here (not redistributed — see below)
├── notebooks/
│   ├── 01_prediction_signal.ipynb
│   ├── 02_counterfactual_sensitivity.ipynb
│   ├── 03_targeting_frontier.ipynb
│   ├── 04_cross_country.ipynb
│   ├── 05_policy_appendix.ipynb
│   ├── 06_inference.ipynb          # cluster-bootstrap CIs (Modules 1-2), reduced levers, 5-PV baselines
│   ├── 07_robustness.ipynb         # learner x benchmark robustness, split-sample sign screen, sign test
│   ├── 08_optimization.ipynb       # MKP, submodular (exact PWL-MILP), robust knapsack, exact Pareto
│   ├── 09_binding.ipynb            # sensitivity: when the optimization structures bind (Appendix D)
│   ├── 10_cross_country_full.ipynb # full 34-system replication + sign test
│   ├── 11_replication.ipynb        # optimization replication in further systems
│   └── 12_convergent.ipynb         # convergent-validity check vs model-free contrast
├── results/
│   ├── figures/          # grayscale .png + .pdf, 600 dpi
│   └── tables/           # .csv (+ .tex)
├── requirements.txt
└── README.md
```

## Data access

This study uses the **IEA ICILS 2023 International Database** (public-use) and its
**National Contexts Survey**. The data are **not redistributed here** under the IEA
licence. To reproduce:

1. Download the ICILS 2023 International Database (SAS format) and the National Contexts
   Survey from the IEA Data Repository (`https://www.iea.nl/data-tools/repository`).
   DOI: `10.58150/ICILS_2023_data_2`.
2. Place the student (`bsg*i3.sas7bdat`) and school (`bcg*i3.sas7bdat`) files for the
   analysed systems, together with `ICILS2023NCS.sav`, directly inside `data/`.

The analysis uses plausible values `PV1CIL–PV5CIL`, final student weights (`TOTWGTS`),
and jackknife-compatible school-clustered resampling.

## Environment

Python 3.13. Install dependencies:

```
pip install -r requirements.txt
```

## Reproduce

Run the notebooks in order (each is a single self-contained cell):

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_prediction_signal.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_counterfactual_sensitivity.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_targeting_frontier.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_cross_country.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_policy_appendix.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_inference.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/07_robustness.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/08_optimization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/09_binding.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/10_cross_country_full.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/11_replication.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/12_convergent.ipynb
```

Notebooks 03, 05, 07 and 08 read tables written by earlier notebooks, so run them in
sequence. Notebook 08 additionally requires `scipy>=1.9` (HiGHS MILP solver).
All figures are written to `results/figures/` as 600-dpi grayscale PNG and PDF (no titles
or captions; captions belong in the manuscript); all tables to `results/tables/`.

## Citation

```
@article{anonymous_targeting,
  title   = {A Counterfactual-Explanation-and-Optimisation Framework for
             Budget-Constrained Intervention Targeting},
  author  = {Anonymous},
  journal = {Anonymous journal (under review)},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The ICILS data are governed by the
IEA licence and are not covered by this repository's licence.
