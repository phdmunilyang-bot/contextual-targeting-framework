#!/usr/bin/env python3
"""M6: student-level test of the disadvantage->responsiveness relation, bypassing
the six-subgroup-mean correlation (whose CI includes zero).

We regress the per-student counterfactual sensitivity dCIL_i on a baseline-advantage
measure at the student level, with school-clustered standard errors. Two advantage
measures: (i) SES index (NISB), (ii) baseline CIL (CIL5). A negative, significant
slope says: more-advantaged students are LESS responsive -- the disadvantage->
responsiveness relation, now on n=3,723 students instead of 6 group means.

Reported: OLS slope (weighted), school-cluster-robust SE and p, and a cluster-
bootstrap CI, for the primary case (Korea).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor

from pathlib import Path as _P
DATA = str((_P("..")/"data")) + "/" if (_P("../data")).exists() else "data/"
PVS = [f"PV{i}CIL" for i in range(1, 6)]
LEV = ["S_LRNINTO", "S_ICTPOSS", "S_GENCLASS", "S_LRNSAFE"]
CTX = ["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
       "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

def wpct(v, wt, p):
    m = ~np.isnan(v); vs = np.sort(v[m]); ww = wt[m][np.argsort(v[m])]
    return np.interp(p, np.cumsum(ww)/ww.sum(), vs)

def wls_cluster(y, X, w, clusters):
    """Weighted LS with cluster-robust (CR0) covariance. X includes intercept col."""
    Wr = w / w.mean()
    XtWX = X.T @ (Wr[:, None] * X)
    XtWXinv = np.linalg.inv(XtWX)
    beta = XtWXinv @ (X.T @ (Wr * y))
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(clusters):
        idx = clusters == c
        Xc = X[idx]; uc = (Wr[idx] * resid[idx])
        sc = Xc.T @ uc
        meat += np.outer(sc, sc)
    V = XtWXinv @ meat @ XtWXinv
    se = np.sqrt(np.diag(V))
    return beta, se

stu, _ = pyreadstat.read_sas7bdat(DATA + "bsgkori3.sas7bdat")
for c in set(LEV+CTX+["NISB"]):
    if c in stu: stu.loc[(stu[c] >= 995) | (stu[c] <= -995), c] = np.nan
stu.loc[(stu.NISB < -6) | (stu.NISB > 6), "NISB"] = np.nan
stu["CIL5"] = stu[PVS].mean(1)
d = stu[stu.TOTWGTS.notna() & stu.IDSCHOOL.notna()].copy()
W = d.TOTWGTS.to_numpy()
Xb = np.array(d[LEV+CTX].to_numpy(float), copy=True); Xc = Xb.copy()
for j, L in enumerate(LEV):
    t = wpct(Xb[:, j], W, 0.75); col = Xc[:, j]
    Xc[:, j] = np.where(np.isnan(col), col, np.maximum(col, t))
dd = np.zeros((len(d), 5))
for k, pv in enumerate(PVS):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=0).fit(Xb, d[pv].to_numpy(float), sample_weight=W)
    dd[:, k] = m.predict(Xc) - m.predict(Xb)
d["dCIL"] = dd.mean(1)
d["sch"] = d.IDSCHOOL.to_numpy()

from scipy.stats import norm
for adv_name, adv_col in [("baseline CIL", "CIL5"), ("SES index (NISB)", "NISB")]:
    sub = d.dropna(subset=[adv_col, "dCIL"]).copy()
    a = sub[adv_col].to_numpy(float)
    # standardise advantage to unit SD for interpretable slope
    a_z = (a - np.average(a, weights=sub.TOTWGTS)) / np.sqrt(np.average((a-np.average(a,weights=sub.TOTWGTS))**2, weights=sub.TOTWGTS))
    y = sub["dCIL"].to_numpy(float)
    w = sub.TOTWGTS.to_numpy(float)
    X = np.column_stack([np.ones(len(a_z)), a_z])
    beta, se = wls_cluster(y, X, w, sub["sch"].to_numpy())
    z = beta[1] / se[1]; p = 2 * norm.cdf(-abs(z))
    # cluster bootstrap CI
    schools = sub["sch"].unique(); rng = np.random.default_rng(0); bs = []
    sidx = {s: np.where(sub["sch"].to_numpy() == s)[0] for s in schools}
    for _ in range(1000):
        pick = rng.choice(schools, len(schools), replace=True)
        rows = np.concatenate([sidx[s] for s in pick])
        Xb2 = X[rows]; yb = y[rows]; wb = w[rows]
        Wr = wb / wb.mean()
        bb = np.linalg.solve(Xb2.T @ (Wr[:, None]*Xb2), Xb2.T @ (Wr*yb))
        bs.append(bb[1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"=== Student-level: dCIL ~ {adv_name} (per 1 SD), n={len(sub)}, {sub['sch'].nunique()} schools ===")
    print(f"  slope = {beta[1]:+.3f} CIL points per SD of advantage")
    print(f"  cluster-robust SE = {se[1]:.3f}, z = {z:.2f}, p = {p:.2e}")
    print(f"  cluster-bootstrap 95% CI = [{lo:+.3f}, {hi:+.3f}]")
    print(f"  --> {'excludes zero' if hi<0 or lo>0 else 'includes zero'}; "
          f"sign is {'NEGATIVE (disadvantaged more responsive)' if beta[1]<0 else 'positive'}")
    print()
