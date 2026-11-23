#!/usr/bin/env python3
"""Verify the correlation-aware separation condition for M1.

Claim to test: adjacent pairwise order stability is governed by the SE of the
DIFFERENCE  s_diff(k) = sqrt(s_k^2 + s_{k+1}^2 - 2 cov_{k,k+1}),  not by the
marginal sum  s_k + s_{k+1}  that Proposition 1 uses. The former predicts the
joint-bootstrap pairwise swap frequency; the latter does not.

We recompute, per adjacent pair:
  - marginal separation      z_marg = gap / (s_k + s_{k+1})            [Prop 1]
  - difference separation     z_diff = gap / s_diff                     [new]
  - empirical swap frequency  P(bootstrap order of the pair reverses)
and check that Phi(-z_diff) tracks the empirical swap frequency while
Phi(-z_marg) is systematically too pessimistic.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import norm

from pathlib import Path as _P
DATA = str((_P("..")/"data")) + "/" if (_P("../data")).exists() else "data/"
PVS = [f"PV{i}CIL" for i in range(1, 6)]
LEV = ["S_LRNINTO", "S_ICTPOSS", "S_GENCLASS", "S_LRNSAFE"]
CTX = ["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
       "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

def wpct(v, wt, p):
    m = ~np.isnan(v); vs = np.sort(v[m]); ww = wt[m][np.argsort(v[m])]
    return np.interp(p, np.cumsum(ww)/ww.sum(), vs)

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
x = d.NISB.to_numpy(); ok = ~np.isnan(x); o = np.argsort(x[ok])
cw = np.cumsum(W[ok][o])/W[ok].sum(); q = np.interp([1/3, 2/3], cw, x[ok][o])
d["SES3"] = pd.Series(np.where(x <= q[0], "Low", np.where(x <= q[1], "Mid", "High"))).where(ok).values
d["SEX"] = d.S_SEX.map({0.0: "Sex0", 1.0: "Sex1"})
d = d.dropna(subset=["SES3", "SEX"]).copy(); d["grp"] = d.SES3 + "/" + d.SEX

groups = sorted(d.grp.unique())
pt = d.groupby("grp").apply(lambda z: np.average(z.dCIL, weights=z.TOTWGTS))
order = pt.sort_values(ascending=False).index.tolist()

# cluster bootstrap: full joint distribution of the 6 subgroup Deltas
schools = d.IDSCHOOL.unique()
sidx = {s: np.where(d.IDSCHOOL.to_numpy() == s)[0] for s in schools}
dd_arr = d.dCIL.to_numpy(); w_arr = d.TOTWGTS.to_numpy(); grp_arr = d.grp.to_numpy()
rng = np.random.default_rng(0); B = 2000
boot = np.full((B, len(groups)), np.nan)
for b in range(B):
    pick = rng.choice(schools, len(schools), replace=True)
    rows = np.concatenate([sidx[s] for s in pick])
    sg = grp_arr[rows]; sd = dd_arr[rows]; sw = w_arr[rows]
    for gi, g in enumerate(groups):
        mm = sg == g
        if mm.any(): boot[b, gi] = np.average(sd[mm], weights=sw[mm])
boot = pd.DataFrame(boot, columns=groups).dropna()
cov = np.cov(boot.values, rowvar=False)
se = boot.std().to_dict()
gidx = {g: i for i, g in enumerate(groups)}

print("point order:", " > ".join(order))
print(f"\n{'pair':<22}{'gap':>7}{'s_k+s_k1':>10}{'s_diff':>8}{'z_marg':>8}{'z_diff':>8}"
      f"{'Phi(-zm)':>9}{'Phi(-zd)':>9}{'emp.swap':>9}")
rows_out = []
for k in range(len(order)-1):
    a, bb = order[k], order[k+1]
    ia, ib = gidx[a], gidx[bb]
    gap = pt[a] - pt[bb]
    smarg = se[a] + se[bb]
    sdiff = np.sqrt(cov[ia, ia] + cov[ib, ib] - 2*cov[ia, ib])
    z_marg = gap / smarg
    z_diff = gap / sdiff
    # empirical swap freq for this adjacent pair in the bootstrap
    emp = np.mean(boot[bb].values > boot[a].values)
    rows_out.append((f"{a} > {bb}", gap, smarg, sdiff, z_marg, z_diff,
                     norm.cdf(-z_marg), norm.cdf(-z_diff), emp))
    print(f"{a+' > '+bb:<22}{gap:>7.2f}{smarg:>10.2f}{sdiff:>8.2f}{z_marg:>8.2f}"
          f"{z_diff:>8.2f}{norm.cdf(-z_marg):>9.3f}{norm.cdf(-z_diff):>9.3f}{emp:>9.3f}")

R = pd.DataFrame(rows_out, columns=["pair","gap","s_marg","s_diff","z_marg","z_diff",
                                    "phi_zmarg","phi_zdiff","emp_swap"])
# how well does each predicted swap prob track the empirical?
from numpy import corrcoef
print("\nCorrelation of predicted vs empirical swap frequency across pairs:")
print(f"  marginal (Prop 1)  : r = {corrcoef(R.phi_zmarg, R.emp_swap)[0,1]:.3f}  "
      f"mean abs err = {np.mean(np.abs(R.phi_zmarg-R.emp_swap)):.3f}")
print(f"  difference (new)   : r = {corrcoef(R.phi_zdiff, R.emp_swap)[0,1]:.3f}  "
      f"mean abs err = {np.mean(np.abs(R.phi_zdiff-R.emp_swap)):.3f}")
print("\nmean bootstrap correlation between adjacent subgroup Deltas:")
cc = []
for k in range(len(order)-1):
    ia, ib = gidx[order[k]], gidx[order[k+1]]
    cc.append(cov[ia, ib]/np.sqrt(cov[ia, ia]*cov[ib, ib]))
print(f"  mean adjacent corr = {np.mean(cc):.3f}  (positive => marginal sum overstates the diff SE)")
R.to_csv("../results/tables/m1_corr_separation.csv", index=False)
print("\nwrote m1_corr_separation.csv")
