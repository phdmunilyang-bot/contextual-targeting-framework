#!/usr/bin/env python3
"""M2: does the Bertsimas-Sim (box, independence-based) robust rule remain valid
when the estimation errors co-move?

We compare three uncertainty models for the SAME subgroup targeting problem:
  (a) NOMINAL    : point estimates, no hedging.
  (b) BOX / Gamma: Bertsimas-Sim budgeted box, marginal SEs, the authors' rule.
  (c) ELLIPSOID  : covariance-aware robust using the FULL bootstrap covariance Sigma,
                   i.e. worst case over { Delta : (Delta-hat)^T Sigma^{-1} (Delta-hat) <= kappa^2 }.
The ellipsoid is the correct uncertainty set when errors co-move; comparing the
robust ORDER and the price of robustness under (b) vs (c) shows the direction and
size of the error from assuming independence.

Key quantities:
  - worst-case value of the selected set under each model,
  - whether the robust ORDER differs,
  - price of robustness under each.
We also report, at matched coverage probability, whether the box set is
conservative (contains the ellipsoid -> box price >= ellipsoid price) or anti-
conservative. For a given confidence, a positively-correlated Sigma makes the
axis-aligned box that covers it LARGER in volume, so the box rule is conservative:
it over-hedges, which is safe for a planner (never understates risk).
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
mN = d.groupby("grp").apply(lambda z: z.TOTWGTS.sum())
Delta = pt[groups].to_numpy(); mg = mN[groups].to_numpy()

# cluster bootstrap joint law -> covariance of Delta
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
boot = boot[~np.isnan(boot).any(1)]
Sigma = np.cov(boot, rowvar=False)        # covariance of Delta
se = np.sqrt(np.diag(Sigma))
# value vector and its covariance (scale by m_g)
val = mg * Delta
Sig_v = np.outer(mg, mg) * Sigma           # Cov(m_g Delta_g)
se_v = np.sqrt(np.diag(Sig_v))

# correlation structure summary
corr = Sigma / np.outer(se, se)
offdiag = corr[np.triu_indices(len(groups), 1)]
print("=== Bootstrap correlation of subgroup sensitivities ===")
print(f"mean off-diagonal corr = {offdiag.mean():.3f}  (min {offdiag.min():.2f}, max {offdiag.max():.2f})")
print(f"fraction positive       = {np.mean(offdiag>0):.2f}")

# --- Budget: fund ~one-third of weight, pick subgroups to maximise value ---
budget = 0.33 * mg.sum()
order_nom = np.argsort(-val)  # value order (greedy, equal unit cost -> by Delta)

def greedy_select(scores):
    idx = np.argsort(-scores); chosen = []; used = 0.0
    for i in idx:
        if used + mg[i] <= budget: chosen.append(i); used += mg[i]
    return set(chosen)

# (a) nominal
S_nom = greedy_select(val)
# (b) box/Gamma robust: rank by lower bound val - z*se_v (Gamma >= all funded)
z = 1.0
S_box = greedy_select(val - z*se_v)
# (c) ellipsoid robust at matched confidence: worst-case of a selected set S is
#     sum_{g in S} val_g - kappa * sqrt(x_S^T Sig_v x_S). Rank by marginal robust value.
#     For a linear-in-x objective with ellipsoidal uncertainty the robust score of the
#     WHOLE set is value(S) - kappa*sqrt(x^T Sig_v x); we select greedily on robust
#     marginal contribution.
kappa = z  # matched radius (1 sd)
def robust_value_ellipsoid(Sset):
    xs = np.zeros(len(groups))
    for i in Sset: xs[i] = 1.0
    return val @ xs - kappa*np.sqrt(xs @ Sig_v @ xs)
# greedy on ellipsoidal robust marginal gain
def greedy_ellipsoid():
    chosen = []; used = 0.0; remaining = set(range(len(groups)))
    while remaining:
        best = None
        for i in list(remaining):
            if used + mg[i] > budget: continue
            cand = chosen + [i]
            xs = np.zeros(len(groups));
            for j in cand: xs[j] = 1.0
            rv = val @ xs - kappa*np.sqrt(xs @ Sig_v @ xs)
            if best is None or rv > best[0]: best = (rv, i)
        if best is None: break
        chosen.append(best[1]); used += mg[best[1]]; remaining.discard(best[1])
    return set(chosen)
S_ell = greedy_ellipsoid()

def name(s): return [groups[i] for i in sorted(s, key=lambda i:-val[i])]
print("\n=== Selected sets (fund ~1/3 of weight) ===")
print("nominal   :", name(S_nom))
print("box/Gamma :", name(S_box))
print("ellipsoid :", name(S_ell))
print("box == nominal set? ", S_box == S_nom)
print("ellipsoid == nominal set? ", S_ell == S_nom)
print("box == ellipsoid set? ", S_box == S_ell)

# worst-case value under the ellipsoid for each selected set (common yardstick)
print("\n=== Worst-case value of each selected set under the ellipsoidal (covariance) model ===")
for lab, S in [("nominal", S_nom), ("box/Gamma", S_box), ("ellipsoid", S_ell)]:
    print(f"  {lab:10s}: worst-case value = {robust_value_ellipsoid(S):.1f}, nominal value = {val@np.array([1.0 if i in S else 0 for i in range(len(groups))]):.1f}")

# price of robustness under each model (relative to nominal nominal-value)
nomval = val @ np.array([1.0 if i in S_nom else 0 for i in range(len(groups))])
for lab, S in [("box/Gamma", S_box), ("ellipsoid", S_ell)]:
    rv = val @ np.array([1.0 if i in S else 0 for i in range(len(groups))])
    print(f"  price of robustness ({lab}) = {100*(nomval-rv)/nomval:.2f}%")

# Direction-of-conservativeness check:
# For matched confidence 1-alpha, does the box set contain the ellipsoid?
# Box half-widths z*se_v cover the ellipsoid {u: u^T Sig_v^{-1} u <= z^2} iff
# z*se_v_g >= max_u in ellipsoid |u_g| = z*sqrt(Sig_v_gg) = z*se_v_g -> equality on axes,
# but the ellipsoid extends LESS than the box in off-axis directions when corr>0.
# Equivalent statement: box volume / ellipsoid volume.
from numpy.linalg import det
n = len(groups)
vol_box = np.prod(2*z*se_v)
# ellipsoid {u: u^T Sig_v^{-1} u <= z^2} volume = (pi^{n/2}/Gamma(n/2+1)) * z^n * sqrt(det Sig_v)
from math import gamma as G, pi
vol_ell = (pi**(n/2)/G(n/2+1)) * z**n * np.sqrt(det(Sig_v))
print(f"\n=== Uncertainty-set volumes at radius z={z} ===")
print(f"box volume / ellipsoid volume = {vol_box/vol_ell:.1f}  (>1 => box strictly larger => box rule is CONSERVATIVE)")
pd.DataFrame(dict(group=groups, Delta=Delta, se=se, m=mg, val=val, se_val=se_v)).to_csv(
    "../results/tables/m2_robust_groups.csv", index=False)

# --- Stress test: inflate uncertainty (kappa scale) and check box price >= ellipsoid price ---
print("\n=== Stress: price of robustness, box vs ellipsoid, as uncertainty scales ===")
print(f"{'scale':>6}{'box_price%':>12}{'ellip_price%':>14}{'box>=ellip?':>13}")
for scale in [1,2,3,5,8]:
    sev = scale*se_v
    S_box_s = greedy_select(val - sev)
    # ellipsoid greedy at radius=scale
    def greedy_ell_s(kap):
        chosen=[]; used=0.0; rem=set(range(len(groups)))
        while rem:
            best=None
            for i in list(rem):
                if used+mg[i]>budget: continue
                cand=chosen+[i]; xs=np.zeros(len(groups))
                for j in cand: xs[j]=1.0
                rv=val@xs - kap*np.sqrt(xs@Sig_v@xs)
                if best is None or rv>best[0]: best=(rv,i)
            if best is None: break
            chosen.append(best[1]); used+=mg[best[1]]; rem.discard(best[1])
        return set(chosen)
    S_ell_s=greedy_ell_s(scale)
    def pv(S): 
        rv=val@np.array([1.0 if i in S else 0 for i in range(len(groups))]); return 100*(nomval-rv)/nomval
    bp,ep=pv(S_box_s),pv(S_ell_s)
    print(f"{scale:>6}{bp:>12.2f}{ep:>14.2f}{str(bp>=ep-1e-9):>13}")
