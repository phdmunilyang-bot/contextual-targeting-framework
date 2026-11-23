#!/usr/bin/env python3
"""M7: strengthen the causal-forest comparison by running it across several systems.

The paper compares counterfactual sensitivity vs a causal-forest CATE for Korea only
(single-lever intervention) and finds fine-ranking disagreement (tau=0.33) but coarse
agreement (both rank a low-SES group first; both give a negative baseline-response
relation). A referee can read one-system coarse agreement as a coincidence. We run the
same comparison across several systems and report, per system: whether BOTH estimators
place a low-SES group first, the sign of each estimator's baseline-response
correlation, and the Kendall tau between the two orders. The claim to support is that
the COARSE decision (who is funded first; the sign) is estimation-route-invariant even
though the fine order is not.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor
from econml.grf import CausalForest
from scipy.stats import kendalltau

from pathlib import Path as _P
DATA = str((_P("..")/"data")) + "/" if (_P("../data")).exists() else "data/"
PVS = [f"PV{i}CIL" for i in range(1, 6)]
TREAT = "S_LRNINTO"
OTHER_LEV = ["S_ICTPOSS", "S_GENCLASS", "S_LRNSAFE"]
CTX = ["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
       "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]
SYSTEMS = ["kor", "deu", "ita", "fra", "dnk", "fin", "aut", "prt"]

def wpct(v, wt, p):
    m = ~np.isnan(v); vs = np.sort(v[m]); ww = wt[m][np.argsort(v[m])]
    return np.interp(p, np.cumsum(ww)/ww.sum(), vs)

def run_system(cc):
    stu, _ = pyreadstat.read_sas7bdat(DATA + f"bsg{cc}i3.sas7bdat")
    for c in set([TREAT]+OTHER_LEV+CTX+["NISB"]):
        if c in stu: stu.loc[(stu[c] >= 995) | (stu[c] <= -995), c] = np.nan
    stu.loc[(stu.NISB < -6) | (stu.NISB > 6), "NISB"] = np.nan
    stu["CIL5"] = stu[PVS].mean(1)
    d = stu[stu.TOTWGTS.notna() & stu.IDSCHOOL.notna()].dropna(subset=[TREAT]).copy()
    W = d.TOTWGTS.to_numpy()
    thr = wpct(d[TREAT].to_numpy(), W, 0.50)
    Tb = (d[TREAT].to_numpy() >= thr).astype(float)
    feat = CTX + OTHER_LEV
    Xf = np.array(d[feat].to_numpy(float), copy=True)
    cmn = np.nanmean(Xf, axis=0); inds = np.where(np.isnan(Xf)); Xf[inds] = np.take(cmn, inds[1])
    Y = d.CIL5.to_numpy(float)
    x = d.NISB.to_numpy(); ok = ~np.isnan(x); o = np.argsort(x[ok])
    cw = np.cumsum(W[ok][o])/W[ok].sum(); q = np.interp([1/3, 2/3], cw, x[ok][o])
    d["SES3"] = pd.Series(np.where(x <= q[0], "Low", np.where(x <= q[1], "Mid", "High"))).where(ok).values
    d["SEX"] = d.S_SEX.map({0.0: "Sex0", 1.0: "Sex1"})
    # (a) counterfactual single-lever sensitivity
    Xb = np.array(d[[TREAT]+OTHER_LEV+CTX].to_numpy(float), copy=True)
    cm = np.nanmean(Xb, axis=0); ii = np.where(np.isnan(Xb)); Xb[ii] = np.take(cm, ii[1])
    Xc = Xb.copy(); Xc[:, 0] = np.maximum(Xc[:, 0], thr)
    dd = np.zeros((len(d), 5))
    for k, pv in enumerate(PVS):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0).fit(Xb, d[pv].to_numpy(float), sample_weight=W)
        dd[:, k] = m.predict(Xc) - m.predict(Xb)
    d["cf"] = dd.mean(1)
    # (b) causal forest
    cf = CausalForest(n_estimators=1000, min_samples_leaf=25, random_state=0)
    cf.fit(Xf, Tb, Y, sample_weight=W)
    d["cate"] = cf.predict(Xf).ravel()
    g = d.dropna(subset=["SES3", "SEX"]).groupby(["SES3", "SEX"]).apply(
        lambda z: pd.Series(dict(base=np.average(z.CIL5, weights=z.TOTWGTS),
                                 cf=np.average(z.cf, weights=z.TOTWGTS),
                                 cate=np.average(z.cate, weights=z.TOTWGTS)))).reset_index()
    g["key"] = g.SES3 + "/" + g.SEX
    order_cf = g.sort_values("cf", ascending=False).key.tolist()
    order_cate = g.sort_values("cate", ascending=False).key.tolist()
    tau = kendalltau([order_cf.index(k) for k in g.key], [order_cate.index(k) for k in g.key])[0]
    corr_cf = np.corrcoef(g.base, g.cf)[0, 1]
    corr_cate = np.corrcoef(g.base, g.cate)[0, 1]
    first_cf_low = order_cf[0].startswith("Low")
    first_cate_low = order_cate[0].startswith("Low")
    return dict(sys=cc.upper(), tau=round(tau, 2),
                corr_cf=round(corr_cf, 2), corr_cate=round(corr_cate, 2),
                cf_first=order_cf[0], cate_first=order_cate[0],
                both_low_first=first_cf_low and first_cate_low,
                both_neg=(corr_cf < 0) and (corr_cate < 0))

rows = []
for cc in SYSTEMS:
    try:
        r = run_system(cc); rows.append(r)
        print(f"  {r['sys']}: tau={r['tau']:+.2f}  corr_cf={r['corr_cf']:+.2f} corr_cate={r['corr_cate']:+.2f}  "
              f"cf_first={r['cf_first']:<9} cate_first={r['cate_first']:<9} "
              f"both_low_first={r['both_low_first']} both_neg={r['both_neg']}", flush=True)
    except Exception as e:
        print(f"  [SKIP] {cc}: {e}", flush=True)
R = pd.DataFrame(rows)
R.to_csv("../results/tables/m7_cate_multisystem.csv", index=False)
print()
print(f"systems run: {len(R)}")
print(f"both estimators rank a LOW-SES group first: {int(R.both_low_first.sum())}/{len(R)}")
print(f"both give NEGATIVE baseline-response corr : {int(R.both_neg.sum())}/{len(R)}")
print(f"mean fine-order Kendall tau (cf vs cate)  : {R.tau.mean():.2f}  (range {R.tau.min():.2f}..{R.tau.max():.2f})")
