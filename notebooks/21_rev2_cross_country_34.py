#!/usr/bin/env python3
"""Faithful re-run of the authors' 10_cross_country_full.ipynb logic over the
proper set of 34 participating COUNTRIES (benchmark participant DNW excluded,
Sweden included). Same estimator, levers, 3-fold CV, 5 plausible values.
Writes an authoritative cross_country_34.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadstat, glob, os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from math import comb

from pathlib import Path as _P
DATA = str((_P("..")/"data")) + "/" if (_P("../data")).exists() else "data/"
PVS = [f"PV{i}CIL" for i in range(1, 6)]
BENCHMARK = {"DNW"}  # Denmark Danish-language benchmark participant (IDCNTRY 276001)

def wR2(y, yh, w):
    yb = np.average(y, weights=w); return 1 - np.sum(w*(y-yh)**2)/np.sum(w*(y-yb)**2)
def wpct(v, wt, p):
    m = ~np.isnan(v); vs = np.sort(v[m]); ww = wt[m][np.argsort(v[m])]
    return np.interp(p, np.cumsum(ww)/ww.sum(), vs)

LEV = ["S_LRNINTO", "S_ICTPOSS", "S_GENCLASS", "S_LRNSAFE"]
CTX = ["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF",
       "S_ICTPOSG","S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]
RES = ["C_ICTDEV","C_ICTSTD","C_RATDEV","C_RATSTD","C_RATSMB","C_ICTRES","C_RESTOOL","C_RESUTIL","C_HINRES"]
PART = ["S_EXCOMP","S_ICTSTUD","S_LRNINTS","S_LRNINTO","S_ACMULT","S_GENEFF","S_SPECEFF",
        "S_CTCLS","S_LRNSAFE","S_GENCLASS","S_SPECLASS","S_ICTPOSG","S_ICTFUT","S_ICTPOSS"]

def clean(df):
    for c in set(PART+CTX+["NISB"]):
        if c in df: df.loc[(df[c] >= 995) | (df[c] <= -995), c] = np.nan
    if "NISB" in df: df.loc[(df.NISB < -6) | (df.NISB > 6), "NISB"] = np.nan
    return df

def oos(df, cols):
    cols = [c for c in cols if c in df]
    X = df[cols].to_numpy(float); w = df.TOTWGTS.to_numpy(); g = df.IDSCHOOL.to_numpy()
    gkf = GroupKFold(3); r2 = []
    for pv in PVS:
        y = df[pv].to_numpy(float); pr = np.zeros(len(df))
        for tr, te in gkf.split(X, y, groups=g):
            m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(
                X[tr], y[tr], sample_weight=w[tr]); pr[te] = m.predict(X[te])
        r2.append(wR2(y, pr, w))
    return np.mean(r2)

out = []
errors = []
OUTCSV = "../results/tables/cross_country_34.csv"
for f in sorted(glob.glob(DATA + "bsg???i3.sas7bdat")):
    cc = os.path.basename(f)[3:6].upper()
    if cc in BENCHMARK:
        continue
    fc = f.replace("bsg", "bcg")
    if not os.path.exists(fc):
        continue
    try:
      stu, _ = pyreadstat.read_sas7bdat(f); sch, _ = pyreadstat.read_sas7bdat(fc)
      if "PV1CIL" not in stu.columns:
        continue
      df = stu.merge(sch[["IDCNTRY","IDSCHOOL"] + [c for c in RES if c in sch]],
                   on=["IDCNTRY","IDSCHOOL"], how="left")
      df = clean(df); df = df[df.TOTWGTS.notna() & df.IDSCHOOL.notna()].copy()
      df["CIL5"] = df[PVS].mean(1)
      if len(df) < 500:
        continue
      # migrant pct (weighted) for the appendix table
      if "S_IMMBGR" in df:
        iv = df["S_IMMBGR"].to_numpy(float); wv = df.TOTWGTS.to_numpy()
        mm = ~np.isnan(iv)
        immig = 100*np.average((iv[mm] > 0).astype(float), weights=wv[mm]) if mm.sum() else np.nan
      else:
        immig = np.nan
      w = df.TOTWGTS.to_numpy(); x = df.NISB.to_numpy(); ok = ~np.isnan(x)
      if ok.sum() < 300:
        continue
      o = np.argsort(x[ok]); cw = np.cumsum(w[ok][o])/w[ok].sum(); q = np.interp([1/3, 2/3], cw, x[ok][o])
      df["SES3"] = pd.Series(np.where(x <= q[0], "Low", np.where(x <= q[1], "Mid", "High"))).where(ok).values
      df["SEX"] = df.S_SEX.map({0.0: "Sex0", 1.0: "Sex1"})
      d2 = df.dropna(subset=["SES3", "SEX"]).copy()
      if d2.groupby(["SES3", "SEX"]).ngroups < 6:
        continue
      W = d2.TOTWGTS.to_numpy(); Xb = d2[LEV+CTX].to_numpy(float); Xc = Xb.copy()
      for j, L in enumerate(LEV):
        t = wpct(Xb[:, j], W, 0.75); col = Xc[:, j]
        Xc[:, j] = np.where(np.isnan(col), col, np.maximum(col, t))
      dd = np.zeros((len(d2), 5))
      for k, pv in enumerate(PVS):
        mm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0).fit(Xb, d2[pv].to_numpy(float), sample_weight=W)
        dd[:, k] = mm.predict(Xc) - mm.predict(Xb)
      d2 = d2.assign(dCIL=dd.mean(1))
      gb = d2.groupby(["SES3", "SEX"]).apply(
        lambda z: pd.Series(dict(base=np.average(z.CIL5, weights=z.TOTWGTS),
                                 dCIL=np.average(z.dCIL, weights=z.TOTWGTS)))).reset_index()
      r_res = oos(df, [c for c in RES if c in df] + ["S_INTNET"]); r_par = oos(df, PART)
      corr = np.corrcoef(gb.base, gb.dCIL)[0, 1]
      out.append(dict(cc=cc, N=len(df), immig_pct=round(immig, 1),
                    R2_res=round(r_res, 3), R2_part=round(r_par, 3), corr=round(corr, 2)))
      print(f"  {cc}: N={len(df)} R2_res={r_res:.3f} R2_part={r_par:.3f} corr={corr:.2f}", flush=True)

    except Exception as e:
        errors.append((cc, repr(e)))
        print(f'  [SKIP] {cc}: {type(e).__name__}: {e}', flush=True)
        continue
    # incremental save so a later crash preserves progress
    pd.DataFrame(out).to_csv(OUTCSV, index=False)

C = pd.DataFrame(out).sort_values("cc").reset_index(drop=True)
C.to_csv("../results/tables/cross_country_34.csv", index=False)
n = len(C); neg = int((C["corr"] < 0).sum())
p = sum(comb(n, k) for k in range(neg, n+1))/2**n
print("\n==== SUMMARY (34 countries, DNW excluded) ====")
print(f"systems={n}")
print(f"participation R2 range: {C.R2_part.min():.3f}-{C.R2_part.max():.3f}")
print(f"resource R2<=0.05 in {int((C.R2_res<=0.05).sum())}/{n}")
print(f"negative corr: {neg}/{n}; sign-test p={p:.3e}; median corr={C['corr'].median():.2f}")
print("positive-corr systems:", list(C.loc[C['corr']>=0,'cc']))

if errors:
    print("\n==== SKIPPED (errors) ====")
    for cc, e in errors:
        print(f"  {cc}: {e}")
