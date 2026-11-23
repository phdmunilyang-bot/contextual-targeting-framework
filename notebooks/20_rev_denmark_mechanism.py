# Why does Denmark carry a high price of fairness (20.4%) at only moderate
# baseline-sensitivity correlation (-0.40), while France (corr -0.52) pays 10.7%?
# Inspect the subgroup structure: which groups are both high-sensitivity and
# high-baseline, how large they are, and where the gap-reducing budget must go.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat

BASE=Path("..") if (Path("../data")).exists() else Path(".")
DATA=BASE/"data"
PVS=[f"PV{i}CIL" for i in range(1,6)]
def wpct(v,wt,p):
    m=~np.isnan(v); vs=np.sort(v[m]); ww=wt[m][np.argsort(v[m])]
    return np.interp(p,np.cumsum(ww)/ww.sum(),vs)
from sklearn.ensemble import HistGradientBoostingRegressor
LEV=["S_LRNINTO","S_ICTPOSS","S_GENCLASS","S_LRNSAFE"]
CTX=["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
     "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

def subgroups(cc):
    stu,_=pyreadstat.read_sas7bdat(str(DATA/("bsg"+cc+"i3.sas7bdat")))
    for c in set(LEV+CTX+["NISB"]):
        if c in stu: stu.loc[(stu[c]>=995)|(stu[c]<=-995),c]=np.nan
    stu.loc[(stu.NISB<-6)|(stu.NISB>6),"NISB"]=np.nan; stu["CIL5"]=stu[PVS].mean(1)
    d=stu[stu.TOTWGTS.notna()&stu.IDSCHOOL.notna()].copy(); W=d.TOTWGTS.to_numpy()
    Xb=np.array(d[LEV+CTX].to_numpy(float),copy=True); Xc=Xb.copy()
    for j,L in enumerate(LEV):
        t=wpct(Xb[:,j],W,0.75); col=Xc[:,j]; Xc[:,j]=np.where(np.isnan(col),col,np.maximum(col,t))
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,
        l2_regularization=1.0,random_state=0).fit(Xb,d.PV1CIL.to_numpy(float),sample_weight=W)
    d["dCIL"]=m.predict(Xc)-m.predict(Xb)
    x=d.NISB.to_numpy(); ok=~np.isnan(x); o=np.argsort(x[ok])
    cw=np.cumsum(W[ok][o])/W[ok].sum(); q=np.interp([1/3,2/3],cw,x[ok][o])
    d["SES3"]=pd.Series(np.where(x<=q[0],"Low",np.where(x<=q[1],"Mid","High"))).where(ok).values
    d["SEX"]=d.S_SEX.map({0.0:"Sex0",1.0:"Sex1"}); d=d.dropna(subset=["SES3","SEX"])
    g=d.groupby(["SES3","SEX"]).apply(lambda z:pd.Series(dict(
        base=np.average(z.CIL5,weights=z.TOTWGTS),
        dCIL=np.average(z.dCIL,weights=z.TOTWGTS),
        wN=z.TOTWGTS.sum()))).reset_index()
    g["grp"]=g.SES3+"/"+g.SEX
    g["share"]=100*g.wN/g.wN.sum()
    g["is_high"]=g.SES3.eq("High"); g["is_low"]=g.SES3.eq("Low")
    return g.sort_values("dCIL",ascending=False)

for cc,nm in [("dnk","Denmark"),("fra","France")]:
    g=subgroups(cc)
    print(f"\n===== {nm} ({cc}) =====")
    print(g[["grp","base","dCIL","wN","share"]].round(2).to_string(index=False))
    # key diagnostic: is the most-responsive group an ADVANTAGED (High) group?
    top=g.iloc[0]
    print(f"most-responsive group: {top.grp}  (base={top.base:.0f}, dCIL={top.dCIL:.1f}, share={top.share:.1f}%)")
    # efficiency order puts high-dCIL first; equity wants Low first. Conflict if a High
    # group has high dCIL. Measure: weighted dCIL captured by High groups vs Low groups.
    hi=g[g.is_high]; lo=g[g.is_low]
    print(f"High-SES groups: mean dCIL={np.average(hi.dCIL,weights=hi.wN):.1f}, share={hi.share.sum():.1f}%")
    print(f"Low-SES  groups: mean dCIL={np.average(lo.dCIL,weights=lo.wN):.1f}, share={lo.share.sum():.1f}%")
    # how much efficiency sits in High groups the equity rule must skip:
    eff_hi=(hi.wN*hi.dCIL).sum(); eff_tot=(g.wN*g.dCIL).sum()
    print(f"share of total effect sitting in High-SES groups: {100*eff_hi/eff_tot:.1f}%")
    # baseline gap the equity rule must close
    print(f"High-Low baseline gap: {np.average(hi.base,weights=hi.wN)-np.average(lo.base,weights=lo.wN):.1f} CIL")
