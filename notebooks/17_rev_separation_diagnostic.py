# M1: an a-priori diagnostic for when the simple greedy order is safe,
# i.e. when the robust apparatus is NOT needed. The diagnostic is the ratio of
# estimation noise to the spread of the group sensitivities: if adjacent groups'
# sensitivities are separated by many standard errors, no plausible uncertainty
# realisation reorders them, so the point-estimate greedy is already robust.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor

BASE=Path("..") if (Path("../data")).exists() else Path(".")
DATA,TAB=BASE/"data",BASE/"results/tables"
PVS=[f"PV{i}CIL" for i in range(1,6)]
def wpct(v,wt,p):
    m=~np.isnan(v); vs=np.sort(v[m]); ww=wt[m][np.argsort(v[m])]
    return np.interp(p,np.cumsum(ww)/ww.sum(),vs)
LEV=["S_LRNINTO","S_ICTPOSS","S_GENCLASS","S_LRNSAFE"]
CTX=["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
     "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

def subgroup_delta(cc, n_boot=200, seed=0):
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
    d=d.assign(dCIL=m.predict(Xc)-m.predict(Xb))
    x=d.NISB.to_numpy(); ok=~np.isnan(x); o=np.argsort(x[ok])
    cw=np.cumsum(W[ok][o])/W[ok].sum(); q=np.interp([1/3,2/3],cw,x[ok][o])
    d["SES3"]=pd.Series(np.where(x<=q[0],"Low",np.where(x<=q[1],"Mid","High"))).where(ok).values
    d["SEX"]=d.S_SEX.map({0.0:"Sex0",1.0:"Sex1"}); d=d.dropna(subset=["SES3","SEX"])
    d["grp"]=d.SES3+"/"+d.SEX
    pt=d.groupby("grp").apply(lambda z:np.average(z.dCIL,weights=z.TOTWGTS))
    # cluster (school) bootstrap SE of each subgroup Delta
    rng=np.random.default_rng(seed); schools=d.IDSCHOOL.unique()
    be=[]
    for _ in range(n_boot):
        samp=rng.choice(schools,size=len(schools),replace=True)
        bd=pd.concat([d[d.IDSCHOOL==s] for s in samp])
        be.append(bd.groupby("grp").apply(lambda z:np.average(z.dCIL,weights=z.TOTWGTS)))
    B=pd.DataFrame(be); se=B.std()
    g=pd.DataFrame({"delta":pt,"se":se}).sort_values("delta",ascending=False)
    # separation diagnostic: adjacent gap in units of pooled SE
    gaps=np.diff(g.delta.to_numpy())*-1
    pooled=np.sqrt(g.se.to_numpy()[:-1]**2+g.se.to_numpy()[1:]**2)
    sep=gaps/pooled
    return g, sep

for cc in ["kor"]:
    g,sep=subgroup_delta(cc)
    print(f"=== {cc.upper()} subgroup sensitivities with cluster-bootstrap SE ===")
    print(g.round(3).to_string())
    print("\nadjacent separation (gap / pooled SE):", np.round(sep,2))
    print(f"min adjacent separation = {sep.min():.2f} SE")
    print("DIAGNOSTIC: if min adjacent separation >~ 1 SE, the point-estimate greedy order")
    print("is already robust and the Gamma-apparatus is not needed; below that, use the robust rule.")
    out=g.reset_index(); out["rank"]=range(1,len(out)+1)
    out.to_csv(TAB/"m1_diagnostic_kor.csv",index=False)
    pd.DataFrame({"adjacent_pair":[f"{i+1}->{i+2}" for i in range(len(sep))],
                  "separation_SE":np.round(sep,3)}).to_csv(TAB/"m1_separation_kor.csv",index=False)
    print("\nwrote m1_diagnostic_kor.csv and m1_separation_kor.csv")
