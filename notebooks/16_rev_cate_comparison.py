# M4: does the counterfactual-explanation route to Delta_g change the decision,
# versus a causal-forest CATE plugged into the SAME subgroup knapsack pipeline?
# Treatment: raising out-of-school ICT learning (S_LRNINTO) to >= national median.
# We compare (a) counterfactual sensitivity Delta_g^cf, (b) causal-forest CATE_g,
# both aggregated to the 6 SES x sex subgroups, then the targeting orders they induce.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor
from econml.grf import CausalForest
from scipy.stats import kendalltau

BASE=Path("..") if (Path("../data")).exists() else Path(".")
DATA,TAB=BASE/"data",BASE/"results/tables"
PVS=[f"PV{i}CIL" for i in range(1,6)]
def wpct(v,wt,p):
    m=~np.isnan(v); vs=np.sort(v[m]); ww=wt[m][np.argsort(v[m])]
    return np.interp(p,np.cumsum(ww)/ww.sum(),vs)

TREAT="S_LRNINTO"                       # the lever treated as a binary intervention
OTHER_LEV=["S_ICTPOSS","S_GENCLASS","S_LRNSAFE"]
CTX=["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
     "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

stu,_=pyreadstat.read_sas7bdat(str(DATA/"bsgkori3.sas7bdat"))
for c in set([TREAT]+OTHER_LEV+CTX+["NISB"]):
    if c in stu: stu.loc[(stu[c]>=995)|(stu[c]<=-995),c]=np.nan
stu.loc[(stu.NISB<-6)|(stu.NISB>6),"NISB"]=np.nan; stu["CIL5"]=stu[PVS].mean(1)
d=stu[stu.TOTWGTS.notna()&stu.IDSCHOOL.notna()].copy()
# confounders X = context + the other levers (held fixed); drop rows missing treatment
feat=CTX+OTHER_LEV
d=d.dropna(subset=[TREAT]).copy()
W=d.TOTWGTS.to_numpy()
# binary treatment: at/above national weighted median of the lever
thr=wpct(d[TREAT].to_numpy(),W,0.50)
Tb=(d[TREAT].to_numpy()>=thr).astype(float)
Xf=np.array(d[feat].to_numpy(float),copy=True)
# impute confounders by weighted mean (CausalForest needs complete X)
col_means=np.nanmean(Xf,axis=0); inds=np.where(np.isnan(Xf)); Xf[inds]=np.take(col_means,inds[1])
Y=d.CIL5.to_numpy(float)

# subgroups
x=d.NISB.to_numpy(); ok=~np.isnan(x); o=np.argsort(x[ok])
cw=np.cumsum(W[ok][o])/W[ok].sum(); q=np.interp([1/3,2/3],cw,x[ok][o])
d["SES3"]=pd.Series(np.where(x<=q[0],"Low",np.where(x<=q[1],"Mid","High"))).where(ok).values
d["SEX"]=d.S_SEX.map({0.0:"Sex0",1.0:"Sex1"})

# ---- (a) counterfactual sensitivity for the SAME single-lever intervention ----
Xb=np.array(d[[TREAT]+OTHER_LEV+CTX].to_numpy(float),copy=True)
cm=np.nanmean(Xb,axis=0); ii=np.where(np.isnan(Xb)); Xb[ii]=np.take(cm,ii[1])
Xc=Xb.copy(); Xc[:,0]=np.maximum(Xc[:,0],thr)      # raise only the treated lever
dd=np.zeros((len(d),5))
for k,pv in enumerate(PVS):
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,
        l2_regularization=1.0,random_state=0).fit(Xb,d[pv].to_numpy(float),sample_weight=W)
    dd[:,k]=m.predict(Xc)-m.predict(Xb)
d["cf"]=dd.mean(1)

# ---- (b) causal-forest CATE for the binary treatment ----
cf=CausalForest(n_estimators=2000,min_samples_leaf=25,max_depth=None,random_state=0)
cf.fit(Xf,Tb,Y,sample_weight=W)
d["cate"]=cf.predict(Xf).ravel()

# aggregate both to subgroups
g=d.dropna(subset=["SES3","SEX"]).groupby(["SES3","SEX"]).apply(
    lambda z:pd.Series(dict(base=np.average(z.CIL5,weights=z.TOTWGTS),
                            cf=np.average(z.cf,weights=z.TOTWGTS),
                            cate=np.average(z.cate,weights=z.TOTWGTS),
                            wN=z.TOTWGTS.sum()))).reset_index()
g["key"]=g.SES3+"/"+g.SEX
# equal unit cost -> order by sensitivity
ocf  =g.sort_values("cf",  ascending=False).key.tolist()
ocate=g.sort_values("cate",ascending=False).key.tolist()
rcf ={k:i for i,k in enumerate(ocf)}; rcate={k:i for i,k in enumerate(ocate)}
keys=sorted(rcf); tau,_=kendalltau([rcf[k] for k in keys],[rcate[k] for k in keys])

print(g[["SES3","SEX","base","cf","cate"]].round(2).to_string(index=False))
print("\ncorr(base, counterfactual) =", round(np.corrcoef(g.base,g.cf)[0,1],3))
print("corr(base, causal-forest CATE) =", round(np.corrcoef(g.base,g.cate)[0,1],3))
print("corr(counterfactual, CATE) across subgroups =", round(np.corrcoef(g.cf,g.cate)[0,1],3))
print("\ncounterfactual order:", " > ".join(ocf))
print("CATE order         :", " > ".join(ocate))
print("Kendall tau (CATE order vs counterfactual order) =", round(tau,3))
print("lowest-priority  cf:",ocf[-1]," | cate:",ocate[-1])
print("highest-priority cf:",ocf[0], " | cate:",ocate[0])

g[["SES3","SEX","base","cf","cate","wN"]].round(4).to_csv(TAB/"m4_cate_vs_cf.csv",index=False)
pd.DataFrame({"metric":["corr_base_cf","corr_base_cate","corr_cf_cate","kendall_tau_orders"],
             "value":[round(np.corrcoef(g.base,g.cf)[0,1],3),
                      round(np.corrcoef(g.base,g.cate)[0,1],3),
                      round(np.corrcoef(g.cf,g.cate)[0,1],3),
                      round(tau,3)]}).to_csv(TAB/"m4_summary.csv",index=False)
print("\nwrote m4_cate_vs_cf.csv and m4_summary.csv")
