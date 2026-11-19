# M7: lever-split robustness.
# Re-estimate Module-2 sensitivities and the targeting order using ONLY the
# directly-deliverable levers, and compare to the full-lever order.
# Mirrors 02_counterfactual_sensitivity.ipynb exactly except for the lever set.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import kendalltau

BASE = Path("..") if (Path("../data")).exists() else Path(".")
DATA, TAB = BASE/"data", BASE/"results/tables"
PVS=[f"PV{i}CIL" for i in range(1,6)]

stu,_=pyreadstat.read_sas7bdat(str(DATA/"bsgkori3.sas7bdat"))
USED=["NISB","S_HOMLIT","S_INTNET","S_HISEI","S_EXCOMP","S_ICTSTUD","S_LRNINTS","S_LRNINTO",
      "S_ACMULT","S_GENEFF","S_SPECEFF","S_CTCLS","S_LRNSAFE","S_GENCLASS","S_SPECLASS","S_ICTPOSG","S_ICTFUT","S_ICTPOSS"]
for c in USED: stu.loc[(stu[c]>=995)|(stu[c]<=-995),c]=np.nan
stu.loc[(stu.NISB<-6)|(stu.NISB>6),"NISB"]=np.nan
w=stu.TOTWGTS.to_numpy(); x=stu.NISB.to_numpy(); ok=~np.isnan(x)
o=np.argsort(x[ok]); cw=np.cumsum(w[ok][o])/w[ok].sum(); q=np.interp([1/3,2/3],cw,x[ok][o])
stu["SES3"]=pd.Series(np.where(x<=q[0],"Low",np.where(x<=q[1],"Mid","High"))).where(ok).values
stu["SEX"]=stu.S_SEX.map({0.0:"Sex0",1.0:"Sex1"})
df=stu.dropna(subset=["SES3","SEX"]).copy()

# Full lever set = the CANONICAL set used in the manuscript (notebook 06: 4 levers,
# S_EXCOMP is a control, not a lever). The directly-deliverable subset drops the one
# attitudinal lever (S_ICTPOSS, perceptions of learning with ICT).
LEVERS_FULL=["S_LRNINTO","S_ICTPOSS","S_GENCLASS","S_LRNSAFE"]     # canonical manuscript levers
LEVERS_DELIV=["S_LRNINTO","S_GENCLASS","S_LRNSAFE"]               # directly deliverable only
CTX_BASE=["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
     "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

def sensitivities(levers):
    # attitudinal levers that are NOT in the active lever set move to controls
    ctx = CTX_BASE + [l for l in LEVERS_FULL if l not in levers]
    W=df.TOTWGTS.to_numpy(); Xb=df[levers+ctx].to_numpy(float); Xc=Xb.copy()
    for j,L in enumerate(levers):
        v=df[L].to_numpy(); mm=~np.isnan(v); vs=np.sort(v[mm]); ww=W[mm][np.argsort(v[mm])]
        tgt=np.interp(0.75,np.cumsum(ww)/ww.sum(),vs); col=Xc[:,j]
        Xc[:,j]=np.where(np.isnan(col),col,np.maximum(col,tgt))
    dd=np.zeros((len(df),5))
    for k,pv in enumerate(PVS):
        m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,
            l2_regularization=1.0,random_state=0).fit(Xb,df[pv].to_numpy(float),sample_weight=W)
        dd[:,k]=m.predict(Xc)-m.predict(Xb)
    d=dd.mean(1)
    rows=[]
    for (s,x2),g in df.assign(dCIL=d).groupby(["SES3","SEX"]):
        ww=g.TOTWGTS.to_numpy()
        rows.append(dict(SES=s,SEX=x2,n=len(g),wN=ww.sum(),
                         base_CIL=np.average(g.PV1CIL,weights=ww),dCIL=np.average(g.dCIL,weights=ww)))
    return pd.DataFrame(rows).sort_values("dCIL",ascending=False).reset_index(drop=True)

Rfull  = sensitivities(LEVERS_FULL)
Rdeliv = sensitivities(LEVERS_DELIV)

# align by subgroup key for rank comparison
key=lambda r: r.SES+"/"+r.SEX
Rfull["key"]=Rfull.apply(key,axis=1); Rdeliv["key"]=Rdeliv.apply(key,axis=1)
order_full = list(Rfull.key)
order_deliv= list(Rdeliv.key)
rank_full = {k:i for i,k in enumerate(order_full)}
rank_deliv= {k:i for i,k in enumerate(order_deliv)}
keys=sorted(rank_full)
tau,_=kendalltau([rank_full[k] for k in keys],[rank_deliv[k] for k in keys])

corr_full = np.corrcoef(Rfull.base_CIL,Rfull.dCIL)[0,1]
corr_deliv= np.corrcoef(Rdeliv.base_CIL,Rdeliv.dCIL)[0,1]

print("=== FULL lever set (EXCOMP, LRNINTO, ICTPOSS, GENCLASS, LRNSAFE) ===")
print(Rfull[["SES","SEX","base_CIL","dCIL"]].round(2).to_string(index=False))
print("order:", " > ".join(order_full))
print("corr(base,sens) =", round(corr_full,3))
print()
print("=== DELIVERABLE-ONLY lever set (LRNINTO, GENCLASS, LRNSAFE) ===")
print(Rdeliv[["SES","SEX","base_CIL","dCIL"]].round(2).to_string(index=False))
print("order:", " > ".join(order_deliv))
print("corr(base,sens) =", round(corr_deliv,3))
print()
print("Kendall tau (deliverable-only order vs full order) =", round(tau,3))
print("lowest-priority group  full:", order_full[-1], "| deliverable:", order_deliv[-1])
print("highest-priority group full:", order_full[0],  "| deliverable:", order_deliv[0])

out=pd.DataFrame({
  "lever_set":["full","deliverable_only"],
  "levers":[";".join(LEVERS_FULL),";".join(LEVERS_DELIV)],
  "order":[" > ".join(order_full)," > ".join(order_deliv)],
  "corr_base_sens":[round(corr_full,3),round(corr_deliv,3)],
  "kendall_tau_vs_full":[1.0,round(tau,3)],
})
out.to_csv(TAB/"m7_lever_split.csv",index=False)
Rdeliv[["SES","SEX","n","base_CIL","dCIL"]].round(3).to_csv(TAB/"m7_deliverable_sensitivity.csv",index=False)
print("\nwrote m7_lever_split.csv and m7_deliverable_sensitivity.csv")
