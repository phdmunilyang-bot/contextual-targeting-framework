# R2: joint-bootstrap reordering probability (fast version matching notebook 06).
# Model is fit once (as in notebook 06's Module-2 bootstrap); the school-clustered
# bootstrap resamples the predicted dCIL and recomputes the subgroup aggregates and
# the greedy order, counting the fraction of replicates whose order differs from the
# point-estimate order.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import kendalltau

BASE=Path("..") if (Path("../data")).exists() else Path(".")
DATA,TAB=BASE/"data",BASE/"results/tables"
PVS=[f"PV{i}CIL" for i in range(1,6)]
def wpct(v,wt,p):
    m=~np.isnan(v); vs=np.sort(v[m]); ww=wt[m][np.argsort(v[m])]
    return np.interp(p,np.cumsum(ww)/ww.sum(),vs)
LEV=["S_LRNINTO","S_ICTPOSS","S_GENCLASS","S_LRNSAFE"]
CTX=["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
     "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

stu,_=pyreadstat.read_sas7bdat(str(DATA/"bsgkori3.sas7bdat"))
for c in set(LEV+CTX+["NISB"]):
    if c in stu: stu.loc[(stu[c]>=995)|(stu[c]<=-995),c]=np.nan
stu.loc[(stu.NISB<-6)|(stu.NISB>6),"NISB"]=np.nan; stu["CIL5"]=stu[PVS].mean(1)
d=stu[stu.TOTWGTS.notna()&stu.IDSCHOOL.notna()].copy()
x=d.NISB.to_numpy(); ok=~np.isnan(x); W=d.TOTWGTS.to_numpy(); o=np.argsort(x[ok])
cw=np.cumsum(W[ok][o])/W[ok].sum(); q=np.interp([1/3,2/3],cw,x[ok][o])
d["SES3"]=pd.Series(np.where(x<=q[0],"Low",np.where(x<=q[1],"Mid","High"))).where(ok).values
d["SEX"]=d.S_SEX.map({0.0:"Sex0",1.0:"Sex1"}); d=d.dropna(subset=["SES3","SEX"]).copy()
d["grp"]=d.SES3+"/"+d.SEX

# fit once, predict dCIL (5-PV averaged) — identical to notebook 06
Wd=d.TOTWGTS.to_numpy()
Xb=np.array(d[LEV+CTX].to_numpy(float),copy=True); Xc=Xb.copy()
for j,L in enumerate(LEV):
    t=wpct(Xb[:,j],Wd,0.75); col=Xc[:,j]; Xc[:,j]=np.where(np.isnan(col),col,np.maximum(col,t))
dd=np.zeros((len(d),5))
for k,pv in enumerate(PVS):
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,
        l2_regularization=1.0,random_state=0).fit(Xb,d[pv].to_numpy(float),sample_weight=Wd)
    dd[:,k]=m.predict(Xc)-m.predict(Xb)
d["dCIL"]=dd.mean(1)

groups=sorted(d.grp.unique())
# precompute per-school arrays for fast resampling
schools=d.IDSCHOOL.unique()
sidx={s:d.index.get_indexer(d[d.IDSCHOOL==s].index) for s in schools}
dd_arr=d.dCIL.to_numpy(); w_arr=d.TOTWGTS.to_numpy(); grp_arr=d.grp.to_numpy()

def order_from(rows):
    sub_d=dd_arr[rows]; sub_w=w_arr[rows]; sub_g=grp_arr[rows]
    vals={}
    for g in groups:
        mask=sub_g==g
        if mask.sum()==0: return None
        vals[g]=np.average(sub_d[mask],weights=sub_w[mask])
    return sorted(groups,key=lambda g:-vals[g])

point_order=order_from(np.arange(len(d)))
print("point order:", " > ".join(point_order))

rng=np.random.default_rng(0); B=400; changed=0; tau_sum=0.0; low=0; top=0; valid=0
for b in range(B):
    pick=rng.choice(schools,len(schools),replace=True)
    rows=np.concatenate([sidx[s] for s in pick])
    bo=order_from(rows)
    if bo is None: continue
    valid+=1
    if bo!=point_order: changed+=1
    tau,_=kendalltau(range(len(point_order)),[bo.index(k) for k in point_order]); tau_sum+=tau
    if bo[-1]==point_order[-1]: low+=1
    if bo[0]==point_order[0]: top+=1

print(f"\nvalid replicates: {valid}")
print(f"reordering probability (full order changes): {100*changed/valid:.1f}%")
print(f"mean Kendall tau to point order: {tau_sum/valid:.3f}")
print(f"lowest-priority group unchanged: {100*low/valid:.1f}% of replicates")
print(f"top-priority group unchanged:    {100*top/valid:.1f}% of replicates")
pd.DataFrame({"metric":["reorder_prob_pct","mean_kendall_tau","lowest_stable_pct","top_stable_pct","B"],
             "value":[round(100*changed/valid,1),round(tau_sum/valid,3),
                      round(100*low/valid,1),round(100*top/valid,1),valid]}).to_csv(
    TAB/"r2_reorder_prob.csv",index=False)
print("\nwrote r2_reorder_prob.csv")
