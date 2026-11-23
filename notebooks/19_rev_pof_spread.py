# M6: price-of-fairness spread across systems spanning the correlation spectrum.
# Reuses the exact optimisation pipeline of notebook 11 (integrality gap, price of
# robustness, exact epsilon-constraint price of fairness), applied to representative
# systems from strong-negative to positive baseline-sensitivity correlation.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import milp, LinearConstraint, Bounds, linprog

BASE=Path("..") if (Path("../data")).exists() else Path(".")
DATA,TAB=BASE/"data",BASE/"results/tables"
PVS=[f"PV{i}CIL" for i in range(1,6)]
def wpct(v,wt,p):
    m=~np.isnan(v); vs=np.sort(v[m]); ww=wt[m][np.argsort(v[m])]
    return np.interp(p,np.cumsum(ww)/ww.sum(),vs)
LEV=["S_LRNINTO","S_ICTPOSS","S_GENCLASS","S_LRNSAFE"]
CTX=["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
     "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

def run(cc):
    stu,_=pyreadstat.read_sas7bdat(str(DATA/("bsg"+cc+"i3.sas7bdat")))
    sch,_=pyreadstat.read_sas7bdat(str(DATA/("bcg"+cc+"i3.sas7bdat")))
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
    g=d.groupby("IDSCHOOL").apply(lambda z:pd.Series(dict(
        mN=z.TOTWGTS.sum(),delta=np.average(z.dCIL,weights=z.TOTWGTS),n=len(z),
        sd=np.sqrt(np.cov(z.dCIL,aweights=z.TOTWGTS)) if len(z)>1 else 0.0))).reset_index()
    g["se"]=g.sd/np.sqrt(g.n); N=len(g); val=(g.mN*g.delta).to_numpy(); c1=g.mN.to_numpy()
    shh=sch[["IDSCHOOL","C_HINRES"]].copy(); shh.loc[(shh.C_HINRES>=995)|(shh.C_HINRES<=-995),"C_HINRES"]=np.nan
    g=g.merge(shh,on="IDSCHOOL",how="left")
    hz=((g.C_HINRES-np.nanmean(g.C_HINRES))/np.nanstd(g.C_HINRES)).fillna(0)
    c2=(g.mN*(1.0+0.5*np.clip(hz,-1.5,1.5))).to_numpy(); B1=0.33*c1.sum(); B2=0.33*c2.sum()
    def mkp(vals,integer):
        r=milp(c=-vals,constraints=LinearConstraint(np.vstack([c1,c2]),-np.inf,[B1,B2]),
               integrality=(np.ones(N) if integer else np.zeros(N)),bounds=Bounds(0,1),options={"disp":False})
        return -r.fun,r.x
    lp,_=mkp(val,False); ip,_=mkp(val,True); gap=100*(lp-ip)/lp
    dev=(g.mN*g.se).to_numpy(); best=None
    for th in np.unique(np.append(dev,0.0)):
        adj=val-np.maximum(dev-th,0.0); _,xx=mkp(adj,True); xx=xx.round()
        obj=val@xx-np.maximum(dev-th,0.0)@xx-30*th
        if best is None or obj>best[0]: best=(obj,xx)
    por=100*(ip-val@best[1])/ip
    # subgroup-level exact price of fairness
    x=d.NISB.to_numpy(); ok=~np.isnan(x); o=np.argsort(x[ok])
    cw=np.cumsum(W[ok][o])/W[ok].sum(); q=np.interp([1/3,2/3],cw,x[ok][o])
    d["SES3"]=pd.Series(np.where(x<=q[0],"Low",np.where(x<=q[1],"Mid","High"))).where(ok).values
    d["SEX"]=d.S_SEX.map({0.0:"Sex0",1.0:"Sex1"}); d2=d.dropna(subset=["SES3","SEX"])
    gg=d2.groupby(["SES3","SEX"]).apply(lambda z:pd.Series(dict(
        base=np.average(z.CIL5,weights=z.TOTWGTS),dCIL=np.average(z.dCIL,weights=z.TOTWGTS),
        wN=z.TOTWGTS.sum()))).reset_index()
    grp=(gg.SES3+"_"+gg.SEX).tolist(); Dg=gg.dCIL.to_numpy(); mg=gg.wN.to_numpy()
    corr=np.corrcoef(gg.base,gg.dCIL)[0,1]
    hi=np.array([x2.startswith("High") for x2 in grp]); lo=np.array([x2.startswith("Low") for x2 in grp])
    MH=mg[hi].sum(); ML=mg[lo].sum()
    gred=(lo*mg*Dg/ML)-(hi*mg*Dg/MH); eff=mg*Dg; cap=mg; budget=0.33*mg.sum(); nG=len(grp)
    emax=-linprog(c=-gred,A_ub=np.array([cap]),b_ub=[budget],bounds=[(0,1)]*nG,method="highs").fun
    par=lambda e:(lambda r:(-r.fun if r.success else np.nan))(
        linprog(c=-eff,A_ub=np.vstack([cap,-gred]),b_ub=[budget,-e],bounds=[(0,1)]*nG,method="highs"))
    e0=par(0.0); e1=par(max(emax,0)); pof=100*(e0-e1)/e0 if e0 else 0
    return dict(country=cc.upper(),schools=N,corr_base_sens=round(corr,2),
                integrality_gap_pct=round(gap,3),price_of_robustness_pct=round(por,2),
                price_of_fairness_pct=round(pof,2))

# representative spectrum: strong-neg -> moderate-neg -> weak-neg -> positive
systems=["hun","kor","ita","fra","dnk","aut","deu","ury","aze"]
rows=[]
for cc in systems:
    try:
        r=run(cc); rows.append(r); print(r)
    except Exception as e:
        print(cc,"FAILED",str(e)[:60])
R=pd.DataFrame(rows).sort_values("corr_base_sens")
R.to_csv(TAB/"m6_pof_spread.csv",index=False)
print("\n=== price-of-fairness spread ===")
print(R.to_string(index=False))
print("\nwrote m6_pof_spread.csv")
