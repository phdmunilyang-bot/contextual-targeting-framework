# M5: finer subgroup partition (SES x sex x migration) where migrant n permits.
# Korea cannot support it (<1% migrant); we use high-migrant systems and recompute
# the baseline-sensitivity correlation on more than six cells.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat
from sklearn.ensemble import HistGradientBoostingRegressor
import sys

BASE=Path("..") if (Path("../data")).exists() else Path(".")
DATA,TAB=BASE/"data",BASE/"results/tables"
PVS=[f"PV{i}CIL" for i in range(1,6)]
def wpct(v,wt,p):
    m=~np.isnan(v); vs=np.sort(v[m]); ww=wt[m][np.argsort(v[m])]
    return np.interp(p,np.cumsum(ww)/ww.sum(),vs)
LEV=["S_LRNINTO","S_ICTPOSS","S_GENCLASS","S_LRNSAFE"]
CTX=["NISB","S_HISEI","S_HOMLIT","S_INTNET","S_EXCOMP","S_GENEFF","S_SPECEFF","S_ICTPOSG",
     "S_ICTFUT","S_ACMULT","S_LRNINTS","S_CTCLS","S_SPECLASS","S_ICTSTUD"]

def analyse(cc):
    stu,_=pyreadstat.read_sas7bdat(str(DATA/("bsg"+cc+"i3.sas7bdat")))
    for c in set(LEV+CTX+["NISB","S_IMMBGR"]):
        if c in stu: stu.loc[(stu[c]>=995)|(stu[c]<=-995),c]=np.nan
    stu.loc[(stu.NISB<-6)|(stu.NISB>6),"NISB"]=np.nan; stu["CIL5"]=stu[PVS].mean(1)
    d=stu[stu.TOTWGTS.notna()&stu.IDSCHOOL.notna()].copy()
    migshare=np.average((d.S_IMMBGR==1).astype(float),weights=d.TOTWGTS) if "S_IMMBGR" in d else np.nan
    W=d.TOTWGTS.to_numpy(); Xb=d[LEV+CTX].to_numpy(float); Xc=Xb.copy()
    for j,L in enumerate(LEV):
        t=wpct(Xb[:,j],W,0.75); col=Xc[:,j]; Xc[:,j]=np.where(np.isnan(col),col,np.maximum(col,t))
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,
        l2_regularization=1.0,random_state=0).fit(Xb,d.PV1CIL.to_numpy(float),sample_weight=W)
    d["dCIL"]=m.predict(Xc)-m.predict(Xb)
    x=d.NISB.to_numpy(); ok=~np.isnan(x); o=np.argsort(x[ok])
    cw=np.cumsum(W[ok][o])/W[ok].sum(); q=np.interp([1/3,2/3],cw,x[ok][o])
    d["SES3"]=pd.Series(np.where(x<=q[0],"Low",np.where(x<=q[1],"Mid","High"))).where(ok).values
    d["SEX"]=d.S_SEX.map({0.0:"Sex0",1.0:"Sex1"})
    d["MIG"]=d.S_IMMBGR.map({0.0:"Native",1.0:"Migrant"})
    def corr_on(cells):
        g=d.dropna(subset=cells).groupby(cells).apply(
            lambda z:pd.Series(dict(base=np.average(z.CIL5,weights=z.TOTWGTS),
                                    dCIL=np.average(z.dCIL,weights=z.TOTWGTS),
                                    n=len(z)))).reset_index()
        g=g[g.n>=80]                       # keep only reasonably-sized cells
        if len(g)<4: return None
        return len(g), np.corrcoef(g.base,g.dCIL)[0,1], int(g.n.min())
    six =corr_on(["SES3","SEX"])
    fine=corr_on(["SES3","SEX","MIG"])
    return dict(cc=cc.upper(), migrant_share=round(float(migshare),3),
                six_cells=(six[0] if six else None),  six_corr=(round(six[1],3) if six else None),
                fine_cells=(fine[0] if fine else None), fine_corr=(round(fine[1],3) if fine else None),
                fine_min_cell_n=(fine[2] if fine else None))

systems=["deu","fra","aut","swe","nld","lux","nor","dnk","usa","ita","kor"]
rows=[]
for cc in systems:
    try: rows.append(analyse(cc))
    except Exception as e: rows.append(dict(cc=cc.upper(),migrant_share=None,note=str(e)[:40]))
R=pd.DataFrame(rows)
R.to_csv(TAB/"m5_finer_partition.csv",index=False)
pd.set_option("display.width",160)
print(R.to_string(index=False))
print("\nwrote m5_finer_partition.csv")
