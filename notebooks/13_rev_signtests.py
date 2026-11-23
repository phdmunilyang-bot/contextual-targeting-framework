# M5: conservative inference on the cross-national baseline-sensitivity sign.
import numpy as np, pandas as pd
from math import comb
from scipy.stats import binomtest, wilcoxon

C=pd.read_csv("../results/tables/cross_country_35.csv")
n=len(C); corr=C["corr"].to_numpy()
neg=int((corr<0).sum()); pos=int((corr>0).sum())
print(f"n={n} systems | negative corr: {neg} | positive: {pos} | median corr={np.median(corr):.3f}\n")

# (1) Naive sign test (what the paper currently reports): H0 p=0.5
p_naive=sum(comb(n,k) for k in range(neg,n+1))/2**n
print(f"(1) Naive binomial sign test (H0: P[neg]=0.5):  p = {p_naive:.3e}")

# (2) Resolved-only: drop weak |corr|<0.5 as uninformative, test the resolved subset.
strong=C[np.abs(C["corr"])>=0.5]
ns=len(strong); negs=int((strong["corr"]<0).sum())
bt=binomtest(negs, ns, 0.5, alternative="greater")
print(f"(2) Resolved systems only (|corr|>=0.5): {negs}/{ns} negative, p = {bt.pvalue:.3e}")

# (3) Design-effect discount on effective sample size (shared ICILS instrument -> dependence)
print("(3) Design-effect discount on effective sample size:")
deff_rows=[]
for deff in (1.5, 2.0, 3.0):
    neff=int(round(n/deff)); negeff=int(round(neg/deff))
    peff=sum(comb(neff,k) for k in range(negeff,neff+1))/2**neff
    print(f"     DEFF={deff}: n_eff={neff}, neg_eff={negeff}  ->  p = {peff:.3e}")
    deff_rows.append((f"deff_{deff}", f"n_eff={neff},neg={negeff}", peff))

# (4) Adversarial breakdown point at alpha=0.05
k=neg
while k>0:
    p=sum(comb(n,j) for j in range(k,n+1))/2**n
    if p>0.05: break
    k-=1
thresh=k+1
print(f"(4) Breakdown point: significant (alpha=.05) down to {thresh}/{n} negative "
      f"(up to {neg-thresh} systems could flip and it stays significant).")

# (5) Wilcoxon signed-rank that the median correlation < 0 (uses magnitudes)
w=wilcoxon(corr, alternative="less")
print(f"(5) Wilcoxon signed-rank (H0 median>=0 vs <0): stat={w.statistic:.1f}, p = {w.pvalue:.3e}")

rows=[("naive_sign",f"{neg}/{n}",p_naive),
      ("resolved_only_|r|>=.5",f"{negs}/{ns}",bt.pvalue)]+deff_rows+[
      ("wilcoxon_median<0",f"median={np.median(corr):.2f}",w.pvalue)]
pd.DataFrame(rows,columns=["test","detail","p_value"]).to_csv(
    "../results/tables/m5_conservative_signtests.csv",index=False)
print("\nwrote m5_conservative_signtests.csv")
