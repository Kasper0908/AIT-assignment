"""Permutation (placebo) inference for the NOR->CHN salmon sanction effect, preferred spec (Table 1, col 3).
Usage: python permutation.py output/salmon_panel.csv [all|large]
Each placebo assigns SANC (2011-16), POST (2017-19) and ANN (2010) to one other pair and re-estimates
Table 1, column (3); the true NOR->CHN indicators stay in as controls.
  all   = pairs with positive trade in every year 2002-2010 ("stable pairs")
  large = stable pairs in the top quartile of average 2002-2010 trade (NOR->CHN itself is in this group)
Output: table3_permutation_<set>.txt"""
import sys, time, warnings
import numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")
NOR, CHN = 579, 156
FE = "exp_year + imp_year + pair"
p = pd.read_csv(sys.argv[1])
which = sys.argv[2] if len(sys.argv) > 2 else "large"
p["pair"] = p["pair"].astype(str)
true_id = f"{NOR}_{CHN}"
p["ann"] = ((p.pair == true_id) & (p.t == 2010)).astype(int)

pre = p[p.t.between(2002, 2010)]
g = pre.groupby("pair").agg(n=("v", lambda x: (x > 0).sum()), m=("v", "mean"))
stable = g[(g.n == 9) & (g.index != true_id)]
q75 = stable.m.quantile(0.75)
pairs = stable.index.tolist() if which == "all" else stable[stable.m >= q75].index.tolist()
print(f"NOR->CHN pre-period mean trade {g.loc[true_id,'m']:.0f} vs top-quartile cutoff {q75:.0f}")

true_b = pf.fepois(f"v ~ sanc + post_norm + ann | {FE}", data=p, vcov="iid").coef()["sanc"]
b, t0 = [], time.time()
for k in pairs:
    isk = p.pair == k
    d = p.assign(pl_sanc=(isk & p.t.between(2011, 2016)).astype(int),
                 pl_post=(isk & (p.t >= 2017)).astype(int),
                 pl_ann=(isk & (p.t == 2010)).astype(int))
    try:
        b.append(pf.fepois(f"v ~ pl_sanc + pl_post + pl_ann + sanc + post_norm + ann | {FE}",
                           data=d, vcov="iid").coef()["pl_sanc"])
    except Exception:
        pass
b = np.array([x for x in b if np.isfinite(x)])
r = int((b <= true_b).sum())
tab = pd.DataFrame([{"placebo set": which, "n placebos": len(b), "true coef": round(true_b, 3),
                     "placebo mean": round(b.mean(), 3), "placebo median": round(np.median(b), 3),
                     "placebos <= true": r, "p (r+1)/(N+1)": round((r + 1) / (len(b) + 1), 3),
                     "true minus placebo mean": round(true_b - b.mean(), 3)}])
print(f"{len(b)} placebos in {time.time()-t0:.0f}s"); print(tab.to_string(index=False))
open(f"table3_permutation_{which}.txt", "w").write(tab.to_string(index=False))
np.save(f"placebo_coefs_{which}.npy", b)
