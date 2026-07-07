# 15_school_type.py — Professor round-2 question: can school type be inferred?
# ANSWER: partially, from school-name strings (both Unicode forms normalized):
#   - "Chuyên"/"Năng khiếu"  -> specialized (trường chuyên)
#   - "Nội trú"/"DTNT"/"Dân tộc" -> ethnic-minority boarding
#   - "Tư thục"/"Dân lập"/"Quốc tế" -> private/international (partial: many private
#     schools carry no marker, so this class is a LOWER BOUND)
#   - everything else -> mainstream (default; includes unmarked private schools)
# Computes, per type: n schools / n students, mean OOF signed bias (attribute-blind
# model) with school-cluster bootstrap CIs, and the share of between-school-mean
# residual variance explained by the type labels.
# Outputs: analysis/outputs/school_type_analysis.csv (+ console summary)

import os
import re
import unicodedata

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SEED = 42
N_BOOT = 500

def nfc_lower(s):
    return unicodedata.normalize("NFC", str(s)).lower()

d = pd.read_csv(os.path.join(HERE, "data_deid.csv"))
pred = np.load(os.path.join(OUT, "rev_oof_noprotected.npy"))
d["_resid"] = d["Điểm HSA"].to_numpy() - pred
d["_school"] = d["Tỉnh"].astype(str) + "||" + d["Trường"].astype(str)
d["_nm"] = d["Trường"].map(nfc_lower)

PAT = {
    "specialized": nfc_lower("chuyên|năng khiếu"),
    "ethnic_boarding": nfc_lower("nội trú|dtnt|dân tộc"),
    "private": nfc_lower("tư thục|dân lập|quốc tế"),
}
def classify(nm):
    for label, p in PAT.items():
        if re.search(p, nm):
            return label
    return "mainstream"
d["_stype"] = d["_nm"].map(classify)

print("=== Classification (from school-name strings) ===")
sch = d.groupby("_school").agg(n=("_resid", "size"), stype=("_stype", "first"),
                               resid_mean=("_resid", "mean"))
for t in ["specialized", "ethnic_boarding", "private", "mainstream"]:
    ns = int((sch["stype"] == t).sum())
    nst = int(d.loc[d["_stype"] == t].shape[0])
    print(f"  {t:16s}: {ns:5d} schools / {nst:6d} students")

# signed bias per type + school-cluster bootstrap CIs
rng = np.random.default_rng(SEED)
schools = d["_school"].unique()
stats = {}
for t in PAT.keys() | {"mainstream"}:
    sub = d[d["_stype"] == t]
    grp = sub.groupby("_school")["_resid"].agg(["sum", "count"])
    stats[t] = (grp["sum"].to_dict(), grp["count"].to_dict())
boot = {t: [] for t in stats}
for _ in range(N_BOOT):
    draw = rng.choice(schools, size=len(schools), replace=True)
    for t, (sums, counts) in stats.items():
        tot = cnt = 0
        for s in draw:
            c = counts.get(s)
            if c:
                tot += sums[s]
                cnt += c
        if cnt:
            boot[t].append(tot / cnt)

rows = []
print("\n=== Signed bias (observed - predicted) by school type, cluster 95% CI ===")
for t in ["specialized", "ethnic_boarding", "private", "mainstream"]:
    est = d.loc[d["_stype"] == t, "_resid"].mean()
    bs = np.array(boot[t])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    mae = d.loc[d["_stype"] == t, "_resid"].abs().mean()
    rows.append(dict(school_type=t, n_schools=int((sch["stype"] == t).sum()),
                     n_students=int((d["_stype"] == t).sum()),
                     MAE=round(mae, 3), signed_bias=round(est, 3),
                     ci95_lo=round(lo, 3), ci95_hi=round(hi, 3),
                     sig95=bool(lo > 0 or hi < 0)))
    print(f"  {t:16s}: bias {est:+.3f} [{lo:+.3f}, {hi:+.3f}]  MAE {mae:.3f}")

# share of between-school-mean residual variance explained by type (schools n>=30,
# matching the paper's decomposition unit)
big = sch[sch["n"] >= 30].copy()
w = big["n"].to_numpy(float)
y = big["resid_mean"].to_numpy(float)
mu = np.average(y, weights=w)
ss_tot = np.average((y - mu) ** 2, weights=w)
grand = {t: np.average(big.loc[big["stype"] == t, "resid_mean"],
                       weights=big.loc[big["stype"] == t, "n"])
         for t in big["stype"].unique()}
fitted = big["stype"].map(grand).to_numpy(float)
ss_res = np.average((y - fitted) ** 2, weights=w)
r2_type = 1 - ss_res / ss_tot
print(f"\nBetween-school-mean residual variance explained by school-type labels "
      f"(schools n>=30, weighted): R² = {r2_type:.3f}")
rows.append(dict(school_type="R2_type_between_school", n_schools=len(big),
                 n_students=int(big['n'].sum()), MAE=None,
                 signed_bias=round(r2_type, 4), ci95_lo=None, ci95_hi=None, sig95=None))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "school_type_analysis.csv"), index=False)
print("\nWritten: outputs/school_type_analysis.csv")
