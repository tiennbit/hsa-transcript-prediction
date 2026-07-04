# 07_cluster_bootstrap.py — Reviewer gap: CIs that respect the clustered (school) design.
#
# Students are nested in schools; iid (student-level) bootstrap CIs are too narrow
# because rows within a school are correlated. This script computes CLUSTER bootstrap
# CIs (resample SCHOOLS with replacement, rebuild the sample from their rows) and
# compares CI widths to the naive iid (row-level) bootstrap for the same quantities.
#
# Quantities: overall R2, overall MAE, and signed bias (mean(y-pred), + = under-predict)
# for male, female, KV1, and the 3 largest provinces.
# Predictions: attribute-blind (transcript-only, NO gender/KV) 5-fold OOF, seed 42 --
# this is the model the paper's equity audit (Section 4.3, Figure 8) reports biases from,
# so the point estimates here reproduce the manuscript's signed-bias numbers.
# N_BOOT = 500 resamples. Output: analysis/outputs/cluster_bootstrap_cis.csv

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

RNG = 42
N_BOOT = 500
OUT = "analysis/outputs"
os.makedirs(OUT, exist_ok=True)

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}

d = pd.read_csv("analysis/data_deid.csv")
for col in d.columns:
    if "Học lực" in col:
        d[col] = d[col].map(HOC_LUC)
    elif "Hạnh kiểm" in col:
        d[col] = d[col].map(HANH_KIEM)
d["gioi_tinh"] = (d["Giới tính"] == "Nam").astype(int)
d = pd.concat([d, pd.get_dummies(d["khuVuc"], prefix="kv")], axis=1)

transcript_cols = [c for c in d.columns
                   if c[:3] in ("10.", "11.", "12.") and pd.api.types.is_numeric_dtype(d[c])]
# attribute-blind feature set (no gender / no priority zone) -> matches the paper's audit
feat = transcript_cols
san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in feat}

X = d[feat].astype(float).rename(columns=san)
y = d["Điểm HSA"].astype(float).values
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=RNG, n_jobs=-1)

print("Computing full-model 5-fold OOF predictions (seed 42) ...")
oof = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=RNG).split(X):
    oof[te] = XGBRegressor(**XGB_PARAMS).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
err = y - oof  # + = under-prediction
r2_pt = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"  point estimate: R2={r2_pt:.4f}  MAE={np.abs(err).mean():.4f}")

# group membership masks
gender = d["Giới tính"].values
kv = d["khuVuc"].values
prov = d["Tỉnh"].values
top_prov = list(pd.Series(prov).value_counts().head(3).index)
print("  3 largest provinces:", top_prov)

masks = {
    "bias_male": gender == "Nam",
    "bias_female": gender == "Nữ",
    "bias_KV1": kv == "KV1",
}
for p in top_prov:
    masks[f"bias_prov::{p}"] = prov == p


def stats_on(idx):
    """Compute the vector of quantities on a (bootstrap) set of row indices."""
    yy, oo, ee = y[idx], oof[idx], err[idx]
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    out = {"R2": 1 - (ee ** 2).sum() / ss_tot if ss_tot > 0 else np.nan,
           "MAE": np.abs(ee).mean()}
    for name, m in masks.items():
        mm = m[idx]
        out[name] = ee[mm].mean() if mm.any() else np.nan
    return out


quantities = ["R2", "MAE"] + list(masks.keys())
point = stats_on(np.arange(len(y)))

# ---- precompute row indices per school for the cluster bootstrap ----
school = d["Trường"].values
sch_to_rows = {s: np.where(school == s)[0] for s in np.unique(school)}
schools = np.array(list(sch_to_rows.keys()))
n_sch = len(schools)
N = len(y)

rng = np.random.RandomState(RNG)
clus = {q: [] for q in quantities}
iid = {q: [] for q in quantities}
for b in range(N_BOOT):
    # cluster: resample schools with replacement, concatenate their rows
    pick = schools[rng.randint(0, n_sch, n_sch)]
    idx_c = np.concatenate([sch_to_rows[s] for s in pick])
    sc = stats_on(idx_c)
    # iid: resample rows with replacement
    idx_i = rng.randint(0, N, N)
    si = stats_on(idx_i)
    for q in quantities:
        clus[q].append(sc[q])
        iid[q].append(si[q])

rows = []
for q in quantities:
    c = np.array(clus[q]); i = np.array(iid[q])
    c_lo, c_hi = np.nanpercentile(c, [2.5, 97.5])
    i_lo, i_hi = np.nanpercentile(i, [2.5, 97.5])
    rows.append(dict(
        quantity=q, estimate=round(point[q], 4),
        cluster_lo=round(c_lo, 4), cluster_hi=round(c_hi, 4),
        cluster_ci_width=round(c_hi - c_lo, 4),
        iid_lo=round(i_lo, 4), iid_hi=round(i_hi, 4),
        iid_ci_width=round(i_hi - i_lo, 4),
        width_ratio_cluster_over_iid=round((c_hi - c_lo) / (i_hi - i_lo), 3),
    ))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/cluster_bootstrap_cis.csv", index=False)
print(f"\nN_BOOT={N_BOOT}  schools resampled per draw={n_sch}")
print(res.to_string(index=False))
print(f"\nMedian cluster/iid CI-width ratio: {res['width_ratio_cluster_over_iid'].median():.2f}x")
print(f"Wrote {OUT}/cluster_bootstrap_cis.csv")
