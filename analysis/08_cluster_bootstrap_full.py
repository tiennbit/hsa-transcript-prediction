# 08_cluster_bootstrap_full.py — Extends 07: school-cluster bootstrap CIs for ALL
# audited groups (gender x2, priority zones x4, and every province with n>=200),
# plus the Bonferroni-significance recount for the province audit under
# cluster-robust inference (replaces the manuscript's iid-based "14 of 22" tally).
#
# Uses the saved attribute-blind OOF predictions (rev_oof_noprotected.npy) so no
# model refit is needed; resamples SCHOOLS (province-school pairs) with replacement,
# 500 draws, seed 42.
#
# Outputs: analysis/outputs/cluster_bootstrap_full.csv
#   columns: quantity, n, estimate, boot_sd, ci95_lo, ci95_hi,
#            sig95_cluster, bonferroni_sig_cluster (provinces only; z = 3.05 for k=22)

import numpy as np
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
N_BOOT = 500
SEED = 42

d = pd.read_csv(os.path.join(HERE, "data_deid.csv"))
pred = np.load(os.path.join(OUT, "rev_oof_noprotected.npy"))
assert len(pred) == len(d)
y = d["Điểm HSA"].to_numpy()
resid = y - pred  # signed bias convention: observed - predicted (+ = under-predicted)

d = d.assign(_resid=resid)
d["_school"] = d["Tỉnh"].astype(str) + "||" + d["Trường"].astype(str)

prov_counts = d["Tỉnh"].value_counts()
provs = sorted(prov_counts[prov_counts >= 200].index.tolist())
zones = sorted(d["khuVuc"].dropna().unique().tolist())
genders = sorted(d["Giới tính"].dropna().unique().tolist())

groups = [("gender", g) for g in genders] + [("zone", z) for z in zones] + [("prov", p) for p in provs]

def group_mask(df, kind, val):
    col = {"gender": "Giới tính", "zone": "khuVuc", "prov": "Tỉnh"}[kind]
    return df[col] == val

# point estimates on the full sample
est = {}
n_g = {}
for kind, val in groups:
    m = group_mask(d, kind, val)
    est[(kind, val)] = d.loc[m, "_resid"].mean()
    n_g[(kind, val)] = int(m.sum())

# cluster bootstrap: resample schools with replacement
rng = np.random.default_rng(SEED)
schools = d["_school"].unique()
by_school = {s: g["_resid"].to_numpy() for s, g in d.groupby("_school")}
# per-school per-group sums/counts for fast recompute
school_stats = {}
for kind, val in groups:
    m = group_mask(d, kind, val)
    sub = d.loc[m]
    grp = sub.groupby("_school")["_resid"].agg(["sum", "count"])
    school_stats[(kind, val)] = (grp["sum"].to_dict(), grp["count"].to_dict())

boot = {g: [] for g in groups}
for b in range(N_BOOT):
    draw = rng.choice(schools, size=len(schools), replace=True)
    for g in groups:
        sums, counts = school_stats[g]
        tot = 0.0
        cnt = 0
        for s in draw:
            c = counts.get(s)
            if c:
                tot += sums[s]
                cnt += c
        if cnt > 0:
            boot[g].append(tot / cnt)

Z_BONF_22 = 3.0545  # two-sided z for alpha = .05/22

rows = []
for kind, val in groups:
    bs = np.array(boot[(kind, val)])
    sd = bs.std(ddof=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    e = est[(kind, val)]
    sig95 = (lo > 0) or (hi < 0)
    bonf = abs(e) > Z_BONF_22 * sd if kind == "prov" else None
    rows.append({
        "quantity": f"{kind}::{val}", "n": n_g[(kind, val)], "estimate": round(e, 4),
        "boot_sd": round(sd, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
        "sig95_cluster": sig95, "bonferroni_sig_cluster": bonf,
    })

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "cluster_bootstrap_full.csv"), index=False)

n_bonf = int(res.loc[res["quantity"].str.startswith("prov"), "bonferroni_sig_cluster"].sum())
n_sig95 = int(res.loc[res["quantity"].str.startswith("prov"), "sig95_cluster"].sum())
n_abs2 = int((res.loc[res["quantity"].str.startswith("prov"), "estimate"].abs() > 2).sum())
print(f"Provinces audited (n>=200): {len(provs)}")
print(f"  |bias| > 2 points: {n_abs2}")
print(f"  cluster-95% CI excludes 0: {n_sig95}")
print(f"  Bonferroni-significant under cluster SE (|est| > {Z_BONF_22}*sd): {n_bonf}  (manuscript iid claim: 14)")
print()
print(res.to_string(index=False))
