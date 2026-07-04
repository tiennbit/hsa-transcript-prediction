# 03_revision_analyses.py — Analyses required by the round-1 review (MAJOR REVISION)
# Addresses editor must-fix items:
#  #2 out-of-fold (OOF) predictions over the FULL sample -> fairness audit with 95% CIs,
#     more provinces clearing n>=200, Bonferroni multiplicity check
#  #3 re-audit with a model WITHOUT protected attributes (no gender, no KV)
#  #4 registration-time model (grade 10 + 11 + 12 semester-1 features only)
#  #6 own-sample grade means/SDs by year (compression check) + Steiger's z for
#     dependent correlation differences (grade-11 vs grade-12 each subject)
#     + school-vs-province variance decomposition of OOF residuals
#  #7 empirical coverage (share |err|<=8) + 80% prediction-interval width
# Outputs: analysis/outputs/rev_*.csv

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

RNG = 42
OUT = "analysis/outputs"
os.makedirs(OUT, exist_ok=True)

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}
SUBJECTS = ["Toán", "Văn", "Vật lí", "Hóa học", "Sinh học", "Lịch sử", "Địa lí", "GDCD", "Ngoại ngữ"]

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
demo_cols = ["gioi_tinh"] + [c for c in d.columns if c.startswith("kv_")]
san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in transcript_cols + demo_cols}

y = d["Điểm HSA"].astype(float)
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=RNG, n_jobs=-1)

def oof_predict(features):
    X = d[features].astype(float).rename(columns=san)
    oof = np.zeros(len(X))
    for tr, te in KFold(5, shuffle=True, random_state=RNG).split(X):
        m = XGBRegressor(**XGB_PARAMS).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof

def metrics(pred, ytrue=y):
    e = ytrue - pred
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()),
                R2=1 - (e**2).sum() / ((ytrue - ytrue.mean())**2).sum())

print("=== OOF predictions: full model (with gender+KV) ===")
oof_full = oof_predict(transcript_cols + demo_cols)
print("  ", {k: round(v, 4) for k, v in metrics(oof_full).items()})

print("=== OOF predictions: NO protected attributes (transcript only) ===")
oof_nopro = oof_predict(transcript_cols)
print("  ", {k: round(v, 4) for k, v in metrics(oof_nopro).items()})

np.save(f"{OUT}/rev_oof_full.npy", oof_full)
np.save(f"{OUT}/rev_oof_noprotected.npy", oof_nopro)
pd.DataFrame([dict(model="full_with_protected", **metrics(oof_full)),
              dict(model="no_protected_attrs", **metrics(oof_nopro))]
             ).round(4).to_csv(f"{OUT}/rev_oof_metrics.csv", index=False)

# ---------- Fairness audit on OOF (n = 56,882), both model variants ----------
def audit(pred, gcol, min_n=200):
    e = y - pred
    g = pd.DataFrame({"err": e, "abs": np.abs(e), "grp": d[gcol]}).groupby("grp")
    t = pd.DataFrame({
        "n": g.size(),
        "MAE": g["abs"].mean(),
        "MAE_ci95": 1.96 * g["abs"].std() / np.sqrt(g.size()),
        "bias": g["err"].mean(),
        "bias_ci95": 1.96 * g["err"].std() / np.sqrt(g.size()),
    })
    return t[t["n"] >= min_n].round(3)

for tag, pred in [("full", oof_full), ("nopro", oof_nopro)]:
    print(f"=== Audit ({tag}) ===")
    for gcol, fname in [("Giới tính", "gender"), ("khuVuc", "region"), ("Tỉnh", "province")]:
        t = audit(pred, gcol)
        t.to_csv(f"{OUT}/rev_audit_{tag}_{fname}.csv")
        if gcol != "Tỉnh":
            print(f"-- {gcol} --\n{t.to_string()}")
        else:
            k = len(t)
            sig_bonf = (np.abs(t['bias']) > stats.norm.ppf(1 - 0.025 / k) * t['bias_ci95'] / 1.96).sum()
            print(f"-- Tỉnh: {k} provinces n>=200; bias range {t['bias'].min():.2f}..{t['bias'].max():.2f}; "
                  f"|bias|>2: {(t['bias'].abs() > 2).sum()}; Bonferroni-significant: {sig_bonf}")

# ---------- Registration-time model (no grade-12 sem-2 / full-year info) ----------
print("=== Registration-time model ===")
reg_cols = [c for c in transcript_cols
            if c[:3] in ("10.", "11.") or (c.startswith("12.") and "HK I" in c and "HK II" not in c)]
X_full = d[transcript_cols + demo_cols].astype(float).rename(columns=san)
X_reg = d[reg_cols + demo_cols].astype(float).rename(columns=san)
tr_idx, te_idx = train_test_split(np.arange(len(d)), test_size=0.2, random_state=RNG)
rows = []
for name, Xs in [("full_113", X_full), ("registration_time", X_reg)]:
    m = XGBRegressor(**XGB_PARAMS).fit(Xs.iloc[tr_idx], y.iloc[tr_idx])
    pred = m.predict(Xs.iloc[te_idx])
    met = metrics(pred, y.iloc[te_idx])
    rows.append(dict(model=name, n_features=Xs.shape[1], **{k: round(v, 4) for k, v in met.items()}))
    print(f"  {name:18s} k={Xs.shape[1]:3d}  MAE={met['MAE']:.3f}  RMSE={met['RMSE']:.3f}  R2={met['R2']:.4f}")
pd.DataFrame(rows).to_csv(f"{OUT}/rev_registration_time.csv", index=False)

# ---------- Grade distributions by year (compression check) ----------
print("=== Grade means/SDs by year ===")
dist_rows = []
for s in SUBJECTS + ["Điểm tổng kết"]:
    for yr in ("10", "11", "12"):
        col = f"{yr}.{s} CN"
        if col in d.columns:
            dist_rows.append(dict(subject=s, year=yr,
                                  mean=round(d[col].mean(), 3), sd=round(d[col].std(), 3)))
dist = pd.DataFrame(dist_rows).pivot(index="subject", columns="year", values=["mean", "sd"]).round(3)
dist.to_csv(f"{OUT}/rev_grade_distributions.csv")
print(dist.to_string())

# ---------- Steiger's z: r(g11,HSA) vs r(g12,HSA), dependent (shared HSA) ----------
def steiger_z(r12, r13, r23, n):
    # Steiger (1980) test for two dependent correlations sharing one variable
    rm2 = (r12**2 + r13**2) / 2
    f = (1 - r23) / (2 * (1 - rm2))
    h = (1 - f * rm2) / (1 - rm2)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    return (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - r23) * h))

print("=== Steiger's z (grade-11 vs grade-12 CN correlation with HSA) ===")
st_rows = []
for s in SUBJECTS + ["Điểm tổng kết"]:
    c11, c12 = f"11.{s} CN", f"12.{s} CN"
    sub = d[[c11, c12, "Điểm HSA"]].dropna()
    r11 = sub[c11].corr(sub["Điểm HSA"]); r12_ = sub[c12].corr(sub["Điểm HSA"])
    r1112 = sub[c11].corr(sub[c12])
    zval = steiger_z(r11, r12_, r1112, len(sub))
    pval = 2 * (1 - stats.norm.cdf(abs(zval)))
    st_rows.append(dict(subject=s, r_g11=round(r11, 3), r_g12=round(r12_, 3),
                        r_g11_g12=round(r1112, 3), steiger_z=round(zval, 2),
                        p=("<.001" if pval < 0.001 else round(pval, 4)), n=len(sub)))
    print(f"  {s:14s} r11={r11:.3f} r12={r12_:.3f} z={zval:7.2f}")
pd.DataFrame(st_rows).to_csv(f"{OUT}/rev_steiger.csv", index=False)

# ---------- School vs province variance decomposition of OOF residuals ----------
print("=== Variance decomposition of residuals (no-protected model) ===")
res = pd.DataFrame({"err": y - oof_nopro, "school": d["Trường"], "prov": d["Tỉnh"]})
sch = res.groupby(["prov", "school"]).agg(m=("err", "mean"), n=("err", "size")).reset_index()
sch = sch[sch["n"] >= 30]
grand = np.average(sch["m"], weights=sch["n"])
prov_means = sch.groupby("prov").apply(lambda g: np.average(g["m"], weights=g["n"]), include_groups=False)
sch = sch.join(prov_means.rename("prov_m"), on="prov")
ss_between_prov = (sch["n"] * (sch["prov_m"] - grand) ** 2).sum()
ss_within_prov = (sch["n"] * (sch["m"] - sch["prov_m"]) ** 2).sum()
share_prov = ss_between_prov / (ss_between_prov + ss_within_prov)
print(f"  schools (n>=30): {len(sch)}; school-mean residual SD = {sch['m'].std():.2f}")
print(f"  between-province share of school-level residual variance: {share_prov:.1%}")
print(f"  within-province (between-school) share: {1 - share_prov:.1%}")
pd.DataFrame([dict(n_schools=len(sch), school_mean_resid_sd=round(sch['m'].std(), 3),
                   between_province_share=round(share_prov, 4),
                   between_school_within_prov_share=round(1 - share_prov, 4))]
             ).to_csv(f"{OUT}/rev_variance_decomposition.csv", index=False)

# ---------- Coverage / prediction-interval width ----------
print("=== Empirical coverage (OOF, full model) ===")
e = y - oof_full
cov8 = (np.abs(e) <= 8).mean()
q10, q90 = np.quantile(e, [0.1, 0.9])
print(f"  share |err|<=8: {cov8:.1%}   80% PI: [{q10:.2f}, {q90:.2f}] width {q90 - q10:.1f}")
pd.DataFrame([dict(share_within_8=round(cov8, 4), pi80_lo=round(q10, 2),
                   pi80_hi=round(q90, 2), pi80_width=round(q90 - q10, 2))]
             ).to_csv(f"{OUT}/rev_coverage.csv", index=False)

# ---------- Sample regional composition ----------
top = d["Tỉnh"].value_counts()
top.to_csv(f"{OUT}/rev_province_counts.csv")
print("=== Top-10 provinces by sample size ===")
print(top.head(10).to_string())
print("\nDone.")
