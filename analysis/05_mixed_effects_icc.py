# 05_mixed_effects_icc.py — Reviewer gap: how much does SCHOOL / PROVINCE matter,
# before vs after accounting for transcript grades (ICC).
#
# Method (documented, reproducible; seed=42, same feature construction as 01-04):
#   * Unconditional ICC: one-way random-effects ANOVA variance decomposition of the
#     raw HSA total, grouped by school (and separately by province). This is the
#     standard unbiased ICC(1) estimator; it equals a random-intercept-only model's
#     variance ratio and is fast/stable at 56,882 x 987 groups.
#   * Conditional ICC: same decomposition applied to the OUT-OF-FOLD residuals of a
#     transcript-only prediction model (XGBoost, same params as 03_revision_analyses).
#     "How much does school still matter AFTER grades are known."
#   * Cross-check: statsmodels MixedLM intercept-only REML for the unconditional
#     school model (guarded; ANOVA remains the reported primary estimator).
#
# Outputs: analysis/outputs/icc_results.csv  (model, grouping, var_between, var_within, icc)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

RNG = 42
OUT = "analysis/outputs"
os.makedirs(OUT, exist_ok=True)

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}

# ---------- Load + identical feature construction to 01-04 ----------
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
import re
san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in transcript_cols + demo_cols}

y = d["Điểm HSA"].astype(float).values
school = d["Trường"].values
prov = d["Tỉnh"].values
print(f"n={len(y)}  transcript_features={len(transcript_cols)}  "
      f"schools={d['Trường'].nunique()}  provinces={d['Tỉnh'].nunique()}")

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=RNG, n_jobs=-1)


def anova_icc(values, groups):
    """One-way random-effects ANOVA variance decomposition -> unbiased ICC(1).
    Returns (var_between, var_within, icc, k_groups, n_total)."""
    dfv = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)})
    dfv = dfv.dropna(subset=["v"])
    N = len(dfv)
    grand = dfv["v"].mean()
    grp = dfv.groupby("g")["v"]
    n_i = grp.size().values
    m_i = grp.mean().values
    k = len(n_i)
    ss_between = float((n_i * (m_i - grand) ** 2).sum())
    # within SS = total SS - between SS
    ss_total = float(((dfv["v"] - grand) ** 2).sum())
    ss_within = ss_total - ss_between
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)          # design-effect denominator
    var_between = max((ms_between - ms_within) / n0, 0.0)
    var_within = ms_within
    icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0
    return var_between, var_within, icc, k, N


def oof_transcript_residuals():
    """5-fold OOF residuals of a transcript-only XGBoost model (seed 42)."""
    X = d[transcript_cols].astype(float).rename(columns=san)
    oof = np.zeros(len(X))
    for tr, te in KFold(5, shuffle=True, random_state=RNG).split(X):
        m = XGBRegressor(**XGB_PARAMS).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    return y - oof, oof


print("\n=== Unconditional ICC (raw HSA total) ===")
rows = []
for gname, g in [("school", school), ("province", prov)]:
    vb, vw, icc, k, N = anova_icc(y, g)
    rows.append(dict(model="unconditional", grouping=gname,
                     var_between=round(vb, 4), var_within=round(vw, 4), icc=round(icc, 4)))
    print(f"  {gname:9s}  k={k:4d}  var_between={vb:8.3f}  var_within={vw:8.3f}  ICC={icc:.4f}")

print("\n=== Conditional ICC (OOF residuals after transcript grades) ===")
print("  fitting transcript-only XGBoost OOF (5-fold, seed 42) ...")
resid, oof = oof_transcript_residuals()
oof_r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"  transcript-only OOF R2 = {oof_r2:.4f}  (residual variance = {resid.var():.3f})")
for gname, g in [("school", school), ("province", prov)]:
    vb, vw, icc, k, N = anova_icc(resid, g)
    rows.append(dict(model="conditional_transcript", grouping=gname,
                     var_between=round(vb, 4), var_within=round(vw, 4), icc=round(icc, 4)))
    print(f"  {gname:9s}  k={k:4d}  var_between={vb:8.3f}  var_within={vw:8.3f}  ICC={icc:.4f}")

# ---------- Cross-check: statsmodels MixedLM intercept-only (unconditional, school) ----------
print("\n=== Cross-check: MixedLM intercept-only REML (school, unconditional) ===")
try:
    import statsmodels.formula.api as smf
    mdf = pd.DataFrame({"y": y, "school": school})
    md = smf.mixedlm("y ~ 1", mdf, groups=mdf["school"])
    mfit = md.fit(reml=True, method="lbfgs")
    var_school = float(mfit.cov_re.iloc[0, 0])
    var_resid = float(mfit.scale)
    icc_mm = var_school / (var_school + var_resid)
    rows.append(dict(model="unconditional_mixedlm", grouping="school",
                     var_between=round(var_school, 4), var_within=round(var_resid, 4),
                     icc=round(icc_mm, 4)))
    print(f"  MixedLM  var_school={var_school:.3f}  var_resid={var_resid:.3f}  ICC={icc_mm:.4f}")
except Exception as e:  # pragma: no cover
    print(f"  MixedLM skipped ({type(e).__name__}: {e}); ANOVA estimator is the reported value.")

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/icc_results.csv", index=False)
print(f"\nWrote {OUT}/icc_results.csv")
print(out.to_string(index=False))
