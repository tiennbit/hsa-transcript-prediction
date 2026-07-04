# 06_school_heldout.py — Reviewer gap: does the model generalize to UNSEEN schools?
#
# Student-level CV lets rows from the same school appear in both train and test, so the
# model can memorize school-specific grading. This script uses GroupKFold BY SCHOOL so
# every test-set school is absent from training. Same model / params / feature set /
# seed as 01-04 for comparability.
#
# Reports (pooled OOF across folds, plus fold-level spread):
#   * school-held-out R2 / MAE / RMSE
#   * student-level (iid) 5-fold OOF R2 / MAE / RMSE on the identical feature set
#   * the fixed 80/20 holdout R2 (comparison anchor; paper reports ~.423)
# Output: analysis/outputs/school_heldout_metrics.csv

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from xgboost import XGBRegressor

RNG = 42
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
demo_cols = ["gioi_tinh"] + [c for c in d.columns if c.startswith("kv_")]
feat = transcript_cols + demo_cols
san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in feat}

X = d[feat].astype(float).rename(columns=san)
y = d["Điểm HSA"].astype(float).values
groups = d["Trường"].values
print(f"n={len(y)}  features={X.shape[1]}  schools={d['Trường'].nunique()}")

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=RNG, n_jobs=-1)


def m_pool(ytrue, pred):
    e = ytrue - pred
    return (float(1 - (e ** 2).sum() / ((ytrue - ytrue.mean()) ** 2).sum()),
            float(np.abs(e).mean()),
            float(np.sqrt((e ** 2).mean())))


def run_cv(splitter, split_args, tag):
    oof = np.zeros(len(y))
    fold_rows = []
    for k, (tr, te) in enumerate(splitter.split(X, y, *split_args)):
        mdl = XGBRegressor(**XGB_PARAMS).fit(X.iloc[tr], y[tr])
        oof[te] = mdl.predict(X.iloc[te])
        r2, mae, rmse = m_pool(y[te], oof[te])
        fold_rows.append(dict(fold=k, n_test=len(te),
                              n_test_schools=len(np.unique(groups[te])),
                              R2=r2, MAE=mae, RMSE=rmse))
        print(f"  [{tag}] fold {k}: n_test={len(te):6d} "
              f"schools={len(np.unique(groups[te])):4d}  R2={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    r2, mae, rmse = m_pool(y, oof)  # pooled across all OOF predictions
    fr = pd.DataFrame(fold_rows)
    return dict(scheme=tag, R2_pooled=round(r2, 4), MAE_pooled=round(mae, 4),
                RMSE_pooled=round(rmse, 4),
                R2_fold_mean=round(fr["R2"].mean(), 4), R2_fold_sd=round(fr["R2"].std(), 4),
                MAE_fold_mean=round(fr["MAE"].mean(), 4))


rows = []
print("=== School-held-out CV (GroupKFold by school, 5 folds) ===")
rows.append(run_cv(GroupKFold(n_splits=5), (groups,), "school_heldout"))

print("=== Student-level CV (KFold shuffle, 5 folds, seed 42) — same features ===")
rows.append(run_cv(KFold(n_splits=5, shuffle=True, random_state=RNG), (), "student_iid"))

print("=== Fixed 80/20 holdout (seed 42) — comparison anchor ===")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RNG)
mdl = XGBRegressor(**XGB_PARAMS).fit(Xtr, ytr)
r2, mae, rmse = m_pool(yte, mdl.predict(Xte))
rows.append(dict(scheme="student_holdout_80_20", R2_pooled=round(r2, 4),
                 MAE_pooled=round(mae, 4), RMSE_pooled=round(rmse, 4),
                 R2_fold_mean=np.nan, R2_fold_sd=np.nan, MAE_fold_mean=np.nan))
print(f"  R2={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/school_heldout_metrics.csv", index=False)

sh = res.loc[res["scheme"] == "school_heldout", "R2_pooled"].iloc[0]
si = res.loc[res["scheme"] == "student_iid", "R2_pooled"].iloc[0]
print(f"\nR2 drop (student iid -> school held-out): {si:.4f} -> {sh:.4f} "
      f"(Δ = {si - sh:.4f}, {100 * (si - sh) / si:.1f}% relative)")
print(f"Wrote {OUT}/school_heldout_metrics.csv")
print(res.to_string(index=False))
