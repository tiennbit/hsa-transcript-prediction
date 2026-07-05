# 11_equal_tuning.py — Professor/reviewer gap: 02_experiments.py tuned only HistGBM
# (RandomizedSearchCV, 12 candidates, 3-fold on the training partition) while XGBoost
# and LightGBM used fixed settings, making the booster comparison asymmetric.
# This script gives XGBoost and LightGBM the SAME tuning budget and protocol
# (12-candidate randomized search over an equivalent-complexity grid, 3-fold CV on
# the identical 80/20 training partition, seed 42), then evaluates the tuned models
# with 5-fold CV on the full sample and on the fixed holdout — directly comparable
# to Table 2. No existing script or output is modified.
# Output: analysis/outputs/equal_tuning_results.csv

import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_validate,
                                     train_test_split)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
RNG = 42

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}

d = pd.read_csv(os.path.join(HERE, "data_deid.csv"))
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
feats = transcript_cols + demo_cols
san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in feats}
X = d[feats].astype(float).rename(columns=san)
y = d["Điểm HSA"].astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RNG)
print(f"n={len(X)}  features={X.shape[1]}  train={len(X_tr)}  test={len(X_te)}")

# Equivalent-complexity grids (same dimensions as the HistGBM grid in 02:
# learning rate, tree count, leaves/depth, min leaf, L2)
GRIDS = {
    "XGBoost": (XGBRegressor(tree_method="hist", random_state=RNG, n_jobs=-1),
                dict(learning_rate=[0.03, 0.05, 0.08, 0.12],
                     n_estimators=[300, 500, 800],
                     max_depth=[5, 6, 7],
                     min_child_weight=[20, 50, 100],
                     reg_lambda=[0.0, 0.1, 1.0],
                     subsample=[0.8], colsample_bytree=[0.8])),
    "LightGBM": (LGBMRegressor(random_state=RNG, n_jobs=-1, verbosity=-1),
                 dict(learning_rate=[0.03, 0.05, 0.08, 0.12],
                      n_estimators=[300, 500, 800],
                      num_leaves=[31, 63, 127],
                      min_child_samples=[20, 50, 100],
                      reg_lambda=[0.0, 0.1, 1.0],
                      subsample=[0.8], colsample_bytree=[0.8])),
}

cv5 = KFold(n_splits=5, shuffle=True, random_state=RNG)
rows = []
for name, (est, grid) in GRIDS.items():
    print(f"--- tuning {name} (RandomizedSearch, 12 iter, 3-fold on train) ---")
    search = RandomizedSearchCV(est, grid, n_iter=12, cv=3, random_state=RNG,
                                n_jobs=-1, scoring="neg_root_mean_squared_error")
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print(f"  best params: {search.best_params_}")

    res = cross_validate(best, X, y, cv=cv5, n_jobs=1,
                         scoring=["neg_mean_absolute_error",
                                  "neg_root_mean_squared_error", "r2"])
    cv_mae = -res["test_neg_mean_absolute_error"].mean()
    cv_rmse = -res["test_neg_root_mean_squared_error"].mean()
    cv_r2 = res["test_r2"].mean()
    cv_r2_sd = res["test_r2"].std()

    best.fit(X_tr, y_tr)
    p = best.predict(X_te)
    ho_mae = mean_absolute_error(y_te, p)
    ho_rmse = root_mean_squared_error(y_te, p)
    ho_r2 = r2_score(y_te, p)

    print(f"  {name}_tuned  CV: MAE={cv_mae:.3f} RMSE={cv_rmse:.3f} R2={cv_r2:.4f} ({cv_r2_sd:.4f})"
          f"  |  holdout: MAE={ho_mae:.3f} RMSE={ho_rmse:.3f} R2={ho_r2:.4f}")
    rows.append(dict(model=f"{name}_tuned", best_params=str(search.best_params_),
                     CV_MAE=round(cv_mae, 4), CV_RMSE=round(cv_rmse, 4),
                     CV_R2=round(cv_r2, 4), CV_R2_sd=round(cv_r2_sd, 4),
                     holdout_MAE=round(ho_mae, 4), holdout_RMSE=round(ho_rmse, 4),
                     holdout_R2=round(ho_r2, 4)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "equal_tuning_results.csv"), index=False)
print("\nReference (from 02): HistGBM_tuned holdout R2=0.4226 MAE=8.177;"
      " fixed XGBoost CV R2=0.4315, holdout 0.4228; fixed LightGBM CV R2=0.4308, holdout 0.4219")
print("Written: outputs/equal_tuning_results.csv")
