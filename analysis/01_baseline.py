# 01_baseline.py — De-identification + first baseline for HSA score prediction
# Data: 2024_Diem_HSA_hocba_THPT-full.xlsx (56,882 students)
# Primary scenario: predict HSA total score from TRANSCRIPT + DEMOGRAPHICS only
# (THPT national-exam columns toan..lich_su are EXCLUDED from predictors —
#  they are administered AFTER the HSA window; kept in data for a separate
#  comparison analysis.)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

RAW = "2024_Diem_HSA_hocba_THPT-full.xlsx"
DEID = "analysis/data_deid.csv"

PII = ["Unnamed: 0", "CCCD", "Họ và tên", "Ngày sinh"]
THPT_EXAM = ["toan", "ngu_van", "ngoai_ngu", "vat_li", "hoa_hoc", "sinh_hoc", "lich_su"]
TARGETS = ["Điểm HSA", "Điểm HSA P1", "Điểm HSA P2", "Điểm HSA P3"]

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}

print("Loading raw data ...")
df = pd.read_excel(RAW)

# --- De-identification: working copy carries no direct identifiers ---
deid = df.drop(columns=[c for c in PII if c in df.columns])
deid.insert(0, "student_id", range(len(deid)))  # surrogate key
deid.to_csv(DEID, index=False)
print(f"De-identified working file -> {DEID}  shape={deid.shape}")

# --- Feature engineering (baseline) ---
d = deid.copy()
for col in d.columns:
    if "Học lực" in col:
        d[col] = d[col].map(HOC_LUC)
    elif "Hạnh kiểm" in col:
        d[col] = d[col].map(HANH_KIEM)
d["gioi_tinh"] = (d["Giới tính"] == "Nam").astype(int)
d = pd.concat([d, pd.get_dummies(d["khuVuc"], prefix="kv")], axis=1)

transcript_cols = [
    c for c in d.columns
    if c[:3] in ("10.", "11.", "12.") and pd.api.types.is_numeric_dtype(d[c])
]
feature_cols = transcript_cols + ["gioi_tinh"] + [c for c in d.columns if c.startswith("kv_")]

X = d[feature_cols].astype(float)
y = d["Điểm HSA"].astype(float)
print(f"Features: {len(feature_cols)} | Samples: {len(X)}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

def report(name, model):
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = root_mean_squared_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"{name:28s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:.3f}")
    return pred

print("\n=== Baseline models (transcript + demographics only) ===")
lin = LinearRegression().fit(X_tr.fillna(X_tr.median()), y_tr)
pred_lin = lin.predict(X_te.fillna(X_tr.median()))
print(f"{'LinearRegression':28s} MAE={mean_absolute_error(y_te, pred_lin):6.2f}  "
      f"RMSE={root_mean_squared_error(y_te, pred_lin):6.2f}  R2={r2_score(y_te, pred_lin):.3f}")

gbm = HistGradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
pred_gbm = report("HistGradientBoosting", gbm)

# --- RQ3 preview: error by region / gender ---
print("\n=== Test-set MAE by group (RQ3 preview) ===")
te = d.loc[X_te.index]
err = np.abs(y_te - pred_gbm)
for gcol in ["khuVuc", "Giới tính"]:
    print(f"-- {gcol} --")
    print(err.groupby(te[gcol]).agg(["mean", "count"]).round(2).to_string())

# --- RQ2 preview: permutation importance (sampled for speed) ---
print("\n=== Top-15 permutation importance (RQ2 preview) ===")
rng = np.random.RandomState(0)
idx = rng.choice(len(X_te), size=min(4000, len(X_te)), replace=False)
imp = permutation_importance(gbm, X_te.iloc[idx], y_te.iloc[idx],
                             n_repeats=3, random_state=0, n_jobs=-1)
order = np.argsort(imp.importances_mean)[::-1][:15]
for i in order:
    print(f"  {feature_cols[i]:35s} {imp.importances_mean[i]:.3f}")
