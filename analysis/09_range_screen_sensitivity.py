# 09_range_screen_sensitivity.py — Reviewer gap: four grade cells exceed the 0-10
# scale in data_deid.csv (decimal-point entry errors, x10, two students):
#   student 53567: '10.Sinh học HK II' = 84, '10.Ngoại ngữ CN' = 91
#   student 21516: '10.Điểm tổng kết HK I' = 83, '10.GDCD HK II' = 82
# This script (a) applies the decimal-shift (/10) correction, (b) recomputes the
# manuscript's Table 3 (full-year subject-grade vs HSA-total Pearson r) and Table 4
# (grade M/SD by year) on corrected data, and (c) re-runs the tuned HistGBM 80/20
# holdout (seed 42, params from run_meta.json) on original vs corrected features to
# verify model metrics are unchanged to reported precision.
# Outputs: outputs/range_screen_sensitivity.csv (+ console summary)

import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
RNG = 42

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}
CELLS = [(53567, "10.Sinh học HK II"), (53567, "10.Ngoại ngữ CN"),
         (21516, "10.Điểm tổng kết HK I"), (21516, "10.GDCD HK II")]

def build_xy(d):
    d = d.copy()
    for col in d.columns:
        if "Học lực" in col:
            d[col] = d[col].map(HOC_LUC)
        elif "Hạnh kiểm" in col:
            d[col] = d[col].map(HANH_KIEM)
    d["gioi_tinh"] = (d["Giới tính"] == "Nam").astype(int)
    d = pd.concat([d, pd.get_dummies(d["khuVuc"], prefix="kv")], axis=1)
    transcript = [c for c in d.columns
                  if c[:3] in ("10.", "11.", "12.") and pd.api.types.is_numeric_dtype(d[c])]
    demo = ["gioi_tinh"] + [c for c in d.columns if c.startswith("kv_")]
    feats = transcript + demo
    san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in feats}
    return d[feats].astype(float).rename(columns=san), d["Điểm HSA"].astype(float)

def holdout_metrics(d):
    X, y = build_xy(d)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RNG)
    params = json.load(open(os.path.join(OUT, "run_meta.json")))["best_params"]
    m = HistGradientBoostingRegressor(random_state=RNG, **params).fit(X_tr, y_tr)
    p = m.predict(X_te)
    return r2_score(y_te, p), mean_absolute_error(y_te, p)

d0 = pd.read_csv(os.path.join(HERE, "data_deid.csv"))
d1 = d0.copy()
for sid, col in CELLS:
    idx = d1["student_id"] == sid
    assert (d1.loc[idx, col] > 10).all(), f"cell {sid}/{col} not out-of-range?"
    d1.loc[idx, col] = d1.loc[idx, col] / 10.0

# (b) Table 3 / Table 4 on corrected data
SUBJ_FY = {"Mathematics": "Toán CN", "Physics": "Vật lí CN", "Overall GPA": "Điểm tổng kết CN",
           "Chemistry": "Hóa học CN", "Biology": "Sinh học CN", "Foreign language": "Ngoại ngữ CN",
           "Geography": "Địa lí CN", "History": "Lịch sử CN", "Civic education": "GDCD CN",
           "Literature": "Văn CN"}
tot = d1["Điểm HSA"]
rows = []
print("Table 3 (corrected) — Pearson r with HSA total:")
for name, stem in SUBJ_FY.items():
    rs = []
    for g in ("10", "11", "12"):
        col = f"{g}.{stem}"
        r = d1[col].corr(tot) if col in d1.columns else np.nan
        rs.append(r)
        rows.append({"table": "T3", "subject": name, "grade": g, "value_corrected": round(r, 3)})
    print(f"  {name:18s} {rs[0]:.3f} {rs[1]:.3f} {rs[2]:.3f}")
print("\nTable 4 (corrected) — full-year M (SD):")
for name in ["Overall GPA", "Mathematics", "Literature", "Physics", "Chemistry", "Foreign language"]:
    stem = SUBJ_FY[name]
    out = []
    for g in ("10", "11", "12"):
        s = d1[f"{g}.{stem}"]
        out.append(f"{s.mean():.2f} ({s.std():.2f})")
        rows.append({"table": "T4", "subject": name, "grade": g,
                     "value_corrected": f"{s.mean():.2f} ({s.std():.2f})"})
    print(f"  {name:18s} " + "  ".join(out))

# (c) model sensitivity
r2_0, mae_0 = holdout_metrics(d0)
r2_1, mae_1 = holdout_metrics(d1)
print(f"\nTuned HistGBM 80/20 holdout: original R2={r2_0:.4f} MAE={mae_0:.4f}"
      f" | corrected R2={r2_1:.4f} MAE={mae_1:.4f}")
rows.append({"table": "model", "subject": "holdout_R2", "grade": "-",
             "value_corrected": f"{r2_1:.4f} (orig {r2_0:.4f})"})
rows.append({"table": "model", "subject": "holdout_MAE", "grade": "-",
             "value_corrected": f"{mae_1:.4f} (orig {mae_0:.4f})"})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "range_screen_sensitivity.csv"), index=False)
print("\nWritten: outputs/range_screen_sensitivity.csv")
