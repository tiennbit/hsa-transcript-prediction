# 02_experiments.py — Full experiment suite for the HSA prediction paper
# Inputs : analysis/data_deid.csv (de-identified working data, from 01_baseline.py)
# Outputs: analysis/outputs/*.csv + *.png  (all tables/figures used in the paper)
#
# Sections
#   A. Feature engineering (transcript + demographics; THPT-exam columns excluded
#      from the primary feature set — administered after HSA; used only in Section E)
#   B. RQ1  Model comparison, 5-fold CV + tuned models on a fixed 80/20 holdout
#   C. RQ2  Interpretation: permutation importance (by feature / year / subject),
#           per-subject correlations, SHAP on LightGBM
#   D. RQ3  Equity audit: group-conditional MAE + signed bias by gender/region/province
#   E. Secondary: transcript-only vs THPT-exam-only vs combined feature sets
#   F. P1/P2/P3 component targets with the best model

import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RNG = 42
OUT = "analysis/outputs"
os.makedirs(OUT, exist_ok=True)

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}
THPT_EXAM = ["toan", "ngu_van", "ngoai_ngu", "vat_li", "hoa_hoc", "sinh_hoc", "lich_su"]
SUBJECTS = ["Toán", "Văn", "Vật lí", "Hóa học", "Sinh học", "Lịch sử", "Địa lí", "GDCD", "Ngoại ngữ"]

# ---------- A. Load + features ----------
print("=== A. Load + feature engineering ===")
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
feat_primary = transcript_cols + demo_cols

# sanitized names for LGBM/XGB
san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in feat_primary + THPT_EXAM}

X = d[feat_primary].astype(float).rename(columns=san)
y = d["Điểm HSA"].astype(float)
print(f"primary features={X.shape[1]}  n={len(X)}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RNG)
te_idx = X_te.index

def eval_pred(y_true, pred):
    return dict(MAE=mean_absolute_error(y_true, pred),
                RMSE=root_mean_squared_error(y_true, pred),
                R2=r2_score(y_true, pred))

# ---------- B. RQ1 model comparison ----------
print("=== B. RQ1 model comparison (5-fold CV on full data) ===")
models = {
    "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
    "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                          n_jobs=-1, random_state=RNG),
    "HistGBM": HistGradientBoostingRegressor(random_state=RNG),
    "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            random_state=RNG, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RNG, n_jobs=-1, verbosity=-1),
}
cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
rows = []
for name, mdl in models.items():
    # RF on imputed data (no native NaN)
    Xc = X.fillna(X.median()) if name == "RandomForest" else X
    res = cross_validate(mdl, Xc, y, cv=cv, n_jobs=1,
                         scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    rows.append(dict(model=name,
                     MAE=-res["test_neg_mean_absolute_error"].mean(),
                     MAE_sd=res["test_neg_mean_absolute_error"].std(),
                     RMSE=-res["test_neg_root_mean_squared_error"].mean(),
                     RMSE_sd=res["test_neg_root_mean_squared_error"].std(),
                     R2=res["test_r2"].mean(), R2_sd=res["test_r2"].std()))
    print(f"  {name:14s} MAE={rows[-1]['MAE']:.3f}  RMSE={rows[-1]['RMSE']:.3f}  R2={rows[-1]['R2']:.4f}")
cv_df = pd.DataFrame(rows).sort_values("RMSE")
cv_df.to_csv(f"{OUT}/metrics_cv.csv", index=False)

print("--- tuning HistGBM (RandomizedSearch, 12 iter, 3-fold on train) ---")
search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=RNG),
    dict(learning_rate=[0.03, 0.05, 0.08, 0.12],
         max_iter=[300, 500, 800],
         max_leaf_nodes=[31, 63, 127],
         min_samples_leaf=[20, 50, 100],
         l2_regularization=[0.0, 0.1, 1.0]),
    n_iter=12, cv=3, random_state=RNG, n_jobs=-1,
    scoring="neg_root_mean_squared_error")
search.fit(X_tr, y_tr)
best = search.best_estimator_
print("  best params:", search.best_params_)

hold_rows = []
for name, mdl in {**models, "HistGBM_tuned": best}.items():
    Xtr2 = X_tr.fillna(X_tr.median()) if name == "RandomForest" else X_tr
    Xte2 = X_te.fillna(X_tr.median()) if name == "RandomForest" else X_te
    mdl.fit(Xtr2, y_tr)
    m = eval_pred(y_te, mdl.predict(Xte2))
    hold_rows.append(dict(model=name, **m))
    print(f"  holdout {name:14s} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.4f}")
hold_df = pd.DataFrame(hold_rows).sort_values("RMSE")
hold_df.to_csv(f"{OUT}/metrics_holdout.csv", index=False)

best_name = hold_df.iloc[0]["model"]
best_model = best if best_name == "HistGBM_tuned" else models[best_name]
print(f"  BEST on holdout: {best_name}")
pred_te = best_model.predict(X_te)

plt.figure(figsize=(5.2, 5))
plt.hexbin(y_te, pred_te, gridsize=45, cmap="Blues", mincnt=1)
lims = [25, 135]
plt.plot(lims, lims, "r--", lw=1)
plt.xlabel("Observed HSA score"); plt.ylabel("Predicted HSA score")
plt.title(f"{best_name}: holdout predictions (n={len(y_te):,})")
plt.colorbar(label="count"); plt.tight_layout()
plt.savefig(f"{OUT}/fig_pred_vs_actual.png", dpi=200); plt.close()

# ---------- C. RQ2 interpretation ----------
print("=== C. RQ2 importance ===")
rng = np.random.RandomState(0)
sub = rng.choice(len(X_te), size=min(6000, len(X_te)), replace=False)
imp = permutation_importance(best_model, X_te.iloc[sub], y_te.iloc[sub],
                             n_repeats=5, random_state=0, n_jobs=-1,
                             scoring="neg_root_mean_squared_error")
inv_san = {v: k for k, v in san.items()}
imp_df = pd.DataFrame({
    "feature": [inv_san.get(c, c) for c in X.columns],
    "importance": imp.importances_mean,
    "importance_sd": imp.importances_std,
}).sort_values("importance", ascending=False)
imp_df.to_csv(f"{OUT}/importance_by_feature.csv", index=False)

def year_of(f):
    return f[:2] if f[:3] in ("10.", "11.", "12.") else "demo"

def subject_of(f):
    for s in SUBJECTS:
        if f".{s} " in f or f.endswith(f".{s} CN") or f".{s} HK" in f:
            return s
    if "Điểm tổng kết" in f: return "Tổng kết (GPA)"
    if "Học lực" in f: return "Học lực"
    if "Hạnh kiểm" in f: return "Hạnh kiểm"
    return "Nhân khẩu" if year_of(f) == "demo" else "Khác"

imp_df["year"] = imp_df["feature"].map(year_of)
imp_df["subject"] = imp_df["feature"].map(subject_of)
by_year = imp_df.groupby("year")["importance"].sum().sort_values(ascending=False)
by_subject = imp_df.groupby("subject")["importance"].sum().sort_values(ascending=False)
by_year.to_csv(f"{OUT}/importance_by_year.csv")
by_subject.to_csv(f"{OUT}/importance_by_subject.csv")
print("  by year:\n", by_year.to_string())
print("  by subject (top):\n", by_subject.head(8).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
top20 = imp_df.head(20).iloc[::-1]
ax[0].barh(top20["feature"], top20["importance"], xerr=top20["importance_sd"], color="#4878d0")
ax[0].set_title("Top-20 permutation importance (ΔRMSE)"); ax[0].tick_params(labelsize=7)
by_subject.head(10).iloc[::-1].plot.barh(ax=ax[1], color="#6acc64")
ax[1].set_title("Aggregated importance by subject")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_importance.png", dpi=200); plt.close()

# per-subject-per-year correlation with HSA (đối-sánh-style table)
corr_rows = []
for yr in ("10", "11", "12"):
    for s in SUBJECTS + ["Điểm tổng kết"]:
        col = f"{yr}.{s} CN"
        if col in d.columns and pd.api.types.is_numeric_dtype(d[col]):
            corr_rows.append(dict(year=yr, subject=s,
                                  pearson_r=d[col].corr(d["Điểm HSA"]),
                                  n=d[col].notna().sum()))
corr_df = pd.DataFrame(corr_rows).pivot(index="subject", columns="year", values="pearson_r").round(3)
corr_df.to_csv(f"{OUT}/correlations_subject_year.csv")
print("  correlations (CN grade vs HSA):\n", corr_df.to_string())

# SHAP on LightGBM (well-supported tree explainer)
print("--- SHAP (LightGBM) ---")
import shap
lgbm = models["LightGBM"].fit(X_tr, y_tr)
expl = shap.TreeExplainer(lgbm)
sv = expl.shap_values(X_te.iloc[sub[:3000]])
plt.figure()
shap.summary_plot(sv, X_te.iloc[sub[:3000]].rename(columns=inv_san),
                  max_display=20, show=False, plot_size=(9, 7))
plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_beeswarm.png", dpi=200); plt.close()

# ---------- D. RQ3 equity audit ----------
print("=== D. RQ3 group-conditional errors ===")
te = d.loc[te_idx]
err = y_te - pred_te          # signed: + = under-prediction (model predicts too low)
abserr = np.abs(err)
def group_table(gcol, min_n=200):
    g = pd.DataFrame({"err": err, "abs": abserr, "y": y_te, "pred": pred_te,
                      "grp": te[gcol]}).groupby("grp")
    t = pd.DataFrame({
        "n": g.size(),
        "MAE": g["abs"].mean(),
        "bias_mean_err": g["err"].mean(),     # systematic over/under-prediction
        "RMSE": g["err"].apply(lambda e: np.sqrt((e**2).mean())),
        "mean_actual": g["y"].mean(),
        "mean_pred": g["pred"].mean(),
    })
    return t[t["n"] >= min_n].round(3)

tabs = {}
for gcol in ["Giới tính", "khuVuc", "Tỉnh"]:
    tabs[gcol] = group_table(gcol)
    tabs[gcol].to_csv(f"{OUT}/fairness_{san.get(gcol, gcol).strip('_') or 'group'}.csv")
print("  gender:\n", tabs["Giới tính"].to_string())
print("  region:\n", tabs["khuVuc"].to_string())
prov = tabs["Tỉnh"].sort_values("bias_mean_err")
print(f"  provinces: bias range {prov['bias_mean_err'].min():.2f} .. {prov['bias_mean_err'].max():.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tabs["khuVuc"]["MAE"].plot.bar(ax=ax[0], color="#4878d0", rot=0)
ax[0].set_title("MAE by priority region"); ax[0].set_xlabel("")
tabs["khuVuc"]["bias_mean_err"].plot.bar(ax=ax[1], color="#d65f5f", rot=0)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_title("Signed bias by priority region (+ = under-predicted)"); ax[1].set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_fairness_region.png", dpi=200); plt.close()

# ---------- E. transcript vs THPT-exam feature sets ----------
print("=== E. Feature-set comparison (construct analysis) ===")
Xe = d[THPT_EXAM].astype(float).rename(columns=san)
Xb = pd.concat([X, Xe], axis=1)
sets = {"transcript_only": X, "thpt_exam_only": Xe, "combined": Xb}
set_rows = []
for name, Xs in sets.items():
    Xtr2, Xte2 = Xs.loc[X_tr.index], Xs.loc[te_idx]
    m = HistGradientBoostingRegressor(random_state=RNG, **{k: v for k, v in search.best_params_.items()})
    m.fit(Xtr2, y_tr)
    met = eval_pred(y_te, m.predict(Xte2))
    set_rows.append(dict(feature_set=name, n_features=Xs.shape[1], **met))
    print(f"  {name:16s} MAE={met['MAE']:.3f}  RMSE={met['RMSE']:.3f}  R2={met['R2']:.4f}")
pd.DataFrame(set_rows).to_csv(f"{OUT}/feature_set_comparison.csv", index=False)

# ---------- F. component targets ----------
print("=== F. P1/P2/P3 components (tuned HistGBM) ===")
comp_rows = []
for part, label in [("Điểm HSA P1", "P1 quantitative"), ("Điểm HSA P2", "P2 qualitative"),
                    ("Điểm HSA P3", "P3 science"), ("Điểm HSA", "Total")]:
    yc = d[part].astype(float)
    m = HistGradientBoostingRegressor(random_state=RNG, **{k: v for k, v in search.best_params_.items()})
    m.fit(X_tr, yc.loc[X_tr.index])
    met = eval_pred(yc.loc[te_idx], m.predict(X_te))
    sd = yc.std()
    comp_rows.append(dict(target=label, sd=round(sd, 2), **{k: round(v, 3) for k, v in met.items()},
                          RMSE_over_SD=round(met["RMSE"] / sd, 3)))
    print(f"  {label:16s} R2={met['R2']:.4f}  RMSE={met['RMSE']:.2f}  (SD={sd:.2f})")
pd.DataFrame(comp_rows).to_csv(f"{OUT}/component_targets.csv", index=False)

json.dump({"best_model": str(best_name), "best_params": search.best_params_,
           "n": int(len(X)), "n_features": int(X.shape[1])},
          open(f"{OUT}/run_meta.json", "w"), indent=2, default=str)
print("\nAll outputs in", OUT)
