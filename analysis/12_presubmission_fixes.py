# 12_presubmission_fixes.py — Pre-submission audit fixes (2026-07-05):
#  (a) Fig 1: school count corrected to 1,124 province-school units (was "987 schools");
#      slightly larger box font per Springer legibility guidance.
#  (b) Fig 6 (correlation heatmap): regenerated from the range-screened data
#      (four decimal-shift grade cells corrected, cf. 09_range_screen_sensitivity.py),
#      so the grade-10 foreign-language cell matches Table 3 (.348 -> ".35"); cell
#      annotations use ROUND_HALF_UP to two decimals for consistency with the tables.
#      Writes outputs/correlations_subject_year_corrected.csv (original CSV untouched).
#  (c) HistGBM (tuned) 5-fold CV on the full sample, so Table 2's tuned rows all
#      carry CV estimates. Writes outputs/histgbm_tuned_cv.csv.
# No existing script or output is modified. Figures overwrite paper/figures/Fig1.png
# and Fig6.png (previous versions recoverable by rerunning 04_figures.py).

import os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "savefig.facecolor": "white",
})
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "paper", "figures")
OUT = os.path.join(HERE, "outputs")

CELLS = [(53567, "10.Sinh học HK II"), (53567, "10.Ngoại ngữ CN"),
         (21516, "10.Điểm tổng kết HK I"), (21516, "10.GDCD HK II")]

# ---------- (a) Fig 1: study design schematic, corrected school count ----------
fig, ax = plt.subplots(figsize=(6.85, 2.6))
ax.axis("off")
boxes = [
    (0.01, 0.55, 0.17, "Administrative records\n56,882 students\n1,124 province–school\nunits, 49 provinces", CB[0]),
    (0.21, 0.55, 0.17, "De-identification\n+ 113 pre-exam features\n(transcript, demographics)", CB[0]),
    (0.41, 0.78, 0.17, "RQ1 Accuracy\n5 models, 5-fold CV\n+ registration-time variant", CB[1]),
    (0.41, 0.50, 0.17, "RQ2 Signal structure\npermutation importance,\ncorrelations, SHAP", CB[2]),
    (0.41, 0.22, 0.17, "RQ3 Equity audit\nOOF signed bias ± 95% CI\n(attribute-blind model)", CB[3]),
    (0.62, 0.55, 0.17, "Construct comparison\ntranscript vs national\nexam vs combined", CB[4]),
    (0.82, 0.55, 0.17, "Implications\ncounseling, score\nconversion, equity", CB[5]),
]
for x, yy, w, text, c in boxes:
    ax.add_patch(FancyBboxPatch((x, yy - 0.13), w, 0.26, boxstyle="round,pad=0.012",
                                fc="white", ec=c, lw=1.4, transform=ax.transAxes))
    ax.text(x + w / 2, yy, text, ha="center", va="center", fontsize=7.8, transform=ax.transAxes)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"))
arrow(0.18, 0.55, 0.21, 0.55)
for yy in (0.78, 0.50, 0.22):
    arrow(0.38, 0.55, 0.41, yy)
for yy in (0.78, 0.50, 0.22):
    arrow(0.58, yy, 0.62, 0.55)
arrow(0.79, 0.55, 0.82, 0.55)
ax.text(0.5, 0.02, "Post-exam graduation scores excluded from all predictive feature sets (construct analysis only)",
        ha="center", fontsize=7.4, style="italic", color="0.35", transform=ax.transAxes)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "Fig1.png"), dpi=600, bbox_inches="tight")
plt.close()
print("Fig1 regenerated (1,124 province-school units)")

# ---------- (b) Fig 6: correlation heatmap from range-screened data ----------
d = pd.read_csv(os.path.join(HERE, "data_deid.csv"))
for sid, col in CELLS:
    idx = d["student_id"] == sid
    d.loc[idx, col] = d.loc[idx, col] / 10.0
tot = d["Điểm HSA"].astype(float)
SUBJ = {"Toán": "Mathematics", "Vật lí": "Physics", "Điểm tổng kết": "Overall GPA",
        "Hóa học": "Chemistry", "Sinh học": "Biology", "Ngoại ngữ": "Foreign language",
        "Địa lí": "Geography", "Lịch sử": "History", "GDCD": "Civic education", "Văn": "Literature"}
rows = {}
for vi, en in SUBJ.items():
    rows[en] = [d[f"{g}.{vi} CN"].astype(float).corr(tot) for g in ("10", "11", "12")]
corr = pd.DataFrame(rows, index=["10", "11", "12"]).T
corr.to_csv(os.path.join(OUT, "correlations_subject_year_corrected.csv"))
corr = corr.sort_values("11", ascending=False)

def fmt2(v):  # ROUND_HALF_UP to 2 decimals, APA-style leading-zero strip
    return str(Decimal(str(round(v, 4))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)).lstrip("0")

fig, ax = plt.subplots(figsize=(3.3, 3.0))
im = ax.imshow(corr.values, cmap="YlGnBu", vmin=0.1, vmax=0.6, aspect="auto")
ax.set_xticks(range(3), [f"Grade {c}" for c in corr.columns])
ax.set_yticks(range(len(corr)), corr.index)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        v = corr.values[i, j]
        ax.text(j, i, fmt2(v), ha="center", va="center",
                fontsize=7, color="white" if v > 0.42 else "0.2")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Pearson r with HSA total", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "Fig6.png"), dpi=600, bbox_inches="tight")
plt.close()
print("Fig6 regenerated from corrected data; FL G10 cell:", fmt2(corr.loc["Foreign language", "10"]))

# ---------- (c) HistGBM (tuned) 5-fold CV for Table 2 ----------
import json
import re
HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}
dd = pd.read_csv(os.path.join(HERE, "data_deid.csv"))
for col in dd.columns:
    if "Học lực" in col:
        dd[col] = dd[col].map(HOC_LUC)
    elif "Hạnh kiểm" in col:
        dd[col] = dd[col].map(HANH_KIEM)
dd["gioi_tinh"] = (dd["Giới tính"] == "Nam").astype(int)
dd = pd.concat([dd, pd.get_dummies(dd["khuVuc"], prefix="kv")], axis=1)
tcols = [c for c in dd.columns if c[:3] in ("10.", "11.", "12.") and pd.api.types.is_numeric_dtype(dd[c])]
feats = tcols + ["gioi_tinh"] + [c for c in dd.columns if c.startswith("kv_")]
san = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in feats}
X = dd[feats].astype(float).rename(columns=san)
y = dd["Điểm HSA"].astype(float)
params = json.load(open(os.path.join(OUT, "run_meta.json")))["best_params"]
m = HistGradientBoostingRegressor(random_state=42, **params)
res = cross_validate(m, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=1,
                     scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
out = dict(model="HistGBM_tuned",
           CV_MAE=round(-res["test_neg_mean_absolute_error"].mean(), 4),
           CV_RMSE=round(-res["test_neg_root_mean_squared_error"].mean(), 4),
           CV_R2=round(res["test_r2"].mean(), 4),
           CV_R2_sd=round(res["test_r2"].std(), 4))
pd.DataFrame([out]).to_csv(os.path.join(OUT, "histgbm_tuned_cv.csv"), index=False)
print(f"HistGBM_tuned CV: MAE={out['CV_MAE']:.3f} RMSE={out['CV_RMSE']:.3f} "
      f"R2={out['CV_R2']:.4f} ({out['CV_R2_sd']:.4f})")
