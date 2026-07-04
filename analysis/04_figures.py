# 04_figures.py — Publication figures for EAIT submission (Springer artwork standards)
# Naming: paper/figures/Fig1..Fig8 (.png, RGB). Fonts: Arial 8-9pt. Colorblind-safe palette.
# Line/combination art 600 dpi; dense raster (hexbin, SHAP) 300 dpi.

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
})
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]  # Okabe-Ito
FIG = "paper/figures"
OUT = "analysis/outputs"
os.makedirs(FIG, exist_ok=True)

HOC_LUC = {"Giỏi": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1, "Kém": 0}
HANH_KIEM = {"Tốt": 4, "Khá": 3, "Trung bình": 2, "Yếu": 1}
d = pd.read_csv("analysis/data_deid.csv")
y = d["Điểm HSA"].astype(float)
oof = np.load(f"{OUT}/rev_oof_full.npy")

# ---------- Fig 1: study design schematic (double column) ----------
fig, ax = plt.subplots(figsize=(6.85, 2.6))
ax.axis("off")
boxes = [
    (0.01, 0.55, 0.17, "Administrative records\n56,882 students\n987 schools, 49 provinces", CB[0]),
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
    ax.text(x + w / 2, yy, text, ha="center", va="center", fontsize=7.2, transform=ax.transAxes)
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
        ha="center", fontsize=7, style="italic", color="0.35", transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig1.png", dpi=600, bbox_inches="tight"); plt.close()
print("Fig1 done")

# ---------- Fig 2: outcome distributions (double column) ----------
fig, axes = plt.subplots(1, 4, figsize=(6.85, 1.9))
specs = [("Điểm HSA", "HSA total (/150)", CB[0]),
         ("Điểm HSA P1", "P1 quantitative (/50)", CB[1]),
         ("Điểm HSA P2", "P2 qualitative (/50)", CB[2]),
         ("Điểm HSA P3", "P3 science (/50)", CB[3])]
for ax, (col, label, c) in zip(axes, specs):
    v = d[col].astype(float)
    ax.hist(v, bins=40, color=c, alpha=0.85, edgecolor="white", linewidth=0.2)
    ax.axvline(v.mean(), color="0.2", lw=0.8, ls="--")
    ax.set_xlabel(label); ax.set_yticks([])
    ax.text(0.97, 0.92, f"M={v.mean():.1f}\nSD={v.std():.1f}", ha="right", va="top",
            transform=ax.transAxes, fontsize=7)
axes[0].set_ylabel("Students")
plt.tight_layout()
plt.savefig(f"{FIG}/Fig2.png", dpi=600, bbox_inches="tight"); plt.close()
print("Fig2 done")

# ---------- Fig 3: observed vs predicted, OOF full sample (single column) ----------
fig, ax = plt.subplots(figsize=(3.3, 3.1))
hb = ax.hexbin(y, oof, gridsize=48, cmap="Blues", mincnt=1, linewidths=0)
lims = [25, 132]
ax.plot(lims, lims, color=CB[3], ls="--", lw=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Observed HSA score")
ax.set_ylabel("Predicted HSA score (out-of-fold)")
cb = fig.colorbar(hb, ax=ax, shrink=0.85); cb.set_label("Students", fontsize=8)
ax.text(0.03, 0.95, f"N = 56,882\nR² = .43,  MAE = 8.1", transform=ax.transAxes,
        va="top", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig3.png", dpi=300, bbox_inches="tight"); plt.close()
print("Fig3 done")

# ---------- Fig 4: grade inflation/compression by year (double column) ----------
from scipy.stats import gaussian_kde
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.4))
for ax, (sub, label) in zip(axes, [("Điểm tổng kết", "Overall GPA"), ("Toán", "Mathematics")]):
    for i, yr in enumerate(("10", "11", "12")):
        v = d[f"{yr}.{sub} CN"].dropna().astype(float)
        xs = np.linspace(5.5, 10, 300)
        ax.plot(xs, gaussian_kde(v)(xs), color=CB[i], lw=1.4,
                label=f"Grade {yr}: M={v.mean():.2f}, SD={v.std():.2f}")
    ax.set_xlabel(f"{label}, full-year mark"); ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig4.png", dpi=600, bbox_inches="tight"); plt.close()
print("Fig4 done")

# ---------- Fig 5: correlation heatmap subject × year (single column) ----------
corr = pd.read_csv(f"{OUT}/correlations_subject_year.csv", index_col=0)
name_en = {"Toán": "Mathematics", "Vật lí": "Physics", "Điểm tổng kết": "Overall GPA",
           "Hóa học": "Chemistry", "Sinh học": "Biology", "Ngoại ngữ": "Foreign language",
           "Địa lí": "Geography", "Lịch sử": "History", "GDCD": "Civic education", "Văn": "Literature"}
corr.index = [name_en.get(i, i) for i in corr.index]
corr = corr.sort_values("11", ascending=False)
fig, ax = plt.subplots(figsize=(3.3, 3.0))
im = ax.imshow(corr.values, cmap="YlGnBu", vmin=0.1, vmax=0.6, aspect="auto")
ax.set_xticks(range(3), [f"Grade {c}" for c in corr.columns])
ax.set_yticks(range(len(corr)), corr.index)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                fontsize=7, color="white" if v > 0.42 else "0.2")
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Pearson r with HSA total", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig5.png", dpi=600, bbox_inches="tight"); plt.close()
print("Fig5 done")

# ---------- Fig 6: permutation importance (double column, 2 panels) ----------
impf = pd.read_csv(f"{OUT}/importance_by_feature.csv").head(15).iloc[::-1]
imps = pd.read_csv(f"{OUT}/importance_by_subject.csv", index_col=0)["importance"].sort_values().tail(9)
sub_en = {"Toán": "Mathematics", "Ngoại ngữ": "Foreign language", "Hóa học": "Chemistry",
          "Tổng kết (GPA)": "Overall GPA", "Nhân khẩu": "Demographics", "Văn": "Literature",
          "Vật lí": "Physics", "Sinh học": "Biology", "Lịch sử": "History", "Địa lí": "Geography",
          "GDCD": "Civic education", "Học lực": "Academic standing", "Hạnh kiểm": "Conduct"}
def feat_en(f):
    f = f.replace("Điểm tổng kết", "GPA").replace("Toán", "Math").replace("Ngoại ngữ", "Foreign lang.") \
         .replace("Hóa học", "Chemistry").replace("Vật lí", "Physics").replace("Văn", "Literature") \
         .replace("Sinh học", "Biology").replace("HK I", "S1").replace("HK II", "S2").replace("CN", "year")
    f = re.sub(r"^(\d+)\.", r"G\1 ", f)
    return {"gioi_tinh": "Gender", "kv_KV2_NT": "Zone KV2-NT", "kv_KV1": "Zone KV1"}.get(f, f)
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.8))
axes[0].barh([feat_en(f) for f in impf["feature"]], impf["importance"],
             xerr=impf["importance_sd"], color=CB[0], error_kw=dict(lw=0.6))
axes[0].set_xlabel("Permutation importance (ΔRMSE)")
axes[0].set_title("(a) Top-15 features", loc="left", fontsize=9)
axes[1].barh([sub_en.get(i, i) for i in imps.index], imps.values, color=CB[2])
axes[1].set_xlabel("Summed permutation importance (ΔRMSE)")
axes[1].set_title("(b) Aggregated by subject", loc="left", fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig6.png", dpi=600, bbox_inches="tight"); plt.close()
print("Fig6 done")

# ---------- Fig 7: SHAP beeswarm (double column) ----------
import shap
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
dd = d.copy()
for col in dd.columns:
    if "Học lực" in col: dd[col] = dd[col].map(HOC_LUC)
    elif "Hạnh kiểm" in col: dd[col] = dd[col].map(HANH_KIEM)
dd["gioi_tinh"] = (dd["Giới tính"] == "Nam").astype(int)
dd = pd.concat([dd, pd.get_dummies(dd["khuVuc"], prefix="kv")], axis=1)
tcols = [c for c in dd.columns if c[:3] in ("10.", "11.", "12.") and pd.api.types.is_numeric_dtype(dd[c])]
fcols = tcols + ["gioi_tinh"] + [c for c in dd.columns if c.startswith("kv_")]
X = dd[fcols].astype(float)
X.columns = [feat_en(c) for c in X.columns]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8,
                     colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1).fit(X_tr, y_tr)
rng = np.random.RandomState(0)
idx = rng.choice(len(X_te), 3000, replace=False)
sv = shap.TreeExplainer(lgbm).shap_values(X_te.iloc[idx])
plt.figure()
shap.summary_plot(sv, X_te.iloc[idx], max_display=15, show=False, plot_size=(6.85, 3.6))
plt.gcf().axes[-1].set_ylabel("Feature value", fontsize=8)
plt.xlabel("SHAP value (impact on predicted HSA score)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig7.png", dpi=300, bbox_inches="tight"); plt.close()
print("Fig7 done")

# ---------- Fig 8: equity forest plots (double column) ----------
gen = pd.read_csv(f"{OUT}/rev_audit_nopro_gender.csv", index_col=0)
reg = pd.read_csv(f"{OUT}/rev_audit_nopro_region.csv", index_col=0)
prov = pd.read_csv(f"{OUT}/rev_audit_nopro_province.csv", index_col=0).sort_values("bias")
fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
# panel a: gender + zones
labels = ["Male", "Female", "KV1 (highland)", "KV2-NT (rural)", "KV2 (towns)", "KV3 (urban)"]
vals = [gen.loc["Nam", "bias"], gen.loc["Nữ", "bias"], reg.loc["KV1", "bias"],
        reg.loc["KV2_NT", "bias"], reg.loc["KV2", "bias"], reg.loc["KV3", "bias"]]
cis = [gen.loc["Nam", "bias_ci95"], gen.loc["Nữ", "bias_ci95"], reg.loc["KV1", "bias_ci95"],
       reg.loc["KV2_NT", "bias_ci95"], reg.loc["KV2", "bias_ci95"], reg.loc["KV3", "bias_ci95"]]
ypos = np.arange(len(labels))[::-1]
axes[0].errorbar(vals, ypos, xerr=cis, fmt="o", color=CB[0], ms=4, capsize=2, lw=1)
axes[0].axvline(0, color="0.3", lw=0.7)
axes[0].set_yticks(ypos, labels)
axes[0].set_xlabel("Signed bias, points (± 95% CI)")
axes[0].set_title("(a) Gender and priority zone", loc="left", fontsize=9)
# panel b: provinces
pnames = [p.replace("Thành phố ", "").replace("Tỉnh ", "") for p in prov.index]
ypos = np.arange(len(prov))
colors = [CB[3] if abs(b) - ci > 0 else "0.55" for b, ci in zip(prov["bias"], prov["bias_ci95"])]
axes[1].errorbar(prov["bias"], ypos, xerr=prov["bias_ci95"], fmt="o", ms=3.5, capsize=1.5,
                 lw=0.9, color="0.55", zorder=1)
for i, (b, ci, c) in enumerate(zip(prov["bias"], prov["bias_ci95"], colors)):
    axes[1].plot(b, i, "o", color=c, ms=4, zorder=2)
axes[1].axvline(0, color="0.3", lw=0.7)
axes[1].set_yticks(ypos, pnames, fontsize=7)
axes[1].set_xlabel("Signed bias, points (± 95% CI)")
axes[1].set_title("(b) Provinces (n ≥ 200; red = 95% CI excludes 0)", loc="left", fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig8.png", dpi=600, bbox_inches="tight"); plt.close()
print("Fig8 done\nAll figures in", FIG)
