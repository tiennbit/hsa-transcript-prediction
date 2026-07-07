# 14_fig1_final.py — Fig 1 (study-design schematic), final version.
# Faithful matplotlib recreation of the authors' redesigned flowchart:
# data cylinder -> feature-engineering container (3 feature groups) -> three model
# variants -> training (5 regressors) -> prediction -> validation & metrics ->
# three RQ panels -> post-hoc construct-comparison band; side note for the
# excluded school/province identifiers.
# Two content corrections vs the draft: "XGB ~ LGBM ~ HistGBM" (not "GBM ~ RF",
# which contradicts Table 2) and holdout R^2 = .42-.43. Arial throughout for
# consistency with Figs 2-8. Writes paper/figures/Fig1.png (600 dpi).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "Arial", "savefig.facecolor": "white"})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "paper", "figures")

FW, FH = 6.85, 4.6
fig, ax = plt.subplots(figsize=(FW, FH))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

GRAY_F, GRAY_E = "#f2f2f2", "#9a9a9a"
BLUE_F, BLUE_E = "#dbe9f6", "#4f81a8"
ORAN_F, ORAN_E = "#fbe3c4", "#c07f3a"
PANE_F, PANE_E = "#d7e6f1", "#5b8db8"
CREAM_F, CREAM_E = "#fdf6ec", "#c8a06a"
INK, SUB = "#222222", "#333333"

def rbox(x0, x1, y0, y1, fc, ec, lw=1.2, ls="-", r=0.006):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, clip_on=False))

def txt(x, y, s, fs=7.8, w="normal", c=INK, ha="center", va="center", style="normal", ls_=1.35):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va,
            style=style, linespacing=ls_)

def arrow(x1, y1, x2, y2, ls="-", c="#444444", lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=c, linestyle=ls,
                                shrinkA=0, shrinkB=0))

def line(x1, y1, x2, y2, ls="-", c="#444444", lw=1.1):
    ax.plot([x1, x2], [y1, y2], ls=ls, c=c, lw=lw, clip_on=False,
            solid_capstyle="round")

# ---------- top row: data cylinder ----------
cx, cw = 0.075, 0.075
cy0, cy1 = 0.845, 0.935
eh = 0.032
ax.add_patch(Rectangle((cx - cw / 2, cy0), cw, cy1 - cy0, fc="#ececec",
                       ec="none", clip_on=False))
ax.add_patch(Ellipse((cx, cy0), cw, eh, fc="#ececec", ec=GRAY_E, lw=1.1, clip_on=False))
line(cx - cw / 2, cy0, cx - cw / 2, cy1, c=GRAY_E)
line(cx + cw / 2, cy0, cx + cw / 2, cy1, c=GRAY_E)
ax.add_patch(Ellipse((cx, cy1), cw, eh, fc="#f7f7f7", ec=GRAY_E, lw=1.1, clip_on=False))
txt(cx, 0.985, "HSA\nRecords", fs=8.2, w="bold")

# ---------- top row: feature-engineering container ----------
fe0, fe1 = 0.205, 0.625
rbox(fe0, fe1, 0.775, 0.995, GRAY_F, GRAY_E, lw=1.2)
txt((fe0 + fe1) / 2, 0.965, "Feature Engineering — 113 pre-exam features", fs=8.8, w="bold")
sub_w, gap = 0.128, 0.010
sx = fe0 + 0.012
subs = [("Transcript grades\nG10–G12 · 9 subjects\nsemester & full-year"),
        ("GPA & ratings\nacademic standing\n& conduct (ordinal)"),
        ("Demographics\ngender · priority\nzone KV1–KV3")]
for i, s in enumerate(subs):
    x0 = sx + i * (sub_w + gap)
    rbox(x0, x0 + sub_w, 0.79, 0.925, BLUE_F, BLUE_E, lw=1.1, r=0.003)
    txt(x0 + sub_w / 2, 0.8575, s, fs=7.0, c=SUB)
arrow(cx + cw / 2 + 0.006, 0.89, fe0 - 0.006, 0.89)

# ---------- top row: excluded-note (dashed) ----------
rbox(0.70, 0.975, 0.86, 0.965, "#fafafa", GRAY_E, lw=1.1, ls=(0, (4, 3)))
txt(0.8375, 0.9125, "Excluded from predictors:\nschool & province identifiers\n(retained as audit dimensions)",
    fs=7.3, c=SUB)
line(fe1 + 0.004, 0.9125, 0.696, 0.9125, ls=(0, (3, 3)), c=GRAY_E)

# ---------- model-variant row ----------
MY0, MY1 = 0.625, 0.705
mods = [(0.045, 0.315, "Primary model", "113 features"),
        (0.365, 0.635, "Registration-time model", "89 features (G10–11 + G12 S1)"),
        (0.685, 0.955, "Attribute-blind model", "excl. gender & priority zone")]
for x0, x1, t, b in mods:
    rbox(x0, x1, MY0, MY1, ORAN_F, ORAN_E, lw=1.2)
    txt((x0 + x1) / 2, (MY0 + MY1) / 2 + 0.018, t, fs=8.0, w="bold")
    txt((x0 + x1) / 2, (MY0 + MY1) / 2 - 0.020, b, fs=7.5, c=SUB)
fe_cx = (fe0 + fe1) / 2
line(fe_cx, 0.775, fe_cx, 0.745)
line(0.18, 0.745, 0.82, 0.745)
for x0, x1, _, _ in mods:
    arrow((x0 + x1) / 2, 0.745, (x0 + x1) / 2, MY1 + 0.005)

# ---------- training -> prediction -> validation row ----------
TY0, TY1 = 0.475, 0.585
rbox(0.095, 0.38, TY0, TY1, GRAY_F, GRAY_E)
txt(0.2375, TY1 - 0.022, "Model training — five regressors", fs=7.9, w="bold")
txt(0.2375, (TY0 + TY1) / 2 - 0.022, "Ridge · Random Forest · HistGBM\nXGBoost · LightGBM", fs=7.5, c=SUB)
rbox(0.415, 0.615, TY0 + 0.008, TY1 - 0.008, ORAN_F, ORAN_E)
txt(0.515, (TY0 + TY1) / 2, "Predict HSA total\ncomponents P1 / P2 / P3", fs=7.3, w="bold")
rbox(0.645, 0.93, TY0, TY1, GRAY_F, GRAY_E)
txt(0.7875, TY1 - 0.018, "Validation & metrics", fs=7.9, w="bold")
txt(0.7875, (TY0 + TY1) / 2 - 0.020,
    "5-fold CV · 80/20 holdout\nschool-held-out CV · OOF\nMAE · RMSE · R²", fs=7.1, c=SUB, ls_=1.25)
for x0, x1, _, _ in mods:
    line((x0 + x1) / 2, MY0, (x0 + x1) / 2, 0.605)
line(0.18, 0.605, 0.82, 0.605)
arrow(0.2375, 0.605, 0.2375, TY1 + 0.005)
arrow(0.384, 0.53, 0.411, 0.53)
arrow(0.619, 0.53, 0.641, 0.53)

# ---------- RQ panels ----------
PY0, PY1 = 0.075, 0.415
panels = [
    (0.025, 0.320, "RQ1 · Predictive accuracy",
     "Compare five regressors\nacross validation schemes",
     "XGB ≈ LGBM ≈ HistGBM\nholdout R² = .42–.43 · MAE ≈ 8.2\n(+.05 R² vs ridge)",
     "→ Transcripts explain ≈ 42% of variance"),
    (0.3525, 0.6475, "RQ2 · Signal structure",
     "Permutation importance ·\nSHAP (LightGBM) ·\ngrade–HSA correlations",
     "Rank subjects & school\nyears by predictive signal",
     "→ Which grades drive HSA scores"),
    (0.680, 0.975, "RQ3 · Equity audit",
     "Slicing analysis · signed bias ·\ncluster bootstrap · ICC\n(OOF predictions)",
     "Group error & differential\nprediction: gender · zone · province",
     "→ Where the model over-/under-predicts"),
]
for x0, x1, title, white, orange, take in panels:
    rbox(x0, x1, PY0, PY1, PANE_F, PANE_E, lw=1.3, ls=(0, (5, 3)))
    cxp = (x0 + x1) / 2
    txt(cxp, PY1 - 0.028, title, fs=8.8, w="bold")
    rbox(x0 + 0.018, x1 - 0.018, 0.272, 0.362, "white", "#8aa5b8", lw=1.0, r=0.003)
    txt(cxp, 0.317, white, fs=7.3, c=SUB)
    rbox(x0 + 0.018, x1 - 0.018, 0.135, 0.245, ORAN_F, ORAN_E, lw=1.1, r=0.003)
    txt(cxp, 0.190, orange, fs=7.3, c=SUB)
    txt(cxp, 0.104, take, fs=7.1, style="italic", c="#1f3d5c")
    arrow(cxp, 0.45, cxp, PY1 + 0.005)
line(0.7875, TY0, 0.7875, 0.45)
line(0.1725, 0.45, 0.8275, 0.45)

# ---------- bottom band ----------
rbox(0.025, 0.975, -0.005, 0.048, CREAM_F, CREAM_E, lw=1.2, ls=(0, (5, 3)))
txt(0.5, 0.0215,
    "Post-hoc construct comparison — HSA vs subsequent national graduation-exam scores (excluded from all predictive feature sets)",
    fs=7.6, c="#6b5432")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(os.path.join(FIG, "Fig1.png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
plt.close()
print("Fig1 final ->", os.path.join(FIG, "Fig1.png"))
