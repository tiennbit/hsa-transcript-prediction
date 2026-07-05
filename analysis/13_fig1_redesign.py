# 13_fig1_redesign.py — Fig 1 (study-design schematic), v4.
# Geometry computed against point sizes so no text can overflow its box:
# canvas 6.85x3.0in; title 8.8pt bold (colored), body 7.9pt; RQ column widened;
# line breaks chosen to fit measured box widths. Titles anchored to box tops,
# body blocks centered in the remaining space. Writes paper/figures/Fig1.png.

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "Arial", "savefig.facecolor": "white"})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "paper", "figures")

fig, ax = plt.subplots(figsize=(6.85, 3.0))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

TITLE_FS, BODY_FS = 8.8, 7.9
LT = 0.047   # title line height (y-units)

def box(x, yc, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle="round,pad=0.008",
                                fc=color, ec="none", alpha=0.09, clip_on=False))
    ax.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle="round,pad=0.008",
                                fc="none", ec=color, lw=1.6, clip_on=False))
    ytop = yc + h / 2
    nt = title.count("\n") + 1
    ax.text(x + w / 2, ytop - 0.035, title, ha="center", va="top",
            fontsize=TITLE_FS, fontweight="bold", color=color, linespacing=1.15)
    title_bottom = ytop - 0.035 - LT * nt
    body_center = (title_bottom - 0.02 + (yc - h / 2) + 0.025) / 2
    ax.text(x + w / 2, body_center, body, ha="center", va="center",
            fontsize=BODY_FS, color="0.25", linespacing=1.4)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.45",
                                shrinkA=0, shrinkB=0))

BLUE, SKY, ORANGE, GREEN, VERM, PINK = ("#0072B2", "#56B4E9", "#E69F00",
                                        "#009E73", "#D55E00", "#CC79A7")
W1, W2, WRQ, W4, W5 = 0.170, 0.170, 0.200, 0.175, 0.148
X1 = 0.004
X2 = X1 + W1 + 0.031
X3 = X2 + W2 + 0.031
X4 = X3 + WRQ + 0.031
X5 = X4 + W4 + 0.031
YM, HM = 0.52, 0.52
YRQ, HRQ = (0.85, 0.52, 0.19), 0.28

box(X1, YM, W1, HM, "Administrative\nrecords",
    "56,882 students\n1,124 province–\nschool units\n49 provinces", BLUE)
box(X2, YM, W2, HM, "De-identification\n+ features",
    "113 pre-exam\nfeatures (transcript\n+ demographics)", SKY)
box(X3, YRQ[0], WRQ, HRQ, "RQ1 Accuracy",
    "5 learners · 5-fold CV\nschool-held-out\nregistration-time", ORANGE)
box(X3, YRQ[1], WRQ, HRQ, "RQ2 Signal",
    "permutation importance\ncorrelations · SHAP", GREEN)
box(X3, YRQ[2], WRQ, HRQ, "RQ3 Equity",
    "OOF signed bias\nschool-clustered CIs\n(attribute-blind)", VERM)
box(X4, YM, W4, HM, "Construct\ncomparison",
    "transcript vs\nnational exam vs\ncombined (post hoc)", PINK)
box(X5, YM, W5, HM, "Implications",
    "counseling\nscore conversion\nequity monitoring", BLUE)

arrow(X1 + W1, YM, X2 - 0.004, YM)
for y in YRQ:
    src = YM + (0.15 if y > YM else -0.15 if y < YM else 0)
    arrow(X2 + W2, src, X3 - 0.004, y)
for y in YRQ:
    dst = YM + (0.15 if y > YM else -0.15 if y < YM else 0)
    arrow(X3 + WRQ, y, X4 - 0.004, dst)
arrow(X4 + W4, YM, X5 - 0.004, YM)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(os.path.join(FIG, "Fig1.png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
plt.close()
print("Fig1 v4 ->", os.path.join(FIG, "Fig1.png"))
