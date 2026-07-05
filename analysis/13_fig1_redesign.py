# 13_fig1_redesign.py — Redesigned Fig 1 (study-design schematic).
# Fixes the cramped layout of the previous version: wider boxes, bold title line
# separated from body text, light color-tinted fills, clear gaps between the RQ
# boxes, and updated contents (school-held-out CV, school-clustered CIs).
# Writes paper/figures/Fig1.png (600 dpi).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "Arial", "savefig.facecolor": "white"})
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "paper", "figures")

fig, ax = plt.subplots(figsize=(6.85, 3.2))
ax.axis("off")

def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.010",
                                fc=color, ec=color, lw=1.5, alpha=0.10,
                                transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.010",
                                fc="none", ec=color, lw=1.5, transform=ax.transAxes))
    n_body = body.count("\n") + 1
    ax.text(x + w / 2, y + h / 2 - 0.075, title, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="0.15", transform=ax.transAxes)
    ax.text(x + w / 2, y + h / 2 - 0.135 - 0.048 * n_body / 2, body, ha="center",
            va="center", fontsize=7.6, color="0.25", linespacing=1.45,
            transform=ax.transAxes)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.45",
                                shrinkA=1, shrinkB=1))

W, H = 0.165, 0.34          # standard box size (mid-row)
HRQ = 0.27                  # RQ box height
X1, X2, X3, X4, X5 = 0.005, 0.215, 0.425, 0.645, 0.845
YMID = 0.52
YRQ = (0.855, 0.52, 0.185)

box(X1, YMID, W, H, "Administrative\nrecords",
    "56,882 students\n1,124 province–school\nunits · 49 provinces", CB[0])
box(X2, YMID, W, H, "De-identification\n+ features",
    "113 pre-exam features\n(transcript +\ndemographics)", CB[5])
box(X3, YRQ[0], W + 0.015, HRQ, "RQ1 Accuracy",
    "5 learners · 5-fold CV\nschool-held-out ·\nregistration-time", CB[1])
box(X3, YRQ[1], W + 0.015, HRQ, "RQ2 Signal",
    "permutation importance\ncorrelations · SHAP", CB[2])
box(X3, YRQ[2], W + 0.015, HRQ, "RQ3 Equity",
    "OOF signed bias\nschool-clustered CIs\n(attribute-blind)", CB[3])
box(X4, YMID, W, H, "Construct\ncomparison",
    "transcript vs national\nexam vs combined\n(post hoc)", CB[4])
box(X5, YMID, W - 0.015, H, "Implications",
    "counseling ·\nscore conversion ·\nequity monitoring", CB[0])

arrow(X1 + W, YMID, X2, YMID)
for y in YRQ:
    arrow(X2 + W, YMID if abs(y - YMID) < 0.01 else YMID + (0.10 if y > YMID else -0.10), X3, y)
for y in YRQ:
    arrow(X3 + W + 0.015, y, X4, YMID if abs(y - YMID) < 0.01 else YMID + (0.10 if y > YMID else -0.10))
arrow(X4 + W, YMID, X5, YMID)

ax.text(0.5, 0.005, "Post-exam graduation scores are excluded from all predictive feature sets (construct analysis only)",
        ha="center", fontsize=7.2, style="italic", color="0.40", transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "Fig1.png"), dpi=600, bbox_inches="tight")
plt.close()
print("Fig1 redesigned ->", os.path.join(FIG, "Fig1.png"))
