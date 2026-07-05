# 10_fig8_cluster.py — Regenerates paper/figures/Fig8.png with SCHOOL-CLUSTERED
# bootstrap 95% CIs (from 08_cluster_bootstrap_full.py -> cluster_bootstrap_full.csv)
# replacing the naive iid/analytic CIs drawn by 04_figures.py. Style matches 04
# (Okabe-Ito palette, Arial, forest plots, 600 dpi). Provinces whose cluster CI
# excludes 0 are highlighted; names of the six Bonferroni-significant provinces
# (|est| > z_{.05/22} * boot SD) are marked with an asterisk.
# Run from the project root:  python3 analysis/10_fig8_cluster.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

cb = pd.read_csv(f"{OUT}/cluster_bootstrap_full.csv")
cb[["kind", "name"]] = cb["quantity"].str.split("::", expand=True)
cb = cb.set_index(["kind", "name"])

def row(kind, name):
    r = cb.loc[(kind, name)]
    return float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"]), bool(r["sig95_cluster"])

fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})

# ---- panel a: gender + zones (cluster CIs, asymmetric) ----
spec = [("Male", ("gender", "Nam")), ("Female", ("gender", "Nữ")),
        ("KV1 (highland)", ("zone", "KV1")), ("KV2-NT (rural)", ("zone", "KV2_NT")),
        ("KV2 (towns)", ("zone", "KV2")), ("KV3 (urban)", ("zone", "KV3"))]
labels = [s[0] for s in spec]
vals, los, his = [], [], []
for _, key in spec:
    e, lo, hi, _ = row(*key)
    vals.append(e); los.append(e - lo); his.append(hi - e)
ypos = np.arange(len(labels))[::-1]
axes[0].errorbar(vals, ypos, xerr=[los, his], fmt="o", color=CB[0], ms=4, capsize=2, lw=1)
axes[0].axvline(0, color="0.3", lw=0.7)
axes[0].set_yticks(ypos, labels)
axes[0].set_xlabel("Signed bias, points (school-clustered 95% CI)")
axes[0].set_title("(a) Gender and priority zone", loc="left", fontsize=9)

# ---- panel b: provinces (cluster CIs; orange = CI excludes 0; * = Bonferroni-sig) ----
prov = cb.loc["prov"].sort_values("estimate")
pnames = [(p.replace("Thành phố ", "").replace("Tỉnh ", "")
           + ("*" if bool(r["bonferroni_sig_cluster"]) else ""))
          for p, r in prov.iterrows()]
est = prov["estimate"].to_numpy()
lo_err = est - prov["ci95_lo"].to_numpy()
hi_err = prov["ci95_hi"].to_numpy() - est
sig = prov["sig95_cluster"].astype(bool).to_numpy()
ypos = np.arange(len(prov))
axes[1].errorbar(est, ypos, xerr=[lo_err, hi_err], fmt="o", ms=3.5, capsize=1.5,
                 lw=0.9, color="0.55", zorder=1)
for i, (b, s) in enumerate(zip(est, sig)):
    axes[1].plot(b, i, "o", color=CB[3] if s else "0.55", ms=4, zorder=2)
axes[1].axvline(0, color="0.3", lw=0.7)
axes[1].set_yticks(ypos, pnames, fontsize=7)
axes[1].set_xlabel("Signed bias, points (school-clustered 95% CI)")
axes[1].set_title("(b) Provinces (n ≥ 200; orange = CI excludes 0; * = Bonferroni-significant)",
                  loc="left", fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/Fig8.png", dpi=600, bbox_inches="tight")
plt.close()
n_sig = int(sig.sum())
n_bonf = int(prov["bonferroni_sig_cluster"].astype(bool).sum())
print(f"Fig8 regenerated with cluster CIs -> {FIG}/Fig8.png "
      f"({n_sig}/22 CI-excludes-0, {n_bonf}/22 Bonferroni-significant)")
