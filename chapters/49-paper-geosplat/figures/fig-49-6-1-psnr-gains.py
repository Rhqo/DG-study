"""Figure 49.6.1: GeoSplat minus GeoGaussian, scene by scene, from [GeoSplat25, Tables 1-2] (Section 49.6).

No new experiment: the numbers are the printed table values (lib/geo49.py, TABLE1 and TABLE2).
(a) PSNR difference (dB) for the 8 Replica scenes and the 4 ICL scenes, both GeoSplat variants, with the mean of each
    dataset as a horizontal segment.
(b) SSIM difference (in units of 0.001) for the same cells; a negative bar is a cell where GeoSplat is below GeoGaussian.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-6-1-psnr-gains.py
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import geo49  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
REP = ["R0", "R1", "R2", "OFF0", "OFF1", "OFF2", "OFF3", "OFF4"]
scenes = REP + geo49.ICL
gain = {}
for v in ("Manifold", "Varifold"):
    gain[v] = {}
    for met in ("PSNR", "SSIM"):
        a = np.array(geo49.TABLE1[v][met]) - np.array(geo49.TABLE1["GeoGaussian"][met])
        b = np.array(geo49.TABLE2[v][met]) - np.array(geo49.TABLE2["GeoGaussian"][met])
        gain[v][met] = np.r_[a, b]
for v in gain:
    print(v, "PSNR", np.round(gain[v]["PSNR"], 2), "SSIM", np.round(1000 * gain[v]["SSIM"]))
mean_rep = {v: gain[v]["PSNR"][:8].mean() for v in gain}
mean_icl = {v: gain[v]["PSNR"][8:].mean() for v in gain}
print("means", mean_rep, mean_icl)
assert np.all(gain["Manifold"]["PSNR"] > 0) and np.all(gain["Varifold"]["PSNR"] > 0)
assert abs(mean_rep["Manifold"] - 0.78) < 0.005 and abs(mean_icl["Manifold"] - 0.38) < 0.005
assert np.isclose(gain["Varifold"]["SSIM"][1], -0.004) and np.sum(gain["Varifold"]["SSIM"] < 0) == 1
assert np.sum(gain["Manifold"]["SSIM"] < 0) == 0

x = np.r_[np.arange(8), np.arange(4) + 9.0]
w = 0.38
fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
styles = {"Manifold": (C["main"], "manifold-based GeoSplat"), "Varifold": (C["tangent"], "varifold-based GeoSplat")}
for k, (met, scale, ylab) in enumerate([("PSNR", 1, "PSNR difference (dB)"), ("SSIM", 1000, r"SSIM difference ($\times 10^{-3}$)")]):
    ax = axes[k]
    for j, (v, (col, lab)) in enumerate(styles.items()):
        ax.bar(x + (j - 0.5) * w, scale * gain[v][met], width=w, color=col, alpha=0.85 if j == 0 else 0.75, label=lab)
    ax.axhline(0, color=C["aux"], lw=0.8)
    ax.axvline(8.0, color=C["aux"], lw=0.8, ls=":")
    ax.set_ylabel(ylab)
    if met == "PSNR":
        for v, (col, _) in styles.items():
            ls = "--" if v == "Manifold" else "-."
            ax.plot([-0.5, 7.5], [mean_rep[v]] * 2, color=col, lw=1.0, ls=ls)
            ax.plot([8.5, 12.5], [mean_icl[v]] * 2, color=col, lw=1.0, ls=ls)
        ax.text(3.5, 1.45, "Replica (Table 1)", ha="center", fontsize=9)
        ax.text(10.5, 1.45, "ICL (Table 2)", ha="center", fontsize=9)
        ax.set_ylim(0, 1.6)
        ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.9))
        ax.set_title("(a) GeoSplat minus GeoGaussian, PSNR (dashed: mean per dataset)", fontsize=10)
    else:
        ax.set_ylim(-6, 27)
        ax.set_title("(b) GeoSplat minus GeoGaussian, SSIM", fontsize=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenes, rotation=45, ha="right", fontsize=8.5)
fig.tight_layout()
dgfig.save(fig, __file__)
