"""Figure 46.9.1: a map of the four papers (schematic).

Four boxes (paper and year, the question it asks, the chapter that reads it). Solid arrows: the later paper cites
the earlier one and uses the named result. Dashed arrows: no citation, but the later paper works with the same object
(relation by content only). Thin grey lines from the top: all four papers cite Simon's Lectures on Geometric Measure
Theory (1983) for varifolds.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-9-1-map-of-papers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

C = dgfig.COLORS

BOX = {
    "BLM17": (0.0, 2.4, "BLM17  (2017)", "mean curvature of any varifold,\nwith convergence", "Chapter 47"),
    "BLM22": (0.0, 0.0, "BLM22  (2022)", "the whole second fundamental\nform, weakly", "Chapter 48"),
    "GeoSplat": (3.85, 2.4, "GeoSplat  (2025)", "curvature priors for\ntraining 3DGS", "Chapter 49"),
    "ManifoldGS": (3.85, 0.0, "Manifold-GS  (2026)", "Gaussians as certified\ngeometry", "Chapter 50"),
}
W, H = 2.75, 1.15

fig, ax = plt.subplots(figsize=(6.4, 5.0))
for key, (x, y, title, question, chap) in BOX.items():
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor="white",
                                edgecolor=C["main"], lw=1.2))
    ax.text(x + W / 2, y + H - 0.17, title, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(x + W / 2, y + H / 2 - 0.08, question, ha="center", va="center", fontsize=8.5, linespacing=1.25)
    ax.text(x + W - 0.08, y + 0.1, chap, ha="right", va="bottom", fontsize=8, color=C["tangent"])

# Simon 1983 at the top
sx, sy = 3.3, 4.3
ax.add_patch(FancyBboxPatch((sx - 1.35, sy - 0.2), 2.7, 0.4, boxstyle="round,pad=0.03,rounding_size=0.06",
                            facecolor=C["surface"], edgecolor=C["aux"], lw=0.8))
ax.text(sx, sy, "Simon 1983 (GMT lectures)", ha="center", va="center", fontsize=8.5)
ax.text(sx, sy - 0.36, "cited by all four for varifolds", ha="center", va="center", fontsize=7.5, color=C["aux"])


def arrow(p, q, dashed, label, lpos, rot=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.4,
                                 color=C["main"] if not dashed else C["normal"], linestyle="--" if dashed else "-",
                                 shrinkA=2, shrinkB=2, zorder=3))
    ax.text(*lpos, label, fontsize=8, ha=ha, va="center", rotation=rot, color=C["main"] if not dashed else C["normal"],
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=4)


arrow((1.375, 2.4), (1.375, 1.15), False, "mean curvature\n→ full tensor", (0.62, 1.78))
arrow((2.75, 3.15), (3.85, 3.15), False, "varifold,\nsmoothing", (3.3, 3.55))
arrow((2.6, 1.15), (4.0, 2.4), False, "approximate\nWSFF", (4.05, 2.08), ha="left")
arrow((2.6, 2.4), (4.0, 1.15), True, "point cloud varifold,\nBL bound", (4.05, 1.47), ha="left")
arrow((2.75, 0.45), (3.85, 0.45), True, "fundamental-form\ncompatibility", (3.3, -0.3))
arrow((5.85, 2.4), (5.85, 1.15), True, "mass,\nrefinement", (5.95, 1.78), ha="left")

ax.set_xlim(-0.15, 6.85)
ax.set_ylim(-0.65, 4.6)
ax.set_aspect("equal")
ax.set_axis_off()

dgfig.save(fig, __file__)
