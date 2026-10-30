"""Figure 49.1.2: the GeoSplat pipeline (schematic).

Five boxes: SfM points -> geometry estimation (manifold App. E / varifold App. D) -> initialization (warm-up (5),
upsampling (6)) -> training (loss + L_scale (8) + L_rot (9), truncated update (7)) -> densification (split (10),
clone (11)), and a return arrow "re-estimate at evenly spaced iterations" (Section 3.2 of [GeoSplat25]).
The equation numbers are those of the paper.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-1-2-pipeline.py
"""
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

C = dgfig.COLORS

boxes = {  # name: (x, y, w, h, title, lines, edge colour)
    "sfm": (0.03, 0.66, 0.27, 0.27, "SfM points", [r"centers $\boldsymbol{\mu}^i$", "(COLMAP)"], C["main"]),
    "est": (0.35, 0.66, 0.33, 0.27, "estimate", [r"$\tilde{\mathbf{u}}^i_3$, $(\tau^i_d, \tilde{\mathbf{w}}^i_d)$",
                                                          "manifold / varifold"], C["normal"]),
    "init": (0.72, 0.66, 0.26, 0.27, "initialize", ["warm-up (5)", "upsample (6)"], C["tangent"]),
    "train": (0.60, 0.10, 0.38, 0.30, "train", ["loss + (8) + (9)", "truncated step (7)"], C["tangent"]),
    "dens": (0.12, 0.10, 0.36, 0.30, "densify", ["split (10)", "clone (11)"], C["tangent"]),
}

fig, ax = plt.subplots(figsize=(6.2, 3.9))
for key, (x, y, w, h, title, lines, ec) in boxes.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc="white", ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=ec)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.14 - 0.075 * k, line, ha="center", va="center", fontsize=10.5)


def arrow(p, q, color=C["main"], rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


arrow((0.30, 0.795), (0.35, 0.795))
arrow((0.68, 0.795), (0.72, 0.795))
arrow((0.85, 0.66), (0.85, 0.40))
arrow((0.60, 0.25), (0.48, 0.25))
# densify -> train (the loop of 3DGS)
arrow((0.36, 0.10), (0.72, 0.10), rad=0.35, color=C["aux"])
# re-estimation
arrow((0.30, 0.40), (0.47, 0.66), color=C["normal"], ls="--")
ax.text(0.205, 0.53, "re-estimate at\nevenly spaced\niterations", ha="center", va="center", fontsize=9.5, color=C["normal"])
ax.text(0.54, 0.005, "iterate", ha="center", va="bottom", fontsize=9.5, color=C["aux"])

assert len(boxes) == 5
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 0.96)
ax.set_axis_off()
dgfig.save(fig, __file__)
