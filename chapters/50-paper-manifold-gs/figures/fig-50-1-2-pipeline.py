"""Figure 50.1.2 (schematic): the pipeline of Manifold-GS as described in [ManifoldGS26] §1, §3–§6.

A trained Gaussian scene is split into geometry atoms (mu_i, P_i, q_i) and rendering parameters (alpha_i, c_i) (eq. (5)).
The geometry atoms form the Gaussian varifold surrogate V_G (Definition 1). Three checks act on it: the mass ledger of
refinement (§4), the realizability diagnostics (§5) and the observation gate (§6). The export is a hybrid bundle (§6).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-1-2-pipeline.py
"""
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

C = dgfig.COLORS
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.set_xlim(-0.15, 10.15)
ax.set_ylim(-0.15, 7.05)
ax.set_axis_off()


def box(x0, y0, w, h, lines, edge, face="white", title=None, fs=10.5):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                                edgecolor=edge, facecolor=face, lw=1.2))
    y = y0 + h - 0.38
    if title:
        ax.text(x0 + w / 2, y, title, ha="center", va="center", fontsize=fs + 0.5, fontweight="bold")
        y -= 0.52
    for ln in lines:
        ax.text(x0 + w / 2, y, ln, ha="center", va="center", fontsize=fs)
        y -= 0.5
    return (x0, y0, w, h)


def arrow(p, q, color=C["main"], rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


box(0.05, 1.9, 2.7, 1.6, [r"$\{(\mu_i, \Sigma_i, \alpha_i, c_i)\}$", "(3DGS)"], C["main"], title="trained scene")
box(3.4, 4.6, 3.0, 2.2, [r"$(\mu_i,\ P_i,\ q_i)$", r"$V_{\mathcal{G}} = \sum_i q_i\,\delta_{(\mu_i, P_i)}$"],
    C["tangent"], face="#EAF3FA", title="geometry")
box(3.4, 0.5, 3.0, 1.6, [r"$(\alpha_i,\ c_i)$"], C["aux"], face="#F2F2F2", title="rendering")
box(7.0, 4.3, 2.95, 2.5, ["mass ledger", "realizability", "observation gate"], C["normal"], face="#FBEDE5",
    title="checks", fs=10)
box(7.0, 0.05, 2.95, 3.65, ["certified patches", "attached Gaussians", "residual Gaussians", "source-ID map",
                            "collision candidate"], C["main"], title="bundle", fs=10)
arrow((2.8, 3.1), (3.35, 5.4))
arrow((2.8, 2.4), (3.35, 1.5))
arrow((6.45, 5.7), (6.95, 5.7))
arrow((8.45, 4.25), (8.45, 3.75))
arrow((6.45, 1.3), (6.95, 1.3))
ax.text(3.25, 2.75, "eq. (5)", fontsize=9.5, color=C["aux"], ha="center", va="center")
dgfig.save(fig, __file__)
