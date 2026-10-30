"""Figure 49.4.1: what the varifold estimator (12) of [GeoSplat25] measures -- the tilt of the own normal seen from
a neighbour's tangent plane, per unit distance (Section 49.4).

Cross-section: a circle of radius rc (curvature 1/rc). Point i at angle 0 with outward normal u_i; neighbour j at
angle theta with normal u_j and tangent line T_j. The projection V_j u_i = u_i - <u_j, u_i> u_j of u_i onto T_j has
length sin(theta), and |mu_i - mu_j| = 2 rc sin(theta/2), so the ratio is cos(theta/2)/rc -> 1/rc.
(a) the geometry for rc = 1, theta = 40 degrees. (b) |V_j u_i| / |mu_i - mu_j| against |mu_i - mu_j| for rc = 1, 2.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-4-1-normal-tilt.py
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS


def tilt_ratio(rc, theta):
    ui = np.array([1.0, 0.0])
    uj = np.array([np.cos(theta), np.sin(theta)])
    mi, mj = rc * ui, rc * uj
    proj = ui - (uj @ ui) * uj
    return np.linalg.norm(proj) / np.linalg.norm(mi - mj), np.linalg.norm(mi - mj)


# self-checks: ratio = cos(theta/2)/rc
for rc in (1.0, 2.0):
    for th in (0.1, 0.5, 1.0):
        ratio, dist = tilt_ratio(rc, th)
        assert abs(ratio - np.cos(th / 2) / rc) < 1e-12 and abs(dist - 2 * rc * np.sin(th / 2)) < 1e-12

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
t = np.linspace(-0.35, 1.25, 200)
ax.plot(np.cos(t), np.sin(t), color=C["main"], lw=1.6)
th = np.radians(40)
mi, mj = np.array([1.0, 0]), np.array([np.cos(th), np.sin(th)])
ui, uj = mi.copy(), mj.copy()
tj = np.array([-np.sin(th), np.cos(th)])
ax.plot(*np.stack([mj - 0.75 * tj, mj + 0.55 * tj]).T, color=C["tangent"], lw=1.0)
ax.text(*(mj + 0.58 * tj + np.array([0.02, 0.0])), r"$T_j$", color=C["tangent"], fontsize=11)
for p, v, lab, off in ((mi, ui, r"$\tilde{\mathbf{u}}^i_3$", (0.05, -0.08)), (mj, uj, r"$\tilde{\mathbf{u}}^j_3$", (0.04, 0.04))):
    ax.annotate("", xy=p + 0.5 * v, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.5))
    ax.text(*(p + 0.5 * v + np.array(off)), lab, color=C["normal"], fontsize=11)
proj = ui - (uj @ ui) * uj
ax.annotate("", xy=mj + 0.7 * proj, xytext=mj, arrowprops=dict(arrowstyle="-|>", color=C["third"], lw=1.8))
ax.text(*(mj + 0.7 * proj + np.array([0.06, 0.0])), r"$\tilde{\mathbf{V}}^j\tilde{\mathbf{u}}^i_3$", color=C["third"],
        fontsize=11)
ax.plot([mi[0], mj[0]], [mi[1], mj[1]], color=C["aux"], ls="--", lw=1.0)
ax.text(0.83, 0.12, r"$\boldsymbol{\mu}^i - \boldsymbol{\mu}^j$", color=C["aux"], fontsize=9.5, ha="right")
for p, lab, off in ((mi, r"$\boldsymbol{\mu}^i$", (0.04, -0.12)), (mj, r"$\boldsymbol{\mu}^j$", (-0.2, -0.02))):
    ax.plot(*p, "o", color=C["main"], ms=4)
    ax.text(*(p + np.array(off)), lab, fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(0.0, 1.75)
ax.set_ylim(-0.4, 1.3)
ax.set_axis_off()
ax.set_title(r"(a) neighbour $j$ sees $\tilde{\mathbf{u}}^i_3$ tilted", fontsize=10.5)

for rc, ls in ((1.0, "-"), (2.0, "--")):
    ths = np.linspace(0.01, 1.2 / rc, 200)
    vals = np.array([tilt_ratio(rc, a) for a in ths])
    ax2.plot(vals[:, 1], vals[:, 0], color=C["third"], ls=ls, lw=1.8, label=f"circle of radius {rc:g}")
    ax2.axhline(1 / rc, color=C["aux"], lw=0.8, ls=":")
ax2.text(1.0, 1.02, r"$1/r = 1$", color=C["aux"], fontsize=9)
ax2.text(1.0, 0.52, r"$1/r = 0.5$", color=C["aux"], fontsize=9)
ax2.set_xlim(0, 1.2)
ax2.set_ylim(0, 1.15)
ax2.set_xlabel(r"distance $\|\boldsymbol{\mu}^i - \boldsymbol{\mu}^j\|$")
ax2.set_ylabel(r"tilt per distance $\|\tilde{\mathbf{V}}^j\tilde{\mathbf{u}}^i_3\| / \|\boldsymbol{\mu}^i - \boldsymbol{\mu}^j\|$",
               fontsize=9)
ax2.legend(loc="lower left", fontsize=8.5, frameon=False)
ax2.set_title("(b) the ratio tends to the curvature", fontsize=10.5)
fig.tight_layout()
dgfig.save(fig, __file__)
