"""Figure 0.6.2: the two stereographic charts of S² and their transition map u ↦ u/|u|² (Section 0.6).

Left: chart σ (from the north pole N), coordinates u. Right: chart σ̃ (from the south pole S), coordinates v.
Both panels show the images of the circles of latitude with colatitude θ = 30°, 60°, 90°, 120°, 150°
(same colour = same circle on S²) and of eight meridians (grey rays).
A circle of latitude with colatitude θ goes to the circle of radius cot(θ/2) under σ and tan(θ/2) under σ̃,
so the product of the two radii is 1. The marked point x has colatitude 60° and longitude 30°:
σ(x) = (3/2, √3/2), σ̃(x) = (1/2, √3/6) = σ(x)/|σ(x)|².
σ(S) = 0 and σ̃(N) = 0 are the origins.

Formulas come from dgsym.stereographic(2) (GUIDELINES §7).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-6-2-transition.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

st = dgsym.stereographic(2)
X, U = st["coords"], st["chart_coords"]
sigma = sp.lambdify(X, list(st["sigma"]), "numpy")
sigma_s = sp.lambdify(X, list(st["sigma_south"]), "numpy")
trans = sp.lambdify(U, list(st["transition"]), "numpy")


def sph(theta, lon):
    return np.array([np.sin(theta) * np.cos(lon), np.sin(theta) * np.sin(lon), np.cos(theta)])


thetas = [30, 60, 90, 120, 150]
cols = [C["accent"], C["tangent"], C["main"], C["third"], C["normal"]]
lon = np.linspace(0, 2 * np.pi, 400)
x = sph(np.deg2rad(60), np.deg2rad(30))
u_x, v_x = np.array(sigma(*x)), np.array(sigma_s(*x))

# self-checks (§12.1)
for th in thetas:
    t = np.deg2rad(th)
    pts = np.array([sph(t, a) for a in lon[::40]])
    ru = np.linalg.norm(np.array(sigma(*pts.T)), axis=0)
    rv = np.linalg.norm(np.array(sigma_s(*pts.T)), axis=0)
    assert np.allclose(ru, 1 / np.tan(t / 2)) and np.allclose(rv, np.tan(t / 2)) and np.allclose(ru * rv, 1)
assert np.allclose(u_x, [1.5, np.sqrt(3) / 2]) and np.allclose(v_x, [0.5, np.sqrt(3) / 6])
assert np.allclose(np.array(trans(*u_x)), v_x)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.3))
R = 4.0
for k, ax in enumerate(axs):
    dgfig.schematic_axes(ax, (-R, R), (-R, R))
    ax.add_patch(plt.Rectangle((-R, -R), 2 * R, 2 * R, facecolor=C["region"], alpha=0.12, lw=0))
    for a in np.deg2rad(np.arange(0, 360, 45)):
        ax.plot([0, 1.6 * R * np.cos(a)], [0, 1.6 * R * np.sin(a)], color=C["aux"], lw=0.6, zorder=1)
    for th, col in zip(thetas, cols):
        t = np.deg2rad(th)
        rad = 1 / np.tan(t / 2) if k == 0 else np.tan(t / 2)
        ax.plot(rad * np.cos(lon), rad * np.sin(lon), color=col, lw=1.4, zorder=3)
    q = u_x if k == 0 else v_x
    ax.plot(*q, "o", color=C["main"], ms=4, zorder=6)
    ax.plot(0, 0, "o", ms=4, mfc="white", mec=C["main"], zorder=6)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

box = dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85)
axs[0].text(u_x[0] + 0.2, u_x[1] + 0.25, r"$\sigma(x)$", fontsize=12, bbox=box, zorder=8)
axs[1].text(v_x[0] + 0.55, v_x[1] + 0.75, r"$\tilde\sigma(x)$", fontsize=12, bbox=box, zorder=8)
axs[1].annotate("", xy=v_x, xytext=(v_x[0] + 0.55, v_x[1] + 0.75),
                arrowprops=dict(arrowstyle="-", color=C["main"], lw=0.6), zorder=7)
axs[0].text(-0.35, -0.55, r"$\sigma(S)$", fontsize=11, ha="right", va="top", bbox=box, zorder=8)
axs[1].text(-0.35, -0.55, r"$\tilde\sigma(N)$", fontsize=11, ha="right", va="top", bbox=box, zorder=8)
axs[0].set_title(r"$u = \sigma(x)$", fontsize=12)
axs[1].set_title(r"$v = \tilde\sigma(x)$", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.03, wspace=0.42)
dgfig.map_arrow(fig, axs[0], axs[1], r"$u \mapsto \dfrac{u}{|u|^2}$", xy_from=(1.0, 0.55), xy_to=(0.0, 0.55),
                rad=-0.45, fontsize=12, label_offset=(0.0, 0.02))
dgfig.save(fig, __file__)
