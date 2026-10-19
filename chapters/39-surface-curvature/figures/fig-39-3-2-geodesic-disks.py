"""Figure 39.3.2: small geodesic disks see the Gaussian curvature.

(a) unit sphere: the geodesic disk of radius eps = 1 about the north pole (the polar cap theta <= 1, region color),
one meridian arc of length eps (orange), and for comparison the flat disk of radius 1 in the tangent plane z = 1
(blue outline). The cap's boundary circle has Euclidean radius sin 1 = 0.841 and the cap has area
2 pi (1 - cos 1) = 2.888 < pi (ratio 0.919).
(b) A(eps)/(pi eps^2) for constant curvature K = 1 (unit sphere: 2 pi (1 - cos eps)), K = 0 (plane: pi eps^2) and
K = -1 (hyperbolic plane: 2 pi (cosh eps - 1)), solid, with the two-term approximation 1 - K eps^2/12 dashed.

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-3-2-geodesic-disks.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg39  # noqa: E402

C = dgfig.COLORS
EPS = 1.0
area_cap = 2 * np.pi * (1 - np.cos(EPS))
assert abs(area_cap - 2.888366) < 1e-6 and area_cap < np.pi
eps = np.linspace(1e-3, 2.0, 400)
ratio = {1: 2 * (1 - np.cos(eps)) / eps ** 2, 0: np.ones_like(eps), -1: 2 * (np.cosh(eps) - 1) / eps ** 2}
for Kc in (1, -1):
    e = 0.05
    exact = {1: 2 * np.pi * (1 - np.cos(e)), -1: 2 * np.pi * (np.cosh(e) - 1)}[Kc]
    assert abs(exact - (np.pi * e ** 2 - np.pi / 12 * Kc * e ** 4)) < 1e-9          # Theorem 39.3.5, O(eps^6) here

fig = plt.figure(figsize=(7.2, 3.6))
ax = dgfig.axes3d(fig, 121, elev=18, azim=-60)
U, V = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60), indexing="ij")
S = ddg39.sphere(U, V)
dgfig.surface(ax, *S, alpha=0.22, grid_every=5)
Uc, Vc = np.meshgrid(np.linspace(0, EPS, 12), np.linspace(0, 2 * np.pi, 60), indexing="ij")
cap = ddg39.sphere(Uc, Vc) * 1.003
ax.plot_surface(*cap, color=C["region"], alpha=0.55, linewidth=0, zorder=3, shade=False)
ph = np.linspace(0, 2 * np.pi, 200)
view = dgfig.view_vector(ax)
bd = ddg39.sphere(EPS + 0 * ph, ph).T
dgfig.curve3d(ax, bd, role="tangent", lw=1.4, visible=bd @ view > 0, zorder=6)
fl = np.stack([np.cos(ph), np.sin(ph), np.ones_like(ph)], 1)
ax.plot(*fl.T, color=C["tangent"], lw=1.0, ls=(0, (4, 2)), zorder=7)
tt = np.linspace(0, EPS, 50)
mer = ddg39.sphere(tt, -0.9 + 0 * tt).T
ax.plot(*mer.T, color=C["accent"], lw=2.0, zorder=8)
dgfig.point3d(ax, [0, 0, 1], None, size=14)
ax.text(*(mer[25] + np.array([0.08, -0.05, 0.05])), r"$\varepsilon$", color=C["accent"], fontsize=12)
ax.text(0.15, 0.25, 1.1, r"$p$", fontsize=12)
dgfig.equal_aspect(ax, S.reshape(3, -1).T, fl, zoom=1.3)
ax.set_title("(a)", loc="left", fontsize=12, y=0.96)

ax = fig.add_subplot(122)
cols = {1: C["normal"], 0: C["main"], -1: C["tangent"]}
names = {1: r"$K = 1$", 0: r"$K = 0$", -1: r"$K = -1$"}
for Kc in (1, 0, -1):
    ax.plot(eps, ratio[Kc], color=cols[Kc], lw=1.8, label=names[Kc])
    ax.plot(eps, 1 - Kc * eps ** 2 / 12, color=cols[Kc], lw=1.0, ls=(0, (4, 3)))
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$A(\varepsilon)/\pi\varepsilon^2$")
ax.set_xlim(0, 2)
ax.set_ylim(0.6, 1.45)
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.plot([EPS], [ratio[1][np.argmin(abs(eps - EPS))]], "o", color=C["normal"], ms=4)
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.0, right=0.98, bottom=0.14, top=0.92, wspace=0.18)
dgfig.save(fig, __file__)
print(f"cap area {area_cap:.6f}, ratio at eps=1: {area_cap / np.pi:.4f}")
