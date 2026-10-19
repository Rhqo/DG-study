"""Figure 39.4.1: Gauss-Bonnet for polar caps of the unit sphere.

Unit sphere, outward N. Cap S_0 = {theta <= theta0}, boundary circle traversed with phi increasing (positive:
N x t points into the cap, Example 39.2.6). (a) theta0 = pi/3 (light blue cap) and theta0 = 2 pi/3 (outline), with
t (blue) and N x t (green) at three points of each boundary circle (length 0.35). (b) against theta0:
int K dA = 2 pi (1 - cos theta0) (vermilion), int kappa_g ds = 2 pi cos theta0 (green) and their sum 2 pi (black).

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-4-1-polar-caps.py
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
TH = [np.pi / 3, 2 * np.pi / 3]
# self-checks: kappa_g from the Darboux frame, GB balance from the profile formula
for th0 in TH:
    kap, kn, kg, t, N = ddg39.darboux(ddg39.sphere, lambda s: (th0, s), 0.3)
    assert abs(kg - 1 / np.tan(th0)) < 1e-6
    A, B = ddg39.cap_terms(lambda t_: (np.sin(t_), np.cos(t_)), th0)
    assert abs(A - 2 * np.pi * (1 - np.cos(th0))) < 1e-5 and abs(B - 2 * np.pi * np.cos(th0)) < 1e-6
    assert abs(A + B - 2 * np.pi) < 1e-5

fig = plt.figure(figsize=(7.2, 3.6))
ax = dgfig.axes3d(fig, 121, elev=34, azim=-55)
U, V = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60), indexing="ij")
S = ddg39.sphere(U, V)
dgfig.surface(ax, *S, alpha=0.2, grid_every=5)
Uc, Vc = np.meshgrid(np.linspace(0, TH[0], 10), np.linspace(0, 2 * np.pi, 60), indexing="ij")
ax.plot_surface(*(ddg39.sphere(Uc, Vc) * 1.004), color=C["region"], alpha=0.5, linewidth=0, shade=False, zorder=3)
view = dgfig.view_vector(ax)
ph = np.linspace(0, 2 * np.pi, 300)
for th0, lw in zip(TH, (1.6, 1.6)):
    bd = ddg39.sphere(th0 + 0 * ph, ph).T
    dgfig.curve3d(ax, bd, lw=lw, visible=bd @ view > 0, zorder=6)
    for phk in (-0.9, -0.2, 0.5):
        p = ddg39.sphere(th0, phk)
        if p @ view < 0:
            continue
        t = np.array([-np.sin(phk), np.cos(phk), 0.0])
        b = np.cross(p, t)
        dgfig.arrow3d(ax, p, 0.35 * t, role="tangent", head=8, lw=1.3, zorder=9)
        dgfig.arrow3d(ax, p, 0.35 * b, role="third", head=8, lw=1.3, zorder=9)
ax.text(*(ddg39.sphere(TH[0], 0.5) + np.array([0.05, 0.1, 0.12])), r"$\theta_0 = \pi/3$", fontsize=11)
ax.text(*(ddg39.sphere(TH[1], 0.5) + np.array([0.1, 0.1, -0.3])), r"$\theta_0 = 2\pi/3$", fontsize=11)
dgfig.equal_aspect(ax, S.reshape(3, -1).T, zoom=1.25)
ax.set_title("(a)", loc="left", fontsize=12, y=0.96)

ax = fig.add_subplot(122)
th = np.linspace(0, np.pi, 300)
ax.axhline(0, color=C["aux"], lw=0.7)
ax.plot(th, 2 * np.pi * (1 - np.cos(th)), color=C["normal"], lw=1.8, label=r"$\int_S K\,dA$")
ax.plot(th, 2 * np.pi * np.cos(th), color=C["third"], lw=1.8, label=r"$\int_{\partial S}\kappa_g\,ds$")
ax.plot(th, 2 * np.pi + 0 * th, color=C["main"], lw=1.8, label=r"sum $= 2\pi$")
for th0 in TH:
    ax.axvline(th0, color=C["aux"], lw=0.7, ls=(0, (3, 2)))
ax.set_xticks([0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
ax.set_xticklabels([r"$0$", r"$\pi/3$", r"$\pi/2$", r"$2\pi/3$", r"$\pi$"])
ax.set_yticks([-2 * np.pi, 0, 2 * np.pi, 4 * np.pi])
ax.set_yticklabels([r"$-2\pi$", r"$0$", r"$2\pi$", r"$4\pi$"])
ax.set_xlabel(r"$\theta_0$")
ax.set_xlim(0, np.pi)
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.0, right=0.98, bottom=0.14, top=0.92, wspace=0.12)
dgfig.save(fig, __file__)
