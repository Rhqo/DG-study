"""Figure 39.1.1: the curvature of a plane curve is the speed at which its normal turns.

Ellipse gamma(t) = (2 cos t, sin t) (running example), counterclockwise. 20 points equally spaced in arc length
(starting at (2, 0)). (a) the curve with the signed unit normals n_s = J t (vermilion, length 0.35).
(b) the same normals drawn from the origin: their tips on the unit circle S^1 (the Gauss map of the curve).
Consecutive tips are kappa_s ds apart (to first order): far apart near (+-2, 0) where kappa = 2, crowded near
(0, +-1) where kappa = 1/4.

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-1-1-ellipse-normals.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
M = 20

t = np.linspace(0, 2 * np.pi, 20001)
gam = np.stack([2 * np.cos(t), np.sin(t)], 1)
d = np.stack([-2 * np.sin(t), np.cos(t)], 1)
speed = np.linalg.norm(d, axis=1)
s = np.concatenate([[0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))])
Ltot = s[-1]
ts = np.interp(np.arange(M) * Ltot / M, s, t)
P = np.stack([2 * np.cos(ts), np.sin(ts)], 1)
T = np.stack([-2 * np.sin(ts), np.cos(ts)], 1)
T /= np.linalg.norm(T, axis=1, keepdims=True)
Nn = np.stack([-T[:, 1], T[:, 0]], 1)          # n_s = J t
kappa = 2 / (4 * np.sin(ts) ** 2 + np.cos(ts) ** 2) ** 1.5

# self-checks
assert abs(Ltot - 9.688448220547676) < 1e-6                 # perimeter of the ellipse (2, 1)
assert np.allclose(np.einsum("ij,ij->i", T, Nn), 0)
assert abs(kappa[0] - 2) < 1e-12 and abs(2 / 8 - 2 / (4 + 0) ** 1.5) < 1e-12
ang = np.unwrap(np.arctan2(Nn[:, 1], Nn[:, 0]))
dphi = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
assert abs(dphi.sum() - 2 * np.pi) < 1e-9                    # the normal turns once
ds = Ltot / M
# spacing of tips: largest next to (2, 0), smallest next to (0, 1)
assert abs(dphi[0] - 0.7621) < 1e-3 and abs(dphi.min() - 0.1240) < 1e-3 and np.argmin(dphi) == 4

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axes[0]
ax.plot(gam[:, 0], gam[:, 1], color=C["main"], lw=dgfig.LW["main"])
for p, n in zip(P, Nn):
    ax.annotate("", xy=p + 0.35 * n, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.2,
                                                              mutation_scale=9, shrinkA=0, shrinkB=0))
ax.plot(*P.T, "o", color=C["main"], ms=2.6)
ax.annotate(r"$\kappa_s = 2$", xy=(2.0, 0.0), xytext=(1.25, -1.45), fontsize=11,
            arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.7))
ax.annotate(r"$\kappa_s = 1/4$", xy=(0.0, 1.0), xytext=(-0.6, 1.42), fontsize=11)
ax.text(-2.45, -1.45, r"$\gamma$", fontsize=12)
ax.annotate("", xy=(-1.05, np.sqrt(1 - 1.05 ** 2 / 4) + 0.0), xytext=(-0.85, np.sqrt(1 - 0.85 ** 2 / 4)),
            arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=1.4, mutation_scale=12))
ax.text(1.1, 0.25, r"$\mathbf{n}_s$", color=C["normal"], fontsize=12)
dgfig.schematic_axes(ax, (-2.5, 2.5), (-1.6, 1.65))
ax.set_title("(a)", loc="left", fontsize=12)

ax = axes[1]
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color=C["aux"], lw=0.9, ls=(0, (4, 3)))
for n in Nn:
    ax.annotate("", xy=n, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=0.9,
                                                         mutation_scale=8, shrinkA=0, shrinkB=0, alpha=0.75))
ax.plot(*Nn.T, "o", color=C["main"], ms=3.0, zorder=5)
ax.annotate("", xy=1.22 * Nn[1], xytext=1.22 * Nn[0],
            arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=1.2, connectionstyle="arc3,rad=0.25"))
ax.text(1.3 * np.cos(-2.4), 1.3 * np.sin(-2.4) - 0.05, r"$\approx\kappa_s\,\Delta s$", fontsize=11, ha="right")
ax.text(0.62, 0.95, r"$S^1$", fontsize=12, color=C["aux"])
ax.text(0.0, -1.32, r"$\mathbf{n}_s(s_k)$", fontsize=11, color=C["normal"], ha="center")
dgfig.schematic_axes(ax, (-1.45, 1.45), (-1.45, 1.45))
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.08)
dgfig.save(fig, __file__)
print(f"perimeter {Ltot:.4f}, ds {ds:.4f}, dphi max {dphi.max():.4f} min {dphi.min():.4f}")
