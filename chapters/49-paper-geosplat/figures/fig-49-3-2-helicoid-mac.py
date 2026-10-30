"""Figure 49.3.2: mean curvature versus mean absolute curvature (MAC) on the helicoid (Section 49.3, [GeoSplat25, §3.1.1]).

Helicoid x(u, v) = (v cos u, v sin u, c u) with c = 1. Its principal curvatures are +-c/(c^2 + v^2), so the mean
curvature H is 0 everywhere while MAC = (|tau_1| + |tau_2|)/2 = c/(c^2 + v^2) is positive.
(a) the helicoid for 0 <= u <= 2 pi, -2 <= v <= 2, coloured by MAC.
(b) H and MAC along a ruling (fixed u) as functions of v.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-3-2-helicoid-mac.py
"""
import numpy as np
import sympy as sp

import dgfig
import dgsym

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402

C = dgfig.COLORS
c = 1.0

# self-check: principal curvatures from the shape operator (dgsym) at a few points
u_, v_ = sp.symbols("u v", real=True)
Xs = sp.Matrix([v_ * sp.cos(u_), v_ * sp.sin(u_), u_])
W = dgsym.shape_operator(Xs, u_, v_)
for uu, vv in ((0.3, 0.0), (1.0, 0.5), (2.0, -1.5)):
    ev = sorted(float(e) for e in W.subs({u_: uu, v_: vv}).evalf().eigenvals(multiple=True))
    k = 1 / (1 + vv ** 2)
    assert np.allclose(ev, [-k, k], atol=1e-12), ev

fig = plt.figure(figsize=(7.6, 3.4))
ax = dgfig.axes3d(fig, pos=121, elev=24, azim=-60)
U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 80), np.linspace(-2, 2, 33))
Xh, Yh, Zh = V * np.cos(U), V * np.sin(U), c * U
mac = c / (c ** 2 + V ** 2)
norm = colors.Normalize(0, 1)
cmap = plt.get_cmap("Blues")
ax.plot_surface(Xh, Yh, Zh, facecolors=cmap(norm(0.15 + 0.85 * mac)), rstride=1, cstride=1, linewidth=0,
                antialiased=True, shade=False, rasterized=True, alpha=0.95)
ax.plot(np.zeros(50), np.zeros(50), np.linspace(0, 2 * np.pi, 50), color=C["main"], lw=1.2)
dgfig.equal_aspect(ax, Xh, Yh, Zh, zoom=1.2)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cb.set_label("MAC", fontsize=9)
ax.set_title("(a) helicoid coloured by MAC", fontsize=10.5)

ax2 = fig.add_subplot(122)
v = np.linspace(-3, 3, 400)
ax2.plot(v, c / (c ** 2 + v ** 2), color=C["tangent"], lw=1.8, label=r"MAC $= (|\tau_1| + |\tau_2|)/2$")
ax2.plot(v, 0 * v, color=C["aux"], lw=1.8, ls="--", label=r"mean curvature $(\tau_1 + \tau_2)/2$")
ax2.plot(v, c / (c ** 2 + v ** 2), color=C["normal"], lw=0.9, ls=":", label=r"$\tau_1 = c/(c^2 + v^2)$ (= MAC)")
ax2.plot(v, -c / (c ** 2 + v ** 2), color=C["normal"], lw=0.9, ls=":", label=r"$\tau_2 = -c/(c^2 + v^2)$")
ax2.set_xlabel(r"$v$ (signed distance from the axis)")
ax2.set_ylabel("curvature")
ax2.set_ylim(-1.15, 1.6)
ax2.legend(loc="upper center", fontsize=7.2, frameon=False, ncol=2, columnspacing=0.8, handlelength=1.6)
ax2.set_title("(b) along a ruling", fontsize=10.5)
assert abs(c / (c ** 2 + 0.0 ** 2) - 1.0) < 1e-15 and abs(c / (c ** 2 + 2.0 ** 2) - 0.2) < 1e-15
fig.subplots_adjust(left=0.0, right=0.98, wspace=0.25, bottom=0.15)
dgfig.save(fig, __file__)
