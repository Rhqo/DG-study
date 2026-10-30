"""Figure 49.3.1: three ways to shape the initial Gaussian at q = (2.8, 0, 0) on the torus T_{2,0.8} (Section 49.3).

Principal curvatures |tau_1| = 1.25 (direction w1 = e_z) and |tau_2| = 1/2.8 = 0.357 (direction w2 = e_y); s_nbr = 0.2,
so every disc below has the same tangent area pi s1 s2 = pi (s_nbr/2)^2 (Proposition 3.1 of [GeoSplat25]).
  isotropic (3DGS-style):       s1 = s2 = 0.1
  GeoSplat (5):                 s1/s2 = |tau_1/tau_2| = 3.5    -> s1 = 0.1871 along w2, s2 = 0.0535 along w1
  square root (this chapter):   s1/s2 = sqrt|tau_1/tau_2|      -> s1 = 0.1368 along w2, s2 = 0.0731 along w1
(a) the 1-sigma ellipses in the tangent plane (horizontal: w2, vertical: w1).
(b) the distance from the boundary point q + s1 cos(phi) w2 + s2 sin(phi) w1 of each ellipse to the torus, against phi.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-3-1-warmup.py
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
k1, k2 = 1.25, 1 / 2.8
s_nbr = 0.2
q = np.array([2.8, 0.0, 0.0])
w1, w2 = np.array([0.0, 0, 1]), np.array([0.0, 1, 0])
iso = np.array([0.1, 0.1])
geo = geo49.warmup_scales(-k1, -k2, s_nbr)[:2]
rt = (s_nbr / 2) * np.array([(k1 / k2) ** 0.25, (k2 / k1) ** 0.25])
cases = [("isotropic", iso, C["aux"], "--"), ("GeoSplat (5)", geo, C["main"], "-"),
         ("square-root ratio", rt, C["third"], "-.")]

phi = np.linspace(0, 2 * np.pi, 721)
dev = {}
for name, s, _, _ in cases:
    P = q + np.outer(s[0] * np.cos(phi), w2) + np.outer(s[1] * np.sin(phi), w1)
    dev[name] = geo49.torus_distance(P)

# self-checks (numbers quoted in Example 49.3.2 and Example 49.3.3)
assert np.allclose(geo, [0.18708, 0.05345], atol=1e-5)
for s in (iso, geo, rt):
    assert abs(s[0] * s[1] - 0.01) < 1e-12
assert abs(dev["isotropic"].max() - 0.00623) < 5e-6 and abs(dev["GeoSplat (5)"].max() - 0.00624) < 5e-6
assert abs(dev["square-root ratio"].max() - 0.00334) < 5e-6 and abs(dev["square-root ratio"].min() - 0.00333) < 5e-6
assert abs(dev["isotropic"][0] - 0.00179) < 5e-6 and abs(dev["GeoSplat (5)"][180] - 0.00178) < 5e-6

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
for name, s, col, ls in cases:
    ax.plot(s[0] * np.cos(phi), s[1] * np.sin(phi), color=col, ls=ls, lw=1.6, label=name)
ax.annotate("", xy=(0.24, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.3))
ax.annotate("", xy=(0, 0.13), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.3))
ax.text(0.245, 0.012, r"$\tilde{\mathbf{w}}_2$", color=C["tangent"], fontsize=11)
ax.text(0.01, 0.135, r"$\tilde{\mathbf{w}}_1$", color=C["tangent"], fontsize=11)
ax.text(0.15, -0.115, r"$|\tau_2| = 0.357$" "\n" r"$|\tau_1| = 1.25$", fontsize=8.5, color=C["main"])
ax.plot(0, 0, "o", color=C["main"], ms=3)
ax.set_aspect("equal")
ax.set_xlim(-0.21, 0.29)
ax.set_ylim(-0.15, 0.16)
ax.set_xlabel(r"along $\tilde{\mathbf{w}}_2$ (low curvature)")
ax.set_ylabel(r"along $\tilde{\mathbf{w}}_1$ (high curvature)")
ax.set_title("(a) 1-sigma discs, equal area", fontsize=10.5)

for name, s, col, ls in cases:
    ax2.plot(np.degrees(phi), 1e3 * dev[name], color=col, ls=ls, lw=1.6, label=name)
ax2.set_xlim(0, 360)
ax2.set_xticks([0, 90, 180, 270, 360])
ax2.set_xlabel(r"boundary angle $\varphi$ (degrees; $0$ = along $\tilde{\mathbf{w}}_2$)")
ax2.set_ylabel(r"distance to torus ($\times 10^{-3}$)")
ax2.set_ylim(0, 7.2)
ax2.set_title("(b) gap between disc edge and surface", fontsize=10.5)
h, l = ax2.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, handlelength=2.6)
fig.tight_layout(rect=(0, 0.08, 1, 1))
dgfig.save(fig, __file__)
