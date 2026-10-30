"""Figure 47.2.1: volumetric and point cloud varifolds built from the same cell data ([BLM17] Def 2.11, 2.12,
Prop 2.13, Remark 2.14).

The circle of radius 0.5 centred at 0 is cut by a square grid of side a (grid lines x = 0.123 a + k a,
y = 0.071 a + k a). Each cell K gets m_K = length of the circle in K and a plane P_K (top eigenvector of the
mass-weighted mean tangent projection). (a) volumetric varifold: mass m_K spread uniformly over K (shade = m_K/|K|),
plane P_K drawn as a segment. (b) point cloud varifold: the same (m_K, P_K) at the cell centre x_K (dot area ~ m_K).
(c) Delta^{1,1}(V^vol, V^pt) against a: exactly (sqrt2 + ln(1 + sqrt2))/6 * a * M for centres x_K (M = pi), and
the bound delta * M of Prop 2.13 with delta = sqrt2 a. (d) total variation |delta V^vol|(R^2) against a, which grows
like 1/a, while the smooth circle has |delta V|(R^2) = 2 pi (Remark 2.14).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-2-1-volumetric-point-cloud.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import var47 as v  # noqa: E402

C = dgfig.COLORS
R = 0.5
A0 = 0.1
OFF = (0.123, 0.071)
cells = v.circle_cells(A0, R, (OFF[0] * A0, OFF[1] * A0))
mass = sum(m for m, _ in cells.values())
assert abs(mass - math.pi * 2 * R) < 1e-12

fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2), gridspec_kw={"hspace": 0.42, "wspace": 0.32})
tt = np.linspace(0, 2 * np.pi, 400)
dmax = max(m for m, _ in cells.values()) / A0 ** 2
for ax, kind in ((axes[0, 0], "vol"), (axes[0, 1], "pt")):
    for (i, j), (m, P) in cells.items():
        x0, y0 = OFF[0] * A0 + i * A0, OFF[1] * A0 + j * A0
        c = np.array([x0 + A0 / 2, y0 + A0 / 2])
        u = np.linalg.eigh(P)[1][:, -1]
        if kind == "vol":
            ax.add_patch(Rectangle((x0, y0), A0, A0, facecolor=C["region"], alpha=0.15 + 0.6 * (m / A0 ** 2) / dmax,
                                   edgecolor="white", lw=0.6))
            ax.plot(*np.array([c - 0.4 * A0 * u, c + 0.4 * A0 * u]).T, color=C["tangent"], lw=1.1)
        else:
            ax.add_patch(Rectangle((x0, y0), A0, A0, facecolor="none", edgecolor=C["aux"], lw=0.3))
            ax.plot(*np.array([c - 0.4 * A0 * u, c + 0.4 * A0 * u]).T, color=C["tangent"], lw=1.1)
            ax.plot(*c, "o", color=C["main"], ms=1.2 + 30 * m)
    ax.plot(R * np.cos(tt), R * np.sin(tt), color=C["aux"], lw=0.6, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.62, 0.62)
    ax.set_axis_off()
axes[0, 0].set_title(r"(a) volumetric $V^{\mathrm{vol}}_{\mathcal{K}}$, $a = 0.1$", fontsize=11)
axes[0, 1].set_title(r"(b) point cloud $V^{\mathrm{pt}}_{\mathcal{K}}$", fontsize=11)

As = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
dist = v.MEAN_DIST_SQUARE * As * mass
bound = math.sqrt(2) * As * mass
tv = np.array([v.volumetric_tv(v.circle_cells(a, R, (OFF[0] * a, OFF[1] * a)), a) for a in As])
assert np.all(dist < bound) and abs(dist[0] / bound[0] - 0.2705) < 1e-4
slope = np.polyfit(np.log(As), np.log(tv), 1)[0]
assert -1.08 < slope < -0.95, slope
assert np.all(tv > 2 * np.pi)

ax = axes[1, 0]
ax.loglog(As, bound, "--", color=C["aux"], lw=1.2, label=r"bound $\delta M$")
ax.loglog(As, dist, "o-", color=C["main"], ms=3.5, lw=1.2, label=r"$\Delta^{1,1}(V^{\mathrm{vol}}, V^{\mathrm{pt}})$")
ax.set_xlabel(r"cell side $a$")
ax.set_title(r"(c) distance, slope $1$", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax = axes[1, 1]
ax.loglog(As, tv, "o-", color=C["normal"], ms=3.5, lw=1.2, label=r"$|\delta V^{\mathrm{vol}}|(\mathbb{R}^2)$")
ax.axhline(2 * np.pi, color=C["main"], ls="--", lw=1.0, label=r"circle: $2\pi$")
ax.set_xlabel(r"cell side $a$")
ax.set_title(r"(d) total variation, slope $-1$", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_ylim(3, 3000)
dgfig.save(fig, __file__)
print("a:", As)
print("Delta exact:", dist)
print("TV:", tv, "slope", slope, "a*TV", tv * As)
