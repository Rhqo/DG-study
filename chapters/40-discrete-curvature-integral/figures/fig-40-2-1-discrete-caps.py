"""Figure 40.2.1: Gauss-Bonnet with boundary on discrete polar caps of icosphere(3).

Cap(theta0) = the faces of dgmesh.icosphere(3) whose three vertices have colatitude <= theta0 (ddg40.icosphere_cap); every
cap used here is a disk (chi = 1). (a) the cap for theta0 = 2 pi / 3 seen from the side: faces of the cap (gray) on the
faint unit sphere, interior vertices (vermilion dots), the boundary polygon (green), and its vertices with turning
angle tau_i = pi - Theta_i > 0 (filled green) and < 0 (open green). (b) for 240 values of theta0: the sum of the angle
defects of the interior vertices (vermilion), the sum of the boundary turning angles (green) and their total (black),
which is 2 pi for every cap (Theorem 40.2.7). Thin dashed curves: the smooth polar cap of (39.4.3),
2 pi (1 - cos theta0) and 2 pi cos theta0.

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-2-1-discrete-caps.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402

C = dgfig.COLORS
LEVEL = 3
th0s = np.linspace(0.25, np.pi - 0.15, 240)
rows = []
for t in th0s:
    Vs, Fs = ddg40.icosphere_cap(LEVEL, t)
    if len(Fs) == 0:
        continue
    assert dm.euler_characteristic(Fs) == 1 and dm.is_manifold(Fs)
    d = dm.angle_defect(Vs, Fs)
    b = np.array(dm.boundary_vertices(Fs))
    inner = np.setdiff1d(np.arange(len(Vs)), b)
    rows.append((t, d[inner].sum(), d[b].sum()))
rows = np.array(rows)
assert np.allclose(rows[:, 1] + rows[:, 2], 2 * np.pi, atol=1e-11)

fig = plt.figure(figsize=(7.0, 3.3))
ax = fig.add_axes([0.0, 0.03, 0.42, 0.9])
cam = dm.Ortho(elev=-18, azim=-60)
Vf, Ff = dm.icosphere(LEVEL)
dm.draw_mesh_2d(ax, Vf, Ff, cam, (0.92, 0.92, 0.92, 1.0), edgecolor="#d9d9d9", lw=0.25, cull=True, zorder=1)
Vs, Fs = ddg40.icosphere_cap(LEVEL, 2 * np.pi / 3)
col = dm.shade(Vs, Fs, dm.hex_to_rgb(C["surface"]))
dm.draw_mesh_2d(ax, Vs * 1.002, Fs, cam, col, edgecolor="#777777", lw=0.3, cull=True, zorder=2)
d = dm.angle_defect(Vs, Fs)
loop = dm.boundary_vertices(Fs)
vis = lambda P: (P @ cam.view) > -0.05  # noqa: E731
P2 = cam(Vs * 1.004)
for a, bb in dm.boundary_edges(Fs):
    if vis(Vs[a]) and vis(Vs[bb]):
        ax.plot(*P2[[a, bb]].T, color=C["third"], lw=1.8, zorder=4)
inner = np.setdiff1d(np.arange(len(Vs)), loop)
iv = [i for i in inner if vis(Vs[i])]
ax.scatter(*P2[iv].T, s=1.5, color=C["normal"], zorder=5)
for i in loop:
    if vis(Vs[i]):
        ax.scatter(*P2[i], s=16, zorder=6, edgecolors=C["third"], linewidths=1.0,
                   facecolors=C["third"] if d[i] > 0 else "white")
ax.set_aspect("equal")
ax.set_xlim(-1.08, 1.08)
ax.set_ylim(-1.08, 1.08)
ax.set_axis_off()
ax.set_title("(a)", loc="left", fontsize=12)

ax2 = fig.add_axes([0.53, 0.17, 0.45, 0.73])
t = rows[:, 0]
ax2.plot(t, 2 * np.pi * (1 - np.cos(t)), color=C["normal"], lw=0.8, ls=(0, (3, 2)))
ax2.plot(t, 2 * np.pi * np.cos(t), color=C["third"], lw=0.8, ls=(0, (3, 2)))
ax2.plot(t, rows[:, 1], ".", color=C["normal"], ms=2.8, label=r"interior: $\Sigma\,\delta_i$")
ax2.plot(t, rows[:, 2], ".", color=C["third"], ms=2.8, label=r"boundary: $\Sigma\,\tau_i$")
ax2.plot(t, rows[:, 1] + rows[:, 2], ".", color=C["main"], ms=2.8, label="total")
ax2.axvline(2 * np.pi / 3, color=C["aux"], lw=0.7, ls=(0, (1, 2)))
ax2.set_xticks([0, np.pi / 2, np.pi])
ax2.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
ax2.set_yticks([-2 * np.pi, 0, 2 * np.pi, 4 * np.pi])
ax2.set_yticklabels([r"$-2\pi$", r"$0$", r"$2\pi$", r"$4\pi$"])
ax2.set_xlabel(r"$\theta_0$")
ax2.set_xlim(0, np.pi)
ax2.legend(fontsize=9, frameon=False, loc="lower left", markerscale=3)
ax2.set_title("(b)", loc="left", fontsize=12)
dgfig.save(fig, __file__)
at = np.argmin(np.abs(rows[:, 0] - 2 * np.pi / 3))
print(f"cap 2 pi / 3: interior {rows[at, 1]:.4f}, boundary {rows[at, 2]:.4f}; smooth {2 * np.pi * (1 - np.cos(rows[at, 0])):.4f}")
print(f"(a): {len(Vs)} vertices, {len(loop)} on the boundary, tau > 0 at {int(np.sum(d[loop] > 0))}, tau < 0 at {int(np.sum(d[loop] < 0))}")
