"""Figure 39.2.1: the Darboux frame of a circle of latitude and the splitting of its curvature vector.

Unit sphere, outward N = x, circle of latitude theta0 = pi/3 traversed with phi increasing (Example 39.2.6).
(a) 3D: the circle (black) and the Darboux frame (t blue, N x t green, N vermilion) at three points; at the point
phi0 = 0.35 the curvature vector t' (orange) as well.
(b) the meridian plane through the point phi0 (spanned by N x t and N, which contains the z-axis): unit circle =
section of the sphere, the point p, the center c of the circle of latitude on the z-axis, t' = (cot theta0) N x t - N
(orange) pointing to c, its components kappa_g N x t (green dashed) and kappa_n N (vermilion dashed).
Also writes an interactive version.

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-2-1-darboux-latitude.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg39  # noqa: E402

C = dgfig.COLORS
th0 = np.pi / 3
phis = [0.35, 0.35 + 1.6]


def frame(ph):
    p = ddg39.sphere(th0, ph)
    kap, kn, kg, t, N = ddg39.darboux(ddg39.sphere, lambda s: (th0, s), ph)
    return p, t, np.cross(N, t), N, kap, kn, kg


p0, t0, b0, N0, kap, kn, kg = frame(phis[0])
assert abs(kn + 1) < 1e-6 and abs(kg - 1 / np.tan(th0)) < 1e-6 and abs(kap - 1 / np.sin(th0)) < 1e-6
assert abs(kap ** 2 - kn ** 2 - kg ** 2) < 1e-6                    # Proposition 39.2.4
assert np.allclose(N0, p0) and b0[2] > 0                           # outward N; N x t points north
tp = kg * b0 + kn * N0                                             # t' = kappa_g N x t + kappa_n N
center = np.array([0, 0, np.cos(th0)])
assert np.allclose(p0 + tp / kap ** 2, center, atol=1e-6)          # t'/kappa^2 reaches the center of the circle

fig = plt.figure(figsize=(7.2, 3.6))
ax = dgfig.axes3d(fig, 121, elev=50, azim=np.rad2deg(0.35) + 70)
U, V = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60), indexing="ij")
S = ddg39.sphere(U, V)
dgfig.surface(ax, *S, alpha=0.25, grid_every=5)
ph = np.linspace(0, 2 * np.pi, 300)
circ = ddg39.sphere(th0 + 0 * ph, ph).T
view = dgfig.view_vector(ax)
dgfig.curve3d(ax, circ, visible=circ @ view > 0, lw=1.8, zorder=6)
eq = ddg39.sphere(np.pi / 2 + 0 * ph, ph).T
dgfig.curve3d(ax, eq, role="aux", lw=0.7, visible=eq @ view > 0, hidden=None, zorder=4)
for k, phk in enumerate(phis):
    p, t, b, N, *_ = frame(phk)
    if p @ view < -0.05:
        continue
    sc = 0.5
    dgfig.arrow3d(ax, p, sc * t, role="tangent", zorder=10, head=8, lw=1.3)
    dgfig.arrow3d(ax, p, sc * b, role="third", zorder=10, head=8, lw=1.3)
    dgfig.arrow3d(ax, p, sc * N, role="normal", zorder=10, head=8, lw=1.3)
dgfig.arrow3d(ax, p0, 0.5 * tp, role="accent", zorder=11, head=9, lw=1.6)
ax.text(*(p0 + 0.55 * t0 + np.array([0, 0.05, 0.02])), r"$\mathbf{t}$", color=C["tangent"], fontsize=12)
ax.text(*(p0 + 0.55 * b0 + np.array([0, 0, 0.05])), r"$\mathbf{N}\times\mathbf{t}$", color=C["third"], fontsize=11)
ax.text(*(p0 + 0.45 * N0 + np.array([0, 0, 0.1])), r"$\mathbf{N}$", color=C["normal"], fontsize=12)
ax.text(*(p0 + 0.5 * tp + np.array([0, -0.05, -0.12])), r"$\mathbf{t}'/2$", color=C["accent"], fontsize=11)
dgfig.equal_aspect(ax, S.reshape(3, -1).T, zoom=1.15)
ax.set_title("(a)", loc="left", fontsize=12, y=0.96)

ax = fig.add_subplot(122)
# meridian plane coordinates: horizontal = distance from the z-axis in the direction of p, vertical = z
e_r = np.array([np.cos(phis[0]), np.sin(phis[0]), 0.0])
to2 = lambda w: np.array([w @ e_r, w[2]])  # noqa: E731
P, B, Nn, T2 = to2(p0), to2(b0), to2(N0), to2(tp)
thc = np.linspace(-np.pi / 2, np.pi / 2, 200)
ax.plot(np.cos(thc), np.sin(thc), color=C["main"], lw=1.2)
ax.plot([0, 0], [-0.95, 1.45], color=C["aux"], lw=0.7)
ax.plot([0, P[0]], [P[1], P[1]], color=C["aux"], lw=0.7, ls=(0, (3, 2)))
ax.plot([0, P[0]], [0, P[1]], color=C["aux"], lw=0.7, ls=(0, (3, 2)))
arr = lambda a, d, col, ls="-", lw=1.5: ax.annotate(  # noqa: E731
    "", xy=a + d, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls, mutation_scale=12,
                                            shrinkA=0, shrinkB=0))
arr(P, Nn, C["normal"])
arr(P, B, C["third"])
arr(P, kg * B, C["third"], ls=(0, (3, 2)), lw=1.2)
arr(P, kn * Nn, C["normal"], ls=(0, (3, 2)), lw=1.2)
arr(P, T2, C["accent"], lw=1.9)
ax.plot(*P, "o", color=C["main"], ms=4)
ax.plot(0, np.cos(th0), "o", color=C["main"], ms=3.5)
ax.text(P[0] + 0.06, P[1] - 0.12, r"$p$", fontsize=12)
ax.text(-0.17, np.cos(th0) + 0.06, r"$c$", fontsize=12)
ax.text(*(P + Nn + np.array([0.04, 0.0])), r"$\mathbf{N}$", color=C["normal"], fontsize=12)
ax.text(*(P + B + np.array([-0.32, 0.05])), r"$\mathbf{N}\times\mathbf{t}$", color=C["third"], fontsize=11)
ax.text(*(P + 0.5 * T2 + np.array([-0.05, 0.08])), r"$\mathbf{t}'$", color=C["accent"], fontsize=12)
ax.text(*(P + kn * Nn + np.array([0.02, -0.12])), r"$\kappa_n\mathbf{N}$", color=C["normal"], fontsize=11)
ax.text(*(P + kg * B + np.array([0.06, 0.02])), r"$\kappa_g\,\mathbf{N}\times\mathbf{t}$", color=C["third"],
        fontsize=11)
ax.text(-0.13, 1.36, r"$z$", fontsize=11)
ax.text(0.09, 0.33, r"$\theta_0$", fontsize=11)
dgfig.schematic_axes(ax, (-0.3, 1.95), (-0.95, 1.5))
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.0, right=0.99, bottom=0.02, top=0.93, wspace=0.02)
dgfig.save(fig, __file__)

# interactive version
Vs, Fs = dm.icosphere(3)
tr = [dm.plotly_mesh(Vs, Fs, color="#D0D0D0", opacity=0.45, name="unit sphere")]
import plotly.graph_objects as go  # noqa: E402
tr.append(go.Scatter3d(x=circ[:, 0], y=circ[:, 1], z=circ[:, 2], mode="lines", line=dict(color="#000000", width=4),
                       name="circle of latitude", hoverinfo="skip"))
Ps, Ts, Bs, Ns = [], [], [], []
for phk in np.linspace(0, 2 * np.pi, 9)[:-1]:
    p, t, b, N, *_ = frame(phk)
    Ps.append(p), Ts.append(t), Bs.append(b), Ns.append(N)
Ps = np.array(Ps)
tr += dm.plotly_arrows(Ps, 0.4 * np.array(Ts), color=C["tangent"], cone=0.08, name="t")
tr += dm.plotly_arrows(Ps, 0.4 * np.array(Bs), color=C["third"], cone=0.08, name="N x t")
tr += dm.plotly_arrows(Ps, 0.4 * np.array(Ns), color=C["normal"], cone=0.08, name="N")
tr += dm.plotly_arrows(p0[None], 0.5 * tp[None], color=C["accent"], cone=0.1, name="t'/2")
dm.plotly_save(tr, __file__, "Darboux frames (t blue, N x t green, N vermilion) on the circle of latitude theta = pi/3",
               "fig-39-2-1")
