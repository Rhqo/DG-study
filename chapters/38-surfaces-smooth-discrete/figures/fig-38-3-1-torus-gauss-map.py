"""Figure 38.3.1: the Gauss map of the torus of revolution (R, r) = (2, 0.8), N = x_u x x_v / |x_u x x_v| (inward,
GUIDELINES 7): N(u, v) = -(cos u cos v, cos u sin v, sin u).

(a) the torus with three coordinate curves and unit normals along them: the outer equator u = 0 (blue), the top
circle u = pi/2 (green) and the meridian v = 0 (orange). (b) the unit sphere with the images under N: the outer
equator goes once around the equator of S^2, the whole top circle goes to the south pole (0, 0, -1), and the
meridian goes once around a great circle.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-3-1-torus-gauss-map.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg38  # noqa: E402

C = dgfig.COLORS
R, r = ddg38.R_TORUS, ddg38.r_TORUS
v_mer = 0.0


def normal(u, v):
    xu, xv = ddg38.partials(ddg38.torus, np.asarray(u, float), np.asarray(v, float))
    n = np.cross(xu, xv, axis=0)
    return n / np.linalg.norm(n, axis=0)


# self-checks: closed form, inward, top circle collapses to the south pole
uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 13), np.linspace(0, 2 * np.pi, 9), indexing="ij")
Nn = normal(uu, vv)
assert np.allclose(Nn, -np.array([np.cos(uu) * np.cos(vv), np.cos(uu) * np.sin(vv), np.sin(uu)]), atol=1e-7)
center = np.array([R * np.cos(vv), R * np.sin(vv), 0 * vv])                 # point on the core circle
assert np.all(np.sum(Nn * (center - ddg38.torus(uu, vv)), 0) > 0)          # N points toward the core circle
assert np.allclose(normal(np.full(5, np.pi / 2), np.linspace(0, 6, 5)), np.array([[0, 0, -1]] * 5).T, atol=1e-7)

curves = [(lambda s: (0 * s, s), C["tangent"]), (lambda s: (np.pi / 2 + 0 * s, s), C["third"]),
          (lambda s: (s, v_mer + 0 * s), C["accent"])]
s = np.linspace(0, 2 * np.pi, 400)
sk = np.linspace(0, 2 * np.pi, 13)[:-1]

fig = plt.figure(figsize=(7.2, 3.7))
ax = dgfig.axes3d(fig, 121, elev=40, azim=-60)
U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(0, 2 * np.pi, 97), indexing="ij")
X, Y, Z = ddg38.torus(U, V)
dgfig.surface(ax, X, Y, Z, alpha=0.22, grid=False)
view = dgfig.view_vector(ax)
for cur, col in curves:
    P = ddg38.torus(*cur(s))
    ax.plot(*P, color=col, lw=1.8, zorder=6)
    for t0 in sk:
        uq, vq = cur(np.array(t0))
        p = ddg38.torus(uq, vq)
        n = normal(uq, vq)
        if (p - 0) @ view > -0.6 or col == C["third"]:
            dgfig.arrow3d(ax, p, 0.45 * n, role="normal", zorder=9, head=6, lw=1.0)
dgfig.equal_aspect(ax, np.stack([X, Y, Z], -1), zoom=1.35)
ax.set_title("(a)", loc="left", fontsize=12, y=0.95)

ax2 = dgfig.axes3d(fig, 122, elev=20, azim=-60)
TH, PH = np.meshgrid(np.linspace(0, np.pi, 37), np.linspace(0, 2 * np.pi, 73), indexing="ij")
dgfig.surface(ax2, *ddg38.sphere(TH, PH), alpha=0.22, grid_every=6)
view2 = dgfig.view_vector(ax2)
for cur, col in curves[:1] + curves[2:]:
    Q = normal(*cur(s))
    dgfig.curve3d(ax2, Q.T, role=col, lw=2.0, visible=Q.T @ view2 > 0)
dgfig.point3d(ax2, [0, 0, -1], None, role=C["third"], size=40)
ax2.text(0.12, 0.0, -1.32, r"$\mathbf{N}(\pi/2, v)$", color=C["third"], fontsize=11)
dgfig.equal_aspect(ax2, np.array([[-1, -1, -1], [1, 1, 1.0]]), zoom=1.25)
ax2.set_title("(b)", loc="left", fontsize=12, y=0.95)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
dgfig.save(fig, __file__)
