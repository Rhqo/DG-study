"""Figure 38.6.1: the dual edge of a hinge is measured along the surface (ddg38.hinge).

Hinge with x0 = (0, 0, 0), x1 = (1, 0, 0), x2 = (0.5, 0.8, 0) and x3 = (0.3, -0.9 cos psi, 0.9 sin psi), drawn for
psi = 0, 60, 120 degrees. Points: circumcenters c_f of (0, 1, 2) and c_g of (1, 0, 3) (both triangles are acute, so
both lie inside), the midpoint m of the shared edge. Blue: the dual edge e* = [c_g, m] + [m, c_f]; each piece is
perpendicular to the shared edge inside its own triangle, and its length |e*| = h_f + h_g = 0.5770833 does not depend
on psi, so star_1 = |e*| / |e| = (cot theta_2 + cot theta_3) / 2 is the same in all panels. Orange, dashed: the
straight chord |c_f - c_g| in space, which shrinks as the hinge folds (0.5771, 0.5018, 0.2988 at 0, 60, 120 degrees;
chord^2 = h_f^2 + h_g^2 + 2 h_f h_g cos psi with h_f = 0.24375, h_g = 1/3).

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-6-1-hinge-dual-edge.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402

C = dgfig.COLORS


def circumcenter(a, b, c):
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    return a + (ac @ ac * np.cross(n, ab) + ab @ ab * np.cross(ac, n)) / (2 * n @ n)


PSIS = np.deg2rad([0.0, 60.0, 120.0])
data = []
for psi in PSIS:
    V, F = ddg38.hinge(psi)
    cf, cg = circumcenter(*V[F[0]]), circumcenter(*V[F[1]])
    m = (V[0] + V[1]) / 2
    dual = np.linalg.norm(cf - m) + np.linalg.norm(cg - m)
    chord = np.linalg.norm(cf - cg)
    # self-checks: both pieces are perpendicular to the edge, |e*| = (cot + cot)/2 * |e| = star_1, L_01 is the same
    e = V[1] - V[0]
    assert abs((cf - m) @ e) < 1e-12 and abs((cg - m) @ e) < 1e-12
    assert abs(dual - 0.5770833333333333) < 1e-12
    assert abs(dd.hodge1(V, F)[0] - dual) < 1e-12 and abs(dm.cotan_laplacian(V, F)[0, 1] - dual) < 1e-12
    data.append((V, F, cf, cg, m, dual, chord))
chords = [d[6] for d in data]
assert abs(chords[0] - data[0][5]) < 1e-12 and chords[0] > chords[1] > chords[2]
assert np.allclose(np.round(chords, 4), [0.5771, 0.5018, 0.2988])
hf, hg = 0.5 * 0.4875, 0.5 * 2 / 3                        # h = (|e| / 2) cot(opposite angle)
assert np.allclose(chords, np.sqrt(hf ** 2 + hg ** 2 + 2 * hf * hg * np.cos(PSIS)))

box = np.array([[0.0, -0.95, -0.2], [1.0, 0.85, 0.8]])          # common bounding box of the three hinges
fig = plt.figure(figsize=(7.2, 3.4))
for k, (V, F, cf, cg, m, dual, chord) in enumerate(data):
    ax = fig.add_axes([k / 3, 0.0, 1 / 3, 0.84], projection="3d")
    ax.view_init(elev=32, azim=-78)
    ax.set_axis_off()
    ax.add_collection3d(Poly3DCollection([V[F[0]]], facecolors=C["surface"], edgecolors="#555555", linewidths=0.7,
                                         alpha=0.55))
    ax.add_collection3d(Poly3DCollection([V[F[1]]], facecolors=C["region"], edgecolors="#555555", linewidths=0.7,
                                         alpha=0.35))
    ax.plot(*V[[0, 1]].T, color=C["main"], lw=2.2, zorder=6)
    ax.plot(*np.array([cg, m, cf]).T, color=C["tangent"], lw=2.0, zorder=9)
    ax.plot(*np.array([cg, cf]).T, color=C["accent"], lw=1.6, ls="--", zorder=8)
    for p, lab, off in ((cf, r"$c_f$", (0.05, 0.0, 0.0)), (cg, r"$c_g$", (-0.13, -0.02, 0.02)),
                        (m, r"$m$", (0.0, -0.12, -0.08))):
        dgfig.point3d(ax, p, lab, size=10, label_offset=off, fontsize=10)
    dgfig.equal_aspect(ax, box, zoom=1.6)
    fig.text(k / 3 + 1 / 6, 0.95, rf"$\psi = {np.degrees(PSIS[k]):.0f}^\circ$", ha="center", fontsize=11)
    fig.text(k / 3 + 1 / 6, 0.87, rf"$|e^\star| = {dual:.3f}$,  $|c_f - c_g| = {chord:.3f}$", ha="center",
             fontsize=9.5)
dgfig.save(fig, __file__)
