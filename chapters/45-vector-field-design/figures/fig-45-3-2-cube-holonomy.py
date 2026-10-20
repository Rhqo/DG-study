"""Figure 45.3.2: Levi-Civita holonomy around a cube corner equals its angle defect pi/2 (Section 45.3, Example 45.3.11).

Mesh: ch45.unit_cube() (each square split by one diagonal); the corner x_7 = (1, 1, 1) lies on three diagonals, so six
triangles f_1, ..., f_6 (counterclockwise seen from outside) meet there, each with a 45 degree corner (Theta = 3 pi/2,
delta = pi/2). A unit vector w_1 at angle 30 degrees from r_{f_1} is carried across the shared edges f_k -> f_{k+1}
by psi -> psi +- rho_e (Definition 45.3.1; + when the crossing follows the dual orientation of Remark 36.5.1).
(a) the cube seen from (1, 1, 1) (elev 35.26, azim 45; back faces culled): the six triangles at x_7 light blue, the
    transported vectors at their centroids (blue), the final vector w_7 after one turn drawn in f_1 (orange), and the
    arc between w_1 and w_7 (black).
(b) the six triangles unfolded around x_7 into the plane (corner angles 45 degrees, true edge lengths), with the copy of
    f_1 that the loop returns to (dashed): in the unfolding the transported vector is one constant vector (blue);
    measured against the returning copy of f_1 it has turned by 2 pi - 3 pi/2 = pi/2.
Self-checks: the six crossings return w rotated by exactly delta_7 = pi/2 counterclockwise (mod 2 pi); the same
holds at every corner (ch45.transport_angles and -d0^T rho).

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-3-2-cube-holonomy.py
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Polygon  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
V, F = ch45.unit_cube()
D0 = dd.d0(F, len(V))
rho = ch45.transport_angles(V, F)
delta = dm.angle_defect(V, F)
assert np.allclose(delta, np.pi / 2) and np.allclose(ch45.wrap(-D0.T @ rho), np.pi / 2)
E, left, right = ch45.left_right_faces(F)
_, idx = dd.edge_list(F)
c = 7
hm = dd.HalfedgeMesh(F)
fan = hm.vertex_faces(c)                       # counterclockwise
ring = hm.vertex_one_ring(c)
assert len(fan) == 6
r, b, N = ch45.face_frames3(V, F)
psi = [np.radians(30)]
for k in range(6):
    f, g = fan[k], fan[(k + 1) % 6]
    e = idx[tuple(sorted(set(F[f]) & set(F[g])))]
    sgn = 1.0 if (right[e] == f and left[e] == g) else -1.0
    assert (right[e], left[e]) in ((f, g), (g, f))
    psi.append(psi[-1] + sgn * rho[e])
turn = ch45.wrap(psi[-1] - psi[0])
assert abs(turn - np.pi / 2) < 1e-12
print(f"holonomy around x_7: {turn:.6f} (pi/2 = {np.pi / 2:.6f})")

cam = dm.Ortho(elev=np.degrees(np.arcsin(1 / np.sqrt(3))), azim=45)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1.0, 1.0]})
Nf = dm.face_normals(V, F)
order = [f for f in np.argsort(cam.depth(V[F].mean(1))) if Nf[f] @ cam.view > 1e-9]
for f in order:
    axa.add_patch(Polygon(cam(V[F[f]]), closed=True, fc="#CFE6F5" if f in fan else "#E6E6E6", ec="#8A8A8A", lw=0.7))


def arrow(ax, p, q, color, lw=1.6, z=5):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10), zorder=z)


s = 0.28
for k in range(6):
    f = fan[k]
    p = V[F[f]].mean(0)
    w = np.cos(psi[k]) * r[f] + np.sin(psi[k]) * b[f]
    arrow(axa, cam(p), cam(p + s * w), col["tangent"])
    axa.text(*(cam(p) + [0.02, -0.07]), rf"$f_{k + 1}$", fontsize=9, color=col["aux"])
p1 = V[F[fan[0]]].mean(0)
w7 = np.cos(psi[6]) * r[fan[0]] + np.sin(psi[6]) * b[fan[0]]
arrow(axa, cam(p1), cam(p1 + s * w7), col["accent"], 2.0)
axa.plot(*cam(V[c]), "o", color=col["main"], ms=4)
axa.text(*(cam(V[c]) + [0.03, 0.03]), r"$\mathbf{x}_7$", fontsize=11)
axa.set_title(r"(a) one turn around $\mathbf{x}_7$: $w_1 \to w_7$", fontsize=11)
# (b) unfolding around x_7: neighbor ring[k] at polar angle o + (sum of the first k corner angles), true distance
ang = dm.corner_angles(V, F)
theta = [ang[f, list(F[f]).index(c)] for f in fan]
assert np.allclose(theta, np.pi / 4)
o = np.pi / 4
phis = o + np.concatenate([[0.0], np.cumsum(theta), [np.sum(theta) + theta[0]]])     # k = 0..7
dist = [np.linalg.norm(V[ring[k % 6]] - V[c]) for k in range(8)]
pts = [dist[k] * np.array([np.cos(phis[k]), np.sin(phis[k])]) for k in range(8)]
for k in range(6):
    axb.add_patch(Polygon(np.array([[0, 0], pts[k], pts[k + 1]]), closed=True, fc="#CFE6F5", ec="#8A8A8A", lw=0.7))
axb.add_patch(Polygon(np.array([[0, 0], pts[6], pts[7]]), closed=True, fc="none", ec=col["aux"], lw=1.0, ls="--"))
rot = lambda v, t: np.array([np.cos(t) * v[0] - np.sin(t) * v[1], np.sin(t) * v[0] + np.cos(t) * v[1]])  # noqa: E731
pos = {c: np.zeros(2), ring[0]: pts[0], ring[1]: pts[1]}                  # first copy of f_1
r2 = pos[int(F[fan[0]][1])] - pos[int(F[fan[0]][0])]
r2 /= np.linalg.norm(r2)
w2 = rot(r2, psi[0])                                                        # the transported vector: constant
for k in range(7):
    cen = (pts[k] + pts[k + 1]) / 3
    arrow(axb, cen, cen + 0.3 * w2, col["tangent"])
Theta = float(np.sum(theta))
cen1 = (pts[0] + pts[1]) / 3
arrow(axb, cen1, cen1 + 0.3 * rot(w2, -Theta), col["accent"], 2.0)       # w_7 carried back into the first copy
assert abs(ch45.wrap(-Theta) - np.pi / 2) < 1e-12
axb.plot(0, 0, "o", color=col["main"], ms=4)
axb.text(-0.2, -0.08, r"$\mathbf{x}_7$", fontsize=11)
axb.text(*(cen1 + [-0.05, -0.12]), r"$f_1$", fontsize=9, color=col["aux"])
axb.text(*((pts[6] + pts[7]) / 3 + [0.03, -0.1]), r"$f_1$ again", fontsize=9, color=col["aux"])
axb.add_patch(Arc((0, 0), 0.5, 0.5, theta1=np.degrees(phis[6]), theta2=np.degrees(o + 2 * np.pi),
                  color=col["normal"], lw=1.4))
gm = (phis[6] + o + 2 * np.pi) / 2
axb.text(0.3 * np.cos(gm) + 0.02, 0.3 * np.sin(gm) + 0.05, r"$\delta = \pi/2$", fontsize=10, color=col["normal"])
axb.set_title(r"(b) unfolded: $\Theta = 3\pi/2$, the return to $f_1$ dashed", fontsize=11)
for ax in (axa, axb):
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06)
dgfig.save(fig, __file__)
