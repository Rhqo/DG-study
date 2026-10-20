"""Figure 44.3.1: a straightest geodesic through a cube corner, and a geodesic triangle around it (Section 44.3).

The three faces of the unit cube at the corner c = (1, 1, 0) (Theta = 3 pi/2, delta = pi/2), unfolded around c and
drawn as seen from outside (counterclockwise in the picture = counterclockwise about the outward normal):
bottom z = 0 -> (y - 1, x - 1) (third quadrant), x = 1 -> (y - 1, z) (second), y = 1 -> (1 - x, z) (first), and a
dashed second copy of y = 1 attached to the bottom along the edge c -> (0, 1, 0), the map (x, 1, z) -> (z, x - 1)
(fourth quadrant: the first-quadrant copy rotated by -90 degrees). The cone of angle 3 pi/2 is drawn with one face
twice.
(a) p = (0.5, 0.2, 0): the straightest geodesic aimed at c (orange; traced with ch44.trace_straightest, length 1.6)
    leaves equal angles 3 pi/4 on both sides; its end point q and the two shortest paths from p to q (blue, green),
    straight in this unfolding, one through x = 1 and one through the copy of y = 1.
(b) the triangle A = (0.5, 1, 0) -> B = (1, 1, 0.5) -> C = (1, 0.5, 0) -> A around c, each side straight in its face;
    turning angle +pi/2 at A, B, C (counterclockwise about the outward normal: c is on the left).
Self-checks: the drawing is orientation preserving on each face (outward normals); the traced path passes through c
with theta_l = theta_r = 3 pi/4; |pq| along the two shortest paths is 1.482 < 1.6; the turning angles of the triangle,
computed in 3D with the outward normals and (44.3.4), are +pi/2 and 3 (pi/2) + delta = 2 pi.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-3-1-cube-corner.py
"""

import pathlib
import sys

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
OUT = {"bottom": np.array([0, 0, -1.0]), "x1": np.array([1.0, 0, 0]), "y1": np.array([0, 1.0, 0])}


def unfold(P, copy=False):
    """3D points on the three faces at c = (1, 1, 0) -> 2D unfolding seen from outside (copy: y = 1 in the fourth
    quadrant)."""
    P = np.atleast_2d(P)
    out = np.zeros((len(P), 2))
    for n, (x, y, z) in enumerate(P):
        if abs(z) < 1e-12:
            out[n] = (y - 1, x - 1)
        elif abs(x - 1) < 1e-12:
            out[n] = (y - 1, z)
        elif abs(y - 1) < 1e-12:
            out[n] = (z, x - 1) if copy else (1 - x, z)
        else:
            raise ValueError(P[n])
    return out


# the drawing is orientation preserving: the images of a positive basis (u, v) with u x v = N have det > 0
for name, base, u, v in (("bottom", np.array([0.5, 0.5, 0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])),
                         ("x1", np.array([1, 0.5, 0.5]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
                         ("y1", np.array([0.5, 1, 0.5]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))):
    assert np.cross(u, v) @ OUT[name] > 0
    for cp in ((False, True) if name == "y1" else (False,)):
        b2 = unfold(base, cp)[0]
        du, dv = unfold(base + 0.1 * u, cp)[0] - b2, unfold(base + 0.1 * v, cp)[0] - b2
        assert du[0] * dv[1] - du[1] * dv[0] > 0, name

V, F = ch44.unit_cube(1)
p = np.array([0.5, 0.2, 0.0])
c = np.array([1.0, 1.0, 0.0])
res = ch44.trace_straightest(V, F, ch44.locate_face(V, F, p), p, ch44.unit(c - p), 1.6)
assert res["events"][0][0] == "vertex" and abs(res["events"][0][2] - 1.5 * np.pi) < 1e-12
path = res["points"]
q = path[-1]
P2 = unfold(path)
ang_in = np.arctan2(*(P2[0] - P2[1])[::-1])
ang_out = np.arctan2(*(P2[2] - P2[1])[::-1])
assert abs(((ang_in - ang_out) % (2 * np.pi)) - 3 * np.pi / 4) < 1e-9        # 135 degrees in the unfolding
q1, q2 = unfold(q)[0], unfold(q, copy=True)[0]
L1, L2 = np.linalg.norm(q1 - P2[0]), np.linalg.norm(q2 - P2[0])
assert abs(L1 - L2) < 1e-9 and L1 < 1.6
d_exact = ch44.cube_distance(p, q[None])[0]
assert abs(d_exact - L1) < 1e-9
print(f"straightest length 1.6, shortest d(p, q) = {L1:.4f} (two paths)")


def transport(w, e, Nf, Ng):
    """(44.3.4): carry the direction w across the edge e from the face with normal Nf to the face with normal Ng."""
    e = e / np.linalg.norm(e)
    return (w @ e) * e + (w @ np.cross(Nf, e)) * np.cross(Ng, e)


A, B, C = np.array([0.5, 1, 0.0]), np.array([1, 1, 0.5]), np.array([1, 0.5, 0.0])
tAB, tBC, tCA = (ch44.unit(B - A), ch44.unit(C - B), ch44.unit(A - C))          # in y = 1, x = 1, bottom
turn3d = []
for t_in, t_out, e, Nf, Ng in ((tAB, tBC, np.array([0, 0, 1.0]), OUT["y1"], OUT["x1"]),
                               (tBC, tCA, np.array([0, 1.0, 0]), OUT["x1"], OUT["bottom"]),
                               (tCA, tAB, np.array([1.0, 0, 0]), OUT["bottom"], OUT["y1"])):
    w = transport(t_in, e, Nf, Ng)
    turn3d.append(np.arctan2(np.cross(w, t_out) @ Ng, w @ t_out))
assert np.allclose(turn3d, np.pi / 2) and abs(sum(turn3d) + np.pi / 2 - 2 * np.pi) < 1e-12
# the same turning angles read off the picture (counterclockwise positive)
a_b, b2, c2 = unfold(A)[0], unfold(B)[0], unfold(C)[0]     # A as a point of the bottom (on its edge y = 1)
a_y = np.array([1 - A[0], A[2]])                            # A as a point of y = 1 (first quadrant)
assert np.allclose(a_y, [0.5, 0]) and np.allclose(b2, [0, 0.5]) and np.allclose(c2, [-0.5, 0]) and np.allclose(a_b, [0, -0.5])
Rm90 = np.array([[0, 1], [-1, 0]])                 # first-quadrant y = 1 -> fourth-quadrant copy (rotation by -90)
turn = []
for u, w in ((b2 - a_y, c2 - b2), (c2 - b2, a_b - c2), (a_b - c2, Rm90 @ (b2 - a_y))):
    turn.append(np.arctan2(u[0] * w[1] - u[1] * w[0], u @ w))
assert np.allclose(turn, turn3d)
print("triangle turning angles:", np.round(turn, 6))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.4))
for ax in (axa, axb):
    for (x0, y0), lab, dashed in (((-1, -1), "bottom", False), ((-1, 0), "x = 1", False), ((0, 0), "y = 1", False),
                                  ((0, -1), "y = 1 (copy)", True)):
        ax.add_patch(plt.Rectangle((x0, y0), 1, 1, fill=not dashed, fc=col["surface"], alpha=0.35 if not dashed else 1,
                                   ec=col["main"], lw=0.8, ls="--" if dashed else "-"))
        ax.text(x0 + 0.5, y0 + 0.5 + (0.38 if y0 == 0 else -0.45), lab, ha="center", fontsize=9, color=col["aux"])
    ax.plot(0, 0, "s", color=col["main"], ms=4, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_axis_off()
axa.text(0.04, -0.13, r"$c$", fontsize=12)
axa.plot(P2[:, 0], P2[:, 1], color=col["accent"], lw=2.4, zorder=4)
axa.plot([P2[0, 0], q1[0]], [P2[0, 1], q1[1]], color=col["tangent"], lw=1.6, zorder=3)
axa.plot([P2[0, 0], q2[0]], [P2[0, 1], q2[1]], color=col["third"], lw=1.6, zorder=3)
for pt, lab, off in ((P2[0], r"$p$", (-0.1, -0.1)), (q1, r"$q$", (0.04, 0.03)), (q2, r"$q$", (0.04, 0.03))):
    axa.plot(*pt, "o", color=col["main"], ms=4, zorder=6)
    axa.text(pt[0] + off[0], pt[1] + off[1], lab, fontsize=12)
axa.text(-0.95, 0.58, r"$\theta_l = \theta_r = 3\pi/4$", fontsize=10)
axa.set_title("(a) straightest vs. shortest", fontsize=11)

xs = [a_y[0], b2[0], c2[0], a_b[0]]
ys = [a_y[1], b2[1], c2[1], a_b[1]]
axb.plot(xs, ys, color=col["accent"], lw=2.2, zorder=4)
axb.plot([a_b[0], a_y[0]], [a_b[1], a_y[1]], color=col["accent"], lw=0.8, ls=":", zorder=3)
for pt, lab, off in ((a_y, r"$A$", (0.04, 0.05)), (b2, r"$B$", (0.05, 0.03)), (c2, r"$C$", (-0.17, 0.05)),
                     (a_b, r"$A$", (0.05, -0.12))):
    axb.plot(*pt, "o", color=col["main"], ms=4, zorder=6)
    axb.text(pt[0] + off[0], pt[1] + off[1], lab, fontsize=12)
axb.text(0.05, -0.14, r"$c$", fontsize=12)
axb.text(-0.3, 0.64, r"$\pi/2$", fontsize=10, color=col["accent"])
axb.text(-0.74, -0.27, r"$\pi/2$", fontsize=10, color=col["accent"])
axb.text(-0.38, -0.7, r"$\pi/2$", fontsize=10, color=col["accent"])
axb.set_title(r"(b) triangle around $c$: $3\cdot\frac{\pi}{2} + \delta = 2\pi$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.06)
dgfig.save(fig, __file__)
