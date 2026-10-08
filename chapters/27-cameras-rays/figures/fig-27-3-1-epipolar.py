"""Figure 27.3.1: the epipolar plane; ray 1 projects onto the epipolar line of camera 2 (Section 27.3).

Camera 1 at the origin (world = camera-1 frame). Camera 2: center c2 = (2, 0, 0.3), rotated by 18 deg about the y axis
(t2 = R t1 + T, T = -R c2). Scene point X = (0.9, -0.5, 5). Both image planes are drawn at depth 1 in their own frames
(normalized cameras, 1.4 x 1.0).
(a) 3D: centers C1, C2, baseline (black), the two rays to X (orange), the epipolar plane (triangle C1 C2 X, light green),
    the epipolar lines in both image planes (green), and points of ray 1 at depths s = 2, 3.5, 8 (blue) with their
    projections into image 2.
(b) Image plane of camera 2 (normalized coordinates): the epipolar line l2 = E pbar1 (green), the epipole e2 (image of C1),
    and the images of the points s pbar1 of ray 1 for s = 0.5, 1, 2, 3.5, 5 (= X), 8, 20 and s -> infinity
    (the vanishing point R pbar1).
Also writes an interactive plotly version.
Self-checks: all image points lie on l2; e2 is the limit s -> 0; the s -> infinity point is R pbar1.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-3-1-epipolar.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

C = dgfig.COLORS


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


a = np.radians(18)
R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1.0, 0], [-np.sin(a), 0, np.cos(a)]])
c2 = np.array([2.0, 0, 0.3])
T = -R @ c2
E = hat(T) @ R
X = np.array([0.9, -0.5, 5.0])
p1 = X / X[2]
q = R @ X + T
p2 = q / q[2]
l2 = E @ p1
assert abs(l2 @ p2) < 1e-12
e2 = T / T[2]
svals = [0.5, 1, 2, 3.5, 5, 8, 20]
imgs = []
for s in svals:
    qq = R @ (s * p1) + T
    imgs.append(qq / qq[2])
    assert abs(l2 @ imgs[-1]) < 1e-12
vinf = R @ p1
vinf = vinf / vinf[2]
assert abs(l2 @ vinf) < 1e-12
qq = R @ (1e-9 * p1) + T
assert np.allclose(qq / qq[2], e2, atol=1e-6)

W2, H2 = 0.45, 0.32
rect_cam = np.array([[-W2, -H2, 1], [W2, -H2, 1], [W2, H2, 1], [-W2, H2, 1]])


def cam2_to_world(v):
    return R.T @ (v - T)


rect1 = rect_cam
rect2 = np.array([cam2_to_world(v) for v in rect_cam])
# plot coordinates (x, z, -y)
Pp = lambda v: np.array([v[0], v[2], -v[1]])


def plane_line_in_image(nrm, pts_frame_to_world, inv=False):
    """epipolar line inside an image rectangle: intersect the epipolar plane with the image plane (in camera frame)"""
    return None


fig = plt.figure(figsize=(8.0, 7.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.08)
ax = dgfig.axes3d(fig, pos=gs[0], elev=26, azim=-68)
for rect, col in ((rect1, C["region"]), (rect2, C["region"])):
    ax.add_collection3d(Poly3DCollection([np.array([Pp(v) for v in rect])], facecolor=col, alpha=0.3,
                                         edgecolor=C["tangent"], lw=0.8))
tri = np.array([Pp(np.zeros(3)), Pp(c2), Pp(X)])
ax.add_collection3d(Poly3DCollection([tri], facecolor=C["third"], alpha=0.12, edgecolor="none"))
ax.plot(*np.array([Pp(np.zeros(3)), Pp(c2)]).T, color="k", lw=1.6)
for cc in (np.zeros(3), c2):
    ax.plot(*np.array([Pp(cc), Pp(X)]).T, color=C["accent"], lw=1.8)
# epipolar lines in image planes: sample the plane through C1, C2, X restricted to each rectangle
nrm1 = np.cross(c2, X)           # plane normal in frame 1 (through origin)
s_ = np.linspace(-W2, W2, 200)
# image 1: points (x, y, 1) with <nrm1, (x,y,1)> = 0
y1 = -(nrm1[0] * s_ + nrm1[2]) / nrm1[1]
m1 = np.abs(y1) <= H2
ax.plot(*np.array([Pp(np.array([x_, y_, 1.0])) for x_, y_ in zip(s_[m1], y1[m1])]).T, color=C["third"], lw=1.8)
# image 2: l2 in camera-2 normalized coordinates
y2 = -(l2[0] * s_ + l2[2]) / l2[1]
m2 = np.abs(y2) <= H2
ax.plot(*np.array([Pp(cam2_to_world(np.array([x_, y_, 1.0]))) for x_, y_ in zip(s_[m2], y2[m2])]).T,
        color=C["third"], lw=1.8)
for s in (2, 3.5, 8):
    Y = s * p1
    ax.scatter(*Pp(Y), color=C["tangent"], s=14, zorder=8)
    ax.text(*(Pp(Y) + np.array([-0.55, 0, 0.05])), rf"$s={s:g}$", fontsize=10, color=C["tangent"])
    qq = R @ Y + T
    im = cam2_to_world(qq / qq[2])
    ax.plot(*np.array([Pp(c2), Pp(Y)]).T, color=C["tangent"], lw=0.6, ls="--")
    ax.scatter(*Pp(im), color=C["tangent"], s=10, zorder=8)
ax.plot(*np.array([Pp(np.zeros(3)), Pp(9 * p1)]).T, color=C["accent"], lw=0.8, ls=":")
for v, lab, off in ((np.zeros(3), r"$C_1$", (-0.35, 0, -0.25)), (c2, r"$C_2$", (0.1, 0, -0.3)),
                    (X, r"$X$", (0.1, 0, 0.12))):
    ax.scatter(*Pp(v), color="k", s=16, zorder=9)
    ax.text(*(Pp(v) + np.array(off)), lab, fontsize=12)
ax.text(*(Pp(0.5 * c2) + np.array([-0.35, 0, -0.45])), "baseline", fontsize=11)
ax.text(*(Pp(0.55 * X) + np.array([-0.55, 0, 0.1])), "ray 1", fontsize=11, color=C["accent"])
ax.text(*(Pp(0.5 * (c2 + X)) + np.array([0.15, 0, 0.0])), "ray 2", fontsize=11, color=C["accent"])
ax.text(*(Pp(np.array([-W2, -H2, 1.0])) + np.array([-0.85, 0, -0.1])), "image 1", fontsize=11, color=C["tangent"])
ax.text(*(Pp(cam2_to_world(np.array([W2, -H2, 1.0]))) + np.array([0.08, 0, 0.05])), "image 2", fontsize=11,
        color=C["tangent"])
ax.set_xlim(-1, 3)
ax.set_ylim(0, 6)
ax.set_zlim(-1.5, 1.5)
ax.set_box_aspect((4, 6, 3), zoom=1.45)
ax.set_title("(a) the epipolar plane through $C_1$, $C_2$, $X$", fontsize=12, y=1.1)

bx = fig.add_subplot(gs[1])
xs = np.linspace(-2.7, 1.1, 50)
bx.plot(xs, -(l2[0] * xs + l2[2]) / l2[1], color=C["third"], lw=1.8)
bx.plot([-W2, W2, W2, -W2, -W2], [-H2, -H2, H2, H2, -H2], color=C["tangent"], lw=0.9)
bx.annotate(r"to $e_2$ at $\bar u_x=-6.0$ (image of $C_1$)", xy=(-2.68, 0.0), xytext=(-2.0, 0.5), fontsize=11,
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
for s, im in zip(svals, imgs):
    bx.plot(*im[:2], "o", color=C["tangent"] if s != 5 else C["accent"], ms=5 if s != 5 else 7, zorder=5)
for s, im, dy in ((0.5, imgs[0], -0.16), (1, imgs[1], -0.16), (2, imgs[2], -0.16), (3.5, imgs[3], 0.24),
                  (5, imgs[4], -0.16), (8, imgs[5], 0.24), (20, imgs[6], -0.16)):
    bx.text(im[0], im[1] + dy, rf"$s={s:g}$", fontsize=10, ha="center", color=C["accent"] if s == 5 else "k")
bx.plot(*vinf[:2], "s", color=C["normal"], ms=6)
bx.text(vinf[0] + 0.04, vinf[1] - 0.18, r"$s\to\infty$: $R\bar p_1$", fontsize=11, color=C["normal"])
bx.text(-1.75, 0.27, r"$l_2=E\bar p_1$", fontsize=12, color=C["third"])
bx.text(-0.44, 0.31, "image 2 frame", fontsize=10, color=C["tangent"], va="top")
bx.set_xlim(-2.7, 1.1)
bx.set_ylim(0.65, -0.55)
bx.set_aspect("equal")
bx.set_xlabel(r"$\bar u_x$ (camera 2)")
bx.set_ylabel(r"$\bar u_y$ (down)")
bx.set_title(r"(b) ray 1 seen by camera 2", fontsize=12)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive
import plotly.graph_objects as go

tr = []
for rect in (rect1, rect2):
    tr.append(go.Mesh3d(x=rect[:, 0], y=rect[:, 1], z=rect[:, 2], i=[0, 0], j=[1, 2], k=[2, 3], color=C["region"],
                        opacity=0.35, hoverinfo="skip", showlegend=False))
tri3 = np.array([np.zeros(3), c2, X])
tr.append(go.Mesh3d(x=tri3[:, 0], y=tri3[:, 1], z=tri3[:, 2], i=[0], j=[1], k=[2], color=C["third"], opacity=0.2,
                    name="epipolar plane", showlegend=True))
tr.append(go.Scatter3d(x=[0, c2[0]], y=[0, c2[1]], z=[0, c2[2]], mode="lines", line=dict(color="black", width=5),
                       name="baseline"))
for cc, nm in ((np.zeros(3), "ray 1"), (c2, "ray 2")):
    tr.append(go.Scatter3d(x=[cc[0], X[0]], y=[cc[1], X[1]], z=[cc[2], X[2]], mode="lines",
                           line=dict(color=C["accent"], width=5), name=nm))
pts_l1 = np.array([[x_, y_, 1.0] for x_, y_ in zip(s_[m1], y1[m1])])
pts_l2 = np.array([cam2_to_world(np.array([x_, y_, 1.0])) for x_, y_ in zip(s_[m2], y2[m2])])
for pl, nm in ((pts_l1, "epipolar line 1"), (pts_l2, "epipolar line 2")):
    tr.append(go.Scatter3d(x=pl[:, 0], y=pl[:, 1], z=pl[:, 2], mode="lines", line=dict(color=C["third"], width=6),
                           name=nm))
for s in (2, 3.5, 8):
    Y = s * p1
    qq = R @ Y + T
    im = cam2_to_world(qq / qq[2])
    tr.append(go.Scatter3d(x=[c2[0], Y[0]], y=[c2[1], Y[1]], z=[c2[2], Y[2]], mode="lines+markers",
                           line=dict(color=C["tangent"], width=2, dash="dash"), marker=dict(size=3, color=C["tangent"]),
                           showlegend=False, hoverinfo="skip"))
tr.append(go.Scatter3d(x=[0, c2[0], X[0]], y=[0, c2[1], X[1]], z=[0, c2[2], X[2]], mode="markers+text",
                       text=["C1", "C2", "X"], marker=dict(size=4, color="black"), textposition="top center",
                       showlegend=False))
pf = go.Figure(tr)
pf.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z",
                            camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=1.2, y=-1.0, z=-1.4))),
                 margin=dict(l=0, r=0, t=40, b=0),
                 title=dict(text="Epipolar plane: both rays and the baseline lie in one plane", x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pf.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-27-3-1-interactive", config={"displaylogo": False})
print(f"saved {out}")
print("e2", e2, "vinf", vinf, "X img", p2)
