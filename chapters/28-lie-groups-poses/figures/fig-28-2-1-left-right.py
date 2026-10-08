"""Figure 28.2.1: the same twist applied on the right and on the left of a camera pose T_wc (Section 28.2).

A camera (OpenCV axes: x right, y down, z forward) sits at c = (3, 0, 1.2) and looks at the world origin; its pose
is T_wc = (R_wc, c). The twist xi = (0, 0, 0, 0, 0, 0.6) (rotation 0.6 rad about the third axis) is applied
  * on the right: T_wc Exp(xi)  -> rotation about the camera's own z axis (the optical axis): the camera rolls in
    place (blue);
  * on the left:  Exp(xi) T_wc  -> rotation about the world z axis through the world origin: the camera orbits the
    origin by 0.6 rad (orange).
The original camera is black. Each camera is a frustum (apex = center, depth 0.8) with a small filled triangle on
the edge that points to camera -y ("up" in the image).
(a) view along the original optical axis from behind the camera: orthographic projection onto the plane spanned by
    the original camera's x axis (right) and -y axis (up). The optical axis points into the page at the center. Only the
    original (black) and the right-perturbed camera (blue) are drawn: their image rectangles at depth 0.8 lie in the
    same plane, and the blue one is the black one rotated by 0.6 rad about the center (roll). The left-perturbed camera
    moves by 2 * 3 sin(0.3) = 1.77 (out of this view) and is shown in (b).
(b) top view (projection to the world xy-plane) with the orbit arc of radius |c_xy| = 3.
Also writes an interactive version (fig-28-2-1-left-right-interactive.html).
Self-check: right: center unchanged and optical axis unchanged; left: center rotated by 0.6 rad about z;
Exp(Ad_T xi) T = T Exp(xi).

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-2-1-left-right.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dglie as L  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
c = np.array([3.0, 0.0, 1.2])
f = -c / np.linalg.norm(c)
r = np.cross(f, [0, 0, 1.0])
r /= np.linalg.norm(r)
d = np.cross(f, r)
Rwc = np.column_stack([r, d, f])
assert np.isclose(np.linalg.det(Rwc), 1.0)
T = L.make_T(Rwc, c)
xi = np.array([0, 0, 0, 0, 0, 0.6])
T_right = T @ L.expSE3(xi)
T_left = L.expSE3(xi) @ T

assert np.allclose(T_right[:3, 3], c)
assert np.allclose(T_right[:3, 2], T[:3, 2])
assert np.allclose(T_left[:3, 3], L.expSO3([0, 0, 0.6]) @ c)
assert np.allclose(L.expSE3(L.Ad_SE3(T) @ xi) @ T, T_right, atol=1e-12)

W, H, D = 0.42, 0.3, 0.8
corners_c = np.array([[-W, -H, D], [W, -H, D], [W, H, D], [-W, H, D]])
tri_c = np.array([[-0.12, -H, D], [0.12, -H, D], [0.0, -H - 0.14, D]])


def frustum(Tm):
    Rm, tm = Tm[:3, :3], Tm[:3, 3]
    cs = (Rm @ corners_c.T).T + tm
    tri = (Rm @ tri_c.T).T + tm
    return tm, cs, tri


cams = [(T, C["main"], "original", "-"), (T_right, C["tangent"], r"right: $T_{wc}\,\mathrm{Exp}(\xi)$", "-"),
        (T_left, C["accent"], r"left: $\mathrm{Exp}(\xi)\,T_{wc}$", "-")]

fig = plt.figure(figsize=(8.8, 4.3))
# ---------------------------------------------------------------- (a) view along the optical axis
E_RIGHT, E_UP = r, -d                                  # original camera x (right) and -y (up) in world coordinates


def proj_a(p):
    q = np.asarray(p, float) - c
    return np.array([q @ E_RIGHT, q @ E_UP])


ax = fig.add_axes([0.05, 0.12, 0.38, 0.78])
roll_check = []
for Tm, col, lab, ls in cams[:2]:
    tm, cs, tri = frustum(Tm)
    P = np.array([proj_a(q) for q in cs])
    roll_check.append(P)
    ax.fill(*P.T, color=col, alpha=0.06, lw=0)
    ax.plot(*np.vstack([P, P[:1]]).T, color=col, lw=1.6, label=lab)
    for q in P:                                         # edges from the center (apex) to the corners, seen end-on
        ax.plot([0, q[0]], [0, q[1]], color=col, lw=0.6, alpha=0.6)
    ax.fill(*np.array([proj_a(q) for q in tri]).T, color=col, alpha=0.9)
    for k in range(2):                                  # camera x_c and y_c (y_c points down in the image)
        e = proj_a(tm + 0.22 * Tm[:3, k])
        ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3,
                                                              mutation_scale=10, shrinkA=0, shrinkB=0))
    ax.text(*(proj_a(tm + 0.27 * Tm[:3, 0]) + [0.0, 0.035]), r"$x_c$", color=col, fontsize=10.5,
            ha="center")
    ax.text(*(proj_a(tm + 0.27 * Tm[:3, 1]) + [0.04, 0.0]), r"$y_c$", color=col, fontsize=10.5, va="center")
# the blue rectangle is the black one rotated about the center by 0.6 rad (clockwise as seen from behind)
Rot = np.array([[np.cos(-0.6), -np.sin(-0.6)], [np.sin(-0.6), np.cos(-0.6)]])
assert np.allclose(roll_check[1], roll_check[0] @ Rot.T, atol=1e-12)
ax.plot(0, 0, "o", ms=9, mfc="white", mec=C["main"], mew=1.2, zorder=6)
ax.plot(0, 0, "x", ms=6, color=C["main"], mew=1.2, zorder=7)      # optical axis into the page
ax.text(-0.05, 0.05, r"$z_c$ (into page)", fontsize=10, color=C["main"], ha="right")
aa = np.linspace(np.pi / 2, np.pi / 2 - 0.6, 30)
ax.plot(0.47 * np.cos(aa), 0.47 * np.sin(aa), color=C["tangent"], lw=1.2)
ax.annotate("", xy=(0.47 * np.cos(aa[-1]), 0.47 * np.sin(aa[-1])), xytext=(0.47 * np.cos(aa[-3]), 0.47 * np.sin(aa[-3])),
            arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.2, mutation_scale=10))
ax.text(0.2, 0.5, r"roll $0.6$ rad about $z_c$", color=C["tangent"], fontsize=10.5)
ax.set_aspect("equal")
ax.set_xlim(-0.62, 0.62)
ax.set_ylim(-0.76, 0.66)
ax.set_xlabel(r"original $x_c$ (right)")
ax.set_ylabel(r"original $-y_c$ (up)")
ax.text(-0.6, -0.74, "black: original", fontsize=9.5, color=C["main"], va="bottom")
ax.text(-0.6, -0.66, r"blue: right $T_{wc}\,\mathrm{Exp}(\xi)$, same center and axis", fontsize=9.5,
        color=C["tangent"], va="bottom")
ax.set_title("(a) view along the optical axis", fontsize=12)

bx = fig.add_axes([0.6, 0.12, 0.38, 0.78])
th = np.linspace(-0.2, 1.0, 100)
rad = np.linalg.norm(c[:2])
bx.plot(rad * np.cos(th), rad * np.sin(th), color=C["aux"], lw=0.8, ls=(0, (3, 3)))
bx.plot(0, 0, "o", color=C["main"], ms=5)
bx.text(0.08, -0.32, r"$O$ ($z_w$ axis)", fontsize=10.5)
for Tm, col, lab, ls in cams:
    tm, cs, tri = frustum(Tm)
    for q in cs:
        bx.plot([tm[0], q[0]], [tm[1], q[1]], color=col, lw=1.2)
    bx.plot(*np.vstack([cs, cs[:1]])[:, :2].T, color=col, lw=1.4)
    bx.fill(*tri[:, :2].T, color=col, alpha=0.9)
    bx.plot(*tm[:2], "o", color=col, ms=5, label=lab)
bx.annotate("", xy=T_left[:2, 3], xytext=c[:2], arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.3,
                                                                connectionstyle="arc3,rad=0.25"))
bx.text(3.05, 0.85, r"$0.6$ rad", color="#B07400", fontsize=11)
bx.set_aspect("equal")
bx.set_xlim(-0.5, 3.9)
bx.set_ylim(-1.0, 2.5)
bx.set_xlabel(r"$x_w$")
bx.set_ylabel(r"$y_w$")
bx.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.95)
bx.set_title("(b) top view (world $xy$-plane)", fontsize=12)

dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go  # noqa: E402

arc = np.array([L.expSO3([0, 0, a]) @ c for a in np.linspace(0, 0.6, 40)])

hexs = {C["main"]: "#000000", C["tangent"]: "#0072B2", C["accent"]: "#E69F00"}
tr = []
for k, col in enumerate(["#0072B2", "#009E73", "#D55E00"]):
    e = np.eye(3)[k] * 0.8
    tr.append(go.Scatter3d(x=[0, e[0]], y=[0, e[1]], z=[0, e[2]], mode="lines", line=dict(color=col, width=5),
                           name=["x_w", "y_w", "z_w"][k]))
for Tm, col, lab, ls in cams:
    tm, cs, tri = frustum(Tm)
    xs, ys, zs = [], [], []
    for q in cs:
        xs += [tm[0], q[0], None]
        ys += [tm[1], q[1], None]
        zs += [tm[2], q[2], None]
    loop = np.vstack([cs, cs[:1], [[np.nan] * 3], tri, tri[:1]])
    xs += list(loop[:, 0])
    ys += list(loop[:, 1])
    zs += list(loop[:, 2])
    name = {"original": "original T_wc"}.get(lab, "right: T_wc Exp(xi)" if col == C["tangent"] else
                                             "left: Exp(xi) T_wc")
    tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=hexs[col], width=5), name=name,
                           connectgaps=False))
tr.append(go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines", line=dict(color="#E69F00", width=3,
                                                                                    dash="dash"),
                       name="orbit of the center (left)"))
pfig = go.Figure(tr)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x_w", yaxis_title="y_w", zaxis_title="z_w"),
                   margin=dict(l=0, r=0, t=40, b=0),
                   title=dict(text="xi = (0,0,0, 0,0,0.6) applied on the right (roll) and on the left (orbit) of T_wc",
                              x=0.5, font=dict(size=13)))
out = pathlib.Path(__file__).with_name(pathlib.Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-28-2-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
