"""Figure 28.4.2: three ways to interpolate between two camera poses; the camera center paths differ (Section 28.4).

Camera poses T_wc (OpenCV camera axes, world z up): T0 at c0 = (0, 0, 0) looking along +x; T1 at c1 = (4, 2, 1)
looking along R_z(120 deg) e_1. For 0 <= s <= 1:
  blue   : Lie-group interpolation  T(s) = T0 Exp(s Log(T0^{-1} T1))  (screw motion; the same curve whether one
           interpolates T_wc or T_cw, on the left or on the right)
  orange : SO(3) x R^3 on T_wc: R_wc(s) = SLERP, c(s) = (1 - s) c0 + s c1   (straight center)
  green  : SO(3) x R^3 on T_cw: R_cw(s) = SLERP, t_cw(s) linear; center c(s) = -R_cw(s)^T t_cw(s)  (dashed)
All three share the same rotations R(s). Camera frusta are drawn at s = 0, 0.25, 0.5, 0.75, 1 for blue and orange.
(a) 3D view, with faint shadows of the three center paths on the plane z = -0.6, dotted drop lines from c0 and c1
and small world x, y, z axes as orientation cues; (b) top view (world xy-plane). Also writes an interactive version.
Self-check: the three rotation paths coincide; the blue curve is invariant under inversion and under left/right
conventions; the orange curve is the geodesic of the left-invariant metric |omega_body|^2 + |c'|^2 and is the
shortest of the three in that metric; the screw axis is vertical (the rotation is about z).

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-4-2-se3-interpolation.py``
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


def cam_pose(center, yaw):
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    r = np.cross(f, [0, 0, 1.0])
    d = np.cross(f, r)
    return L.make_T(np.column_stack([r, d, f]), center)


T0 = cam_pose(np.zeros(3), 0.0)
T1 = cam_pose(np.array([4.0, 2.0, 1.0]), np.radians(120))
ss = np.linspace(0, 1, 201)
xi = L.logSE3(L.inv_T(T0) @ T1)
lie = [T0 @ L.expSE3(s * xi) for s in ss]
dR = L.logSO3(T0[:3, :3].T @ T1[:3, :3])
prod_wc = [L.make_T(T0[:3, :3] @ L.expSO3(s * dR), (1 - s) * T0[:3, 3] + s * T1[:3, 3]) for s in ss]
C0, C1 = L.inv_T(T0), L.inv_T(T1)
dRc = L.logSO3(C0[:3, :3].T @ C1[:3, :3])
prod_cw = [L.inv_T(L.make_T(C0[:3, :3] @ L.expSO3(s * dRc), (1 - s) * C0[:3, 3] + s * C1[:3, 3])) for s in ss]

for a, b, c in zip(lie, prod_wc, prod_cw):
    assert np.allclose(a[:3, :3], b[:3, :3], atol=1e-12) and np.allclose(b[:3, :3], c[:3, :3], atol=1e-12)
lie_cw = [L.inv_T(C0 @ L.expSE3(s * L.logSE3(L.inv_T(C0) @ C1))) for s in ss]
lie_left = [L.expSE3(s * L.logSE3(T1 @ L.inv_T(T0))) @ T0 for s in ss]
for a, b, c in zip(lie, lie_cw, lie_left):
    assert np.allclose(a, b, atol=1e-11) and np.allclose(a, c, atol=1e-11)


def length(Ts):
    tot = 0.0
    for A, Bm in zip(Ts[:-1], Ts[1:]):
        w = L.logSO3(A[:3, :3].T @ Bm[:3, :3])
        tot += np.sqrt(w @ w + np.sum((Bm[:3, 3] - A[:3, 3]) ** 2))
    return tot


lens = [length(lie), length(prod_wc), length(prod_cw)]
assert lens[1] < lens[0] and lens[1] < lens[2]
assert abs(lens[1] - np.sqrt(np.radians(120) ** 2 + 21.0)) < 1e-6
cl, cp, cc = (np.array([T[:3, 3] for T in X]) for X in (lie, prod_wc, prod_cw))
dev = [np.max(np.linalg.norm(np.cross(P - P[0], (P[-1] - P[0]) / np.linalg.norm(P[-1] - P[0])), axis=1))
       for P in (cl, cp, cc)]
print("lengths (lie, prod_wc, prod_cw):", np.round(lens, 4), " max deviation from the chord:", np.round(dev, 3))

W, H, D = 0.28, 0.2, 0.55
corners = np.array([[-W, -H, D], [W, -H, D], [W, H, D], [-W, H, D]])


def frustum_lines(T):
    cs = (T[:3, :3] @ corners.T).T + T[:3, 3]
    segs = [np.stack([T[:3, 3], q]) for q in cs] + [np.vstack([cs, cs[:1]])]
    return segs


marks = [0, 50, 100, 150, 200]
fig = plt.figure(figsize=(8.8, 4.3))
ax = dgfig.axes3d(fig, pos=121, elev=30, azim=-70)
ax.set_position([0.0, 0.03, 0.52, 0.92])
for P, col, ls, lab in [(cl, C["tangent"], "-", "Lie group (screw)"), (cp, C["accent"], "-", r"$\mathrm{SO}(3)\times\mathbb{R}^3$ on $T_{wc}$"),
                        (cc, C["third"], (0, (4, 2)), r"$\mathrm{SO}(3)\times\mathbb{R}^3$ on $T_{cw}$")]:
    ax.plot(*P.T, color=col, ls=ls, lw=2.0, zorder=5)
for X, col in [(lie, C["tangent"]), (prod_wc, C["accent"])]:
    for k in marks:
        for seg in frustum_lines(X[k]):
            ax.plot(*seg.T, color=col, lw=1.0, zorder=6)
ZG = -0.6                                  # ground plane for shadows and the world axes (orientation cues)
for P, col, ls in [(cl, C["tangent"], "-"), (cp, C["accent"], "-"), (cc, C["third"], (0, (4, 2)))]:
    ax.plot(P[:, 0], P[:, 1], np.full(len(P), ZG), color=col, ls=ls, lw=0.9, alpha=0.4, zorder=1)
for cpt in (cl[0], cl[-1]):
    ax.plot([cpt[0]] * 2, [cpt[1]] * 2, [cpt[2], ZG], color=C["aux"], lw=0.8, ls=":", zorder=1)
o3 = np.array([-0.6, 2.2, ZG])
for e, lab in zip(np.eye(3), [r"$x$", r"$y$", r"$z$"]):
    dgfig.arrow3d(ax, o3, 0.9 * e, role="aux", label=lab, lw=1.1, head=8, fontsize=11, zorder=2)
ax.text(*(cl[0] + np.array([-0.5, -0.4, -0.35])), r"$T_0$", fontsize=12)
ax.text(*(cl[-1] + np.array([0.2, 0.2, 0.3])), r"$T_1$", fontsize=12)
dgfig.equal_aspect(ax, np.vstack([cl, cp, cc, [[-0.7, -0.6, ZG], [4.5, 3.1, 1.4]]]), zoom=1.3)
ax.set_title("(a) 3D", fontsize=12, y=0.97)

bx = fig.add_axes([0.58, 0.14, 0.4, 0.76])
for P, col, ls, lab in [(cl, C["tangent"], "-", "Lie group: $T_0\\,\\mathrm{Exp}(s\\,\\xi)$"),
                        (cp, C["accent"], "-", r"$\mathrm{SO}(3)\times\mathbb{R}^3$ on $T_{wc}$"),
                        (cc, C["third"], (0, (4, 2)), r"$\mathrm{SO}(3)\times\mathbb{R}^3$ on $T_{cw}$")]:
    bx.plot(P[:, 0], P[:, 1], color=col, ls=ls, lw=2.0, label=lab)
for X, col in [(lie, C["tangent"]), (prod_wc, C["accent"])]:
    for k in marks:
        T = X[k]
        f = T[:3, 2]
        bx.annotate("", xy=T[:2, 3] + 0.55 * f[:2], xytext=T[:2, 3],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=9, shrinkA=0, shrinkB=0))
        bx.plot(*T[:2, 3], "o", color=col, ms=4)
bx.text(-0.45, -0.45, r"$c_0$", fontsize=12)
bx.text(4.1, 2.05, r"$c_1$", fontsize=12)
bx.set_aspect("equal")
bx.set_xlim(-0.8, 5.0)
bx.set_ylim(-1.6, 3.2)
bx.set_xlabel(r"$x$")
bx.set_ylabel(r"$y$")
bx.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95)
bx.set_title("(b) top view (arrows: viewing direction)", fontsize=11.5)

dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go  # noqa: E402

tr = []
for P, col, dash, name in [(cl, "#0072B2", "solid", "Lie group (screw)"), (cp, "#E69F00", "solid", "SO(3)xR3 on T_wc"),
                           (cc, "#009E73", "dash", "SO(3)xR3 on T_cw")]:
    tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=col, width=6, dash=dash),
                           name=name))
for X, col, name in [(lie, "#0072B2", "Lie group frusta"), (prod_wc, "#E69F00", "SO(3)xR3 (T_wc) frusta")]:
    xs, ys, zs = [], [], []
    for k in marks:
        for seg in frustum_lines(X[k]):
            xs += list(seg[:, 0]) + [None]
            ys += list(seg[:, 1]) + [None]
            zs += list(seg[:, 2]) + [None]
    tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=3), name=name))
pfig = go.Figure(tr)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                   margin=dict(l=0, r=0, t=40, b=0),
                   title=dict(text="Interpolating T0 -> T1: Lie group (screw) vs SO(3)xR3 on T_wc vs on T_cw",
                              x=0.5, font=dict(size=13)))
out = pathlib.Path(__file__).with_name(pathlib.Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-28-4-2-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
