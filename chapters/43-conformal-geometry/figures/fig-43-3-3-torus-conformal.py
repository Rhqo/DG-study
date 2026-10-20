"""Figure 43.3.3: the torus of revolution T_{R,r} is conformally a flat rectangular torus (Example 43.3.15(c)).

Torus x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u), (R, r) = (2, 0.8). With
w(u) = int_0^u r / (R + r cos s) ds = 2r/sqrt(R^2 - r^2) * arctan(sqrt((R - r)/(R + r)) tan(u/2)) the induced metric is
g = r^2 du^2 + rho^2 dv^2 = rho^2 (dv^2 + dw^2), rho = R + r cos u, so (v, w) are conformal coordinates; w runs over
an interval of length W = 2 pi r / sqrt(R^2 - r^2) = 2.7422.
(a) the torus with the grid w = -W/2 + k W/10 (k = 0..9; each is a circle v -> x(u_k, v))
    and v = 2 pi j / 23 (j = 0..22); two cells are orange: one on the outer equator (w in [0, W/10], v near 297 deg) and
    one near the inner equator (w in [3W/10, 4W/10], v near 110 deg).
(b) the rectangle [0, 2 pi] x [-W/2, W/2] in the (v, w) plane with the same lines; every cell is a square of side
    2 pi / 23 x W/10 (aspect 1.004), the two orange cells are the same squares.
Self-check: E = r^2, G = rho^2 and dw/du = r/rho give g = rho^2(dv^2 + dw^2) (numerically at 7 points); W = 2.7422;
on the torus the two orange cells have side ratio (midline lengths) within 2 % of 1; they are small squares of
different sizes, side about rho * 0.27.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-3-3-torus-conformal.py``
"""

import pathlib
import sys

import dgfig  # noqa: E402
import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch43  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
R, r = 2.0, 0.8
W = 2 * np.pi * r / np.sqrt(R * R - r * r)
assert abs(W - 2.7422) < 1e-4


def X(u, v):
    rho = R + r * np.cos(u)
    return np.stack([rho * np.cos(v), rho * np.sin(v), r * np.sin(u)], -1)


def u_of_w(w):
    """inverse of ch43.torus_w on (-W/2, W/2) -> (-pi, pi)"""
    k = np.sqrt(R * R - r * r)
    return 2 * np.arctan(np.tan(w * k / (2 * r)) / np.sqrt((R - r) / (R + r)))


for uu in np.linspace(-2.5, 2.5, 7):
    hh = 1e-6
    dwdu = (ch43.torus_w(uu + hh) - ch43.torus_w(uu - hh)) / (2 * hh)
    assert abs(dwdu - r / (R + r * np.cos(uu))) < 1e-8 and abs(u_of_w(ch43.torus_w(uu)) - uu) < 1e-12
    Xu = (X(uu + hh, 0.3) - X(uu - hh, 0.3)) / (2 * hh)
    Xv = (X(uu, 0.3 + hh) - X(uu, 0.3 - hh)) / (2 * hh)
    rho = R + r * np.cos(uu)
    # metric in (v, w): x_w = x_u du/dw
    Xw = Xu / dwdu
    assert abs(Xv @ Xv - rho ** 2) < 1e-6 and abs(Xw @ Xw - rho ** 2) < 1e-6 and abs(Xv @ Xw) < 1e-8
dv, dw = 2 * np.pi / 23, W / 10
CELLS = (("outer", 19 * dv, 0.0), ("inner", 7 * dv, W / 2 - 2 * dw))
print("W = %.4f, cell %.4f x %.4f (aspect %.4f)" % (W, dv, dw, dw / dv))
side = {}
for name, v0, w0 in CELLS:
    wm = w0 + dw / 2
    ws = np.linspace(w0, w0 + dw - 1e-12, 401)
    rho_ws = R + r * np.cos(u_of_w(ws))
    along_w = np.trapezoid(rho_ws, ws)                       # arc length of the w-side (metric rho^2 dw^2)
    along_v = (R + r * np.cos(u_of_w(wm))) * dv           # length of the v-side through the middle
    side[name] = (along_v, along_w)
    assert abs(along_v / along_w - 1) < 0.02
print("cell sides on the torus (along v, along w):", {k: (round(a, 3), round(b, 3)) for k, (a, b) in side.items()})

fig = plt.figure(figsize=(6.6, 3.0))
ax = dgfig.axes3d(fig, 121, elev=38, azim=-62)
uu, vv = np.meshgrid(np.linspace(-np.pi, np.pi, 60), np.linspace(0, 2 * np.pi, 90))
P = X(uu, vv)
dgfig.surface(ax, P[..., 0], P[..., 1], P[..., 2], alpha=0.45, grid=False)
vis = lambda Q: True  # noqa: E731
for k in range(10):
    u_k = u_of_w(-W / 2 + k * dw) if k else np.pi
    Q = X(np.full(200, u_k), np.linspace(0, 2 * np.pi, 200))
    dgfig.curve3d(ax, Q, role="aux", lw=0.6, visible=(Q - np.c_[np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)), 0 * Q[:, 2]] * R) @ dgfig.view_vector(ax) > -0.2)
for j in range(23):
    Q = X(np.linspace(-np.pi, np.pi, 200), np.full(200, j * dv))
    nrm = Q - R * np.c_[np.cos(j * dv) * np.ones(200), np.sin(j * dv) * np.ones(200), np.zeros(200)]
    dgfig.curve3d(ax, Q, role="aux", lw=0.6, visible=nrm @ dgfig.view_vector(ax) > -0.2)
for name, v0, w0 in CELLS:
    ws = np.linspace(w0, w0 + dw - 1e-12, 12)
    vs = np.linspace(v0, v0 + dv, 12)
    loop = np.r_[X(u_of_w(ws), np.full(12, v0)), X(np.full(12, u_of_w(ws[-1])), vs),
                 X(u_of_w(ws[::-1]), np.full(12, v0 + dv)), X(np.full(12, u_of_w(ws[0])), vs[::-1])]
    ax.plot(*loop.T, color=col["accent"], lw=1.8, zorder=20)
dgfig.equal_aspect(ax, P.reshape(-1, 3), zoom=1.2)
ax.set_title(r"(a) $T_{2,0.8}$, grid in $(v, w)$", fontsize=11)

ax2 = fig.add_subplot(122)
for k in range(11):
    ax2.plot([0, 2 * np.pi], [-W / 2 + k * dw] * 2, color=col["aux"], lw=0.6)
for j in range(24):
    ax2.plot([j * dv] * 2, [-W / 2, W / 2], color=col["aux"], lw=0.6)
for name, v0, w0 in CELLS:
    ax2.fill([v0, v0 + dv, v0 + dv, v0], [w0, w0, w0 + dw, w0 + dw], color=col["accent"], alpha=0.7, lw=0)
ax2.text(19.5 * dv, -0.25, "outer", fontsize=9, ha="center")
ax2.text(7.5 * dv, W / 2 - 2 * dw - 0.25, "inner", fontsize=9, ha="center")
ax2.set_aspect("equal")
ax2.set_xlabel(r"$v$")
ax2.set_ylabel(r"$w$")
ax2.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
ax2.set_yticks([-W / 2, 0, W / 2], [r"$-W/2$", "0", r"$W/2$"])
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title(r"(b) flat rectangle, $W = 2.742$", fontsize=11, pad=14)
fig.subplots_adjust(left=0.0, right=0.98, top=0.9, bottom=0.16, wspace=0.1)
dgfig.save(fig, __file__)

# interactive version: the torus with the conformal grid
traces = [dm.plotly_mesh(*dm.torus_mesh(R, r, 48, 96), color="#D0D0D0", name="torus")]
import plotly.graph_objects as go  # noqa: E402
for k in range(10):
    u_k = u_of_w(-W / 2 + k * dw) if k else np.pi
    Q = X(np.full(200, u_k), np.linspace(0, 2 * np.pi, 200)) * 1.003
    traces.append(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode="lines", line=dict(color="#555555", width=2),
                               hoverinfo="skip", showlegend=False))
for j in range(23):
    Q = X(np.linspace(-np.pi, np.pi, 200), np.full(200, j * dv)) * 1.003
    traces.append(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode="lines", line=dict(color="#555555", width=2),
                               hoverinfo="skip", showlegend=False))
dm.plotly_save(traces, __file__, "torus of revolution (R, r) = (2, 0.8) with the conformal grid (v, w)", "fig-43-3-3")
