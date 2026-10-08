"""Figure 27.1.5: the same room seen through the gnomonic chart (pinhole) and the equidistant chart (fisheye).

Scene in camera coordinates (t_y points down): a box room x in [-2, 2], y in [-1.5, 1.2] (floor y = 1.2,
ceiling y = -1.5), z in [-1.5, 4]; the camera at the origin looks along +z. Drawn: the 12 edges of the room
(black), the floor grid lines x = -1, 0, 1 and z = 0, 1, 2, 3 (blue), and the parts of all lines with t_z <= 0
(on or behind the plane of the camera, orange). The floor line z = 0 lies in the camera plane (theta = 90 deg), so the
equidistant chart draws it exactly on the 90 deg circle.
(a) Gnomonic chart ubar = (t_x/t_z, t_y/t_z), shown for t_z > 0 and |ubar| <= 2.5. Gray circles: theta = 45, 60 deg.
(b) Equidistant chart theta (t_x, t_y)/|(t_x, t_y)| (radius in radians), shown for theta <= 95 deg (a 190 deg lens).
    Gray circles: theta = 45, 60, 90 deg; dashed: 95 deg.
Self-checks: straight 3D lines map to straight image lines under the gnomonic chart (collinearity residual ~ 0);
the equidistant image of a line is not straight; the chart equals the log map of S^2 at e_z.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-1-5-pinhole-vs-fisheye.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
X0, X1, Y0, Y1, Z0, Z1 = -2.0, 2.0, -1.5, 1.2, -1.5, 4.0
corners = np.array([[x, y, z] for x in (X0, X1) for y in (Y0, Y1) for z in (Z0, Z1)])
edges = []
for i in range(8):
    for j in range(i + 1, 8):
        if np.sum(corners[i] != corners[j]) == 1:
            edges.append((corners[i], corners[j], "edge"))
for xg in (-1.0, 0.0, 1.0):
    edges.append((np.array([xg, Y1, Z0]), np.array([xg, Y1, Z1]), "floor"))
for zg in (0.0, 1.0, 2.0, 3.0):
    edges.append((np.array([X0, Y1, zg]), np.array([X1, Y1, zg]), "floor"))
assert len([e for e in edges if e[2] == "edge"]) == 12


def gnomonic(P):
    return P[:, :2] / P[:, 2:3]


def equidistant(P):
    rxy = np.hypot(P[:, 0], P[:, 1])
    th = np.arctan2(rxy, P[:, 2])
    return th[:, None] * P[:, :2] / rxy[:, None], th


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# self-check: gnomonic images of lines are straight, equidistant are not (floor line z = 2)
L = np.linspace(0, 1, 50)[:, None] * (np.array([X1, Y1, 2.0]) - np.array([X0, Y1, 2.0])) + np.array([X0, Y1, 2.0])
g = gnomonic(L)
assert np.abs(cross2(g[-1] - g[0], g - g[0])).max() < 1e-12
e, _ = equidistant(L)
assert np.abs(cross2(e[-1] - e[0], e - e[0])).max() > 0.05
# equidistant = log map at e_z
W = L / np.linalg.norm(L, axis=1, keepdims=True)
thw = np.arccos(W[:, 2])
logv = (thw / np.sin(thw))[:, None] * (W - np.cos(thw)[:, None] * np.array([0, 0, 1.0]))
assert np.allclose(logv[:, :2], e) and np.allclose(logv[:, 2], 0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.6))
colors = {"edge": C["main"], "floor": C["tangent"]}
TH_MAX = np.radians(95)
for a, b, kind in edges:
    P = a + np.linspace(0, 1, 800)[:, None] * (b - a)
    front = P[:, 2] > 1e-6
    # (a) gnomonic, only t_z > 0
    uv = np.full((len(P), 2), np.nan)
    uv[front] = gnomonic(P[front])
    far = np.linalg.norm(uv, axis=1) > 2.5
    uv[far] = np.nan
    ax.plot(uv[:, 0], -uv[:, 1], color=colors[kind], lw=1.3 if kind == "edge" else 0.9)
    # (b) equidistant
    q, th = equidistant(P)
    q[th > TH_MAX] = np.nan
    qf, qb = q.copy(), q.copy()
    qf[~front] = np.nan
    qb[front] = np.nan
    bx.plot(qf[:, 0], -qf[:, 1], color=colors[kind], lw=1.3 if kind == "edge" else 0.9)
    bx.plot(qb[:, 0], -qb[:, 1], color=C["accent"], lw=1.6)
ang = np.linspace(0, 2 * np.pi, 361)
for deg in (45, 60):
    rr = np.tan(np.radians(deg))
    ax.plot(rr * np.cos(ang), rr * np.sin(ang), color=C["aux"], lw=0.7, ls=":")
    ax.text(rr * np.cos(0.8) + 0.03, rr * np.sin(0.8), rf"${deg}^\circ$", fontsize=10, color=C["aux"])
for deg, ls in ((45, ":"), (60, ":"), (90, "-"), (95, "--")):
    rr = np.radians(deg)
    bx.plot(rr * np.cos(ang), rr * np.sin(ang), color=C["aux"] if deg != 95 else "k", lw=0.8, ls=ls)
for deg, a_ in ((45, 0.8), (60, 0.8)):
    rr = np.radians(deg)
    bx.text(rr * np.cos(a_) + 0.03, rr * np.sin(a_), rf"${deg}^\circ$", fontsize=10, color=C["aux"])
bx.text(1.27, 0.12, r"$90^\circ$", fontsize=10, color=C["aux"])
bx.text(-1.62, -1.86, r"$95^\circ$ (190$^\circ$ lens)", fontsize=10)
ax.text(-2.45, -2.4, "straight lines stay straight;\n$\\theta\\geq 90^\\circ$ cannot be shown", fontsize=10)
bx.text(0.47, -1.86, r"orange: $t_z\leq 0$", fontsize=10, color=C["accent"])
# the floor line z = 0 is the image of theta = 90 deg
q0, th0 = equidistant(np.array([[0.0, Y1, 0.0]]))
assert abs(th0[0] - np.pi / 2) < 1e-12

for a_, ttl in ((ax, "(a) pinhole = gnomonic chart"), (bx, "(b) fisheye = equidistant chart")):
    a_.set_aspect("equal")
    a_.set_title(ttl, fontsize=12)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel(r"$\bar u_x=t_x/t_z$")
ax.set_ylabel(r"$-\bar u_y$")
bx.set_xlim(-1.75, 1.75)
bx.set_ylim(-1.95, 1.75)
bx.set_xlabel(r"$\theta\,t_x/|(t_x,t_y)|$ (rad)")
bx.set_ylabel(r"$-\theta\,t_y/|(t_x,t_y)|$ (rad)")
fig.tight_layout()
dgfig.save(fig, __file__)
