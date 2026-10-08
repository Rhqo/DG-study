"""Figure 30.2.4: marching squares (the 2D cross-section of marching cubes) and its ambiguous case (Section 30.2).

(a) the ellipse x^2/4 + y^2 = 1 (thin black), its exact SDF sampled on a grid of spacing h = 0.5 (dots: blue f < 0,
    red f > 0). On every grid edge whose end values have opposite signs a vertex is placed by linear interpolation
    x0 + f0/(f0 - f1) (x1 - x0) (orange dots); the vertices are joined cell by cell (orange polygon).
(b) one cell with corner values f00 = 0.3 (bottom left), f10 = -0.4, f11 = 0.2, f01 = -0.3; gray: level sets of the
    bilinear interpolant; black: its zero set (two hyperbola branches); orange square: its saddle point
    (0.5, 0.583) with value -0.05. The two triangulations that fit the corner signs: orange solid (cut off the two
    positive corners: consistent with the negative saddle value) and blue dashed (cut off the negative corners).
Self-checks: vertices are on the sign-change edges; the saddle is a critical point of the interpolant.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-2-4-marching-squares.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
A, B = 2.0, 1.0


def closest_param(P, n_init=512, iters=40):
    s0 = np.linspace(0, 2 * np.pi, n_init, endpoint=False)
    E = np.stack([A * np.cos(s0), B * np.sin(s0)], 1)
    s = s0[np.argmin(((P[:, None, :] - E[None]) ** 2).sum(-1), 1)]
    for _ in range(iters):
        c, si = np.cos(s), np.sin(s)
        dx, dy = A * c - P[:, 0], B * si - P[:, 1]
        g1 = -A * si * dx + B * c * dy
        g2 = (A * si) ** 2 + (B * c) ** 2 - A * c * dx - B * si * dy
        s = s - g1 / np.where(np.abs(g2) > 1e-12, g2, 1e-12)
    return s


def sdf(P):
    s = closest_param(P)
    d = np.linalg.norm(P - np.stack([A * np.cos(s), B * np.sin(s)], 1), axis=1)
    return np.where(P[:, 0] ** 2 / A ** 2 + P[:, 1] ** 2 / B ** 2 < 1, -d, d)


h = 0.5
xs = np.arange(-2.75, 2.76, h)
ys = np.arange(-1.75, 1.76, h)
X, Y = np.meshgrid(xs, ys)
F = sdf(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)


def interp(p0, p1, f0, f1):
    t = f0 / (f0 - f1)
    assert 0 <= t <= 1
    return p0 + t * (p1 - p0)


segs, verts = [], []
for i in range(len(ys) - 1):
    for j in range(len(xs) - 1):
        corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]          # ccw: bl, br, tr, tl
        P = [np.array([X[a, b], Y[a, b]]) for a, b in corners]
        V = [F[a, b] for a, b in corners]
        pts = []
        for k in range(4):
            a, b = k, (k + 1) % 4
            if V[a] * V[b] < 0:
                pts.append(interp(P[a], P[b], V[a], V[b]))
        assert len(pts) in (0, 2), "no ambiguous cell expected for this grid"
        if len(pts) == 2:
            segs.append(pts)
            verts += pts
verts = np.array(verts)
err = np.abs(np.sqrt(verts[:, 0] ** 2 / A ** 2 + verts[:, 1] ** 2 / B ** 2) - 1)
print(f"{len(segs)} segments; max |x^2/a^2 + y^2/b^2 - 1| at vertices = {err.max():.3f}")

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 8.2), gridspec_kw=dict(height_ratios=[1.0, 1.25], hspace=0.45))

tt = np.linspace(0, 2 * np.pi, 400)
for x0 in xs:
    axa.axvline(x0, color="#BBBBBB", lw=0.6, zorder=0)
for y0 in ys:
    axa.axhline(y0, color="#BBBBBB", lw=0.6, zorder=0)
axa.plot(A * np.cos(tt), B * np.sin(tt), color=C["main"], lw=1.0, zorder=2)
neg = F < 0
axa.plot(X[neg], Y[neg], "o", color=C["tangent"], ms=4.5, zorder=3)
axa.plot(X[~neg], Y[~neg], "o", color="#C0392B", ms=4.5, mfc="white", zorder=3)
for p, q in segs:
    axa.plot([p[0], q[0]], [p[1], q[1]], color=C["accent"], lw=2.4, zorder=4)
axa.plot(verts[:, 0], verts[:, 1], "o", color=C["accent"], ms=4, mec=C["main"], mew=0.5, zorder=5)
axa.set_aspect("equal")
axa.set_xlim(-2.9, 2.9)
axa.set_ylim(-1.9, 1.9)
axa.set_xticks([-2, -1, 0, 1, 2])
axa.set_yticks([-1, 0, 1])
axa.set_title(r"(a) marching squares on the SDF, $h = 0.5$", fontsize=13.5)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="none", color=C["tangent"], ms=6, label=r"sample $f < 0$"),
           Line2D([], [], marker="o", ls="none", color="#C0392B", mfc="white", ms=6, label=r"sample $f > 0$"),
           Line2D([], [], color=C["accent"], lw=2.4, marker="o", ms=4, mec=C["main"], label="extracted polygon"),
           Line2D([], [], color=C["main"], lw=1.0, label=r"true $S$")]
axa.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=11,
           columnspacing=1.2, handletextpad=0.4)

# (b) ambiguous cell
f00, f10, f01, f11 = 0.3, -0.4, -0.3, 0.2
bil = lambda u, v: f00 * (1 - u) * (1 - v) + f10 * u * (1 - v) + f01 * (1 - u) * v + f11 * u * v
den = f00 + f11 - f10 - f01
us_, vs_ = (f00 - f01) / den, (f00 - f10) / den
fs = bil(us_, vs_)
eps = 1e-7
assert abs(bil(us_ + eps, vs_) - bil(us_ - eps, vs_)) < 1e-9 and abs(bil(us_, vs_ + eps) - bil(us_, vs_ - eps)) < 1e-9
assert abs(fs - (f00 * f11 - f10 * f01) / den) < 1e-15 and abs(fs + 0.05) < 1e-12
g = np.linspace(0, 1, 401)
U, V = np.meshgrid(g, g)
Fb = bil(U, V)
axb.contour(U, V, Fb, levels=[-0.3, -0.2, -0.1, 0.1, 0.2], colors="#AAAAAA", linewidths=0.8)
axb.contour(U, V, Fb, levels=[0.0], colors=C["main"], linewidths=2.0)
e_b = (f00 / (f00 - f10), 0.0)          # bottom edge
e_r = (1.0, f10 / (f10 - f11))          # right edge
e_t = (f01 / (f01 - f11), 1.0)          # top edge
e_l = (0.0, f00 / (f00 - f01))          # left edge
# option 1 (orange): isolate the positive corners (0,0) and (1,1)
axb.plot([e_b[0], e_l[0]], [e_b[1], e_l[1]], color=C["accent"], lw=2.6)
axb.plot([e_r[0], e_t[0]], [e_r[1], e_t[1]], color=C["accent"], lw=2.6, label="cut off + corners")
# option 2 (blue dashed): isolate the negative corners (1,0) and (0,1)
axb.plot([e_b[0], e_r[0]], [e_b[1], e_r[1]], color=C["tangent"], lw=2.0, ls=(0, (5, 3)))
axb.plot([e_l[0], e_t[0]], [e_l[1], e_t[1]], color=C["tangent"], lw=2.0, ls=(0, (5, 3)), label="cut off − corners")
axb.plot([us_], [vs_], "s", color=C["accent"], ms=8, mec=C["main"], zorder=6)
axb.annotate("saddle\n" + rf"$f = {fs:.2f}$", xy=(us_, vs_), xytext=(0.6, 0.40), ha="center", va="center",
             fontsize=11.5, arrowprops=dict(arrowstyle="->", lw=0.9), bbox=dict(fc="white", ec="none", pad=1))
for (cu, cv, val) in [(0, 0, f00), (1, 0, f10), (1, 1, f11), (0, 1, f01)]:
    axb.plot([cu], [cv], "o", ms=10, color=(C["tangent"] if val < 0 else "white"),
             mec=(C["tangent"] if val < 0 else "#C0392B"), mew=1.6, zorder=7, clip_on=False)
    axb.text(cu + (0.05 if cu == 0 else -0.05), cv + (0.06 if cv == 0 else -0.08), rf"${val:+.1f}$", fontsize=12.5,
             ha="left" if cu == 0 else "right", va="bottom" if cv == 0 else "top")
axb.set_xlim(-0.02, 1.02)
axb.set_ylim(-0.02, 1.02)
axb.set_aspect("equal")
axb.set_xticks([0, 1])
axb.set_yticks([0, 1])
axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=11.5)
axb.set_title("(b) an ambiguous cell", fontsize=13.5)

dgfig.save(fig, __file__)
