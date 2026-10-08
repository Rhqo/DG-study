"""Figure 31.1.4: curvature concentrates at corners and edges as a rounding radius goes to 0 (Section 31.1).

(a) The square [-1, 1]^2 rounded with radius rho (its boundary at distance rho): the curvature kappa of the
    boundary near one corner, plotted against arc length s measured from the start of the corner arc.
    kappa = 1/rho on an arc of length (pi/2) rho and 0 elsewhere, so the area under each spike is pi/2
    for every rho (rho = 0.3, 0.1, 0.03).
(b) The cube [-1, 1]^3 rounded with rho = 0.25 (all points at distance rho from the cube). Gray: flat
    pieces (K = H = 0); light blue: quarter cylinders along the 12 edges (K = 0, H = -1/(2 rho) for the
    outward normal); orange: eighth spheres at the 8 corners (K = 1/rho^2).
Self-check: numerical integrals of kappa, K and H over the pieces equal pi/2, pi/2 and -(pi/4) * (edge
length 2) for several rho.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-1-4-concentration.py``
"""

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS

fig = plt.figure(figsize=(7.6, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.12)

# ---------------------------------------------------------------- (a) curvature spikes of a rounded square
ax = fig.add_subplot(gs[0, 0])
cols = [C["tangent"], C["third"], C["accent"]]
for rho, col in zip((0.3, 0.1, 0.03), cols):
    arc = np.pi / 2 * rho
    s = np.linspace(-0.4, 0.9, 4001)
    kappa = np.where((s >= 0) & (s <= arc), 1 / rho, 0.0)
    ds = s[1] - s[0]
    area = np.trapezoid(kappa, s)
    assert abs(area - np.pi / 2) < 2 * ds / rho  # = pi/2 up to grid error
    ax.plot(s, kappa, color=col, lw=1.6, label=rf"$\rho = {rho}$")
    if rho == 0.1:
        ax.fill_between(s, kappa, color=col, alpha=0.25, lw=0)
        ax.annotate(r"area $= \pi/2$", xy=(arc / 2, 5), xytext=(0.32, 14), fontsize=11, color=col,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0))
ax.set_xlim(-0.35, 0.85)
ax.set_ylim(0, 36)
ax.set_xlabel(r"arc length $s$ from the start of the corner")
ax.set_ylabel(r"curvature $\kappa$")
ax.legend(loc="upper right", fontsize=10.5, frameon=False)
ax.set_title("(a) rounded square: one corner", fontsize=12)
ax.text(0.47, 5.5, r"$\kappa = 1/\rho$ on an arc" + "\n" + r"of length $\pi\rho/2$", fontsize=10.5)

# ---------------------------------------------------------------- (b) rounded cube
RHO = 0.25
cam = dm.Ortho(elev=24, azim=-58)
polys = []  # (points (k,3), outward normal at centre, color)


def add_grid(fun, nrm, n1, n2, color):
    a = np.linspace(0, 1, n1 + 1)
    b = np.linspace(0, 1, n2 + 1)
    for i in range(n1):
        for j in range(n2):
            corners = [(a[i], b[j]), (a[i + 1], b[j]), (a[i + 1], b[j + 1]), (a[i], b[j + 1])]
            P = np.array([fun(u, v) for u, v in corners])
            uc, vc = (a[i] + a[i + 1]) / 2, (b[j] + b[j + 1]) / 2
            polys.append((P, nrm(uc, vc), color))


flat_col, edge_col, corner_col = C["surface"], C["region"], C["accent"]
for ax_i in range(3):
    for sgn in (-1, 1):
        o = [k for k in range(3) if k != ax_i]

        def fun(u, v, ax_i=ax_i, sgn=sgn, o=o):
            p = np.zeros(3)
            p[ax_i] = sgn * (1 + RHO)
            p[o[0]], p[o[1]] = -1 + 2 * u, -1 + 2 * v
            return p

        def nrm(u, v, ax_i=ax_i, sgn=sgn):
            n = np.zeros(3)
            n[ax_i] = sgn
            return n
        add_grid(fun, nrm, 2, 2, flat_col)
for a_i, b_i in itertools.combinations(range(3), 2):
    c_i = 3 - a_i - b_i
    for sa in (-1, 1):
        for sb in (-1, 1):
            def fun(u, v, a_i=a_i, b_i=b_i, c_i=c_i, sa=sa, sb=sb):
                ph = u * np.pi / 2
                p = np.zeros(3)
                p[a_i] = sa * (1 + RHO * np.cos(ph))
                p[b_i] = sb * (1 + RHO * np.sin(ph))
                p[c_i] = -1 + 2 * v
                return p

            def nrm(u, v, a_i=a_i, b_i=b_i, sa=sa, sb=sb):
                ph = u * np.pi / 2
                n = np.zeros(3)
                n[a_i], n[b_i] = sa * np.cos(ph), sb * np.sin(ph)
                return n
            add_grid(fun, nrm, 4, 2, edge_col)
for sx, sy, sz in itertools.product((-1, 1), repeat=3):
    def unit(u, v):
        th, ph = u * np.pi / 2, v * np.pi / 2
        return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    def fun(u, v, s=np.array([sx, sy, sz])):
        return s * (1 + RHO * unit(u, v))

    def nrm(u, v, s=np.array([sx, sy, sz])):
        return s * unit(u, v)
    add_grid(fun, nrm, 4, 4, corner_col)

axb = fig.add_subplot(gs[0, 1])
light = np.array([0.3, -0.5, 0.9])
light /= np.linalg.norm(light)
vis = [(P, n, c) for P, n, c in polys if n @ cam.view > 1e-9]
vis.sort(key=lambda t: cam.depth(t[0].mean(0)))
for P, n, c in vis:
    k = 0.55 + 0.45 * abs(n @ light)
    rgb = np.clip(dm.hex_to_rgb(c) * k, 0, 1)
    axb.add_patch(Polygon(cam(P), closed=True, fc=rgb, ec=rgb, lw=0.3, zorder=1))
allP = cam(np.concatenate([p for p, _, _ in polys]))
axb.set_xlim(allP[:, 0].min() - 0.05, allP[:, 0].max() + 0.05)
axb.set_ylim(allP[:, 1].min() - 0.75, allP[:, 1].max() + 0.75)
axb.set_aspect("equal")
axb.set_axis_off()
axb.set_title(r"(b) rounded cube, $\rho = 0.25$", fontsize=12)
# labels for one corner and one edge
corner = np.array([1, -1, 1]) * (1 + RHO / np.sqrt(3))
q = cam(corner)
axb.annotate(r"corner: $\iint K\,dA = \pi/2$", xy=q, xytext=(q[0] + 0.3, q[1] + 1.4), fontsize=10.5,
             arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9), ha="center")
edge_pt = np.array([1 + RHO / np.sqrt(2), -1 - RHO / np.sqrt(2), -0.3])
q = cam(edge_pt)
axb.annotate(r"edge: $\iint H\,dA = -\frac{\pi}{4}\ell$", xy=q, xytext=(q[0] + 0.55, q[1] - 1.25), fontsize=10.5,
             arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9), ha="center")
face_pt = np.array([0.0, -(1 + RHO), 0.2])
q = cam(face_pt)
axb.text(q[0] - 0.15, q[1] + 0.05, r"$K = H = 0$", fontsize=10.5, ha="center")

# ---------------------------------------------------------------- self-check of the integrals
for rho in (0.5, 0.25, 0.05):
    # corner: eighth sphere of radius rho, K = 1/rho^2, area pi rho^2 / 2
    th = np.linspace(0, np.pi / 2, 801)
    corner_area = (np.pi / 2) * rho ** 2 * np.trapezoid(np.sin(th), th)
    assert abs(corner_area / rho ** 2 - np.pi / 2) < 1e-5
    # edge: quarter cylinder radius rho, length 2, H = -1/(2 rho) with the outward normal
    edge_H = -1 / (2 * rho) * (np.pi / 2 * rho) * 2.0
    assert abs(edge_H - (-np.pi / 4 * 2.0)) < 1e-12

fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.15)
dgfig.save(fig, __file__)
