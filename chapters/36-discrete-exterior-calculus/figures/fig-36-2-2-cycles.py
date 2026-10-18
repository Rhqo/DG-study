"""Figure 36.2.2: a cycle that bounds and a cycle that does not, on an annulus (Section 36.2, Example 36.2.15).

Annulus 1 <= r <= 2, three rings of 12 vertices (lib/ch36.annulus(3, 12)), faces counterclockwise.
(a) the 2-chain b = the eight faces of the sector 0 <= theta <= pi/3 (shaded) and its boundary, a 1-cycle (blue);
    the cochain R_1(d theta) sums to 0 on it.
(b) the inner circle traversed counterclockwise (vermillion), a 1-cycle: each of its 12 edges carries
    R_1(d theta) = pi/6 (the angle it subtends at the origin), so the sum is 2 pi. One face f is highlighted with the
    values of R_1(d theta) on its three edges, read along its counterclockwise boundary: pi/6, -pi/6, 0, sum 0.
    Since every face boundary sums to 0, every boundary sums to 0, so the inner circle is not a boundary.
Self-check: the boundary of b is a cycle; d1 R_1(d theta) = 0 on every face (Stokes on triangles, d(d theta) = 0);
the inner circle is a cycle with sum 2 pi; it is not in the image of the boundary matrix (least squares residual).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-2-2-cycles.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import ch36  # noqa: E402
import dgdec as dd  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

nt = 12
V, F = ch36.annulus(3, nt)
E, index = dd.edge_list(F)
B1, B2 = dd.boundary1(F, len(V)), dd.boundary2(F)
dtheta = lambda p: np.array([-p[1], p[0]]) / (p[0] ** 2 + p[1] ** 2)  # noqa: E731
a = dd.derham1(dtheta, V, E, order=12)
assert np.abs(dd.d1(F) @ a).max() < 1e-12                      # closed on every triangle
# (a) the sector 0 <= theta <= pi/3: faces whose vertices have angle index in {0, 1, 2}, spanning two steps
sector = [f for f, tri in enumerate(F) if set((np.array(tri) % nt).tolist()) <= {0, 1, 2}]
assert len(sector) == 8
bvec = np.zeros(len(F))
bvec[sector] = 1
cyc_a = B2 @ bvec
assert not np.any(B1 @ cyc_a) and abs(a @ cyc_a) < 1e-12
# (b) the inner circle: vertices 0, 1, ..., nt-1 counterclockwise
inner = np.zeros(len(E))
for j in range(nt):
    i0, i1 = j, (j + 1) % nt
    inner[index[(min(i0, i1), max(i0, i1))]] += 1 if i0 < i1 else -1
assert not np.any(B1 @ inner) and abs(a @ inner - 2 * np.pi) < 1e-12
res = inner - B2 @ np.linalg.lstsq(B2, inner, rcond=None)[0]
assert np.linalg.norm(res) > 0.5                              # not a boundary

C = dgfig.COLORS
BLUE, RED = C["tangent"], C["normal"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.6))
for ax in (axa, axb):
    ch36.draw_mesh2d(ax, V, F, face="#F7F7F7", edge="#CFCFCF", lw=0.6)
    ax.plot(0, 0, "+", color="#555555", ms=8)
for f in sector:
    axa.add_patch(Polygon(V[F[f]], closed=True, facecolor=C["region"], alpha=0.35, edgecolor="none", zorder=1))
for e in np.nonzero(cyc_a)[0]:
    i, j = E[e] if cyc_a[e] > 0 else E[e][::-1]
    ch36.edge_arrow(axa, V[i], V[j], BLUE, lw=2.0, head=10, frac=(0.05, 0.95))
axa.text(0, -2.45, r"$\partial b$: sum of $\widehat{d\theta}$ $= 0$", fontsize=11, ha="center", color=BLUE)
axa.set_title(r"(a) a boundary $\partial b$", fontsize=11)

f0 = [f for f, tri in enumerate(F) if sorted(tri) == [nt, 2 * nt, 2 * nt + 1]][0]
axb.add_patch(Polygon(V[F[f0]], closed=True, facecolor=C["accent"], alpha=0.3, edgecolor="none", zorder=1))
tri = F[f0]
vals = []
for c in range(3):
    i, j = int(tri[c]), int(tri[(c + 1) % 3])
    v = a[index[(min(i, j), max(i, j))]] * (1 if i < j else -1)
    vals.append(v)
    ch36.edge_arrow(axb, V[i], V[j], "#8a5a00", lw=1.6, head=9, frac=(0.15, 0.85))
    lab = {1: r"$\pi/6$", -1: r"$-\pi/6$", 0: r"$0$"}[int(round(v / (np.pi / 6)))]
    axb.text(*ch36.normal_offset(V[i], V[j], -0.3), lab, fontsize=10, color="#8a5a00", ha="center", va="center",
             bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85), zorder=6)
assert abs(sum(vals)) < 1e-12 and sorted(np.round(np.array(vals) / (np.pi / 6)).astype(int).tolist()) == [-1, 0, 1]
for j in range(nt):
    ch36.edge_arrow(axb, V[j], V[(j + 1) % nt], RED, lw=2.2, head=10, frac=(0.1, 0.9))
axb.text(*(0.5 * (V[3] + V[4]) + np.array([0.0, -0.32])), r"$\pi/6$", fontsize=11, color=RED, ha="center")
axb.text(0, -2.45, r"inner circle: sum $= 2\pi \neq 0$", fontsize=11, ha="center", color=RED)
axb.set_title(r"(b) a cycle that is not a boundary", fontsize=11)
for ax in (axa, axb):
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.6, 2.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.04)
dgfig.save(fig, __file__)
