"""Figure 44.6.3: boundary conditions for the heat step on a planar square (Section 44.6, Example 44.6.12).

Mesh dgmesh.grid_mesh on [0, 1]^2 with 32 x 32 squares (alternate diagonals); source: the vertex (0.25, 0.375); exact
distance: Euclidean (the square is convex). Step I with zero Neumann data (L as is, Proposition 42.5.8), with u = 0 on
the boundary (Dirichlet, Proposition 42.5.7), or the average of the two (ch44.HeatMethod bc = ...); step III always
with the natural (Neumann) condition.
(a) |phi - d| along the boundary, parametrised by arc length from (0, 0) counterclockwise, for t = 16 (mean edge)^2.
(b) mean error against m = t / (mean edge length)^2.
Self-checks: for m >= 4 the average has the smallest mean error of the three.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-6-3-boundary.py
"""

import pathlib
import sys

import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
xs = np.linspace(0, 1, 33)
V, F = dm.grid_mesh(xs, xs)
s = int(np.argmin(np.linalg.norm(V - [0.25, 0.375, 0], axis=1)))
ex = np.linalg.norm(V - V[s], axis=1)
m = dm.mass_voronoi(V, F)
loop = dm.boundary_loop(F)
P = V[loop]
start = int(np.argmin(np.linalg.norm(P, axis=1)))
loop = np.roll(loop, -start)
P = V[loop]
arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
styles = {"neumann": (col["tangent"], "Neumann"), "dirichlet": (col["normal"], "Dirichlet"), "average": (col["accent"], "average")}
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))
for bc, (c_, lab) in styles.items():
    phi = ch44.HeatMethod(V, F, m_t=16, bc=bc).distance(s)
    axa.plot(arc, np.abs(phi - ex)[loop], color=c_, lw=1.3, label=lab)
for x_ in (1, 2, 3):
    axa.axvline(x_, color=col["aux"], lw=0.6, ls=":")
axa.set_xlabel("arc length along the boundary from (0, 0)")
axa.set_ylabel(r"$|\varphi - d|$")
axa.legend(frameon=False, fontsize=9)
axa.spines[["top", "right"]].set_visible(False)
axa.set_title(r"(a) $t = 16\,\bar\ell^{\,2}$, boundary values", fontsize=11)
ms = [1, 2, 4, 8, 16, 32, 64]
table = {}
for bc, (c_, lab) in styles.items():
    errs = [float(m @ np.abs(ch44.HeatMethod(V, F, m_t=mm, bc=bc).distance(s) - ex) / m.sum()) for mm in ms]
    table[bc] = errs
    axb.loglog(ms, errs, "o-", color=c_, lw=1.3, ms=3.5, label=lab)
for k, mm in enumerate(ms):
    if mm >= 4:
        assert table["average"][k] < min(table["neumann"][k], table["dirichlet"][k])
print({bc: [round(e, 4) for e in v] for bc, v in table.items()})
axb.set_xlabel(r"$m = t/\bar\ell^{\,2}$")
axb.set_ylabel("mean error")
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("(b) mean error", fontsize=11)
fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.18, wspace=0.32)
dgfig.save(fig, __file__)
