"""Figure 36.3.3: W o R is not the identity, but W R(alpha) converges to alpha (Section 36.3).

1-form alpha = x dy on the unit square (vector proxy (0, x)).
(a) grid with 4 x 4 squares (h = 1/4, alternating diagonals): the vector proxy of W_1 R_1(alpha) at every face
    barycenter (blue) next to the exact value (0, x) at the same point (gray).
(b) n x n squares, h = 1/n, n = 2, ..., 64: the largest error |W_1 R_1(alpha) - alpha| over the face barycenters
    (blue) and over the vertices of every face (orange); the dotted line has slope 1.
Self-check: R_1 W_1 R_1(alpha) = R_1(alpha) (de Rham o Whitney = id); the error vanishes for the constant 1-form
2 dx - dy; the fitted slopes are 1 to within 0.05.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-3-3-whitney-convergence.py``
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
from matplotlib.patches import FancyArrowPatch  # noqa: E402

xdy = lambda p: np.array([0.0, p[0]])  # noqa: E731


def errors(n):
    V, F = ch36.grid(n)
    E, index = dd.edge_list(F)
    a = dd.derham1(xdy, V, E)
    eb, ev = 0.0, 0.0
    for f, tri in enumerate(F):
        for q, b in enumerate([np.full(3, 1 / 3)] + list(np.eye(3))):
            p = b @ V[tri]
            err = np.linalg.norm(dd.whitney_interpolate(V, F, a, f, b, index) - xdy(p))
            if q == 0:
                eb = max(eb, err)
            else:
                ev = max(ev, err)
    return eb, ev


V, F = ch36.grid(4)
E, index = dd.edge_list(F)
a = dd.derham1(xdy, V, E)
Wb = dd.whitney_at_barycenters(V, F, a)
cent = V[F].mean(1)
# de Rham o Whitney = id: integrate the interpolant over every edge from one adjacent face
tq, wq = dd.gauss_legendre_01(4)
back = np.zeros(len(E))
for f, tri in enumerate(F):
    for c in range(3):
        i, j = int(tri[c]), int(tri[(c + 1) % 3])
        if i < j:
            back[index[(i, j)]] = sum(w * dd.whitney_interpolate(V, F, a, f, dd.barycentric_coordinates(
                V[i] + t * (V[j] - V[i]), V[tri]), index) @ (V[j] - V[i]) for t, w in zip(tq, wq))
assert np.allclose(back, a, atol=1e-14)
const = dd.derham1(lambda p: np.array([2.0, -1.0]), V, E)
assert np.allclose(dd.whitney_at_barycenters(V, F, const), [2.0, -1.0])
ns = np.array([2, 4, 8, 16, 32, 64])
errs = np.array([errors(int(n)) for n in ns])
h = 1.0 / ns
sl = [np.polyfit(np.log(h), np.log(errs[:, k]), 1)[0] for k in range(2)]
assert all(abs(s - 1) < 0.05 for s in sl), sl

C = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"width_ratios": [1.0, 1.1]})
ch36.draw_mesh2d(axa, V, F, face="#F7F7F7", edge="#C8C8C8", lw=0.6)
s = 0.18
for p, w in zip(cent, Wb):
    axa.add_patch(FancyArrowPatch(p, p + s * xdy(p), arrowstyle="-|>", mutation_scale=7, color="#9A9A9A", lw=1.6,
                                  shrinkA=0, shrinkB=0, zorder=3))
    axa.add_patch(FancyArrowPatch(p, p + s * w, arrowstyle="-|>", mutation_scale=7, color=C["tangent"], lw=1.1,
                                  shrinkA=0, shrinkB=0, zorder=4))
axa.set_xlim(-0.03, 1.03)
axa.set_ylim(-0.03, 1.2)
axa.set_aspect("equal")
axa.set_axis_off()
axa.set_title(r"(a) $W_1R_1(x\,dy)$ (blue) and $x\,dy$ (gray)", fontsize=10.5)

axb.loglog(h, errs[:, 0], "o-", color=C["tangent"], ms=4, lw=1.4, label="at barycenters")
axb.loglog(h, errs[:, 1], "s-", color=C["accent"], ms=4, lw=1.4, label="at corners")
axb.loglog(h, errs[-1, 1] * h / h[-1] * 1.3, ":", color="#999999", lw=1.0, label="slope 1")
axb.set_xlabel(r"spacing $h$")
axb.set_ylabel("largest error")
axb.legend(fontsize=9, frameon=False, loc="lower right")
axb.set_title(r"(b) error $\to 0$ like $h$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.15, wspace=0.25)
dgfig.save(fig, __file__)
