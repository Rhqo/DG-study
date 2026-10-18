"""Figure 36.1.3: discrete forms are integrated quantities (Section 36.1).

(a) the 2-form x dx^dy on the grid of Example 36.1.2: the number in each triangle is its integral over the
    counterclockwise triangle, i.e. (area 1/8) x (value of x at the centroid); the color is the same number.
(b) refining the grid of the unit square (n x n squares, spacing h = 1/n, n = 2, ..., 64): the largest absolute value
    of the de Rham images of the 0-form x^2 - y^2 (black), the 1-form x dy (blue) and the 2-form x dx^dy (orange).
    Dotted gray lines have slopes 0, 1, 2 on the log-log axes.
Self-check: the face values are area x centroid value exactly (the integrand is linear); the fitted slopes are
0, 1 to within 0.02 and 2 to within 0.06 (1.95: the largest face value is
(h^2 / 2)(1 - h / 3), the face nearest x = 1).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-1-3-integrated-quantities.py``
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

V, F = ch36.grid(2)
w_hat = dd.derham2(lambda p: p[0], V, F)
assert np.allclose(w_hat, 0.125 * V[F].mean(1)[:, 0], atol=1e-15)

ns = np.array([2, 4, 8, 16, 32, 64])
m0, m1, m2 = [], [], []
for n in ns:
    Vn, Fn = ch36.grid(int(n))
    En, _ = dd.edge_list(Fn)
    m0.append(np.abs(dd.derham0(lambda p: p[0] ** 2 - p[1] ** 2, Vn)).max())
    m1.append(np.abs(dd.derham1(lambda p: np.array([0.0, p[0]]), Vn, En)).max())
    m2.append(np.abs(dd.derham2(lambda p: p[0], Vn, Fn)).max())
h = 1.0 / ns
slopes = [np.polyfit(np.log(h), np.log(m), 1)[0] for m in (m0, m1, m2)]
assert abs(slopes[0]) < 0.02 and abs(slopes[1] - 1) < 0.02 and abs(slopes[2] - 2) < 0.06, slopes

C = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"width_ratios": [1.0, 1.15]})
cmap = plt.get_cmap("Oranges")
for f, tri in enumerate(F):
    axa.add_patch(Polygon(V[tri], closed=True, facecolor=cmap(0.15 + 0.7 * w_hat[f] / w_hat.max()),
                          edgecolor="#777777", lw=0.8))
    c = V[tri].mean(0)
    axa.text(*c, f"{w_hat[f]:.4f}", fontsize=8.5, ha="center", va="center")
axa.set_xlim(-0.04, 1.04)
axa.set_ylim(-0.04, 1.04)
axa.set_aspect("equal")
axa.set_axis_off()
axa.set_title(r"(a) $\int_f x\,dx\wedge dy$ on each face", fontsize=11)

axb.loglog(h, m0, "o-", color="black", ms=4, lw=1.4, label=r"$k=0$: $x^2-y^2$")
axb.loglog(h, m1, "s-", color=C["tangent"], ms=4, lw=1.4, label=r"$k=1$: $x\,dy$")
axb.loglog(h, m2, "^-", color=C["accent"], ms=4, lw=1.4, label=r"$k=2$: $x\,dx\wedge dy$")
for p, m in ((0, m0), (1, m1), (2, m2)):
    axb.loglog(h, m[-1] * (h / h[-1]) ** p, ":", color="#999999", lw=1.0)
axb.set_xlabel(r"spacing $h$")
axb.set_ylabel("largest value")
axb.legend(fontsize=9, loc="lower right", frameon=False)
axb.set_title(r"(b) values scale like $h^k$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.15, wspace=0.28)
dgfig.save(fig, __file__)
