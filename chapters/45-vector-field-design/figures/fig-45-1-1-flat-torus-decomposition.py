"""Figure 45.1.1: the discrete Hodge decomposition of a 1-form on a flat torus (Section 45.1, Example 45.1.15).

Mesh: ch45.FlatTorus(12, 12), the equilateral triangulation of R^2 / (W Z x H Z), W = 1, H = sqrt(3)/2 (side 1/12;
144 vertices, 432 edges, 288 faces; star1 = 1/sqrt 3 on every edge). The 1-form is omega = R_1(du + star dv + h) with
u = 0.05 sin(2 pi x/W) cos(2 pi y/H), v = 0.03 cos(2 pi x/W + 0.4) sin(4 pi y/H), h = 0.3 dx + 0.1 dy
(ch45.flat_torus_example). It is split by Proposition 45.1.13 into d0 alpha + star1^{-1} d1^T bt + gamma.
Every panel shows the fundamental rectangle (dashed) with the mesh edges (light gray) and, at the barycenters of 36
faces (the first face of the pair (r, c) for even r and even c), the Whitney interpolant of the 1-form (blue arrows,
0.16 times the true length in all four panels). Shading: PuOr_r, orange positive, purple negative, white 0.
(a) omega, (b) d0 alpha over the faces shaded by alpha (~ u), (c) the co-exact part over the faces shaded by -bt (~ v),
(d) gamma.
Self-checks: |alpha - u| < 3e-5, |bt + v| < 3e-4 (means removed), gamma = R_1(h) to 1e-14, and the three parts are
pairwise orthogonal for <a, b>_1 = a^T star1 b.

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-1-1-flat-torus-decomposition.py
"""

import pathlib
import sys

import dgdec as dd
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
ft = ch45.FlatTorus(12, 12)
F = ft.F
D0, D1 = dd.d0(F), dd.d1(F)
s0, s1, s2 = ch45.stars_from_corners(F, ft.C)
omega, u, v, h = ch45.flat_torus_example(ft)
alpha, bt, gamma = ch45.hodge_decomposition(D0, D1, s1, omega)
ex, co = D0 @ alpha, (D1.T @ bt) / s1
assert np.abs(alpha - (u - u.mean())).max() < 3e-5
assert np.abs(bt + (v - v.mean())).max() < 3e-4
assert np.abs(gamma - h).max() < 1e-14
ip = lambda a, b: a @ (s1 * b)  # noqa: E731
assert max(abs(ip(ex, co)), abs(ip(ex, gamma)), abs(ip(co, gamma))) < 1e-13
print(f"|alpha - u| = {np.abs(alpha - (u - u.mean())).max():.2e}, |bt + v| = {np.abs(bt + (v - v.mean())).max():.2e}, "
      f"|gamma - h| = {np.abs(gamma - h).max():.1e}")

cen = ft.centroids()
sel = np.array([2 * (r * ft.n + c) for r in range(0, ft.rows, 2) for c in range(0, ft.n, 2)])  # 36 faces
scale = 0.16
panels = [(omega, None, r"(a) $\omega$"),
          (ex, alpha[F].mean(1), r"(b) exact: $d_0\alpha \approx \mathrm{grad}\,u$"),
          (co, -bt, r"(c) co-exact $\approx J\,\mathrm{grad}\,v$"),
          (gamma, None, r"(d) harmonic: $\gamma = h$")]
fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.9))
segs = []
for f in range(len(F)):
    for a in range(3):
        segs.append([ft.C[f, a], ft.C[f, (a + 1) % 3]])
for ax, (form, shade, title) in zip(axs.ravel(), panels):
    if shade is not None:
        m = np.abs(shade).max()
        cm = plt.get_cmap("PuOr_r")
        ax.add_collection(PolyCollection(ft.C, facecolors=cm(0.5 + 0.32 * shade / m), edgecolors="none", zorder=0))
    ax.add_collection(LineCollection(segs, colors="#C8C8C8", linewidths=0.35, zorder=1))
    X = ch45.whitney_barycenters(F, ft.C, form)
    ax.quiver(cen[sel, 0], cen[sel, 1], scale * X[sel, 0], scale * X[sel, 1], angles="xy", scale_units="xy", scale=1,
              color=col["tangent"], width=0.008, headwidth=3.4, headlength=4.0, headaxislength=3.6, zorder=3)
    ax.plot([0, ft.W, ft.W, 0, 0], [0, 0, ft.H, ft.H, 0], color=col["aux"], lw=0.8, ls="--", zorder=2)
    ax.set_xlim(-0.06, ft.W + 0.06)
    ax.set_ylim(-0.06, ft.H + 0.06)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.12)
dgfig.save(fig, __file__)
