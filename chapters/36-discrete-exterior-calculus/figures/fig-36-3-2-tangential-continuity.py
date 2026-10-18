"""Figure 36.3.2: the Whitney interpolant of a discrete 1-form is tangentially continuous (Section 36.3).

Mesh: the two triangles f = (0, 1, 2) and g = (1, 3, 2) of lib/ch36.kite (x_0 = (0, 0), x_1 = (2, 0),
x_2 = (1, 1.4), x_3 = (2.2, 1.6)); discrete 1-form alpha = (1.0, 0.5, 1.0, -0.6, 0.3) on the edges
01, 02, 12, 13, 23 (edge_list order).
(a) the vector proxy of W_1 alpha on both triangles (blue arrows on a barycentric grid; f light gray, g light blue);
    the shared edge [1, 2] is orange.
(b) five points on the shared edge: the value of W_1 alpha computed in f (gray arrow) and in g (green arrow) from the
    same base point. Their components along the edge are the same, alpha_12 / |e| (blue arrows along the edge); the
    components across the edge differ.
Self-check: along the edge, alpha(x_2 - x_1) evaluated in f and in g both equal alpha_12 = 1.0 at every point;
the normal components differ; de Rham of the interpolant recovers alpha on all five edges.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-3-2-tangential-continuity.py``
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
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

V, F = ch36.kite()
E, index = dd.edge_list(F)
alpha = np.array([1.0, 0.5, 1.0, -0.6, 0.3])
assert E.tolist() == [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]
e12 = V[2] - V[1]
ts = np.linspace(0.1, 0.9, 5)
vals_f, vals_g = [], []
for t in ts:
    p = V[1] + t * e12
    wf = dd.whitney_interpolate(V, F, alpha, 0, dd.barycentric_coordinates(p, V[F[0]]), index)
    wg = dd.whitney_interpolate(V, F, alpha, 1, dd.barycentric_coordinates(p, V[F[1]]), index)
    vals_f.append(wf)
    vals_g.append(wg)
vals_f, vals_g = np.array(vals_f), np.array(vals_g)
assert np.allclose(vals_f @ e12, alpha[index[(1, 2)]]) and np.allclose(vals_g @ e12, alpha[index[(1, 2)]])
nrm = np.array([-e12[1], e12[0]]) / np.linalg.norm(e12)
assert np.min(np.abs(vals_f @ nrm - vals_g @ nrm)) > 0.1
tq, wq = dd.gauss_legendre_01(4)
for f, tri in enumerate(F):
    for c in range(3):
        a, b = int(tri[c]), int(tri[(c + 1) % 3])
        integral = sum(w * dd.whitney_interpolate(V, F, alpha, f, dd.barycentric_coordinates(
            V[a] + t * (V[b] - V[a]), V[tri]), index) @ (V[b] - V[a]) for t, w in zip(tq, wq))
        sgn = 1 if a < b else -1
        assert abs(integral - sgn * alpha[index[(min(a, b), max(a, b))]]) < 1e-13

C = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": [1.0, 1.0]})
axa.add_patch(Polygon(V[F[0]], closed=True, facecolor="#EEEEEE", edgecolor="#999999", lw=0.9))
axa.add_patch(Polygon(V[F[1]], closed=True, facecolor="#DDEFFA", edgecolor="#999999", lw=0.9))
axa.plot(*V[[1, 2]].T, color=C["accent"], lw=2.6, zorder=1)
m = 6
Bg = np.array([[i / m, j / m, (m - i - j) / m] for i in range(m + 1) for j in range(m + 1 - i)])
for f, tri in enumerate(F):
    P = Bg @ V[tri]
    Wv = np.array([dd.whitney_interpolate(V, F, alpha, f, b, index) for b in Bg])
    inside = Bg.min(1) > 1e-9
    for p, v in zip(P[inside], Wv[inside]):
        axa.add_patch(FancyArrowPatch(p, p + 0.22 * v, arrowstyle="-|>", mutation_scale=6, color=C["tangent"], lw=0.9,
                                      shrinkA=0, shrinkB=0, zorder=3))
for v, off in zip(range(4), ([-0.12, -0.1], [0.08, -0.12], [-0.14, 0.06], [0.1, 0.08])):
    axa.text(*(V[v] + np.array(off)), rf"$x_{v}$", fontsize=11, ha="center", va="center")
axa.text(*V[F[0]].mean(0) + np.array([-0.2, -0.25]), r"$f$", fontsize=13)
axa.text(*V[F[1]].mean(0) + np.array([0.12, -0.1]), r"$g$", fontsize=13)
axa.set_title(r"(a) $W_1\alpha$ on two triangles", fontsize=11)
axa.set_xlim(-0.3, 2.5)
axa.set_ylim(-0.3, 1.8)

# (b) zoom on the shared edge, drawn horizontally: rotate so that e12 points to the right
ang = np.arctan2(e12[1], e12[0])
Rm = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
L = np.linalg.norm(e12)
axb.plot([0, L], [0, 0], color=C["accent"], lw=2.6, zorder=1)
axb.fill_between([0, L], [0, 0], [-0.75, -0.75], color="#DDEFFA", zorder=0)
axb.fill_between([0, L], [0, 0], [0.75, 0.75], color="#EEEEEE", zorder=0)
axb.text(0.05, 0.6, r"side of $f$", fontsize=10, color="#555555")
axb.text(0.05, -0.68, r"side of $g$", fontsize=10, color="#555555")
s = 0.3
for t, wf, wg in zip(ts, vals_f, vals_g):
    base = np.array([t * L, 0.0])
    rf, rg = Rm @ wf, Rm @ wg
    axb.add_patch(FancyArrowPatch(base, base + s * rf, arrowstyle="-|>", mutation_scale=10, color="#555555", lw=1.4,
                                  shrinkA=0, shrinkB=0, zorder=4))
    axb.add_patch(FancyArrowPatch(base, base + s * rg, arrowstyle="-|>", mutation_scale=10, color=C["third"], lw=1.4,
                                  shrinkA=0, shrinkB=0, zorder=4))
    axb.add_patch(FancyArrowPatch(base, base + s * np.array([rf[0], 0.0]), arrowstyle="-|>", mutation_scale=10,
                                  color=C["tangent"], lw=2.2, shrinkA=0, shrinkB=0, zorder=5))
    for r_ in (rf, rg):
        axb.plot([base[0] + s * r_[0]] * 2, [0, s * r_[1]], ls=":", color="#888888", lw=0.8, zorder=3)
axb.text(-0.02, -0.1, r"$x_1$", fontsize=11, ha="right")
axb.text(L + 0.02, -0.1, r"$x_2$", fontsize=11, ha="left")
axb.set_title("(b) along the shared edge", fontsize=11)
axb.set_xlim(-0.25, L + 0.3)
axb.set_ylim(-0.8, 0.8)
for ax in (axa, axb):
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.08)
dgfig.save(fig, __file__)
