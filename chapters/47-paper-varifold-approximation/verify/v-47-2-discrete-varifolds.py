"""Verification for Section 47.2 (Varifolds and Discrete Varifolds).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/verify/v-47-2-discrete-varifolds.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var47 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(472)

# ---------------------------------------------------------------- mean distances in a square cell
k = 2000
g = (np.arange(k) + 0.5) / k - 0.5
GX, GY = np.meshgrid(g, g)
mean_c = np.hypot(GX, GY).mean()
mean_corner = np.hypot(GX + 0.5, GY + 0.5).mean()
close("Example 47.2.7: mean distance from the centre of a unit square = (sqrt2 + ln(1+sqrt2))/6 = 0.3826",
      mean_c, v.MEAN_DIST_SQUARE, 1e-6)
close("0.3826", v.MEAN_DIST_SQUARE, 0.3826, 5e-5)
close("Exercise 47.2.1: mean distance from a corner = (sqrt2 + ln(1+sqrt2))/3 = 0.7652", mean_corner,
      2 * v.MEAN_DIST_SQUARE, 1e-6)
close("0.7652", 2 * v.MEAN_DIST_SQUARE, 0.7652, 5e-5)
close("Example 47.2.7: ratio to the bound delta = sqrt2 a is 0.2705", v.MEAN_DIST_SQUARE / math.sqrt(2), 0.2705, 5e-5)
close("Exercise 47.2.1: corner ratio to the bound is 0.5411", 2 * v.MEAN_DIST_SQUARE / math.sqrt(2), 0.5411, 5e-5)

# ---------------------------------------------------------------- Example 47.2.7: lower bound is attained by phi = dist to centres
a = 0.1
cells = v.circle_cells(a, 0.5, (0.0123, 0.0071))
M = sum(m for m, _ in cells.values())
close("Figure 47.2.1: total mass = length of the circle = pi", M, math.pi, 1e-12)
check("Figure 47.2.1(a): 40 cells meet the circle for a = 0.1", len(cells) == 40)
# integrate phi(x) = dist(x, grid centres) against V^vol by Monte Carlo inside each cell; against V^pt it is 0
tot = 0.0
for (i, j), (m, P) in cells.items():
    x0, y0 = 0.0123 + i * a, 0.0071 + j * a
    U = rng.random((4000, 2)) * a + np.array([x0, y0])
    cg = (np.floor((U - [0.0123, 0.0071]) / a) + 0.5) * a + [0.0123, 0.0071]
    tot += m * np.linalg.norm(U - cg, axis=1).mean()
close("Example 47.2.7: int phi dV^vol = 0.3826 a pi = 0.1202 (Monte Carlo, phi = dist to centres)", tot,
      v.MEAN_DIST_SQUARE * a * math.pi, 2e-3)
close("Example 47.2.7: 0.3826 * 0.1 * pi = 0.1202", v.MEAN_DIST_SQUARE * a * math.pi, 0.1202, 5e-5)
check("Example 47.2.7: phi is admissible (|phi| <= a/sqrt2 <= 1)", a / math.sqrt(2) <= 1)

# ---------------------------------------------------------------- Example 47.2.9: first variation of one volumetric cell
def cell_first_variation(Xfield, DX, P, m, a, n=400):
    """delta V(X) = (m/a^2) int_K div_P X dx on K = [0,a]^2 (midpoint rule) and the boundary form (m/a^2) int_dK X.P nu."""
    s = (np.arange(n) + 0.5) / n * a
    SX, SY = np.meshgrid(s, s)
    pts = np.stack([SX.ravel(), SY.ravel()], 1)
    J = DX(pts)                                  # (N, 2, 2) Jacobian, J[:, i, j] = d X_i / d x_j
    div = np.einsum("ij,nji->n", P, J)           # tr(P DX)
    vol = m / a ** 2 * div.mean() * a ** 2
    bd = 0.0
    for side, nu in (((s * 0 + a, s), (1, 0)), ((s * 0, s), (-1, 0)), ((s, s * 0 + a), (0, 1)), ((s, s * 0), (0, -1))):
        q = np.stack(side, 1)
        bd += m / a ** 2 * np.mean(Xfield(q) @ (P @ np.array(nu, float))) * a
    return vol, bd


Xf = lambda p: np.stack([np.sin(3 * p[:, 0]) + p[:, 1] ** 2, np.cos(2 * p[:, 1]) * p[:, 0]], 1)
DXf = lambda p: np.stack([np.stack([3 * np.cos(3 * p[:, 0]), 2 * p[:, 1]], 1),
                          np.stack([np.cos(2 * p[:, 1]), -2 * np.sin(2 * p[:, 1]) * p[:, 0]], 1)], 1)
Ph = np.array([[1.0, 0.0], [0.0, 0.0]])
vol, bd = cell_first_variation(Xf, DXf, Ph, 0.7, 0.3)
close("Example 47.2.9: (m/|K|) int_K div_P X = (m/|K|) int_dK X . P nu (divergence theorem, P constant)", vol, bd, 1e-5)
# total variation: P nu = 0 on top/bottom, = +-e1 on the sides
m, aa = 0.7, 0.3
close("Example 47.2.9: |delta V|(R^2) = 2 m / a for a horizontal plane", (m / aa ** 2) * aa * 2, 2 * m / aa, 1e-12)
u = np.array([1.0, 1.0]) / math.sqrt(2)
Pd = np.outer(u, u)
tv_d = sum((m / aa ** 2) * aa * np.linalg.norm(Pd @ np.array(nu, float)) for nu in ((1, 0), (-1, 0), (0, 1), (0, -1)))
close("Exercise 47.2.3: diagonal plane gives 2 sqrt2 m / a", tv_d, 2 * math.sqrt(2) * m / aa, 1e-12)
check("Example 47.2.9: volumetric_tv of a single cell with horizontal plane = 2 m/a",
      abs(v.volumetric_tv({(0, 0): (m, Ph)}, aa) - 2 * m / aa) < 1e-12)

# ---------------------------------------------------------------- Figure 47.2.1(d): total variation grows like 1/a
As = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
tv = np.array([v.volumetric_tv(v.circle_cells(a_, 0.5, (0.123 * a_, 0.071 * a_)), a_) for a_ in As])
slope = np.polyfit(np.log(As), np.log(tv), 1)[0]
close("Figure 47.2.1(d): log-log slope of |delta V^vol| against a is about -1", slope, -1.0, 0.05)
check("Figure 47.2.1(d): a |delta V^vol| stays between 4.0 and 4.8", 4.0 < (tv * As).min() and (tv * As).max() < 4.8)
close("Figure 47.2.1(d): |delta V^vol| = 22.5 (a = 0.2) and 1345 (a = 0.003125)", [tv[0], tv[-1]], [22.5, 1345.0], 0.5)
close("smooth circle: |delta V|(R^2) = int |H| ds = 2 pi (any radius)", (1 / 0.5) * math.pi, 2 * math.pi, 1e-12)

# ---------------------------------------------------------------- Definition 47.2.10: polyhedral varifold of a square boundary
# delta V(X) for the boundary of the unit square (four edges): on an edge pq with unit tangent t,
# int div_P X = X(q).t - X(p).t, so delta V(X) = sum over vertices of X(v) . (turning vector at v)
V4 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
direct = 0.0
turn = 0.0
for i in range(4):
    p, q = V4[i], V4[(i + 1) % 4]
    t = (q - p) / np.linalg.norm(q - p)
    P = np.outer(t, t)
    s = (np.arange(4000) + 0.5) / 4000
    pts = p + s[:, None] * (q - p)
    direct += np.linalg.norm(q - p) * np.mean(np.einsum("ij,nji->n", P, DXf(pts)))
    turn += Xf(q[None])[0] @ t - Xf(p[None])[0] @ t
close("Definition 47.2.10: delta V(X) of a polygon = sum of X . (edge tangent) differences (turning vectors)", direct, turn,
      1e-6)

summary()
