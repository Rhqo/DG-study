"""Verification for Section 50.6 (Assessment).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/verify/v-50-6-assessment.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import mgs50 as g  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

# ---------------------------------------------------------------- Example 50.6.1 / Table 50.6.3 / Figure 50.6.1
inh = [g.refined_tiling(4, k, "inherit") for k in range(4)]
ree = [g.refined_tiling(4, k, "reestimate", lower=(k < 3)) for k in range(4)]
pt = g.plane_term(4)
close("Table 50.6.3: atoms 32, 128, 512, 2048", [a[0] for a in inh], [32, 128, 512, 2048], 0)
close("Table 50.6.3: inherit upper bounds (levels 1-3)", [round(a[1], 2) for a in inh[1:]], [6.57, 5.92, 5.67], 1e-9)
close("Table 50.6.3: inherit lower bounds (levels 1-3)", [round(a[2], 2) for a in inh[1:]], [6.54, 5.78, 5.44], 1e-9)
# review: the re-estimated model at level k is tiling(4 2^k); its bounds on a fine equal-area grid (the 8-point
# Gauss-Legendre rule of refined_tiling is about 0.2 % high there because the integrand has a kink at the atom)
fine = [g.fine_bounds(4, 96), g.fine_bounds(8, 48), g.fine_bounds(16, 24), g.fine_bounds(32, 24, lower=False)]
close("Table 50.6.3: level 0 (both models) and re-estimate upper bounds (fine grid)", [round(f[0], 2) for f in fine],
      [8.02, 4.09, 2.05, 1.03], 1e-9)
close("Table 50.6.3: level 0 and re-estimate lower bounds (fine grid)", [round(f[1], 2) for f in fine[:3]],
      [7.95, 4.09, 2.05], 1e-9)
close("Table 50.6.3: the 8-point rule agrees with the fine grid to 0.3 %", [a[1] for a in ree], [f[0] for f in fine], 0.03)
floor = g.parent_plane_floor(4)
close("Example 50.6.1: plane-only lower bound over the 32 parent planes = 4.67", round(floor, 2), 4.67, 1e-9)
check("Example 50.6.1: the plane-only lower bound is below every inherited distance and at most the plane term",
      floor <= pt + 1e-3 and all(floor < a[2] for a in inh))
close("Example 50.6.1: level-0 plane error 4.67", round(pt, 2), 4.67, 1e-9)
for k, (a, b) in enumerate(zip(ree[:-1], ree[1:])):
    close(f"Example 50.6.1: re-estimate halves the distance (level {k} -> {k + 1})", b[1] / a[1], 0.5, 0.03)
for k, a in enumerate(inh):
    check(f"Example 50.6.1 (level {k}): inherit lower bound stays above the level-0 plane error", a[2] > pt)
check("Example 50.6.1: level 0 is the same in both models", abs(inh[0][1] - ree[0][1]) < 1e-12)
# the inherited model satisfies the three conservation laws of Proposition 50.3.1 (checked on one parent cell)
cells, mus, areas = g.tiling(4)
mu, cell = mus[5], cells[5]
P = np.eye(3) - np.outer(mu, mu)
a0, a1, b0, b1 = cell
ta, tb = np.linspace(a0, a1, 5), np.linspace(b0, b1, 5)
cen = np.array([g.sph(0.5 * (ta[i] + ta[i + 1]), 0.5 * (tb[j] + tb[j + 1])) for i in range(4) for j in range(4)])
qs = np.array([(math.cos(ta[i]) - math.cos(ta[i + 1])) * (tb[j + 1] - tb[j]) for i in range(4) for j in range(4)])
pos = mu + (cen - mu) @ P
pos += mu - (qs[:, None] * pos).sum(0) / qs.sum()
close("Example 50.6.1: children masses add up to the parent cell area", qs.sum(), areas[5], 1e-12)
close("Example 50.6.1: recentred children keep the barycentre", (qs[:, None] * pos).sum(0) / qs.sum(), mu, 1e-12)

# ---------------------------------------------------------------- Exercise 50.6.2
r4 = g.forward_consistency(4, lower=False)
close("Exercise 50.6.2: eps_T of n_theta = 4 is 0.737", round(r4["eT"], 3), 0.737, 1e-9)
close("Exercise 50.6.2: 4 pi x 0.737 = 9.26", 4 * math.pi * 0.737, 9.26, 5e-3)
check("Exercise 50.6.2: the plane error 4.67 is below the inherited distances and above the re-estimated ones at 512 atoms",
      ree[2][1] < pt < inh[2][2])

# ---------------------------------------------------------------- Exercise 50.6.3: plane term about sqrt 2 r on the unit sphere
n0 = np.array([0.0, 0.0, 1.0])
for r in (0.05, 0.1):
    n1 = np.array([math.sin(r), 0.0, math.cos(r)])                 # a point at geodesic distance r has its normal tilted by r
    close(f"Exercise 50.6.3 (r = {r}): ||P_j - P_i||_F ~ sqrt 2 r", g.fro_from_normals(n0, n1) / (math.sqrt(2) * r), 1.0, 2e-3)

summary()
