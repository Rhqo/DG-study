"""Verification for Section 50.1 (Problem and Key Idea).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/verify/v-50-1-problem-key-idea.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import mgs50 as g  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

# ---------------------------------------------------------------- Figure 50.1.1: the toy rates
m = g.toy_metrics()
fw, cw, pw = m["watertight"]
fc, cc, pc = m["certified"]
close("Figure 50.1.1: watertight floater 60.5 %", round(fw, 1), 60.5, 1e-9)
close("Figure 50.1.1: watertight coverage 100 %", cw, 100.0, 1e-12)
close("Figure 50.1.1: watertight phantom 35.6 %", round(pw, 1), 35.6, 1e-9)
check("Figure 50.1.1: certified floater 0 and phantom 0", fc == 0.0 and pc == 0.0)
close("Figure 50.1.1: certified coverage 80.2 %", round(cc, 1), 80.2, 1e-9)
close("Figure 50.1.1: unseen share 220/360 = 61.1 % (floater without tolerance)", 100 * 220 / 360, 61.11, 5e-3)
close("Figure 50.1.1: disk share pi / 9 = 34.9 % (phantom of a solid disk in the box)", 100 * math.pi / 9, 34.91, 5e-3)
close("Figure 50.1.1: 110/140 = 78.6 % (certified coverage without tolerance)", 100 * 110 / 140, 78.57, 5e-3)
# the tolerance adds about 2 tol (in radians of the unit circle) at the two ends of the certified arc
close("Figure 50.1.1: 78.6 % + 2 x 0.02 rad / (140 deg) = 80.2 %",
      100 * (math.radians(110) + 2 * 0.02) / math.radians(140), 80.20, 0.02)
# the evaluator itself: a point on the wall is at distance 0, a point at radius 1.1 inside the angle range at 0.1
P = np.array([[math.cos(math.radians(270)), math.sin(math.radians(270))], [0.0, -1.1], [1.0, 0.0]])
d = g.dist_to_arc(P, *g.TOY_GT)
close("toy evaluator: distances to the wall arc", d, [0.0, 0.1, math.dist((1, 0), (math.cos(math.radians(340)),
                                                                             math.sin(math.radians(340))))], 1e-12)

# ---------------------------------------------------------------- Exercise 50.1.2
close("Exercise 50.1.2 (a): 80/140 = 57.1 %", 100 * 80 / 140, 57.14, 5e-3)
arc, chord = math.radians(140), 2 * math.sin(math.radians(70))
close("Exercise 50.1.2 (b): arc 2.443, chord 1.879", [arc, chord], [2.443, 1.879], 5e-4)
close("Exercise 50.1.2 (b): chord floater 43.5 %", 100 * chord / (arc + chord), 43.5, 0.05)

# ---------------------------------------------------------------- Exercise 50.1.1: P depends on n only through n n^T
rng = np.random.default_rng(501)
for _ in range(3):
    n = g.unit(rng.normal(size=3))
    P = g.plane_proj(n)
    check("Exercise 50.1.1: P(n) = P(-n), P^2 = P, tr P = 2",
          np.allclose(P, g.plane_proj(-n)) and np.allclose(P @ P, P) and abs(np.trace(P) - 2) < 1e-12)

summary()
