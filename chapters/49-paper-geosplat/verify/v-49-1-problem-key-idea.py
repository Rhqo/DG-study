"""Verification for Section 49.1 (Problem and Key Idea).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/verify/v-49-1-problem-key-idea.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgsym  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

# ---------------------------------------------------------------- Example 49.1.1: principal curvatures w.r.t. N = e3
u, v = sp.symbols("u v", real=True)
surfaces = {  # graphs z = f(u, v) through the origin with tangent plane z = 0
    "plane": 0 * u,
    "sphere": 1 - sp.sqrt(1 - u ** 2 - v ** 2),
    "cylinder": 1 - sp.sqrt(1 - u ** 2),
    "saddle": u ** 2 - v ** 2,
}
expected = {"plane": (0, 0), "sphere": (1, 1), "cylinder": (1, 0), "saddle": (2, -2)}
for name, f in surfaces.items():
    X = sp.Matrix([u, v, f])
    W = dgsym.shape_operator(X, u, v)          # upward normal (graph surface convention of §7)
    W0 = W.subs({u: 0, v: 0})
    ev = sorted([float(e) for e in W0.eigenvals(multiple=True)], reverse=True)
    check(f"Example 49.1.1: {name} principal curvatures {expected[name]} (N = e3)",
          np.allclose(ev, sorted(expected[name], reverse=True)))
    N0 = dgsym.unit_normal(X, u, v).subs({u: 0, v: 0})
    check(f"Example 49.1.1: {name} normal at the origin is e3", sp.simplify(N0 - sp.Matrix([0, 0, 1])) == sp.zeros(3, 1))

# the saddle is the running example of GUIDELINES §7: K = -4, H = 0 at the origin
e = dgsym.EXAMPLES["saddle"]
K, H = dgsym.K_H(e["expr"], u, v)
check("saddle (§7): K = -4, H = 0 at the origin", K.subs(e["point"]) == -4 and H.subs(e["point"]) == 0)

# deviations quoted in Example 49.1.1 (l = 0.1) and in the text of Figure 49.1.1 (l = 0.3)
close("Example 49.1.1: sphere/cylinder deviation at l = 0.1", 1 - math.sqrt(1 - 0.01), 0.00501, tol=5e-6)
close("Example 49.1.1: saddle deviation +-0.01 along x / y", 0.1 ** 2, 0.01, tol=1e-12)
close("Example 49.1.1: saddle deviation 0 at 45 degrees", (0.1 / math.sqrt(2)) ** 2 - (0.1 / math.sqrt(2)) ** 2, 0.0, tol=1e-15)
close("Figure 49.1.1: sphere deviation at l = 0.3 is 0.046", 1 - math.sqrt(1 - 0.09), 0.046, tol=1e-3)
close("Figure 49.1.1: saddle deviation at l = 0.3 is 0.09", 0.3 ** 2, 0.09, tol=1e-12)

# Euler's formula (49.1.1) agrees with the Hessian of the height at the origin (Proposition 8.2.10)
for name, f in surfaces.items():
    Hs = sp.hessian(f, (u, v)).subs({u: 0, v: 0})
    k1, k2 = expected[name][0], expected[name][1]
    ok = True
    for th in np.linspace(0, math.pi, 13):
        d = sp.Matrix([math.cos(th), math.sin(th)])
        ok &= abs(float((d.T * Hs * d)[0]) - (k1 * math.cos(th) ** 2 + k2 * math.sin(th) ** 2)) < 1e-12
    check(f"(49.1.1): Euler's formula = Hessian quadratic form, {name}", ok)

# ---------------------------------------------------------------- Exercises
close("Exercise 49.1.1: deviation along kappa_1 = 2", 0.5 * 2 * 0.01, 0.01, tol=1e-15)
close("Exercise 49.1.1: deviation along kappa_2 = 0.5", 0.5 * 0.5 * 0.01, 0.0025, tol=1e-15)
close("Exercise 49.1.1: half-axis along kappa_1 for deviation 0.0025", math.sqrt(2 * 0.0025 / 2), 0.05, tol=1e-15)
close("Exercise 49.1.1: half-axis along kappa_2 for deviation 0.0025", math.sqrt(2 * 0.0025 / 0.5), 0.1, tol=1e-15)
close("Exercise 49.1.1: ratio of half-axes sqrt(4) = 2", 0.1 / 0.05, 2.0, tol=1e-12)
close("Exercise 49.1.3: kappa_n(45 deg) = 0 on the saddle", 2 * 0.5 - 2 * 0.5, 0.0, tol=1e-15)

summary()
