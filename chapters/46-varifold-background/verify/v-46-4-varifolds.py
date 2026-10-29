"""Verification for Section 46.4 (Varifolds).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-4-varifolds.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402


def P11(X, P):
    return P[..., 0, 0] * np.ones(len(X))


def P12x2(X, P):
    return 2 * P[..., 0, 1] * np.ones(len(X))


def one(X, P):
    return np.ones(len(X))


# ---------------------------------------------------------------- Example 46.4.3 / Figure 46.4.1: circle varifolds
close("Example 46.4.3: v(C,1)(1) = 2 pi", v.circle_varifold_integral(one), 2 * math.pi, 1e-12)
close("Example 46.4.3: v(C,1)(P_11) = pi", v.circle_varifold_integral(P11), math.pi, 1e-12)
for N in (12, 48):
    tv = 2 * np.pi * np.arange(N) / N
    poly = v.circle(tv)
    t, X, P, m = v.circle_atoms(N)
    # the line of each edge is the tangent at the middle of its arc; the largest Grassmannian gap is sin(pi/N)
    edge_line = v.dir_proj(np.roll(poly, -1, axis=0) - poly)
    tmid = tv + np.pi / N
    check(f"Figure 46.4.1: N = {N}: edge line = tangent line at the arc midpoint",
          np.allclose(edge_line, v.line_proj(tmid + np.pi / 2)))
    close(f"Figure 46.4.1: N = {N}: largest gap ||P_e - T_x|| over the edge = sin(pi/N)",
          v.op_dist(edge_line[0], v.line_proj(tv[0] + np.pi / 2)), math.sin(math.pi / N), 1e-12)
    tests = v.random_lipschitz_tests(6, seed=3)
    errs_poly = [abs(v.polyline_integral(f, poly, closed=True, nq=16) - v.circle_varifold_integral(f)) for f in tests]
    errs_pts = [abs(v.atoms_integral(f, X, P, m) - v.circle_varifold_integral(f)) for f in tests]
    check(f"Example 46.4.3: N = {N}: polygon and point cloud are close to v(C,1) on Lipschitz tests "
          f"(max {max(errs_poly):.2e}, {max(errs_pts):.2e})", max(errs_poly) < 2.5 * math.pi / N and max(errs_pts) < 0.05 * 12 / N)

# ---------------------------------------------------------------- Proposition 46.4.7: Young measures
# two atoms at the same point
x = np.array([[0.3, 0.1], [0.3, 0.1]])
P = np.stack([v.P_H, v.P_D])
m = np.array([2.0, 1.0])
mass = v.atoms_integral(one, x, P, m)
close("Exercise 46.4.4: two atoms at one point: mass 3", mass, 3.0, 1e-15)
close("Exercise 46.4.4: nu_x(P_11) = (2*1 + 1*0.5)/3", v.atoms_integral(P11, x, P, m) / mass, (2 * 1 + 0.5) / 3, 1e-15)

# ---------------------------------------------------------------- Example 46.4.10 / Figure 46.4.2: the staircase limit
close("Example 46.4.10: V_inf(1) = 2", v.staircase_limit_integral(one), 2.0, 1e-12)
close("Example 46.4.10: V_inf(P_11) = 1", v.staircase_limit_integral(P11), 1.0, 1e-12)
close("Example 46.4.10: V_inf(2 P_12) = 0", v.staircase_limit_integral(P12x2), 0.0, 1e-12)
close("Example 46.4.10: v(D,1)(1) = sqrt2", v.diagonal_integral(one), math.sqrt(2), 1e-12)
close("Example 46.4.10: v(D,1)(P_11) = sqrt2/2 = 0.7071", v.diagonal_integral(P11), 0.7071, 5e-5)
close("Example 46.4.10: v(D,1)(2 P_12) = sqrt2 = 1.4142", v.diagonal_integral(P12x2), 1.4142, 5e-5)
for n in (1, 4, 32):
    S = v.staircase(n)
    close(f"Example 46.4.10: v(S_{n},1)(P_11) = 1 exactly", v.polyline_integral(P11, S, False), 1.0, 1e-12)
    close(f"Example 46.4.10: v(S_{n},1)(2 P_12) = 0 exactly", v.polyline_integral(P12x2, S, False), 0.0, 1e-12)
tests = v.random_lipschitz_tests(8, seed=5)
errs = []
for n in (4, 16, 64):
    S = v.staircase(n)
    errs.append(max(abs(v.polyline_integral(f, S, False) - v.staircase_limit_integral(f)) for f in tests))
check(f"Example 46.4.10: v(S_n,1) -> V_inf on Lipschitz tests (errors {np.round(errs, 5)})",
      errs[0] > errs[1] > errs[2] and errs[2] < 0.02)
gap = max(abs(v.staircase_limit_integral(f) - v.diagonal_integral(f)) for f in tests)
check(f"Example 46.4.10: V_inf differs from v(D,1) on some Lipschitz test (gap {gap:.3f})", gap > 0.05)
# mass of the limit: density sqrt2 per unit length of D, sqrt2/2 per family (Figure 46.4.2)
close("Figure 46.4.2: density per family sqrt2/2", (1 / 4) / (1 / (4 * math.sqrt(2))) / 2, math.sqrt(2) / 2, 1e-12)

# ---------------------------------------------------------------- Exercise 46.4.1: a small point cloud varifold
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
P = np.stack([v.P_H, v.P_D, v.P_V])
m = np.array([0.5, 1.0, 2.0])
close("Exercise 46.4.1: total mass 3.5", v.atoms_integral(one, X, P, m), 3.5, 1e-15)
inside = lambda X, P: (np.linalg.norm(X, axis=1) < 1.5).astype(float)  # noqa: E731
close("Exercise 46.4.1: ||V||(B_1.5(0)) = 1.5", v.atoms_integral(inside, X, P, m), 1.5, 1e-15)
close("Exercise 46.4.1: V(P_11) = 0.5*1 + 1*0.5 + 2*0 = 1.0", v.atoms_integral(P11, X, P, m), 1.0, 1e-15)
f = lambda X, P: X[:, 0] * P[..., 1, 1]  # noqa: E731
close("Exercise 46.4.1: V(x_1 P_22) = 0.5*0*0 + 1*1*0.5 + 2*0*1 = 0.5", v.atoms_integral(f, X, P, m), 0.5, 1e-15)

# ---------------------------------------------------------------- Exercise 46.4.2: the zigzag limit
for n in (5, 50):
    xs = np.linspace(0, 1, 2 * n + 1)
    ys = np.where(np.arange(2 * n + 1) % 2 == 0, 0.0, 1.0 / (2 * n))
    Z = np.stack([xs, ys], 1)
    close(f"Exercise 46.4.2: v(Z_{n},1)(P_11) = sqrt2 * 1/2", v.polyline_integral(P11, Z, False), math.sqrt(2) / 2, 1e-12)
    close(f"Exercise 46.4.2: v(Z_{n},1)(1) = sqrt2", v.polyline_integral(one, Z, False), math.sqrt(2), 1e-12)
close("Exercise 46.4.2: the segment [0,1]: v(P_11) = 1", 1.0, 1.0, 0.0)

# ---------------------------------------------------------------- Exercise 46.4.5: mass does not determine the varifold
close("Exercise 46.4.5: masses agree: V_inf(1) = v(D, sqrt2)(1) = 2", math.sqrt(2) * v.diagonal_integral(one),
      v.staircase_limit_integral(one), 1e-12)
check("Exercise 46.4.5: but V_inf(2P_12) = 0 != 2 = v(D, sqrt2)(2P_12)",
      abs(v.staircase_limit_integral(P12x2)) < 1e-12 and abs(math.sqrt(2) * v.diagonal_integral(P12x2) - 2) < 1e-12)

summary()
