"""Verification for Section 46.2 (Measures and Weak Convergence).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-2-measures.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

A = 0.5
CENTER, RADIUS = np.array([-1.0, 0.0]), 0.6


def phi(X):
    return v.bump(X, CENTER, RADIUS)


# ---------------------------------------------------------------- Figure 46.2.2 / Example 46.2.7
lim_vor = v.integrate_1d(lambda t: phi(v.circle(t)), 0, 2 * np.pi, n=64, pieces=64)
lim_eq = 2 * np.pi * v.integrate_1d(lambda u: phi(v.circle(v.warp(u, A))), 0, 1, n=64, pieces=64)
close("Figure 46.2.2: int phi dH^1 = 0.7294", lim_vor, 0.7294, 5e-5)
close("Figure 46.2.2: int phi theta dH^1 = 1.3374", lim_eq, 1.3374, 5e-5)
t, m_vor, m_eq = v.circle_samples(24, a=A)
f = phi(v.circle(t))
close("Figure 46.2.2: N = 24 Voronoi weights give 0.7293", np.sum(m_vor * f), 0.7293, 5e-5)
close("Figure 46.2.2: N = 24 equal weights give 1.3271", np.sum(m_eq * f), 1.3271, 5e-5)
close("Figure 46.2.2: N = 24 unit weights give 5.0692", np.sum(f), 5.0692, 5e-5)
errs = []
for N in (48, 96, 192, 384):
    t, m_vor, m_eq = v.circle_samples(N, a=A)
    f = phi(v.circle(t))
    errs.append(abs(np.sum(m_vor * f) - lim_vor))
    close(f"Example 46.2.7: N = {N} equal weights approach int phi theta", np.sum(m_eq * f), lim_eq, 2e-3)
    close(f"Example 46.2.7: N = {N} unit weights grow like N / (2 pi) * int phi theta", np.sum(f) / N,
          lim_eq / (2 * np.pi), 2e-3 / N * 50)
check("Example 46.2.7: Voronoi-weight error decreases with N", all(a > b for a, b in zip(errs, errs[1:])))
check("Example 46.2.7: Voronoi-weight error at N = 384 below 1e-4", errs[-1] < 1e-4)
# the density of the equal-weight limit: theta = 1 / (1 + a cos 2 pi u)
close("Example 46.2.7: theta at t = pi equals 2", 1 / (1 + A * math.cos(2 * math.pi * 0.5)), 2.0, 1e-15)
close("Example 46.2.7: theta at t = 0 equals 2/3", 1 / (1 + A * math.cos(0.0)), 2 / 3, 1e-15)
# a second test function: a bump centred at (1, 0), where theta = 2/3
lim_v2 = v.integrate_1d(lambda t: v.bump(v.circle(t), (1, 0), 0.6), 0, 2 * np.pi, n=64, pieces=64)
lim_e2 = 2 * np.pi * v.integrate_1d(lambda u: v.bump(v.circle(v.warp(u, A)), (1, 0), 0.6), 0, 1, n=64, pieces=64)
check("Example 46.2.7: near t = 0 the equal-weight limit carries less (ratio about 2/3)", 0.6 < lim_e2 / lim_v2 < 0.75)

# ---------------------------------------------------------------- Example 46.2.7(a): moving Dirac
for k in (1, 10, 100):
    close(f"Example 46.2.7(a): delta_(1/k) of a continuous phi tends to phi(0), k = {k}",
          math.cos(1 / k), 1.0, 1 / k ** 2)

# ---------------------------------------------------------------- Example 46.2.8: the staircase as a measure
def staircase_xy(n):
    S = v.staircase(n)
    return v.polyline_integral(lambda X, P: X[:, 0] * X[:, 1], S, closed=False, nq=4)


for n in (1, 2, 4, 16, 64):
    close(f"Example 46.2.8: int_(S_{n}) xy ds = 2/3 - 1/(6 n^2)", staircase_xy(n), 2 / 3 - 1 / (6 * n * n), 1e-12)
close("Example 46.2.8: n = 4 gives 0.65625", staircase_xy(4), 0.65625, 1e-12)
close("Example 46.2.8: sqrt2 * int_D xy dH^1 = 2/3", math.sqrt(2) * v.diagonal_integral(lambda X, P: X[:, 0] * X[:, 1]),
      2 / 3, 1e-12)
close("Example 46.2.8: int_D xy dH^1 = sqrt2 / 3 = 0.4714", v.diagonal_integral(lambda X, P: X[:, 0] * X[:, 1]),
      0.4714, 5e-5)
# a compactly supported bump test as well
bt = lambda X, P: v.bump(X, (0.4, 0.5), 0.5)  # noqa: E731
lim = math.sqrt(2) * v.diagonal_integral(bt)
bump_err = [abs(v.polyline_integral(bt, v.staircase(n), False, nq=8) - lim) for n in (16, 64, 256)]
check("Example 46.2.8: bump test on S_n approaches sqrt2 * int_D (error decreasing, below 2e-3 at n = 256)",
      bump_err[0] > bump_err[1] > bump_err[2] and bump_err[2] < 2e-3)

# ---------------------------------------------------------------- Example 46.2.5 / Figure 46.2.1: covers of the ellipse
L = v.ellipse_length_total()
close("Example 46.2.5: length of the ellipse = 9.6884", L, 9.6884, 5e-5)
for dl, k, s in ((1.0, 10, 10.0), (0.25, 39, 9.75)):
    kk = math.ceil(L / dl)
    check(f"Example 46.2.5: delta = {dl} needs {k} balls", kk == k)
    close(f"Example 46.2.5: delta = {dl}: sum of diameters = {s}", kk * dl, s, 1e-12)
close("Example 46.2.5: d = 2 sum at delta = 0.01 is 0.0761", math.ceil(L / 0.01) * math.pi * 0.005 ** 2, 0.0761, 5e-5)
check("Example 46.2.5: d = 0 sum at delta = 0.01 is 969 balls", math.ceil(L / 0.01) == 969)
close("Example 46.2.5: omega_1 (delta/2) = delta (sum of diameters for d = 1)", v.OMEGA[1] * 0.5, 1.0, 1e-15)

# ---------------------------------------------------------------- Example 46.2.12: push-forward of the circle to the x-axis
# density 2 / sqrt(1 - x^2) on (-1, 1); substitute x = cos s to integrate without the endpoint singularity
close("Example 46.2.12: total mass of pi_# (H^1 on C) = 2 pi",
      v.integrate_1d(lambda s: 2 / np.sqrt(1 - np.cos(s) ** 2) * np.sin(s), 0, np.pi), 2 * np.pi, 1e-12)
close("Example 46.2.12: int x^2 d(pi_# mu) = int cos^2 t dt = pi",
      v.integrate_1d(lambda s: np.cos(s) ** 2 * 2 / np.sqrt(1 - np.cos(s) ** 2) * np.sin(s), 0, np.pi), math.pi, 1e-12)
close("Example 46.2.12: 2 arccos(0.9) = 0.902", 2 * math.acos(0.9), 0.902, 5e-4)
close("Example 46.2.12: mass of pi_# mu on [0.9, 1] equals 2 arccos(0.9)",
      v.integrate_1d(lambda t: np.ones_like(t), -math.acos(0.9), math.acos(0.9)), 2 * math.acos(0.9), 1e-12)

# ---------------------------------------------------------------- Theorem 46.2.13: density as a ratio of small balls
# nu = theta H^1 on C (equal-weight limit), mu = H^1 on C; at x = (-1, 0) (t = pi, u = 1/2) the ratio tends to theta = 2
def arc_mass_in_ball(x, r, weight):
    ts = np.linspace(0, 2 * np.pi, 200001)
    X = v.circle(ts)
    inside = np.linalg.norm(X - x, axis=1) < r
    dt = ts[1] - ts[0]
    return float(np.sum(weight(ts)[inside]) * dt)


def theta_of_t(ts):
    # invert t = 2 pi u + a sin 2 pi u by Newton
    u = ts / (2 * np.pi)
    for _ in range(30):
        u = u - (2 * np.pi * u + A * np.sin(2 * np.pi * u) - ts) / (2 * np.pi * (1 + A * np.cos(2 * np.pi * u)))
    return 1 / (1 + A * np.cos(2 * np.pi * u))


ratios = []
for r in (0.2, 0.05, 0.01):
    ratios.append(arc_mass_in_ball(np.array([-1.0, 0.0]), r, theta_of_t)
                  / arc_mass_in_ball(np.array([-1.0, 0.0]), r, lambda ts: np.ones_like(ts)))
check(f"Theorem 46.2.13: nu(B_r)/mu(B_r) at t = pi increases to theta = 2 as r -> 0 {np.round(ratios, 4)}",
      ratios[0] < ratios[1] < ratios[2] < 2.0 and 2.0 - ratios[2] < 5e-4)

# ---------------------------------------------------------------- Exercise 46.2.1
# (b) k 1_[0,1/k] L^1 tested with cos: k int_0^(1/k) cos = k sin(1/k) -> 1 = cos(0)
for k in (10, 100, 1000):
    close(f"Exercise 46.2.1(b): k sin(1/k) -> 1, k = {k}", k * math.sin(1 / k), 1.0, 1 / k ** 2)

# ---------------------------------------------------------------- Exercise 46.2.2: Hausdorff measures of simple sets
check("Exercise 46.2.2: omega_0 = 1, omega_1 = 2, omega_2 = pi, omega_3 = 4 pi / 3",
      v.OMEGA[0] == 1 and v.OMEGA[1] == 2 and abs(v.OMEGA[2] - math.pi) < 1e-15 and abs(v.OMEGA[3] - 4 * math.pi / 3) < 1e-15)
# covering the unit square by k^2 squares of side 1/k (diameter sqrt2/k): d = 2 sum -> pi/2 * (upper bound), d = 3 -> 0
for k in (10, 100):
    s2 = k * k * math.pi * (math.sqrt(2) / k / 2) ** 2
    s3 = k * k * (4 * math.pi / 3) * (math.sqrt(2) / k / 2) ** 3
    close(f"Exercise 46.2.2: square cover, d = 2 sum is pi/2 for every k = {k}", s2, math.pi / 2, 1e-12)
    check(f"Exercise 46.2.2: square cover, d = 3 sum -> 0 (k = {k})", s3 < 4 / k)

# ---------------------------------------------------------------- Exercise 46.2.3: equal-weight density
close("Exercise 46.2.3: int theta dH^1 = 2 pi (total mass unchanged)",
      v.integrate_1d(lambda t: theta_of_t(t), 0, 2 * np.pi, n=64, pieces=64), 2 * np.pi, 1e-9)
close("Exercise 46.2.3: int theta dt = int 2 pi du", v.integrate_1d(lambda u: 2 * np.pi * np.ones_like(u), 0, 1), 2 * np.pi, 1e-12)

# ---------------------------------------------------------------- Exercise 46.2.4: staircase pushed to the x-axis
for n in (1, 4, 50):
    S = v.staircase(n)
    # horizontal pieces push to Lebesgue on [0, 1], vertical pieces to (1/n) delta at x = (k+1)/n
    val = 0.5 + sum((1 / n) * ((k + 1) / n) for k in range(n))
    close(f"Exercise 46.2.4: int x d(pi_# H^1 on S_{n}) = 1/2 + (n+1)/(2n)", val, 0.5 + (n + 1) / (2 * n), 1e-12)
    direct = v.polyline_integral(lambda X, P: X[:, 0], S, closed=False, nq=4)
    close(f"Exercise 46.2.4: same value computed on S_{n} directly", direct, 0.5 + (n + 1) / (2 * n), 1e-12)

# ---------------------------------------------------------------- Exercise 46.2.5: a curve is singular for L^2
for r in (0.1, 0.01):
    arc = arc_mass_in_ball(np.array([1.0, 0.0]), r, lambda ts: np.ones_like(ts))
    close(f"Exercise 46.2.5: H^1(C cap B_r)/L^2(B_r) ~ 2/(pi r), r = {r}", arc / (math.pi * r * r), 2 / (math.pi * r),
          0.01 * 2 / (math.pi * r))

summary()
