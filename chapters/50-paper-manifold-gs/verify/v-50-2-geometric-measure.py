"""Verification for Section 50.2 (Gaussian Splats as a Geometric Measure).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/verify/v-50-2-geometric-measure.py
"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import mgs50 as g  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "50-2-geometric-measure.html"

# ---------------------------------------------------------------- (50.2.1) and Example 50.2.2
n = np.array([0.0, 0.6, 0.8])
P = g.plane_proj(n)
check("Example 50.2.2: P = I - n n^T is a rank-2 projection with P n = 0",
      np.allclose(P @ P, P) and abs(np.trace(P) - 2) < 1e-12 and np.allclose(P @ n, 0))
u = np.array([1.0, 0.0, 0.0])
v = np.cross(n, u)
S = 0.09 * np.outer(u, u) + 0.04 * np.outer(v, v) + 0.0004 * np.outer(n, n)
w, U = np.linalg.eigh(S)
check("(50.2.1): the top-2 eigenspace projection of Sigma is u u^T + v v^T = I - n n^T",
      np.allclose(U[:, 1:] @ U[:, 1:].T, P) and np.allclose(np.outer(u, u) + np.outer(v, v), P))
alpha, lam1, lam2 = 0.4, 0.09, 0.04
close("Example 50.2.2: alpha sqrt(lambda1 lambda2) = 0.024", alpha * math.sqrt(lam1 * lam2), 0.024, 1e-12)
close("Example 50.2.2: pi s1 s2 = 0.1885", math.pi * 0.3 * 0.2, 0.1885, 5e-5)
close("Example 50.2.2: 2 pi alpha s1 s2 = 0.1508", 2 * math.pi * alpha * 0.3 * 0.2, 0.1508, 5e-5)
# the rendered coverage of the flat footprint seen along its normal: int alpha exp(-x^T S2^-1 x / 2) dx
xs = np.linspace(-2, 2, 801)
X, Y = np.meshgrid(xs, xs, indexing="ij")
cov = alpha * np.exp(-0.5 * (X ** 2 / 0.09 + Y ** 2 / 0.04)).sum() * (xs[1] - xs[0]) ** 2
close("Example 50.2.2: int alpha g = 2 pi alpha s1 s2 (numerical)", cov, 2 * math.pi * alpha * 0.06, 1e-4)

# ---------------------------------------------------------------- Example 50.2.5 / Table 50.2.1
expect = {4: (0.541, 0.737, 16.06, 7.95, 8.02, 0.0659), 8: (0.276, 0.386, 8.32, 4.09, 4.09, 0.0145),
          16: (0.139, 0.196, 4.20, 2.05, 2.05, 0.0035), 32: (0.069, 0.098, 2.10, None, 1.03, 0.00087)}


FINE = {4: g.fine_bounds(4, 96), 8: g.fine_bounds(8, 48), 16: g.fine_bounds(16, 24), 32: g.fine_bounds(32, 24, lower=False)}
for nt, (h_e, eT_e, b_e, lo_e, up_e, sm_e) in expect.items():
    up, lo = FINE[nt]
    close(f"Table 50.2.1 (n_theta = {nt}): Delta upper (cell form, fine grid)", round(up, 2), up_e, 1e-9)
    if lo_e is not None:
        close(f"Table 50.2.1 (n_theta = {nt}): Delta lower (fine grid)", round(lo, 2), lo_e, 1e-9)
    check(f"Example 50.2.5 (n_theta = {nt}): bound / Delta in [2.00, 2.05] (fine grid)", 2.0 <= 4 * np.pi * (
        g.forward_consistency(nt, lower=False, ntests=1)["h"] + g.forward_consistency(nt, lower=False, ntests=1)["eT"]) / up <= 2.05)
check("Example 50.2.5: upper and lower agree to 1 % (n_theta = 4) and to 3 digits (n_theta = 8, 16)",
      (FINE[4][0] - FINE[4][1]) / FINE[4][0] < 0.01 and all(abs(FINE[n][0] - FINE[n][1]) < 5e-3 for n in (8, 16)))
res = {}
for nt, (h_e, eT_e, b_e, lo_e, up_e, sm_e) in expect.items():
    r = g.forward_consistency(nt, lower=(nt <= 16))
    res[nt] = r
    close(f"Table 50.2.1 (n_theta = {nt}): h, eps_T", [round(r["h"], 3), round(r["eT"], 3)], [h_e, eT_e], 1e-9)
    close(f"Table 50.2.1 (n_theta = {nt}): bound (7)", round(r["bound"], 2), b_e, 1e-9)
    close(f"Table 50.2.1 (n_theta = {nt}): Delta upper, 10-point rule within 0.2 % of the table", r["upper"], up_e, 0.002 * up_e + 0.005)
    if lo_e is not None:
        close(f"Table 50.2.1 (n_theta = {nt}): Delta lower, 10-point rule within 0.2 % of the table", r["lower"], lo_e, 0.002 * lo_e + 0.005)
        check(f"Example 50.2.5 (n_theta = {nt}): lower <= upper <= bound (7)",
              r["lower"] <= r["upper"] <= r["bound"])
    close(f"Table 50.2.1 (n_theta = {nt}): smooth error", r["smooth"], sm_e, 0.006 * sm_e + 1e-5)
    check(f"Example 50.2.5 (n_theta = {nt}): bound / Delta in [2.00, 2.05]", 2.0 <= r["bound"] / r["upper"] <= 2.05)
    close(f"Example 50.2.5 (n_theta = {nt}): eps_T (Frobenius) = sqrt 2 x eps_T (operator)",
          r["eT"] / math.sqrt(2), {4: 0.5210, 8: 0.2733, 16: 0.1383, 32: 0.0694}[nt], 6e-4)
H = np.array([res[nt]["h"] for nt in expect])
close("Example 50.2.5: slope of the cell form against h = 1.00",
      np.polyfit(np.log(H), np.log([res[nt]["upper"] for nt in expect]), 1)[0], 1.00, 0.01)
close("Example 50.2.5: slope of the smooth error against h = 2.10",
      np.polyfit(np.log(H), np.log([res[nt]["smooth"] for nt in expect]), 1)[0], 2.10, 0.01)
for nt, eq_e in zip(expect, (0.34, 0.32, 0.316, 0.315)):
    q = g.forward_consistency(nt, qbias=0.05, lower=False)
    close(f"Example 50.2.5 (n_theta = {nt}): eps_q with q = A(1 + 0.05 z)", round(q["eq"], 3 if nt > 8 else 2), eq_e, 1e-9)
    check(f"Example 50.2.5 (n_theta = {nt}): smooth error with the mass bias in [0.066, 0.068]",
          0.066 <= q["smooth"] <= 0.068)
close("Example 50.2.5: eps_q -> 0.05 int |z| dA = 0.1 pi", 0.05 * 2 * math.pi, 0.3142, 5e-5)

# ---------------------------------------------------------------- Implementation 50.2.1
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = dgcode.run_block(PAGE, "code-50-2-1")
check("Implementation 50.2.1: at most 30 lines", dgcode.code_lines(PAGE, "code-50-2-1") <= 30)
check("Implementation 50.2.1: printed output equals the page", buf.getvalue().strip() == dgcode.code_block(PAGE, "out-50-2-1").strip())
close("Implementation 50.2.1: h, eps_T and the cell form equal the library values",
      [ns["h"], ns["eT"], ns["cell"]], [res[8]["h"], res[8]["eT"], res[8]["upper"]], 1e-10)
check("Implementation 50.2.1: the test function is 1-Lipschitz and |phi| <= 3/4 (a, b unit)",
      abs(np.linalg.norm(ns["a"]) - 1) < 1e-12 and abs(np.linalg.norm(ns["b"]) - 1) < 1e-12)

# ---------------------------------------------------------------- exercises
close("Exercise 50.2.1: operator and Frobenius distances for 10 degrees",
      [g.op_from_normals(np.array([0, 0, 1.0]), np.array([math.sin(math.radians(10)), 0, math.cos(math.radians(10))])),
       g.fro_from_normals(np.array([0, 0, 1.0]), np.array([math.sin(math.radians(10)), 0, math.cos(math.radians(10))]))],
      [0.1736, 0.2456], 5e-5)
eT5 = math.sqrt(2) * math.sin(math.radians(5))
close("Exercise 50.2.2: eps_T = sqrt 2 sin 5 deg = 0.1233", eT5, 0.1233, 5e-5)
close("Exercise 50.2.2: bound in m = 0.3265", 2 * (0.01 + eT5) + 0.06, 0.3265, 5e-5)
close("Exercise 50.2.2: bound in cm = 23065", 20000 * (1 + eT5) + 600, 23065, 0.5)
# Exercise 50.2.4: the midpoint rule and the distance to the atoms on [0, 1]
for N in (8, 32):
    L = 1.0 / N
    mids = (np.arange(N) + 0.5) * L
    xq = np.linspace(0, 1, 200001)
    f = np.cos(3 * xq)
    err = abs(np.trapezoid(f, xq) - L * np.cos(3 * mids).sum())
    check(f"Exercise 50.2.4 (a) (N = {N}): midpoint error <= |f''| L^2 / 24", err <= 9 * L ** 2 / 24)
    dist = np.min(np.abs(xq[:, None] - mids[None, :]), axis=1)
    close(f"Exercise 50.2.4 (b) (N = {N}): int distance to the atoms = L / 4", np.trapezoid(dist, xq), L / 4, 1e-6)

summary()
