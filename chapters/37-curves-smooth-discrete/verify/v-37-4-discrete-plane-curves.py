"""Verification for Section 37.4 (Discrete Plane Curves).

Run from the project root: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/verify/v-37-4-discrete-plane-curves.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "37-4-discrete-plane-curves.html"
check("Implementation 37.4: at most 30 lines", dgcode.code_lines(PAGE, "code-37-4-1") <= 30)
NS = dgcode.run_block(PAGE, "code-37-4-1")
la, cfa, cdef = NS["lengths_and_angles"], NS["curve_from_angles"], NS["closing_defects"]
rng = np.random.default_rng(374)

# ---------------------------------------------------------------- Implementation agrees with dgdec
for _ in range(5):
    X = rng.normal(size=(9, 2))
    ell, th = la(X)
    close("Implementation 37.4: lengths_and_angles = dgdec (random polygon)", np.r_[ell, th],
          np.r_[dd.curve_edge_lengths(X, True), dd.turning_angles(X, True)], 1e-12)

# ---------------------------------------------------------------- Proposition 37.4.1, (37.4.1)
n = 7
D = dd.curve_d0(n, True)
X = rng.normal(size=(n, 2))
close("(37.4.1): D gamma = edge vectors (D = dgdec.curve_d0)", D @ X, dd.curve_edge_vectors(X, True), 1e-12)
close("Prop 37.4.1(b): 1^T D = 0 for a closed curve", np.ones(n) @ D, np.zeros(n), 0)
close("Prop 37.4.1(a): gamma_5 - gamma_1 = e_1 + ... + e_4", X[5] - X[1], (D @ X)[1:5].sum(0), 1e-12)
check("Prop 37.4.1(c): ker D = constants (rank n - 1)", np.linalg.matrix_rank(D) == n - 1
      and np.allclose(D @ np.ones(n), 0))

# ---------------------------------------------------------------- Proposition 37.4.3 (locally injective)
P = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.5, -0.5]])
T = dd.curve_tangents(P, False)
check("Non-example 37.4.5: t_2 = -t_1 at (3, 1)", np.allclose(T[2], -T[1]))
a = 0.3
check("Prop 37.4.3: Gamma(2 - a/l_1) = Gamma(2 + a/l_2) for the backtracking vertex",
      np.allclose(P[2] - a * T[1], P[2] + a * T[2]))
for Xr in (ddg37.sample("ellipse", 16)[0], ddg37.sample("flower", 40)[0]):
    th = dd.turning_angles(Xr, True)
    check("Example 37.4.4(b): sampled ellipse / flower polygons are regular", np.abs(th).max() < np.pi - 1e-6)
thf = dd.turning_angles(ddg37.sample("flower", 40)[0], True)
close("Example 37.4.4(b): flower (n = 40) turning angles in [-1.130, 0.994]", [round(thf.min(), 3), round(thf.max(), 3)],
      [-1.130, 0.994], 1e-12)
a5 = np.pi / 2 + 4 * np.pi * np.arange(5) / 5
P5 = np.stack([np.cos(a5), np.sin(a5)], 1)
close("Example 37.4.4(c) / 37.4.12(b): pentagram turning angles 4 pi/5, k = 2",
      np.r_[dd.turning_angles(P5, True), dd.turning_angles(P5, True).sum() / (2 * np.pi)],
      np.r_[np.full(5, 0.8 * np.pi), 2.0], 1e-12)

# ---------------------------------------------------------------- Theorem 37.4.6 / 37.4.7, Example 37.4.8
for name, nn in (("ellipse", 16), ("flower", 40)):
    X, _ = ddg37.sample(name, nn)
    ell, th = la(X)
    G = cfa(ell, th, (1.0, -2.0), 0.3)
    close(f"Example 37.4.8: {name} n = {nn}: closing distance < 2e-15", min(np.linalg.norm(G[-1] - G[0]), 2e-15),
          np.linalg.norm(G[-1] - G[0]), 0)
    err = ddg37.rigid_align_error(G[:-1], X)
    check(f"Example 37.4.8: {name}: rigid alignment error {err:.1e} < 2e-15", err < 2e-15)
    d1, d2 = cdef(ell, th)
    check(f"Theorem 37.4.7: closing defects vanish for the sampled {name}", d1 < 1e-12 and d2 < 1e-12)
    # Theorem 37.4.6(b): a rigid motion maps one to the other; the angles are rigid invariants
    R = np.array([[np.cos(1.1), -np.sin(1.1)], [np.sin(1.1), np.cos(1.1)]])
    ell2, th2 = la(X @ R.T + np.array([3.0, -1.0]))
    close(f"Theorem 37.4.6(b): lengths and angles are invariant under a rigid motion ({name})", np.r_[ell2, th2],
          np.r_[ell, th], 1e-12)
# random open data: existence and the turning angles of the result
ellr = rng.uniform(0.5, 2, 8)
thr = np.r_[0.0, rng.uniform(-3, 3, 7)]
G = cfa(ellr, thr, (0.5, 0.5), -0.4)
close("Theorem 37.4.6(a): the rebuilt open curve has the prescribed lengths and turning angles",
      np.r_[dd.curve_edge_lengths(G, False), dd.turning_angles(G, False)[1:-1]], np.r_[ellr, thr[1:]], 1e-12)

# ---------------------------------------------------------------- Example 37.4.9, Non-example 37.4.10
for nn in (3, 5, 8):
    d1, d2 = cdef(np.full(nn, 0.7), np.full(nn, 2 * np.pi / nn))
    check(f"Example 37.4.9: regular {nn}-gon data close up", d1 < 1e-12 and d2 < 1e-12)
G = cfa(np.array([1.0, 2.0, 1.0, 1.0]), np.full(4, np.pi / 2))
close("Non-example 37.4.10: l = (1, 2, 1, 1), right angles: gamma_4 - gamma_0 = (0, 1)", G[-1] - G[0], [0, 1], 1e-12)

# ---------------------------------------------------------------- Exercise 37.4.3 (equal lengths and angles)
for k in (1, 2, 3, -2):
    d1, d2 = cdef(np.ones(7), np.full(7, 2 * np.pi * k / 7))
    check(f"Exercise 37.4.3: n = 7, k = {k} closes", d1 < 1e-12 and d2 < 1e-12)
check("Exercise 37.4.3: theta = 0 does not close", cdef(np.ones(7), np.zeros(7))[1] > 1)

# ---------------------------------------------------------------- Exercise 37.4.1, 37.4.2
P1 = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [3.0, 3.0]])
ell1 = dd.curve_edge_lengths(P1, False)
th1 = dd.turning_angles(P1, False)
G1 = cfa(ell1, np.r_[0.0, th1[1:-1]], (0.0, 0.0), np.pi / 2)
close("Exercise 37.4.1: rebuilt vertices (0,0), (0,2), (-1,3), (-3,3) = R_{pi/2} of the original",
      np.r_[G1.ravel(), (P1 @ np.array([[0, -1], [1, 0]]).T).ravel()],
      np.r_[[0, 0, 0, 2, -1, 3, -3, 3], [0, 0, 0, 2, -1, 3, -3, 3]], 1e-12)
X = rng.normal(size=(6, 2))
th = dd.turning_angles(X, True)
close("Exercise 37.4.2: the mirror image negates every turning angle",
      dd.turning_angles(X * np.array([1, -1]), True), -th, 1e-12)
close("Exercise 37.4.2: reversing the order gives theta'_i = -theta_{-i}", dd.turning_angles(X[::-1], True),
      -th[::-1], 1e-12)

# ---------------------------------------------------------------- Example 37.4.12, Proposition 37.4.13
P8, _ = ddg37.sample("figure8", 12, phase=np.pi / 12)
th8 = dd.turning_angles(P8, True)
close("Example 37.4.12(c): figure-eight polygon: k = 0, positive angles on the left loop", th8.sum(), 0, 1e-12)
check("Example 37.4.12(c): theta > 0 where x < 0 and theta < 0 where x > 0",
      np.all(th8[P8[:, 0] < -1e-9] > 0) and np.all(th8[P8[:, 0] > 1e-9] < 0))
tt = np.linspace(0, 2 * np.pi, 40001)


def osc_ok(name, nn):
    phi = ddg37.angle_lift(ddg37.velocity(name, tt))
    idx = np.round(np.linspace(0, 40000, nn + 1)).astype(int)
    return max(phi[idx[i]:idx[i + 1] + 1].max() - phi[idx[i]:idx[i + 1] + 1].min() for i in range(nn)) < np.pi / 2


check("Prop 37.4.13 remark: for the flower the hypothesis holds exactly for n = 10, 20 and 23..40 (n = 3..40)",
      [nn for nn in range(3, 41) if osc_ok("flower", nn)] == [10, 20] + list(range(23, 41)))
_ph16 = ddg37.angle_lift(ddg37.velocity("flower", tt))
_ix16 = np.round(np.linspace(0, 40000, 17)).astype(int)
close("Prop 37.4.13 remark: flower n = 16: largest amplitude 2.00",
      max(np.ptp(_ph16[_ix16[i]:_ix16[i + 1] + 1]) for i in range(16)), 2.00, 5e-3)
check("Exercise 37.4.6: for the limacon the hypothesis holds from n = 12 (fails at 11)",
      (not osc_ok("limacon", 11)) and osc_ok("limacon", 12))
ok = True
for name, k in (("ellipse", 1), ("flower", 1), ("limacon", 2), ("figure8", 0)):
    for nn in (24, 40, 64):
        if osc_ok(name, nn):
            X, _ = ddg37.sample(name, nn)
            ok &= abs(dd.turning_angles(X, True).sum() / (2 * np.pi) - k) < 1e-9
check("Prop 37.4.13: whenever the hypothesis holds, the discrete turning number equals rot", ok)
check("Prop 37.4.13 remark: flower polygons have k = 1 already from n = 3",
      all(abs(dd.turning_angles(ddg37.sample("flower", nn)[0], True).sum() / (2 * np.pi) - 1) < 1e-9
          for nn in range(3, 24)))

# ---------------------------------------------------------------- Example 37.4.15
for lam, k in ((0.6, 1), (0.01, 1), (-0.01, -1), (-0.6, -1)):
    P = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, lam]])
    close(f"Example 37.4.15: lambda = {lam}: k = {k}, min edge length 1",
          [dd.turning_angles(P, True).sum() / (2 * np.pi), dd.curve_edge_lengths(P, True).min()], [k, 1], 1e-12)

# ---------------------------------------------------------------- Exercises 37.4.5, 37.4.6 (printed code and output)
for ex in ("37-4-5", "37-4-6"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dgcode.run_block(PAGE, f"code-ex-{ex}", dict(NS))
    check(f"Exercise {ex.replace('-', '.')}: printed output matches",
          buf.getvalue().strip() == dgcode.code_block(PAGE, f"out-ex-{ex}").strip(), buf.getvalue())
X, _ = ddg37.sample("flower", 40)
ell, th = la(X)
close("Exercise 37.4.5: the gap after l_7 -> 1.5 l_7 equals 0.5 l_7", 0.5 * ell[7], 0.1078, 5e-5)

# ---------------------------------------------------------------- Smooth vs. Discrete box: kappa^D does not determine theta
def _kD(th, a=1.0, b=2.0):
    """kappa^D = 2 sin(theta) / w at a vertex with edges a, b (w = |a t_{i-1} + b t_i|)."""
    return 2 * np.sin(th) / np.sqrt(a * a + b * b + 2 * a * b * np.cos(th))


_lo, _hi = 2.0, np.pi - 1e-12            # kappa^D decreases on [2, pi): solve kappa^D(theta) = kappa^D(0.5) by bisection
for _ in range(100):
    _mid = 0.5 * (_lo + _hi)
    _lo, _hi = (_mid, _hi) if _kD(_mid) > _kD(0.5) else (_lo, _mid)
close("Smooth vs. Discrete 37.4: l = (1, 2): theta = 0.5 and theta = 2.9718 give the same kappa^D = 0.3287",
      [_kD(0.5), _lo], [0.3287, 2.9718], 5e-5)
_P = np.array([[-1.0, 0.0], [0.0, 0.0], [2 * np.cos(2.9718), 2 * np.sin(2.9718)]])
_th = dd.turning_angles(_P, False)[1]
close("Smooth vs. Discrete 37.4: kappa^D of the actual vertex (2 sin theta / w) for theta = 2.9718",
      2 * np.sin(_th) / np.linalg.norm(_P[2] - _P[0]), _kD(2.9718), 1e-12)

summary()
