"""Verification for Section 32.4 (No Free Lunch: Discrete Curvature Flow).

Run from the project root: PYTHONPATH=tools python3 chapters/32-ddg-overview/verify/v-32-4-no-free-lunch.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgdec as dd  # noqa: E402
import dgsym  # noqa: E402
import ddg32  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "32-4-no-free-lunch.html"
rng = np.random.default_rng(324)
NS = ddg32.run_block(PAGE, "code-32-4-1")
cv, flow = NS["curvature_vectors"], NS["flow"]
check("Implementation 32.4: at most 30 lines", ddg32.code_lines(PAGE, "code-32-4-1") <= 30)

# ---------------------------------------------------------------- Proposition 32.4.3 (c) ROUND, symbolic
u, t = sp.symbols("u t", real=True)
r0 = sp.symbols("r0", positive=True)
rt = sp.sqrt(r0 ** 2 - 2 * t)
g = sp.Matrix([rt * sp.cos(u), rt * sp.sin(u)])
ks = dgsym.signed_curvature(g, u, positive=(rt,))
gu = g.diff(u)
Tn = gu / sp.sqrt(sp.simplify(gu.dot(gu)))
Nn = sp.Matrix([-Tn[1], Tn[0]])
sym_equal("Prop 32.4.3(c): d gamma/dt = kappa_s n_s for the circles r(t) = sqrt(r0^2 - 2t)",
          list(g.diff(t)), list(sp.simplify(ks * Nn)), {r0: (1, 2), t: (0, 0.4), u: (0, 6)})
sym_equal("Prop 32.4.3(c): r(t)^2 has slope -2", sp.diff(rt ** 2, t), -2)
# Warning box (not the offset): d gamma/dt = n_s moves a counterclockwise circle to radius r0 - t
go = sp.Matrix([(r0 - t) * sp.cos(u), (r0 - t) * sp.sin(u)])
gou = go.diff(u)
To = gou / (r0 - t)
sym_equal("Warning box: the unit-speed normal motion d gamma/dt = n_s gives the circles of radius r0 - t",
          list(go.diff(t)), [-To[1], To[0]], {r0: (1, 2), t: (0, 0.5), u: (0, 6)})

# ---------------------------------------------------------------- Prop 32.4.2, 32.4.3(a)(b) on smooth curves
M = 20000
tq = 2 * np.pi * np.arange(M) / M
for name in ("ellipse", "flower", "limacon"):
    sp_ = ddg32.speed(name, tq)
    k = ddg32.kappa(name, tq)
    gx = sp.lambdify(ddg32.T_SYM, list(ddg32.CURVES[name].diff(ddg32.T_SYM)), "numpy")
    D1 = np.stack([np.asarray(c) * np.ones_like(tq) for c in gx(tq)], 1)
    T = D1 / sp_[:, None]
    N = T @ ddg32.J.T
    mom = (k[:, None] * N * sp_[:, None]).sum(0) * 2 * np.pi / M
    close(f"Prop 32.4.3(b): int kappa_s n_s ds = 0 ({name})", mom, [0, 0], 1e-10)
    close(f"Prop 32.4.3(a): int kappa_s ds = 2 pi ({name}, simple and counterclockwise)",
          (k * sp_).sum() * 2 * np.pi / M, 2 * np.pi, 1e-9)

# dL/dt = -int kappa^2 ds at t = 0 (ellipse), by a fine approximation of the flow over a short time
E0 = ddg32.sample("ellipse", 20000)[0]
dtt = 1e-4
c0, c1 = ddg32.smooth_flow(E0, [0.0, dtt], n=1600)
dLdt = (dd.curve_length(c1, True) - dd.curve_length(c0, True)) / dtt
close("Prop 32.4.2: dL/dt at t = 0 = -int kappa^2 ds = -6.636 (ellipse, numerical flow)", dLdt, -6.636029752123299,
      2e-2)

# ---------------------------------------------------------------- Figure 32.4.1 and Exercise 32.4.2
F0 = ddg32.sample("flower", 4000)[0]
A0 = np.pi * (1 + 0.3 ** 2 / 2)
tf = [0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.48]
cf = ddg32.smooth_flow(F0, tf, n=400)
areas = np.array([ddg32.polygon_area(X) for X in cf])
close("Exercise 32.4.2: A(0) = 1.045 pi = 3.2830, A(0)/2 pi = 0.5225", [round(A0, 4), round(A0 / (2 * np.pi), 4)],
      [3.2830, 0.5225], 1e-12)
check("Figure 32.4.1 / Exercise 32.4.2: area = A(0) - 2 pi t within 1e-3 at all ten times",
      np.max(np.abs(areas - (A0 - 2 * np.pi * np.array(tf)))) < 1e-3)
convex = [bool(np.all(dd.turning_angles(X, True) > 0)) for X in cf]
check("Figure 32.4.1: not convex at t = 0.06, convex from t = 0.1 on", (not convex[3]) and all(convex[4:]))
cent = [X.mean(0) for X in cf]
check("Warning box: the 5-fold symmetric flower keeps its center (400 equally spaced points, |centroid| < 1e-9)",
      max(np.linalg.norm(c) for c in cent) < 1e-9)
check("Figure 32.4.1: total turning stays 2 pi along the computed flow",
      all(abs(dd.turning_angles(X, True).sum() - 2 * np.pi) < 1e-9 for X in cf))

# ---------------------------------------------------------------- Warning: the limacon centroid moves, (32.4.3)
P = ddg32.point("limacon", tq)
sp_ = ddg32.speed("limacon", tq)
k = ddg32.kappa("limacon", tq)
ds = sp_ * 2 * np.pi / M
L = ds.sum()
c = (P * ds[:, None]).sum(0) / L
dcdt = -((P - c) * (k ** 2 * ds)[:, None]).sum(0) / L
close("Warning (32.4.3): limacon r = 1 + 0.4 cos t: dc/dt at t = 0 = (0.0114, 0)", np.round(dcdt, 4), [0.0114, 0.0],
      1e-12)
check("Warning: the convex limacon is convex (kappa_s > 0)", k.min() > 0)
Lm0 = ddg32.sample("limacon", 20000)[0]


def arc_centroid(X):
    e = np.linalg.norm(np.roll(X, -1, 0) - X, axis=1)
    return ((X + np.roll(X, -1, 0)) / 2 * e[:, None]).sum(0) / e.sum()


m0, m1 = ddg32.smooth_flow(Lm0, [0.0, 1e-3], n=1600)
rate = (arc_centroid(m1) - arc_centroid(m0)) / 1e-3
check(f"Warning: the computed flow (1600 points) moves the arc-length centroid at {rate[0]:.4f} (expected 0.0114)",
      abs(rate[0] - dcdt[0]) < 3e-4 and abs(rate[1]) < 1e-9)
Asym = ddg32.polygon_area(Lm0)
r_ = 1 + 0.4 * np.cos(tq)                                   # area centroid by the polar formula (1/3) int r^3 u dt / A
cA = np.array([(r_ ** 3 * np.cos(tq)).sum(), (r_ ** 3 * np.sin(tq)).sum()]) * (2 * np.pi / M) / 3 / Asym
spt = (P * (k * ds)[:, None]).sum(0) / (2 * np.pi)
check("Warning: the area centroid also moves (d/dt = (2 pi/A)(c_A - Steiner point) != 0)",
      abs(2 * np.pi / Asym * (cA - spt))[0] > 1e-3)

# ---------------------------------------------------------------- Definition 32.4.5 and the Implementation
ok = True
for X in (ddg32.jittered_flower(), ddg32.uneven_circle(), ddg32.sample("ellipse", 13)[0]):
    for kind in "ABCD":
        ok &= np.allclose(cv(X, kind), ddg32.curvature_vectors(X, kind))
    ok &= np.allclose(cv(X, "B"), -dd.length_gradient(X, True))
    kD = dd.curvature_osculating_circle(X, True)
    ok &= np.allclose(np.linalg.norm(cv(X, "D"), axis=1), np.abs(kD))
    ok &= np.all(np.sign((cv(X, "D") * dd.curve_vertex_normals(X, True)).sum(1)) == np.sign(kD))
check("Def 32.4.5: Implementation = ddg32; kappa^B vector = -grad L; |kappa^D vector| = |kappa^D|, side of n_i = "
      "sign kappa^D", ok)
ok = True
for X in (ddg32.jittered_flower(), ddg32.uneven_circle(), rng.normal(size=(9, 2))):
    th = dd.turning_angles(X, True)
    if np.any(np.abs(th) > np.pi - 1e-6) or np.any(th == 0):
        continue
    R = X[::-1]
    for kind in "ABCD":
        ok &= np.allclose(cv(R, kind)[::-1], cv(X, kind))
    ok &= np.allclose(dd.turning_angles(R, True)[::-1], -th)
check("Text after Def 32.4.5: reversing the vertex order flips theta_i and n_i, kappa^X_i vectors unchanged", ok)

# ---------------------------------------------------------------- Proposition 32.4.6 (regular polygons)
ok = True
for n in (5, 9, 16):
    for r in (1.0, 2.5):
        X = ddg32.regular_ngon(n, r)
        h = 0.01
        rho = {"A": 2 * np.pi / n, "B": 2 * np.sin(np.pi / n), "C": 2 * np.tan(np.pi / n), "D": 1 / r}
        for kind in "ABCD":
            Y = flow(X, kind, h, 1)
            ok &= np.allclose(Y, X * (r - h * rho[kind]) / r, atol=1e-13)
check("Prop 32.4.6: one step maps a regular n-gon to the concentric regular n-gon of radius r - h rho_X", ok)
Xr = ddg32.regular_ngon(7, 1.3)
check("Prop 32.4.6 proof: n_i = -gamma_i / r for a regular polygon",
      np.allclose(dd.curve_vertex_normals(Xr, True), -Xr / 1.3))

# ---------------------------------------------------------------- Proposition 32.4.7 (TOTAL) and Figure 32.4.2
Fj = ddg32.sample("flower", 48, 0.25, 11)[0]
lbj = dd.curve_dual_lengths(Fj, True)
dt, Tend = 5e-5, 0.06
steps = int(round(Tend / dt))
res = {}
for kind in "ABCD":
    h = dt if kind == "D" else dt / lbj.mean()
    tr = ddg32.flow(Fj, kind, h, steps, every=10)
    res[kind] = tr
    Yi = flow(Fj, kind, h, steps)
    check(f"Fig 32.4.2: the Implementation flow ({kind}) gives the same final polygon as ddg32.flow",
          np.allclose(Yi, tr[-1], atol=1e-12))
check("Prop 32.4.7: sum kappa^A = 2 pi at every recorded step of all four flows (jittered flower)",
      all(abs(dd.turning_angles(X, True).sum() - 2 * np.pi) < 1e-10 for kind in "ABCD" for X in res[kind]))
tot = {kd: np.array([ddg32.own_total(X, kd) for X in res[kd]]) for kd in "ABCD"}
close("Fig 32.4.2(a): B 6.2724 -> min 6.1777; C max 6.5020; D min 6.1545",
      [round(tot["B"][0], 4), round(tot["B"].min(), 4), round(tot["C"].max(), 4), round(tot["D"].min(), 4)],
      [6.2724, 6.1777, 6.5020, 6.1545], 1e-12)
check("Fig 32.4.2(a): the own totals of B, C, D are not constant", all(np.ptp(tot[k_]) > 0.05 for k_ in "BCD"))
drift = {kd: np.linalg.norm(res[kd][-1].mean(0) - Fj.mean(0)) for kd in "ABCD"}
check("Fig 32.4.2(b): drift at t = 0.06: A 2.9e-5, C 8.8e-5, D 1.6e-3",
      [f"{drift[k_]:.1e}" for k_ in "ACD"] == ["2.9e-05", "8.8e-05", "1.6e-03"], str(drift))
check("Fig 32.4.2(b) / Prop 32.4.8: B drift is round-off only (< 1e-14) at every recorded step",
      max(np.linalg.norm(X.mean(0) - Fj.mean(0)) for X in res["B"]) < 1e-14)
check("Fig 32.4.2: no edge collapses in this time range (min edge > 10% of the initial one)",
      all(min(dd.curve_edge_lengths(X, True).min() for X in res[k_]) > 0.1 * dd.curve_edge_lengths(Fj, True).min()
          for k_ in "ABCD"))
# long runs: short edges can collapse (text after Proposition 32.4.7)
Fu = ddg32.sample("flower", 64)[0]
lbu = dd.curve_dual_lengths(Fu, True)
long = ddg32.flow(Fu, "A", 5e-5 / lbu.mean(), 4000, every=400)
me = min(dd.curve_edge_lengths(X, True).min() for X in long) / dd.curve_edge_lengths(Fu, True).min()
check(f"Text after Prop 32.4.7: the A flow on the flower (64 vertices) shrinks some edge to {me:.3f} of its initial "
      "length by t = 0.2", me < 0.02)

# ---------------------------------------------------------------- Proposition 32.4.8 (DRIFT) in general
ok = True
for _ in range(50):
    X = rng.normal(size=(int(rng.integers(3, 15)), 2))
    th = dd.turning_angles(X, True)
    if np.any(np.abs(th) > np.pi - 1e-6):
        continue
    ok &= np.allclose(cv(X, "B").sum(0), 0, atol=1e-12)
    Y = X + 0.37 * cv(X, "B")
    ok &= np.allclose(Y.mean(0), X.mean(0), atol=1e-14)
check("Prop 32.4.8: sum kappa^B_i = 0 and the vertex average is unchanged for any h (50 random polygons)", ok)

# ---------------------------------------------------------------- Proposition 32.4.9 (ROUND) and Figure 32.4.3
U = ddg32.uneven_circle()
lbU = dd.curve_dual_lengths(U, True)
ok = True
for h in (1e-3, 0.05, 0.3):
    Y = flow(U, "D", h, 1)
    ok &= np.allclose(np.linalg.norm(Y, axis=1), 1 - h) and np.allclose(Y / (1 - h), U)
check("Prop 32.4.9: one D step maps the uneven unit circle to the circle of radius 1 - h (same angles)", ok)
Y = flow(U, "D", 1e-3, 300)
r_m = 1.0
for _ in range(300):
    r_m -= 1e-3 / r_m
close("Prop 32.4.9 / Fig 32.4.3: after 300 steps radius r_m = 0.6328 (recursion), smooth 0.6325",
      [np.linalg.norm(Y, axis=1).max(), np.linalg.norm(Y, axis=1).min(), round(r_m, 4), round(np.sqrt(0.4), 4)],
      [r_m, r_m, 0.6328, 0.6325], 1e-12)
rd = {}
for kind in "ABC":
    Yk = flow(U, kind, 1e-3 / lbU.mean(), 300)
    rd[kind] = ddg32.roundness_defect(Yk)
Yb = U.copy()
for _ in range(300):
    Yb = Yb + 1e-3 * ddg32.curvature_vectors(Yb, "D", normal="bisector")
rd["Dbis"] = ddg32.roundness_defect(Yb)
close("Fig 32.4.3(b): defect at t = 0.3: A 0.117, B 0.115, C 0.121, D with n_i 2.2e-3",
      [round(rd["A"], 3), round(rd["B"], 3), round(rd["C"], 3), round(rd["Dbis"], 4)], [0.117, 0.115, 0.121, 0.0022],
      1e-12)
check("Fig 32.4.3(b): D defect < 1e-13", ddg32.roundness_defect(Y) < 1e-13)
thU = np.degrees(dd.turning_angles(U, True))
kBU = dd.curvature_length_variation(U, True)
close("Text after Fig 32.4.3: theta = 40 deg at 105 and 160 deg, 20 deg at 185 deg (half the two adjacent arcs)",
      [thU[4], thU[5], thU[6]], [40.0, 40.0, 20.0], 1e-9)
check("Text after Fig 32.4.3: kappa^B_i = lbar_i / r within 1% on the uneven circle (B steps scale with lbar_i)",
      np.all(np.abs(kBU / lbU - 1) < 0.01) and np.argmax(kBU) in (4, 5) and np.argmin(kBU) == 6)
cfit, _ = ddg32.fit_circle(Yb)
d0 = np.linalg.norm(Yb, axis=1)
check(f"Exercise 32.4.5: spread about the origin {np.ptp(d0) / d0.mean():.1e} > defect about the fitted center "
      f"{rd['Dbis']:.1e}; the fitted center moved by {np.linalg.norm(cfit):.1e}",
      np.ptp(d0) / d0.mean() > 2 * rd["Dbis"] and np.linalg.norm(cfit) > 1e-4)

# ---------------------------------------------------------------- Exercises 32.4.1, 32.4.3, 32.4.4
ts_ = sp.symbols("t", nonnegative=True)
rr_ = sp.sqrt(1 - 2 * ts_)
sym_equal("Exercise 32.4.1: dL/dt = -int kappa^2 ds = -2 pi / r for the shrinking unit circle",
          sp.diff(2 * sp.pi * rr_, ts_), -(1 / rr_ ** 2) * 2 * sp.pi * rr_, {ts_: (0, 0.49)})
Tri = np.array([[0, 0], [1, 0], [0, 1.0]])
close("Exercise 32.4.3: theta = (pi/2, 3pi/4, 3pi/4)", dd.turning_angles(Tri, True), [np.pi / 2, 3 * np.pi / 4,
                                                                                        3 * np.pi / 4], 1e-12)
close("Exercise 32.4.3: vertex normals", np.round(dd.curve_vertex_normals(Tri, True), 4),
      [[0.7071, 0.7071], [-0.9239, 0.3827], [0.3827, -0.9239]], 1e-12)
close("Exercise 32.4.3: sum kappa^A n = (-0.1644, -0.1644), sum kappa^B n = 0",
      [*np.round(cv(Tri, "A").sum(0), 4), *np.round(cv(Tri, "B").sum(0), 12)], [-0.1644, -0.1644, 0, 0], 1e-12)
close("Exercise 32.4.3: ratios theta/(2 sin(theta/2)) = 1.111, 1.275; shift 0.0775 h",
      [round((np.pi / 2) / (2 * np.sin(np.pi / 4)), 3), round((3 * np.pi / 4) / (2 * np.sin(3 * np.pi / 8)), 3),
       round(np.linalg.norm(cv(Tri, "A").sum(0)) / 3, 4)], [1.111, 1.275, 0.0775], 1e-12)
close("Exercise 32.4.3: 2 sin(3pi/8) (sin(pi/8) - cos(pi/8)) = -1 exactly",
      2 * np.sin(3 * np.pi / 8) * (np.sin(np.pi / 8) - np.cos(np.pi / 8)), -1.0, 1e-12)
r25 = 1.0
for _ in range(25):
    r25 -= 0.01 / r25
close("Exercise 32.4.4: r_25 = 0.70951, exceeds sqrt(0.5) by 0.0024", [round(r25, 5), round(r25 - np.sqrt(0.5), 4)],
      [0.70951, 0.0024], 1e-12)
Ycirc = flow(ddg32.regular_ngon(10), "D", 0.01, 25)
close("Exercise 32.4.4: the Implementation D flow reproduces r_25 on a polygon on the unit circle",
      np.linalg.norm(Ycirc, axis=1), np.full(10, r25), 1e-12)

# ---------------------------------------------------------------- Exercises 32.4.5, 32.4.6 (printed code and output)
for blk in ("code-ex-32-4-5", "code-ex-32-4-6"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ddg32.run_block(PAGE, blk, NS)
    want = ddg32.code_block(PAGE, blk.replace("code", "out")).strip()
    check(f"Exercise {blk[-6:]}: the solution code prints exactly the output shown", buf.getvalue().strip() == want,
          f"got:\n{buf.getvalue()}")
    check(f"Exercise {blk[-6:]}: at most 30 lines", ddg32.code_lines(PAGE, blk) <= 30)

summary()
