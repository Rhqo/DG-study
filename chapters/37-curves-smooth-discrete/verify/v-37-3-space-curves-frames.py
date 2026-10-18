"""Verification for Section 37.3 (Space Curves and Adapted Frames).

Run from the project root: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/verify/v-37-3-space-curves-frames.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgnum  # noqa: E402
import dgsym  # noqa: E402
import ddg37  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "37-3-space-curves-frames.html"
check("Implementation 37.3: at most 30 lines", dgcode.code_lines(PAGE, "code-37-3-1") <= 30)
NS = dgcode.run_block(PAGE, "code-37-3-1")

# ---------------------------------------------------------------- Example 37.3.1
ex = dgsym.EXAMPLES["helix"]
(t,) = ex["coords"]
A, B = ex["params"]
kap_s, tau_s = dgsym.curvature_torsion(ex["expr"].subs({A: 1, B: sp.Rational(3, 10)}), t)
close("Example 37.3.1: helix (1, 0.3) kappa = 0.91743, tau = 0.27523",
      [float(kap_s), float(tau_s), 1.09], [0.91743, 0.27523, 1.0 + 0.09], 5e-6)
kap, tau = ddg37.helix_kappa_tau()

# arc-length Frenet frame of the helix (Example 5.3.3) and Proposition 37.3.2 / (37.3.3)
s = sp.symbols("s", real=True)
a, b = sp.symbols("a b", positive=True)
c = sp.sqrt(a ** 2 + b ** 2)
T = sp.Matrix([-a / c * sp.sin(s / c), a / c * sp.cos(s / c), b / c])
N = -sp.Matrix([sp.cos(s / c), sp.sin(s / c), 0])
Bn = sp.Matrix([b / c * sp.sin(s / c), -b / c * sp.cos(s / c), a / c])
F = sp.Matrix.hstack(T, N, Bn)
sym_equal("Prop 37.3.2: F^T F = I for the helix Frenet frame", list(sp.simplify(F.T * F)), list(sp.eye(3)))
Amat = sp.simplify(F.T * F.diff(s))
sym_equal("(37.3.3): F^T F' = [[0,-k,0],[k,0,-tau],[0,tau,0]] with k = a/c^2, tau = b/c^2", list(Amat),
          list(sp.Matrix([[0, -a / c ** 2, 0], [a / c ** 2, 0, -b / c ** 2], [0, b / c ** 2, 0]])))
sym_equal("Prop 37.3.2: A is antisymmetric", list(Amat + Amat.T), [0] * 9)

# ---------------------------------------------------------------- Conventions box (slides, a^2 + b^2 = 1)
th = sp.symbols("theta", real=True)
aa, bb = sp.cos(th), sp.sin(th)                       # a^2 + b^2 = 1, 0 < theta < pi/2
Tc = sp.Matrix([-aa * sp.sin(s), aa * sp.cos(s), bb])
Nc = sp.Matrix([sp.cos(s), sp.sin(s), 0])             # the slide's outward normal
Bc = Tc.cross(Nc)
sym_equal("Conventions: slide helix B = T x N = (-b sin s, b cos s, -a)", list(Bc), [-bb * sp.sin(s), bb * sp.cos(s), -aa])
sym_equal("Conventions: slide helix <N, T'> = -a (the slide's kappa, read as T' = kappa N)", Nc.dot(Tc.diff(s)), -aa)
sym_equal("Conventions: slide helix <N, B'> = -b (the slide's tau, B' = tau N)", Nc.dot(Bc.diff(s)), -bb)
sym_equal("Conventions: with T' = -kappa N (matrix slide) the same N gives kappa = +a", -Nc.dot(Tc.diff(s)), aa)
# with N = -n, B = -b the slide matrix holds with kappa_C = kappa, tau_C = -tau
Nm, Bm = -N, -Bn
kk, tt_ = a / c ** 2, b / c ** 2
sym_equal("Conventions: N = -n, B = -b satisfy T' = -kappa N, N' = kappa T + tau B, B' = -tau N",
          list(T.diff(s) + kk * Nm) + list(Nm.diff(s) - kk * T - tt_ * Bm) + list(Bm.diff(s) + tt_ * Nm), [0] * 9)
M_crane = sp.Matrix([[0, -1, 0], [1, 0, -1], [0, 1, 0]])
M_ours_rows = sp.Matrix([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])  # row form of (37.3.3) with kappa = tau = 1
check("Conventions: the slide's row matrix is minus the row form of (37.3.3)", M_crane == -M_ours_rows)

# ---------------------------------------------------------------- Example 37.3.3 (Implementation)
cfkt = NS["curve_from_kappa_tau"]
L2 = 4 * np.pi * np.sqrt(1.09)
P = cfkt(lambda s_: kap, lambda s_: tau, L2, 2000)
axis = np.array([tau, 0.0, kap]) / np.hypot(kap, tau)
p0 = np.array([0.0, 1.0, 0.0])
rel = P - p0
dist = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
close("Example 37.3.3: all points at distance kappa/(kappa^2+tau^2) = 1 from the axis", [dist.min(), dist.max()],
      [1.0, 1.0], 1e-11)
close("Example 37.3.3: rise per turn 2 pi tau/(kappa^2+tau^2) = 2 pi 0.3", (P[1000] - P[0]) @ axis,
      2 * np.pi * 0.3, 1e-10)

# ---------------------------------------------------------------- Prop 37.3.7 on a twisted adapted frame of the helix
zeta = sp.Rational(3, 10) * sp.sin(s)
U = sp.cos(zeta) * N + sp.sin(zeta) * Bn
W = T.cross(U)
k1, k2, m_ = T.diff(s).dot(U), T.diff(s).dot(W), U.diff(s).dot(W)
subs = {a: 1, b: sp.Rational(3, 10)}
sym_equal("Prop 37.3.7: t' = k1 u + k2 w", list((T.diff(s) - k1 * U - k2 * W).subs(subs)), [0, 0, 0], {s: (0, 6)})
sym_equal("Prop 37.3.7: u' = -k1 t + m w", list((U.diff(s) + k1 * T - m_ * W).subs(subs)), [0, 0, 0], {s: (0, 6)})
sym_equal("Prop 37.3.7: w' = -k2 t - m u", list((W.diff(s) + k2 * T + m_ * U).subs(subs)), [0, 0, 0], {s: (0, 6)})
sym_equal("Prop 37.3.7: kappa^2 = k1^2 + k2^2", (k1 ** 2 + k2 ** 2 - kk ** 2).subs(subs), 0, {s: (0, 6)})
sym_equal("Prop 37.3.10: m = theta' + tau (theta = zeta)", (m_ - zeta.diff(s) - tt_).subs(subs), 0, {s: (0, 6)})
sym_equal("Exercise 37.3.4: zeta' = m relative to the Bishop frame (theta = zeta + tau s)",
          (m_ - zeta.diff(s) - tt_).subs(subs), 0, {s: (0, 6)})

# ---------------------------------------------------------------- Prop 37.3.10, Example 37.3.11 (Bishop: theta = -tau s)
vth = -tt_ * s
Ub = sp.cos(vth) * N + sp.sin(vth) * Bn
Wb = -sp.sin(vth) * N + sp.cos(vth) * Bn
sym_equal("Prop 37.3.10: theta = -tau s gives m = <u', w> = 0", sp.simplify(Ub.diff(s).dot(Wb)), 0)
sym_equal("Prop 37.3.10: k1 = kappa cos theta", T.diff(s).dot(Ub), kk * sp.cos(vth))
sym_equal("Prop 37.3.10: k2 = -kappa sin theta", T.diff(s).dot(Wb), -kk * sp.sin(vth))
sym_equal("Prop 37.3.10: w = t x u", list(T.cross(Ub) - Wb), [0, 0, 0])
close("Example 37.3.11: one turn rotates the Bishop frame by -2 pi b / c = -1.80546",
      round(-2 * np.pi * 0.3 / np.sqrt(1.09), 5), -1.80546, 1e-12)
close("Example 37.3.11: -1.80546 rad is about -103.4 degrees", round(np.degrees(-2 * np.pi * 0.3 / np.sqrt(1.09)), 1),
      -103.4, 1e-9)

# ---------------------------------------------------------------- Example 37.3.12 and Exercise 37.3.2 (plane curves)
tp = sp.symbols("tp", positive=True)
cub = sp.Matrix([tp, tp ** 3, 0])
Tcub, Ncub, Bcub = dgsym.frenet_frame(cub, tp)
sym_equal("Example 37.3.12: n = (-3t^2, 1, 0)/sqrt(1+9t^4) for t > 0", list(Ncub),
          [-3 * tp ** 2 / sp.sqrt(1 + 9 * tp ** 4), 1 / sp.sqrt(1 + 9 * tp ** 4), 0])
sym_equal("Example 37.3.12: b = e3 for t > 0", list(Bcub), [0, 0, 1])
tn = sp.symbols("tn", negative=True)
_, Nneg, Bneg = dgsym.frenet_frame(sp.Matrix([tn, tn ** 3, 0]), tn)
sym_equal("Example 37.3.12: n = -(-3t^2, 1, 0)/sqrt(1+9t^4) and b = -e3 for t < 0",
          list(Nneg) + list(Bneg), [3 * tn ** 2 / sp.sqrt(1 + 9 * tn ** 4), -1 / sp.sqrt(1 + 9 * tn ** 4), 0, 0, 0, -1])
phi = sp.Function("phi")(s)
tpl = sp.Matrix([sp.cos(phi), sp.sin(phi), 0])
Jt = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
sym_equal("Exercise 37.3.2: (J t)' = -kappa_s t with kappa_s = phi'", list(Jt.diff(s) + phi.diff(s) * tpl), [0, 0, 0])
sym_equal("Exercise 37.3.2: t x J t = e3", list(tpl.cross(Jt)), [0, 0, 1])

# ---------------------------------------------------------------- Prop 37.3.13, Example 37.3.14 (trefoil)
tt = sp.symbols("t", real=True)
tref = sp.Matrix([(2 + sp.cos(3 * tt)) * sp.cos(2 * tt), (2 + sp.cos(3 * tt)) * sp.sin(2 * tt), sp.sin(3 * tt)])
ktr, ttr = dgsym.curvature_torsion(tref, tt)
spd = sp.sqrt(tref.diff(tt).dot(tref.diff(tt)))
fk, ft, fs = (sp.lambdify(tt, e, "numpy") for e in (ktr, ttr, spd))
mq = 200000
tq = 2 * np.pi * np.arange(mq) / mq
Ltr = float(fs(tq).sum() * 2 * np.pi / mq)
Itr = float((ft(tq) * fs(tq)).sum() * 2 * np.pi / mq)
hol = float(np.angle(np.exp(-1j * Itr)))
close("Example 37.3.14: kappa >= 0.385, length 31.899, total torsion -15.593, holonomy 3.0270",
      [round(fk(tq).min(), 3), round(Ltr, 3), round(Itr, 3), round(hol, 4)], [0.385, 31.899, -15.593, 3.0270], 1e-9)
# independent check: integrate the Bishop equation du/dt = -<T_t, u> T around the trefoil (RK4, dgnum)
d1 = tref.diff(tt)
Tun = d1 / spd
fT = sp.lambdify(tt, list(Tun), "numpy")
fTt = sp.lambdify(tt, list(Tun.diff(tt)), "numpy")
T0 = np.array(fT(0.0), float)
u0 = np.cross(T0, [0.0, 0.0, 1.0])
u0 /= np.linalg.norm(u0)
ts = np.linspace(0, 2 * np.pi, 6001)
us = dgnum.rk4(lambda t_, u: -np.dot(np.array(fTt(t_), float), u) * np.array(fT(t_), float), u0, ts)
uL = us[-1]
ang = ddg37.signed_angle_about(u0, uL - np.dot(uL, T0) * T0, T0)
close("Prop 37.3.13: Bishop transport around the trefoil turns u by -int tau ds (mod 2 pi)",
      np.angle(np.exp(1j * (ang - hol))), 0.0, 1e-6)

# ---------------------------------------------------------------- Exercise 37.3.5
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    dgcode.run_block(PAGE, "code-ex-37-3-5", {})
check("Exercise 37.3.5: printed output matches", buf.getvalue().strip() == dgcode.code_block(PAGE, "out-ex-37-3-5").strip(),
      buf.getvalue())

summary()
