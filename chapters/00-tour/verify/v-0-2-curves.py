"""0.2절 Curves: Speed, Arc Length, and Curvature — 본문과 Quick check의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-2-curves.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES

# Example 0.2.2: 원과 helix의 속도, 속력, 길이 ---------------------------------------------------
cir = EX["circle"]
(t,) = cir["coords"]
(r,) = cir["params"]
gc = cir["expr"]
sym_equal("Example 0.2.2: circle 속도 = (−r sin t, r cos t, 0)", gc.diff(t), sp.Matrix([-r * sp.sin(t), r * sp.cos(t), 0]))
sym_equal("Example 0.2.2: circle 속력 = r", sp.sqrt(gc.diff(t).dot(gc.diff(t))), r)

hel = EX["helix"]
(th,) = hel["coords"]
a, b = hel["params"]
gh = hel["expr"]
c = sp.sqrt(a ** 2 + b ** 2)
sp_h = sp.simplify(sp.sqrt(gh.diff(th).dot(gh.diff(th))))
sym_equal("Example 0.2.2: helix 속력 = √(a²+b²)", sp_h, c, hel["domain"])
sym_equal("Example 0.2.2: helix 한 바퀴 길이 = 2π√(a²+b²)", sp.integrate(sp_h, (th, 0, 2 * sp.pi)), 2 * sp.pi * c, hel["domain"])
s = sp.symbols("s", real=True)
gs = gh.subs(th, s / c)
sym_equal("Example 0.2.2 / (0.2.1): arc length parametrization의 속력 = 1",
          sp.simplify(gs.diff(s).dot(gs.diff(s))), 1, hel["domain"])

# Example 0.2.5: curvature ---------------------------------------------------------------------
kc, tc = dgsym.curvature_torsion(gc, t, ())
sym_equal("Example 0.2.5: circle κ = 1/r", kc, cir["expected"]["kappa"], cir["domain"])
sym_equal("Example 0.2.5: 반시계 circle κ_s = +1/r", dgsym.signed_curvature(gc, t), 1 / r, cir["domain"])
gc_rev = gc.subs(t, -t)
sym_equal("Example 0.2.5 / Exercise 0.2.3: 시계 방향 circle κ_s = −1/r", dgsym.signed_curvature(gc_rev, t), -1 / r, cir["domain"])
kh, tauh = dgsym.curvature_torsion(gh, th)
sym_equal("Example 0.2.5: helix κ = a/(a²+b²)", kh, hel["expected"]["kappa"], hel["domain"])
sym_equal("(0.2.5): helix τ = b/(a²+b²)", tauh, hel["expected"]["tau"], hel["domain"])
# (0.2.2)와 (0.2.3)이 arc length parametrization에서 같다: |γ''(s)| = κ
sym_equal("(0.2.2): helix |γ''(s)| = a/(a²+b²)", sp.sqrt(gs.diff(s, 2).dot(gs.diff(s, 2))), hel["expected"]["kappa"], hel["domain"])

# ellipse (Figure 0.1.1, 0.2.2, Exercise 0.2.1) ------------------------------------------------
tt = sp.symbols("t", real=True)
ell = sp.Matrix([2 * sp.cos(tt), sp.sin(tt), 0])
ks = dgsym.signed_curvature(ell, tt)
sym_equal("Exercise 0.2.1: ellipse κ_s = 2/(4 sin²t + cos²t)^{3/2}", ks,
          2 / (4 * sp.sin(tt) ** 2 + sp.cos(tt) ** 2) ** sp.Rational(3, 2), {tt: (0, 2 * sp.pi)})
check("Exercise 0.2.1: κ_s(0) = 2, 반지름 1/2", sp.simplify(ks.subs(tt, 0) - 2) == 0)
check("Exercise 0.2.1: κ_s(π/2) = 1/4, 반지름 4", sp.simplify(ks.subs(tt, sp.pi / 2) - sp.Rational(1, 4)) == 0)
close("Figure 0.2.2: t = π/4에서 반지름 ≈ 1.98", float(1 / ks.subs(tt, sp.pi / 4)), 1.976, 1e-3)
num = sp.diff(2 * sp.cos(tt), tt) * sp.diff(sp.sin(tt), tt, 2) - sp.diff(sp.sin(tt), tt) * sp.diff(2 * sp.cos(tt), tt, 2)
check("Exercise 0.2.1: det(γ', γ'') = 2", sp.simplify(num - 2) == 0)

# (0.2.4): graph y = f(x)의 signed curvature = f''/(1 + f'²)^{3/2} -------------------------------
x = sp.symbols("x", real=True)
f = sp.Function("f")(x)
sym_equal("(0.2.4): graph의 κ_s = f''/(1+f'^2)^{3/2}", dgsym.signed_curvature(sp.Matrix([x, f, 0]), x),
          f.diff(x, 2) / (1 + f.diff(x) ** 2) ** sp.Rational(3, 2))
ksin = dgsym.signed_curvature(sp.Matrix([x, sp.sin(x), 0]), x)
check("Figure 0.2.1: (t, sin t)의 inflection point t = π에서 κ_s = 0", ksin.subs(x, sp.pi) == 0)
check("Figure 0.2.1: 0 < t < π에서 κ_s < 0", all(float(ksin.subs(x, v)) < 0 for v in (0.3, 1.5, 3.0)))

# Theorem 0.2.6: κ_s(s)에서 곡선을 다시 만들면 rigid motion만큼만 다르다 (ellipse, 수치) ------------
T = np.linspace(0, 2 * np.pi, 200001)
P = np.stack([2 * np.cos(T), np.sin(T)], axis=1)
dP = np.stack([-2 * np.sin(T), np.cos(T)], axis=1)
sp_e = np.linalg.norm(dP, axis=1)
kf = sp.lambdify(tt, ks, "numpy")
S = np.concatenate([[0], np.cumsum(0.5 * (sp_e[1:] + sp_e[:-1]) * np.diff(T))])
kap_s = kf(T)
theta = np.concatenate([[0], np.cumsum(0.5 * (kap_s[1:] + kap_s[:-1]) * np.diff(S))])
Q = np.stack([np.concatenate([[0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * np.diff(S))]),
              np.concatenate([[0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * np.diff(S))])], axis=1)
th0 = np.arctan2(dP[0, 1], dP[0, 0])
Rm = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
Q2 = Q @ Rm.T + P[0]
check("Theorem 0.2.6: κ_s(s)로 다시 만든 곡선 = 원래 ellipse (rigid motion 뒤, 수치)", np.max(np.abs(Q2 - P)) < 1e-6)
close("Theorem 0.2.7: ellipse의 ∫κ_s ds = 2π", theta[-1], 2 * np.pi, 1e-8)
close("Theorem 0.2.7: ellipse 둘레 ≈ 9.69", S[-1], 9.688448, 1e-5)

# Frenet–Serret (0.2.5)를 helix에서 --------------------------------------------------------------
Tt, Nn, Bb = dgsym.frenet_frame(gs, s)
kap = hel["expected"]["kappa"]
tau = hel["expected"]["tau"]
dom = {**hel["domain"], s: (0, 5)}
sym_equal("(0.2.5): t' = κ n", Tt.diff(s), kap * Nn, dom)
sym_equal("(0.2.5): n' = −κ t + τ b", Nn.diff(s), -kap * Tt + tau * Bb, dom)
sym_equal("(0.2.5): b' = −τ n", Bb.diff(s), -tau * Nn, dom)
sym_equal("Figure 0.2.3: helix의 n = (−cos, −sin, 0) (축을 향함)", Nn, sp.Matrix([-sp.cos(s / c), -sp.sin(s / c), 0]), dom)

# Figure 0.2.3: a = 1, b = 0.3에서 κ ≈ 0.917, τ ≈ 0.275 ---------------------------------------------
close("Figure 0.2.3: κ(1, 0.3) ≈ 0.917", float(kap.subs({a: 1, b: sp.Rational(3, 10)})), 0.917, 5e-4)
close("Figure 0.2.3: τ(1, 0.3) ≈ 0.275", float(tau.subs({a: 1, b: sp.Rational(3, 10)})), 0.275, 5e-4)

# Exercise 0.2.3: 방향을 뒤집으면 κ_s의 부호가, λ배 확대하면 κ가 1/λ배 -------------------------------
lam = sp.symbols("lambda", positive=True)
ks_rev = dgsym.signed_curvature(ell.subs(tt, -tt), tt)
sym_equal("Exercise 0.2.3(a): 반대 방향 ellipse의 κ_s(−t) = −κ_s(t)", ks_rev.subs(tt, -tt), -ks, {tt: (0, 2 * sp.pi)})
Rot = sp.Matrix([[sp.cos(sp.pi / 6), -sp.sin(sp.pi / 6), 0], [sp.sin(sp.pi / 6), sp.cos(sp.pi / 6), 0], [0, 0, 1]])
sym_equal("Exercise 0.2.3(b): 30° 회전해도 κ_s는 같다", dgsym.signed_curvature(Rot * ell, tt), ks, {tt: (0, 2 * sp.pi)})
sym_equal("Exercise 0.2.3(c): λ배 확대하면 κ_s는 1/λ배", dgsym.signed_curvature(lam * ell, tt), ks / lam, {tt: (0, 2 * sp.pi)})

# Exercise 0.2.2: 극한 ----------------------------------------------------------------------------
check("Exercise 0.2.2: b → 0이면 κ → 1/a, τ → 0", sp.limit(kap, b, 0) == 1 / a and sp.limit(tau, b, 0) == 0)
check("Exercise 0.2.2: b → ∞이면 κ → 0, τ → 0", sp.limit(kap, b, sp.oo) == 0 and sp.limit(tau, b, sp.oo) == 0)
check("Exercise 0.2.2: b = a에서 τ가 최대 1/(2a)",
      sp.simplify(sp.diff(tau, b).subs(b, a)) == 0 and sp.simplify(tau.subs(b, a) - 1 / (2 * a)) == 0)

summary()
