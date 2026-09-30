"""11.2절 차트, 아틀라스, 매끄러운 구조: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/verify/v-11-2-smooth-structures.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(112)
x, u = sp.symbols("x u", real=True)
up = sp.symbols("u", positive=True)

# ---------------------------------------------------------------- 동기, 비예 11.2.3: (R, id)와 (R, x^3)
cbrt = sp.sign(u) * sp.Abs(u) ** sp.Rational(1, 3)
check("비예 11.2.3: ψ∘id^{-1}(x) = x^3의 도함수는 0에서 0", sp.diff(x ** 3, x).subs(x, 0) == 0)
check("비예 11.2.3: id∘ψ^{-1}의 차분몫 h^{1/3}/h = h^{-2/3} → ∞ (h → 0+)",
      sp.limit(up ** sp.Rational(1, 3) / up, up, 0, "+") == sp.oo)
check("비예 11.2.3: 두 사상은 서로 역 (x > 0)", sp.simplify((up ** 3) ** sp.Rational(1, 3) - up) == 0)
xs = rng.normal(size=1000) * 3
check("비예 11.2.3: cbrt(x^3) = x (수치, 부호 포함)", bool(np.allclose(np.cbrt(xs ** 3), xs)))
# 동기의 (11.2.1): f∘ψ^{-1} = (f∘φ^{-1})∘(φ∘ψ^{-1})에서 f(x) = x, φ = id: f∘ψ^{-1}(u) = u^{1/3}
check("동기: f(x) = x를 x^3 차트로 읽으면 u^{1/3} (u > 0)", sp.simplify(sp.Lambda(x, x)(up ** sp.Rational(1, 3)) - up ** sp.Rational(1, 3)) == 0)

# ---------------------------------------------------------------- 예 11.2.12: 극좌표 차트
r, th = sp.symbols("r theta", real=True)
rp = sp.symbols("r", positive=True)
P = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
DP = P.jacobian([r, th])
sym_equal("예 11.2.12: det DP = r", DP.det(), r)
X, Y = sp.symbols("x y", real=True)
rho = sp.sqrt(X ** 2 + Y ** 2)
theta_inv = 2 * sp.atan(Y / (X + rho))
Th = sp.Matrix([rho, theta_inv])
DTh = Th.jacobian([X, Y])
# D(P^{-1})(P(r, θ)) = DP(r, θ)^{-1} = [[cos θ, sin θ], [-sin θ / r, cos θ / r]]
expected = sp.Matrix([[sp.cos(th), sp.sin(th)], [-sp.sin(th) / rp, sp.cos(th) / rp]])
sub = {X: rp * sp.cos(th), Y: rp * sp.sin(th)}
sym_equal("예 11.2.12: DP^{-1} = [[cos, sin], [-sin/r, cos/r]]", DP.subs(r, rp).inv(), expected, {rp: (0.3, 3), th: (-3, 3)})
DTh_sub = DTh.subs(sub)
sym_equal("예 11.2.12 (11.2.5): D(θ-차트)(P(r, θ)) = DP(r, θ)^{-1}", DTh_sub, expected, {rp: (0.3, 3), th: (-3.0, 3.0)})
ok = True
for _ in range(500):
    rr, tt = rng.uniform(0.1, 5), rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
    xx, yy = rr * np.cos(tt), rr * np.sin(tt)
    ok &= np.isclose(np.hypot(xx, yy), rr) and np.isclose(2 * np.arctan(yy / (xx + np.hypot(xx, yy))), tt)
check("예 11.2.12: (r, θ) ↦ P ↦ (√(x²+y²), 2 arctan(y/(x+√(x²+y²)))) = id (수치)", bool(ok))

# ---------------------------------------------------------------- 명제 11.2.14(c): 공을 R^n으로
v = sp.Matrix(sp.symbols("v1:4", real=True))
nv2 = (v.T * v)[0]
Fv = v / sp.sqrt(1 - nv2)
Gv = v / sp.sqrt(1 + nv2)
sym_equal("명제 11.2.14(c): G(F(v)) = v (예 2.3.2(c)의 사상)", Gv.subs({v[i]: Fv[i] for i in range(3)}, simultaneous=True), v,
          {s: (-0.5, 0.5) for s in v})

# ---------------------------------------------------------------- 연습 11.2.1: x + x^3
g = x + x ** 3
check("연습 11.2.1: (x + x^3)' = 1 + 3x^2 > 0", sp.solve(sp.Eq(sp.diff(g, x), 0), x) == [] or all(not s.is_real for s in sp.solve(sp.diff(g, x), x)))
check("연습 11.2.1: x + x^3 → ±∞ (전사)", sp.limit(g, x, sp.oo) == sp.oo and sp.limit(g, x, -sp.oo) == -sp.oo)
# 역함수를 뉴턴법으로 구해 (g^{-1})' = 1/g'(g^{-1}) 확인
ys = np.linspace(-5, 5, 41)
xs = np.zeros_like(ys)
for _ in range(60):
    xs = xs - (xs + xs ** 3 - ys) / (1 + 3 * xs ** 2)
h = 1e-6
xs_h = np.zeros_like(ys)
for _ in range(60):
    xs_h = xs_h - (xs_h + xs_h ** 3 - (ys + h)) / (1 + 3 * xs_h ** 2)
close("연습 11.2.1: (g^{-1})'(y) = 1/(1 + 3 g^{-1}(y)^2) (수치)", (xs_h - xs) / h, 1 / (1 + 3 * xs ** 2), tol=1e-5)

# ---------------------------------------------------------------- 연습 11.2.2: x^k (k 홀수)
for k in (1, 3, 5, 7):
    inv_diff_at0 = sp.limit(up ** sp.Rational(1, k) / up, up, 0, "+")
    check(f"연습 11.2.2: k = {k}이면 u^(1/k)의 0에서의 차분몫 극한 = {inv_diff_at0}", (inv_diff_at0 == 1) == (k == 1))

# ---------------------------------------------------------------- 연습 11.2.4: ((0, ∞), log)
check("연습 11.2.4: log∘exp = id, exp와 log는 서로 역", sp.simplify(sp.log(sp.exp(u)) - u) == 0 and sp.simplify(sp.exp(sp.log(up)) - up) == 0)

# ---------------------------------------------------------------- 연습 11.2.6: ((0, ∞), x^3)은 두 구조에 모두 속한다
w = sp.symbols("w", positive=True)
check("연습 11.2.6: w^{1/3}는 w > 0에서 모든 계의 도함수가 존재 (5계까지 유한)",
      all(sp.diff(w ** sp.Rational(1, 3), w, j).subs(w, sp.Rational(1, 7)).is_finite for j in range(1, 6)))
check("연습 11.2.6: ψ∘θ^{-1} = id ((0, ∞)에서 θ = ψ)", sp.simplify((w ** sp.Rational(1, 3)) ** 3 - w) == 0)

# ---------------------------------------------------------------- 연습 11.2.7: ψ_a(x) = x|x|^{a-1}
a, b = sp.symbols("a b", positive=True)
# ψ_b∘ψ_a^{-1}(u) = u|u|^{b/a - 1} (u > 0에서 u^{b/a})
check("연습 11.2.7: ψ_b(ψ_a^{-1}(w)) = w^{b/a} (w > 0)", sp.simplify((w ** (1 / a)) ** b - w ** (b / a)) == 0)
for c in (sp.Rational(1, 3), sp.Rational(1, 2), sp.Integer(1), sp.Integer(2), sp.Integer(3), sp.Rational(5, 3)):
    fwd = sp.limit(w ** c / w, w, 0, "+")          # u^c의 0에서의 차분몫 (오른쪽)
    bwd = sp.limit(w ** (1 / c) / w, w, 0, "+")    # 역사상 u^{1/c}
    both_diff = fwd.is_finite and bwd.is_finite
    check(f"연습 11.2.7: c = {c}: u^c와 u^(1/c)가 모두 0에서 미분가능 ⇔ c = 1", both_diff == (c == 1))
ok = True
for aa, bb in ((1.0, 3.0), (0.5, 0.7), (2.0, 2.0)):
    uu = np.linspace(-2, 2, 401)
    inv_a = np.sign(uu) * np.abs(uu) ** (1 / aa)
    comp = np.sign(inv_a) * np.abs(inv_a) ** bb
    ok &= bool(np.allclose(comp, np.sign(uu) * np.abs(uu) ** (bb / aa)))
check("연습 11.2.7: ψ_b∘ψ_a^{-1}(u) = u|u|^{b/a-1} (음수 포함, 수치)", ok)

summary()
