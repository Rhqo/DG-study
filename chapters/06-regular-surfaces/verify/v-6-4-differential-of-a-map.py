"""6.4절 사상의 미분: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/verify/v-6-4-differential-of-a-map.py``
"""

import math
import random

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
rng = random.Random(2)
e = EX["sphere"]
th, ph = e["coords"]
(r,) = e["params"]
X = e["expr"]
Xt, Xp = X.diff(th), X.diff(ph)
t = sp.symbols("t", real=True)

# 명제 6.4.2: 곡선으로 계산한 값 = 좌표 공식 (구면 → 타원면 A = diag(a,b,c)) -------------------
a, b, c = sp.symbols("a b c", positive=True)
A = sp.diag(a, b, c)
th0, ph0, da, db = sp.Rational(1, 1), sp.Rational(3, 4), sp.Rational(1, 2), sp.Rational(6, 5)
alpha = X.subs({th: th0 + da * t, ph: ph0 + db * t})
lhs = (A * alpha).diff(t).subs(t, 0)
rhs = (A * Xt).subs({th: th0, ph: ph0}) * da + (A * Xp).subs({th: th0, ph: ph0}) * db
check("명제 6.4.2 / 예 6.4.6: (φ∘α)'(0) = A α'(0) = ā(Ax)_θ + b̄(Ax)_φ (좌표 표현은 항등)", sp.simplify(lhs - rhs) == sp.zeros(3, 1))
p = X.subs({th: th0, ph: ph0})
gradE = sp.Matrix([(A * p)[0] / a ** 2, (A * p)[1] / b ** 2, (A * p)[2] / c ** 2])
check("예 6.4.6: A(T_pS²) ⊆ T_{Ap}E (A x_θ, A x_φ ⊥ 타원면의 기울기)",
      sp.simplify(gradE.dot(A * Xt.subs({th: th0, ph: ph0}))) == 0 and sp.simplify(gradE.dot(A * Xp.subs({th: th0, ph: ph0}))) == 0)
check("예 6.4.6: A∘x는 타원면 위에 있다", sp.simplify(((A * X)[0] / a) ** 2 + ((A * X)[1] / b) ** 2 + ((A * X)[2] / c) ** 2 - r ** 2) == 0)

# 예 6.4.3: 좌표함수의 미분은 쌍대기저 -----------------------------------------------------
u, v = sp.symbols("u v", real=True)
check("예 6.4.3: D(u∘x) = (1, 0), D(v∘x) = (0, 1)", sp.Matrix([[u, v]]).jacobian([u, v]) == sp.eye(2))

# 예 6.4.5 ------------------------------------------------------------------------------
vv = sp.Matrix(sp.symbols("v1 v2 v3", real=True))
P = sp.Matrix(sp.symbols("x y z", real=True))
check("예 6.4.5(a): grad⟨p, v⟩ = v", sp.Matrix([P.dot(vv)]).jacobian(P).T == vv)
p0 = sp.Matrix(sp.symbols("q1 q2 q3", real=True))
check("예 6.4.5(b): grad|p − p0|² = 2(p − p0)", sp.simplify(sp.Matrix([(P - p0).dot(P - p0)]).jacobian(P).T - 2 * (P - p0)) == sp.zeros(3, 1))
# 높이함수 h = z의 좌표 표현 r cosθ: 미분 (−r sinθ, 0)
check("연습 6.4.2: d(z)(x_θ) = −r sinθ, d(z)(x_φ) = 0",
      sp.simplify((r * sp.cos(th)).diff(th) + r * sp.sin(th)) == 0 and sp.diff(r * sp.cos(th), ph) == 0
      and sp.simplify(Xt[2] + r * sp.sin(th)) == 0)

# 예 6.4.6: 회전은 북극의 접평면에서 회전 --------------------------------------------------
al = sp.symbols("alpha", real=True)
Rz = sp.Matrix([[sp.cos(al), -sp.sin(al), 0], [sp.sin(al), sp.cos(al), 0], [0, 0, 1]])
check("예 6.4.6: R_{z,α}는 북극을 고정하고 xy 평면을 xy 평면으로", Rz * sp.Matrix([0, 0, r]) == sp.Matrix([0, 0, r])
      and (Rz * sp.Matrix([1, 0, 0]))[2] == 0 and (Rz * sp.Matrix([0, 1, 0]))[2] == 0)

# 예 6.4.7: 반사의 좌표 계산 ---------------------------------------------------------------
S = sp.diag(1, 1, -1)
hat = {th: sp.pi - th}
check("예 6.4.7: σ(x(θ,φ)) = x(π − θ, φ)", sp.simplify(S * X - X.subs(hat)) == sp.zeros(3, 1))
check("예 6.4.7 (6.4.4): Dσ x_θ(q) = −x_θ(q̂)", sp.simplify(S * Xt + Xt.subs(hat)) == sp.zeros(3, 1))
check("예 6.4.7 (6.4.4): Dσ x_φ(q) = x_φ(q̂)", sp.simplify(S * Xp - Xp.subs(hat)) == sp.zeros(3, 1))

# 예 6.4.12: 평면의 지수 사상 ---------------------------------------------------------------
F = sp.Matrix([sp.exp(u) * sp.cos(v), sp.exp(u) * sp.sin(v)])
check("예 6.4.12: det Dφ̂ = e^{2u}", sp.simplify(F.jacobian([u, v]).det() - sp.exp(2 * u)) == 0)
check("예 6.4.12: φ̂(u, v + 2π) = φ̂(u, v)", sp.simplify(F.subs(v, v + 2 * sp.pi) - F) == sp.zeros(2, 1))

# 연습 6.4.4 ------------------------------------------------------------------------------
ok = True
for _ in range(50):
    R = rng.uniform(0.5, 2)
    q0 = np.array([rng.uniform(-3, 3) for _ in range(3)])
    for sgn in (1, -1):
        pp = sgn * R * q0 / np.linalg.norm(q0)
        # df_p = 0 ⟺ p − p0 ∥ p
        ok &= np.linalg.norm(np.cross(pp - q0, pp)) < 1e-9
check("연습 6.4.4: p = ±r p0/|p0|에서 p − p0 ∥ p", ok)

# 연습 6.4.5: 대척사상의 좌표 표현 -----------------------------------------------------------
hatA = {th: sp.pi - th, ph: ph + sp.pi}
check("연습 6.4.5: x_θ(q̂) = x_θ(q)", sp.simplify(Xt.subs(hatA, simultaneous=True) - Xt) == sp.zeros(3, 1))
check("연습 6.4.5: x_φ(q̂) = −x_φ(q)", sp.simplify(Xp.subs(hatA, simultaneous=True) + Xp) == sp.zeros(3, 1))

# 연습 6.4.6: x + 2y + 2z의 최댓값 ----------------------------------------------------------
vmax = np.array([1.0, 2.0, 2.0])
pts = np.random.default_rng(0).normal(size=(200000, 3))
pts /= np.linalg.norm(pts, axis=1)[:, None]
close("연습 6.4.6: 최댓값 3 (p = (1,2,2)/3)", float(vmax @ (vmax / 3)), 3.0, 1e-12)
check("연습 6.4.6: 무작위 표본의 최댓값은 3 이하", float((pts @ vmax).max()) <= 3.0 + 1e-12)

summary()
