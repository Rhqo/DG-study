"""6.2절 접평면: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/verify/v-6-2-tangent-plane.py``
"""

import math

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
u, v, t, s, a, b = sp.symbols("u v t s a b", real=True)


def Arg(x, y):
    return math.pi - 2 * math.atan(y / (1 - x))


# 비예 6.2.2: 원뿔의 꼭짓점 ------------------------------------------------------------------
w = sp.Matrix(sp.symbols("w1 w2 w3", real=True))
K2 = lambda P: P[0] ** 2 + P[1] ** 2 - P[2] ** 2
check("비예 6.2.2: w ∈ K₂이면 tw ∈ K₂ (K₂(tw) = t² K₂(w))", sp.expand(K2(t * w) - t ** 2 * K2(w)) == 0)
check("비예 6.2.2: (1,0,1) + (−1,0,1) = (0,0,2) ∉ K₂", K2([1, 0, 1]) == 0 and K2([-1, 0, 1]) == 0 and K2([0, 0, 2]) != 0)
# 표본 곡선: K₂ 위의 곡선 (t cos t, t sin t, t)의 0에서의 속도 (1, 0, 1) ∈ K₂
curve = sp.Matrix([t * sp.cos(t), t * sp.sin(t), t])
check("비예 6.2.2: 표본 곡선이 K₂ 위에 있고 속도도 K₂에 있다",
      sp.simplify(K2(curve)) == 0 and K2(list(curve.diff(t).subs(t, 0))) == 0)

# 예 6.2.7: 그래프 곡면 ----------------------------------------------------------------------
g = EX["graph"]
(f,) = g["functions"]
gu, gv = g["coords"]
Xg = g["expr"]
sym_equal("예 6.2.7 (6.2.2): x_u × x_v = (−f_u, −f_v, 1)", Xg.diff(gu).cross(Xg.diff(gv)),
          sp.Matrix([-sp.diff(f(gu, gv), gu), -sp.diff(f(gu, gv), gv), 1]))
sad = EX["saddle"]["expr"]
su, sv = EX["saddle"]["coords"]
n0 = sad.diff(su).cross(sad.diff(sv)).subs({su: 0, sv: 0})
check("예 6.2.7: 안장면 원점의 법선 (0,0,1) (접평면 = xy 평면)", n0 == sp.Matrix([0, 0, 1]))
n1 = sad.diff(su).cross(sad.diff(sv)).subs({su: 1, sv: 1})
check("예 6.2.7: 안장면 (1,1,0)에서 f_u = 2, f_v = −2, 법선 (−2, 2, 1)", n1 == sp.Matrix([-2, 2, 1]))
X_, Y_ = sp.symbols("X Y", real=True)
check("예 6.2.7: 아핀 접평면 z = 2x − 2y는 1차 근사와 같고 (1,1,0)을 지난다",
      sp.expand(0 + 2 * (X_ - 1) - 2 * (Y_ - 1) - (2 * X_ - 2 * Y_)) == 0 and 2 * 1 - 2 * 1 == 0)

# 예 6.2.8: 구면 --------------------------------------------------------------------------
e = EX["sphere"]
th, ph = e["coords"]
(r,) = e["params"]
X = e["expr"]
Xt, Xp = X.diff(th), X.diff(ph)
check("예 6.2.8: ⟨x, x_θ⟩ = ⟨x, x_φ⟩ = 0", sp.simplify(X.dot(Xt)) == 0 and sp.simplify(X.dot(Xp)) == 0)
hemi = sp.Matrix([u, v, sp.sqrt(r ** 2 - u ** 2 - v ** 2)])
nh = hemi.diff(u).cross(hemi.diff(v)).subs({u: 0, v: 0})
check("예 6.2.8: 북극에서 반구 그래프의 x_u × x_v = (0, 0, 1)", sp.simplify(nh - sp.Matrix([0, 0, 1])) == sp.zeros(3, 1))

# 예 6.2.9: 성분 계산 ---------------------------------------------------------------------
al = r / sp.sqrt(2) * sp.Matrix([sp.cos(t), 1, sp.sin(t)])
check("예 6.2.9: α는 구면 위", sp.simplify(al.dot(al) - r ** 2) == 0)
q = {th: sp.pi / 2, ph: sp.pi / 4}
check("예 6.2.9: α(0) = x(π/2, π/4)", sp.simplify(al.subs(t, 0) - X.subs(q)) == sp.zeros(3, 1))
dal = al.diff(t).subs(t, 0)
check("예 6.2.9: x_θ(q) = (0,0,−r), x_φ(q) = (−r/√2, r/√2, 0)",
      sp.simplify(Xt.subs(q) - sp.Matrix([0, 0, -r])) == sp.zeros(3, 1)
      and sp.simplify(Xp.subs(q) - sp.Matrix([-r / sp.sqrt(2), r / sp.sqrt(2), 0])) == sp.zeros(3, 1))
check("예 6.2.9: α'(0) = −(1/√2) x_θ + 0·x_φ",
      sp.simplify(dal - (-1 / sp.sqrt(2) * Xt.subs(q))) == sp.zeros(3, 1))
thet = sp.acos(sp.sin(t) / sp.sqrt(2))
check("예 6.2.9: θ(t) = arccos(sin t/√2), θ'(0) = −1/√2", sp.simplify(sp.diff(thet, t).subs(t, 0) + 1 / sp.sqrt(2)) == 0)
R = 1.3
def phi_of(tt):
    x, y, z = [float(c) for c in (al.subs({r: R, t: tt}))]
    rho = math.hypot(x, y)
    return Arg(x / rho, y / rho)
close("예 6.2.9: φ(0) = π/4", phi_of(0.0), math.pi / 4, 1e-12)
close("예 6.2.9: φ'(0) = 0 (중심 차분)", (phi_of(1e-5) - phi_of(-1e-5)) / 2e-5, 0.0, 1e-8)
check("예 6.2.9: tan(3π/8) = √2 + 1", sp.simplify(sp.tan(3 * sp.pi / 8) - (sp.sqrt(2) + 1)) == 0)

# 비예 6.2.6: 8자 기둥 -----------------------------------------------------------------------
x8 = lambda U, V: sp.Matrix([sp.sin(2 * U), sp.sin(U), V])
al8 = sp.Matrix([-sp.sin(2 * t), sp.sin(t), 0])
check("비예 6.2.6: α(t) = x(π − t, 0)", sp.simplify(al8 - x8(sp.pi - t, 0)) == sp.zeros(3, 1))
check("비예 6.2.6: α(t) = x(−π − t, 0)", sp.simplify(al8 - x8(-sp.pi - t, 0)) == sp.zeros(3, 1))
v8 = al8.diff(t).subs(t, 0)
check("비예 6.2.6: α'(0) = (−2, 1, 0)", v8 == sp.Matrix([-2, 1, 0]))
M = sp.Matrix.hstack(sp.Matrix([2, 1, 0]), sp.Matrix([0, 0, 1]), v8)
check("비예 6.2.6: (−2,1,0) ∉ span{(2,1,0), (0,0,1)} (세 벡터의 행렬식 ≠ 0)", M.det() != 0)
cab = sp.Matrix([-sp.sin(2 * a * t), sp.sin(a * t), b * t])
check("비예 6.2.6: (−sin 2at, sin at, bt)의 속도 (−2a, a, b)", cab.diff(t).subs(t, 0) == sp.Matrix([-2 * a, a, b]))

# 연습 6.2.1 ------------------------------------------------------------------------------
Xz = sp.Matrix([u, v, u * v])
q1 = {u: 1, v: 2}
check("연습 6.2.1: x_u = (1,0,2), x_v = (0,1,1)",
      Xz.diff(u).subs(q1) == sp.Matrix([1, 0, 2]) and Xz.diff(v).subs(q1) == sp.Matrix([0, 1, 1]))
check("연습 6.2.1: 법선 (−2, −1, 1)", Xz.diff(u).cross(Xz.diff(v)).subs(q1) == sp.Matrix([-2, -1, 1]))
check("연습 6.2.1: z = 2x + y − 2는 p를 지나고 법선에 수직",
      2 * 1 + 2 - 2 == 2 and sp.Matrix([1, 0, 2]).dot(sp.Matrix([-2, -1, 1])) == 0)

# 연습 6.2.2 ------------------------------------------------------------------------------
q2 = {th: sp.pi / 3, ph: sp.pi / 4, r: 1}
p2 = X.subs(q2)
w2 = sp.Matrix([1, 0, -sp.sqrt(6) / 2])
check("연습 6.2.2: ⟨p, w⟩ = 0", sp.simplify(p2.dot(w2)) == 0)
check("연습 6.2.2: w = √2 x_θ − (√6/3) x_φ",
      sp.simplify(w2 - (sp.sqrt(2) * Xt.subs(q2) - sp.sqrt(6) / 3 * Xp.subs(q2))) == sp.zeros(3, 1))

# 연습 6.2.4 ------------------------------------------------------------------------------
gc = sp.Matrix([r * sp.sin(t), 0, r * sp.cos(t)])
check("연습 6.2.4: 대원 속도 (r, 0, 0) = r·x_{1,u}", gc.diff(t).subs(t, 0) == sp.Matrix([r, 0, 0])
      and hemi.diff(u).subs({u: 0, v: 0}) == sp.Matrix([1, 0, 0]))

# 연습 6.2.5 ------------------------------------------------------------------------------
rho = sp.sqrt(u ** 2 + v ** 2)
Xc = sp.Matrix([u, v, rho])
check("연습 6.2.5: u x_u + v x_v = p (원뿔)", sp.simplify(u * Xc.diff(u) + v * Xc.diff(v) - Xc) == sp.zeros(3, 1))

# 연습 6.2.6 ------------------------------------------------------------------------------
ss = np.linspace(-math.pi, math.pi, 2001)
check("연습 6.2.6: E 위에서 |x| = 2|y|√(1 − y²)",
      np.allclose(np.abs(np.sin(2 * ss)), 2 * np.abs(np.sin(ss)) * np.sqrt(1 - np.sin(ss) ** 2)))

summary()
