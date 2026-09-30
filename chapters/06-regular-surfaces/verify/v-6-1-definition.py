"""6.1절 정칙곡면의 정의: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/verify/v-6-1-definition.py``
"""

import math
import random

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
u, v, t, s = sp.symbols("u v t s", real=True)
rng = random.Random(0)

# 예 6.1.2: 평면의 역사상 (AᵀA)⁻¹Aᵀ(p − p0) ------------------------------------------------
p0 = sp.Matrix([1, -2, 3])
w1 = sp.Matrix([1, 2, 0])
w2 = sp.Matrix([0, 1, -1])
A = w1.row_join(w2)
xP = p0 + A * sp.Matrix([u, v])
left = (A.T * A).inv() * A.T
check("예 6.1.2: (AᵀA)⁻¹Aᵀ(x(q) − p0) = q", sp.simplify(left * (xP - p0) - sp.Matrix([u, v])) == sp.zeros(2, 1))
check("예 6.1.2: Dx = A는 계수 2", xP.jacobian([u, v]).rank() == 2)

# 비예 6.1.3 --------------------------------------------------------------------------------
h = sp.symbols("h", real=True, nonzero=True)
check("비예 6.1.3(a): |h|/h의 좌·우극한이 다르다(원뿔 사상은 원점에서 미분불가)",
      sp.limit(sp.Abs(h) / h, h, 0, "+") == 1 and sp.limit(sp.Abs(h) / h, h, 0, "-") == -1)
xcusp = sp.Matrix([u ** 2, u ** 3, v])
Dcusp = xcusp.jacobian([u, v])
check("비예 6.1.3(b): 뾰족 기둥의 Dx는 u = 0에서 계수 1", Dcusp.subs(u, 0).rank() == 1)
check("비예 6.1.3(b): u ≠ 0에서 계수 2", Dcusp.subs(u, sp.Rational(1, 3)).rank() == 2)
ok = True
for _ in range(20):
    uu = rng.uniform(-2, 2)
    y = uu ** 3
    ok &= abs(math.copysign(abs(y) ** (1 / 3), y) - uu) < 1e-12
check("비예 6.1.3(b): 역사상 (x, y, z) ↦ (y^{1/3}, z)가 u를 되찾는다", ok)
x8 = sp.Matrix([sp.sin(2 * u), sp.sin(u), v])
xu8 = x8.diff(u)
check("비예 6.1.3(c): cos u = 0이면 cos 2u = −1 (u = ±π/2)",
      all(sp.cos(2 * a) == -1 for a in (sp.pi / 2, -sp.pi / 2)))
vals = [abs(2 * math.cos(2 * a)) + abs(math.cos(a)) for a in np.linspace(-math.pi, math.pi, 2001)]
check("비예 6.1.3(c): (2cos 2u, cos u) ≠ 0 (격자 최솟값 > 0)", min(vals) > 0.5)
close("비예 6.1.3(c): u → π일 때 x(u, 0) → 원점", float(sp.Matrix(x8.subs({u: sp.pi - sp.Rational(1, 10**6), v: 0})).norm()), 0.0, 1e-5)

# 명제 6.1.4: 그래프 곡면 ------------------------------------------------------------------
g = EX["graph"]
(f,) = g["functions"]
gu, gv = g["coords"]
Xg = g["expr"]
Dg = Xg.jacobian([gu, gv])
check("명제 6.1.4: 그래프 곡면 Dx의 위쪽 2×2는 단위행렬", Dg[:2, :] == sp.eye(2))
sym_equal("명제 6.1.4: x_u × x_v = (−f_u, −f_v, 1)", Xg.diff(gu).cross(Xg.diff(gv)),
          sp.Matrix([-sp.diff(f(gu, gv), gu), -sp.diff(f(gu, gv), gv), 1]))
Q = sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
check("명제 6.1.4 증명: 좌표 순서 바꾸기 Q(a,b,c) = (a,c,b)는 가역", Q.det() != 0)

# 예 6.1.5: 반구 ---------------------------------------------------------------------------
r = sp.symbols("r", positive=True)
hemi = sp.Matrix([u, v, sp.sqrt(r ** 2 - u ** 2 - v ** 2)])
check("예 6.1.5: 위쪽 반구 그래프는 |x| = r", sp.simplify(hemi.dot(hemi) - r ** 2) == 0)

# 보조정리 6.1.6: 각도 함수 ---------------------------------------------------------------
def Arg(a, b):
    return math.pi - 2 * math.atan(b / (1 - a))


ok = all(abs(Arg(math.cos(tt), math.sin(tt)) - tt) < 1e-12 for tt in np.linspace(1e-3, 2 * math.pi - 1e-3, 997))
check("보조정리 6.1.6: Arg(cos t, sin t) = t (t ∈ (0, 2π))", ok)
ok = True
for _ in range(200):
    ang = rng.uniform(0, 2 * math.pi)
    a, b = math.cos(ang), math.sin(ang)
    if abs(a - 1) < 1e-9:
        continue
    T = Arg(a, b)
    ok &= abs(math.cos(T) - a) < 1e-12 and abs(math.sin(T) - b) < 1e-12 and 0 < T < 2 * math.pi
check("보조정리 6.1.6: e(Arg(a, b)) = (a, b), 값은 (0, 2π)", ok)
m = sp.symbols("m", real=True)
am, bm = (m ** 2 - 1) / (m ** 2 + 1), 2 * m / (m ** 2 + 1)
check("보조정리 6.1.6 증명: m = b/(1 − a)에서 (a, b)를 되찾는 식",
      sp.simplify(am ** 2 + bm ** 2 - 1) == 0 and sp.simplify(bm / (1 - am) - m) == 0)
sym_equal("보조정리 6.1.6 증명: sin t/(1 − cos t) = cot(t/2)", sp.sin(t) / (1 - sp.cos(t)), sp.cot(t / 2), {t: (0.1, 6.1)})
t0 = sp.symbols("t0", real=True)
R0 = sp.Matrix([[sp.cos(t0), -sp.sin(t0)], [sp.sin(t0), sp.cos(t0)]])
check("보조정리 6.1.6 증명: e(t0 + s) = R_{t0} e(s)",
      sp.simplify(R0 * sp.Matrix([sp.cos(s), sp.sin(s)]) - sp.Matrix([sp.cos(t0 + s), sp.sin(t0 + s)])) == sp.zeros(2, 1))

# 예 6.1.7: 구면의 기준 매개화 -----------------------------------------------------------
e = EX["sphere"]
th, ph = e["coords"]
(rr,) = e["params"]
X = e["expr"]
Xt, Xp = X.diff(th), X.diff(ph)
cr = Xt.cross(Xp)
sym_equal("예 6.1.7 (6.1.5): x_θ × x_φ = r sinθ · x", cr, rr * sp.sin(th) * X, e["domain"])
sym_equal("예 6.1.7 (6.1.5): |x_θ × x_φ|² = r⁴ sin²θ", cr.dot(cr), rr ** 4 * sp.sin(th) ** 2, e["domain"])
J = lambda i, j: sp.simplify(X[i].diff(th) * X[j].diff(ph) - X[i].diff(ph) * X[j].diff(th))
sym_equal("예 6.1.7: ∂(x,y)/∂(θ,φ) = r² sinθ cosθ = 셋째 성분", J(0, 1), cr[2], e["domain"])
sym_equal("예 6.1.7: ∂(y,z)/∂(θ,φ) = 첫째 성분", J(1, 2), cr[0], e["domain"])
sym_equal("예 6.1.7: −∂(x,z)/∂(θ,φ) = 둘째 성분", -J(0, 2), cr[1], e["domain"])
fX = sp.lambdify((th, ph, rr), list(X), "math")
ok = True
for _ in range(300):
    T, P, R = rng.uniform(0.01, math.pi - 0.01), rng.uniform(0.01, 2 * math.pi - 0.01), rng.uniform(0.5, 2)
    x, y, z = fX(T, P, R)
    rho = math.hypot(x, y)
    ok &= abs(math.acos(z / R) - T) < 1e-9 and abs(Arg(x / rho, y / rho) - P) < 1e-9
check("예 6.1.7 (6.1.6): 역사상 식이 (θ, φ)를 되찾는다", ok)
check("예 6.1.7: y = 0이면 φ = π이고 x = −r sinθ < 0 (상은 C와 만나지 않음)",
      sp.simplify(X[1].subs(ph, sp.pi)) == 0 and sp.simplify(X[0].subs(ph, sp.pi) + rr * sp.sin(th)) == 0)
Qs = sp.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
check("예 6.1.7: Q(x,y,z) = (−x, z, y)는 직교, det Q = 1, Q∘Q = id",
      Qs.T * Qs == sp.eye(3) and Qs.det() == 1 and Qs * Qs == sp.eye(3))
# C = {x(θ, 0)}, Q(C) = {Q x(θ, 0)}: Q(C)는 z = 0, x ≤ 0
QC = Qs * X.subs(ph, 0)
check("예 6.1.7: Q(C)는 z = 0이고 x = −r sinθ ≤ 0", QC[2] == 0 and sp.simplify(QC[0] + rr * sp.sin(th)) == 0)
check("예 6.1.7: C ∩ {z = 0} = {(r, 0, 0)}이고 이 점은 Q(C)에 없음 (x > 0)",
      sp.simplify(X.subs({ph: 0, th: sp.pi / 2}) - sp.Matrix([rr, 0, 0])) == sp.zeros(3, 1))

# 명제 6.1.8: 구면에서 P_xy ∘ x의 국소 역과 Π ------------------------------------------------
TH0, PH0 = math.pi / 3, math.pi / 4
ok = True
for _ in range(100):
    T, P = TH0 + rng.uniform(-0.3, 0.3), PH0 + rng.uniform(-0.45, 0.45)
    x, y, z = fX(T, P, 1.0)
    # Π(x, y, z) = (P_xy ∘ x)⁻¹(x, y): θ = arcsin(ρ), φ = Arg (θ < π/2 구역)
    rho = math.hypot(x, y)
    ok &= abs(math.asin(rho) - T) < 1e-9 and abs(Arg(x / rho, y / rho) - P) < 1e-9
check("명제 6.1.8 (6.1.7): 그림 6.1.4의 U₀에서 Π(x(θ,φ)) = (θ,φ)", ok)
check("그림 6.1.4: U₀에서 sinθ cosθ > 0 (P_xy ∘ x의 야코비 행렬식 ≠ 0)",
      min(math.sin(a) * math.cos(a) for a in np.linspace(TH0 - 0.3, TH0 + 0.3, 50)) > 0)

# 비예 6.1.9 --------------------------------------------------------------------------------
beta = lambda a: sp.Matrix([sp.sin(2 * a), sp.sin(a)])
check("비예 6.1.9(d): β(π − s) = (−sin 2s, sin s)", sp.simplify(beta(sp.pi - s) - sp.Matrix([-sp.sin(2 * s), sp.sin(s)])) == sp.zeros(2, 1))
check("비예 6.1.9(d): β(s − π) = (sin 2s, −sin s)", sp.simplify(beta(s - sp.pi) - sp.Matrix([sp.sin(2 * s), -sp.sin(s)])) == sp.zeros(2, 1))
tp = sp.symbols("t", positive=True)
check("비예 6.1.9(c): (t², ±t³, 0)은 x = |y|^{2/3} 위에 있다", sp.simplify((tp ** 3) ** sp.Rational(2, 3) - tp ** 2) == 0)
check("비예 6.1.9(c): |y|^{2/3}/y → ∞ (y → 0+)", sp.limit(tp ** sp.Rational(2, 3) / tp, tp, 0, "+") == sp.oo)

# 연습 6.1.2 --------------------------------------------------------------------------------
Xq = Qs * X
crq = Xq.diff(th).cross(Xq.diff(ph))
sym_equal("연습 6.1.2: x̃_θ × x̃_φ = r sinθ · x̃", crq, rr * sp.sin(th) * Xq, e["domain"])
check("연습 6.1.2: x̃(π/2, π/2) = 북극", sp.simplify(Xq.subs({th: sp.pi / 2, ph: sp.pi / 2}) - sp.Matrix([0, 0, rr])) == sp.zeros(3, 1))

# 연습 6.1.5: 입체사영의 역 -----------------------------------------------------------------
w = u ** 2 + v ** 2 + 1
sig = sp.Matrix([2 * u, 2 * v, u ** 2 + v ** 2 - 1]) / w
su, sv = sig.diff(u), sig.diff(v)
check("연습 6.1.5: σ⁻¹의 상은 단위구면", sp.simplify(sig.dot(sig) - 1) == 0)
check("연습 6.1.5: σ⁻¹_u 식", sp.simplify(su - 2 / w ** 2 * sp.Matrix([1 - u ** 2 + v ** 2, -2 * u * v, 2 * u])) == sp.zeros(3, 1))
check("연습 6.1.5: σ⁻¹_v 식", sp.simplify(sv - 2 / w ** 2 * sp.Matrix([-2 * u * v, 1 + u ** 2 - v ** 2, 2 * v])) == sp.zeros(3, 1))
check("연습 6.1.5: ⟨σ⁻¹_u, σ⁻¹_v⟩ = 0", sp.simplify(su.dot(sv)) == 0)
check("연습 6.1.5: |σ⁻¹_u|² = |σ⁻¹_v|² = 4/w²", sp.simplify(su.dot(su) - 4 / w ** 2) == 0 and sp.simplify(sv.dot(sv) - 4 / w ** 2) == 0)
check("연습 6.1.5: (1 − u² + v²)² + 4u²v² + 4u² = (1 + u² + v²)²",
      sp.expand((1 - u ** 2 + v ** 2) ** 2 + 4 * u ** 2 * v ** 2 + 4 * u ** 2 - w ** 2) == 0)
c5 = su.cross(sv)
check("연습 6.1.5: |σ⁻¹_u × σ⁻¹_v| = 4/w²", sp.simplify(c5.dot(c5) - 16 / w ** 4) == 0)

# 연습 6.1.6 --------------------------------------------------------------------------------
xp = sp.symbols("x", positive=True)
f6 = xp ** sp.Rational(4, 3)
check("연습 6.1.6: f(x, 0) = x^{4/3}의 이계도함수 = (4/9) x^{-2/3}", sp.simplify(sp.diff(f6, xp, 2) - sp.Rational(4, 9) * xp ** sp.Rational(-2, 3)) == 0)
check("연습 6.1.6: 차분몫 x^{4/3}/x = x^{1/3} → 0", sp.limit(f6 / xp, xp, 0, "+") == 0)
X2, Y2 = sp.symbols("X Y", real=True)
F6 = (X2 ** 2 + Y2 ** 2) ** sp.Rational(2, 3)
sym_equal("연습 6.1.6: f_x = (4/3) x (x² + y²)^{-1/3}", sp.diff(F6, X2), sp.Rational(4, 3) * X2 * (X2 ** 2 + Y2 ** 2) ** sp.Rational(-1, 3))

summary()
