"""6.3절 매개변수 변환과 곡면 위의 매끄러운 함수: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/verify/v-6-3-smooth-functions.py``
"""

import math
import random

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
rng = random.Random(1)
e = EX["sphere"]
th, ph = e["coords"]
(r,) = e["params"]
X = e["expr"]
ub, vb = sp.symbols("ubar vbar", real=True)
Y = sp.Matrix([ub, vb, sp.sqrt(r ** 2 - ub ** 2 - vb ** 2)])


def Arg(a, b):
    return math.pi - 2 * math.atan(b / (1 - a))


# 비예 6.3.2: 8자 기둥의 두 사상 ---------------------------------------------------------------
s, v, u = sp.symbols("s v u", real=True)
x8 = lambda U, V: sp.Matrix([sp.sin(2 * U), sp.sin(U), V])
y8 = sp.Matrix([-sp.sin(2 * s), sp.sin(s), v])
check("비예 6.3.2: y(s, v) = x(π − s, v)", sp.simplify(y8 - x8(sp.pi - s, v)) == sp.zeros(3, 1))
check("비예 6.3.2: y(s, v) = x(−π − s, v)", sp.simplify(y8 - x8(-sp.pi - s, v)) == sp.zeros(3, 1))
Dy8 = y8.jacobian([s, v])
vals = [np.linalg.matrix_rank(np.array(Dy8.subs(s, a), dtype=float)) for a in np.linspace(-1.5, 1.5, 31)]
check("비예 6.3.2: Dy는 모든 점에서 계수 2", all(k == 2 for k in vals))
hplus, hminus = (math.pi - 1e-6), (-math.pi + 1e-6)
check("비예 6.3.2: x⁻¹∘y의 s → 0± 극한은 ±π이고 s = 0의 값 0과 다르다", abs(hplus) > 3 and abs(hminus) > 3)

# 예 6.3.3: 기준 매개화와 반구 그래프 -------------------------------------------------------
k = sp.Matrix([r * sp.sin(th) * sp.cos(ph), r * sp.sin(th) * sp.sin(ph)])
check("예 6.3.3: y⁻¹∘x = P_xy∘x = k(θ, φ)", sp.simplify(X[:2, :] - k) == sp.zeros(2, 1))
sym_equal("예 6.3.3: y(k(θ, φ)) = x(θ, φ) (θ < π/2)", Y.subs({ub: k[0], vb: k[1]}), X,
          {th: (0.05, 1.5), ph: (0.05, 6.2), r: (0.5, 2)})
Dk = k.jacobian([th, ph])
sym_equal("예 6.3.3 (6.3.3): det Dk = r² sinθ cosθ", Dk.det(), r ** 2 * sp.sin(th) * sp.cos(th), e["domain"])
rb = sp.sqrt(ub ** 2 + vb ** 2)
check("예 6.3.3: arccos√(1 − s²) = arcsin s (s ∈ (0,1))",
      all(abs(math.acos(math.sqrt(1 - t ** 2)) - math.asin(t)) < 1e-12 for t in np.linspace(0.01, 0.99, 50)))
ok = True
for _ in range(200):
    R = rng.uniform(0.5, 2)
    T, P = rng.uniform(0.02, math.pi / 2 - 0.02), rng.uniform(0.02, 2 * math.pi - 0.02)
    a, b = R * math.sin(T) * math.cos(P), R * math.sin(T) * math.sin(P)
    rho = math.hypot(a, b)
    ok &= abs(math.asin(rho / R) - T) < 1e-9 and abs(Arg(a / rho, b / rho) - P) < 1e-9
check("예 6.3.3 (6.3.4): h(k(θ, φ)) = (θ, φ)", ok)
q = {th: sp.pi / 4, ph: sp.pi / 2}
Dkq = sp.simplify(Dk.subs(q))
check("예 6.3.3: (π/4, π/2)에서 Dk = [[0, −r/√2], [r/√2, 0]]", sp.simplify(Dkq - sp.Matrix([[0, -r / sp.sqrt(2)], [r / sp.sqrt(2), 0]])) == sp.zeros(2, 2))
check("예 6.3.3: k(π/4, π/2) = (0, r/√2)", sp.simplify(k.subs(q) - sp.Matrix([0, r / sp.sqrt(2)])) == sp.zeros(2, 1))
# h의 첫째 성분 arcsin(ρ̄/r)을 직접 미분
th_of = sp.asin(rb / r)
pt = {ub: 0, vb: r / sp.sqrt(2)}
dth = [sp.simplify(sp.diff(th_of, w).subs(pt)) for w in (ub, vb)]
check("예 6.3.3: ∂θ/∂ū = 0, ∂θ/∂v̄ = √2/r", dth[0] == 0 and sp.simplify(dth[1] - sp.sqrt(2) / r) == 0)
# 둘째 성분(각)의 편미분을 수치로: 각 = atan2(v̄, ū)와 같다
R = 1.7
def hnum(a, b):
    rho = math.hypot(a, b)
    return np.array([math.asin(rho / R), Arg(a / rho, b / rho)])
p0 = np.array([0.0, R / math.sqrt(2)])
eps = 1e-6
J = np.column_stack([(hnum(*(p0 + eps * d)) - hnum(*(p0 - eps * d))) / (2 * eps) for d in np.eye(2)])
close("예 6.3.3: Dh(0, r/√2) = (Dk)⁻¹ = [[0, √2/r], [−√2/r, 0]] (수치)", J,
      np.array([[0, math.sqrt(2) / R], [-math.sqrt(2) / R, 0]]), 1e-6)

# 명제 6.3.4 / 연습 6.3.5: 성분 변환 ------------------------------------------------------
Xt = X.diff(th).subs(q)
Yv = Y.diff(vb).subs(pt)
Yu = Y.diff(ub).subs(pt)
check("연습 6.3.5: y_ū = (1, 0, 0), y_v̄ = (0, 1, −1)", sp.simplify(Yu - sp.Matrix([1, 0, 0])) == sp.zeros(3, 1)
      and sp.simplify(Yv - sp.Matrix([0, 1, -1])) == sp.zeros(3, 1))
check("연습 6.3.5: x_θ = (r/√2) y_v̄", sp.simplify(Xt - r / sp.sqrt(2) * Yv) == sp.zeros(3, 1))
# (6.3.5)의 일반 확인: D(x∘k)는 Dx(k) Dk ... 여기서는 y = x∘h ⇔ x = y∘k이므로 Dx = Dy(k) Dk
Dx = X.jacobian([th, ph])
DyK = Y.jacobian([ub, vb]).subs({ub: k[0], vb: k[1]})
sym_equal("명제 6.3.4: Dx = Dy(k)·Dk (연쇄법칙, θ < π/2)", Dx, DyK * Dk, {th: (0.05, 1.5), ph: (0.05, 6.2), r: (0.5, 2)})

# 예 6.3.7(c): z의 두 좌표 표현이 맞는다 ------------------------------------------------------
check("예 6.3.7(c): r cos(arcsin(ρ̄/r)) = √(r² − ρ̄²)", sp.simplify(r * sp.cos(sp.asin(rb / r)) - sp.sqrt(r ** 2 - rb ** 2)) == 0)

# 비예 6.3.8 --------------------------------------------------------------------------------
hh = sp.symbols("h", positive=True)
fz = r * sp.Abs(sp.cos(th))
right = sp.limit((fz.subs(th, sp.pi / 2 + hh) - fz.subs(th, sp.pi / 2)) / hh, hh, 0)
left = sp.limit((fz.subs(th, sp.pi / 2 - hh) - fz.subs(th, sp.pi / 2)) / (-hh), hh, 0)
check("비예 6.3.8(a): r|cos θ|의 θ = π/2에서 우미분 r, 좌미분 −r", sp.simplify(right - r) == 0 and sp.simplify(left + r) == 0)
check("비예 6.3.8(b): u^{1/3}의 차분몫은 발산", sp.limit(hh ** sp.Rational(1, 3) / hh, hh, 0, "+") == sp.oo)

# 예 6.3.12, 연습 6.3.1 ----------------------------------------------------------------------
al = sp.symbols("alpha", real=True)
Rz = sp.Matrix([[sp.cos(al), -sp.sin(al), 0], [sp.sin(al), sp.cos(al), 0], [0, 0, 1]])
check("예 6.3.12: R_{z,α}는 직교행렬이고 역은 R_{z,−α}", sp.simplify(Rz.T * Rz) == sp.eye(3)
      and sp.simplify(Rz * Rz.subs(al, -al)) == sp.eye(3))
check("연습 6.3.1: −x(θ, φ) = x(π − θ, φ + π)", sp.simplify(-X - X.subs({th: sp.pi - th, ph: ph + sp.pi}, simultaneous=True)) == sp.zeros(3, 1))

# 연습 6.3.4: 그래프 곡면과 평면 조각 ------------------------------------------------------------
xx, yy = sp.symbols("x y", real=True)
fg = sp.Function("f")(xx, yy)
psi = sp.Matrix([xx, yy, fg])
check("연습 6.3.4: ψ(x, y, 0) = (x, y, f)는 Γ_f 위에 있고 φ∘ψ = id", psi[2] == fg and psi[:2, :] == sp.Matrix([xx, yy]))
psi_par = psi.subs(fg, xx ** 2 + yy ** 2)
check("연습 6.3.4: 포물면의 경우 ψ의 상은 z = x² + y²", sp.simplify(psi_par[2] - (psi_par[0] ** 2 + psi_par[1] ** 2)) == 0)

# 연습 6.3.2: 위쪽 반구 → 열린 원판 -------------------------------------------------------------
rp = sp.symbols("r", positive=True)
eta = sp.Matrix([xx, yy, sp.sqrt(rp ** 2 - xx ** 2 - yy ** 2)])
check("연습 6.3.2: η(x, y, 0)은 S²(r) 위에 있다", sp.simplify(eta.dot(eta) - rp ** 2) == 0)
check("연습 6.3.2: ψ∘η = id (처음 두 성분)", eta[:2, :] == sp.Matrix([xx, yy]))

# 연습 6.3.6(b): 열린 원판 ↔ 평면 ------------------------------------------------------------------
rho2 = xx ** 2 + yy ** 2
Fd = sp.Matrix([xx, yy]) / sp.sqrt(1 - rho2)   # D_1 → 평면
Gd = sp.Matrix([xx, yy]) / sp.sqrt(1 + rho2)   # 평면 → D_1
GF = Gd.subs({xx: Fd[0], yy: Fd[1]}, simultaneous=True)
FG = Fd.subs({xx: Gd[0], yy: Gd[1]}, simultaneous=True)
pts_d = [(0.3, -0.4), (0.1, 0.8), (-0.6, 0.2)]
check("연습 6.3.6(b): (x,y,0)/√(1+x²+y²)의 값은 열린 원판 안", all(float((Gd.dot(Gd)).subs({xx: a, yy: b})) < 1 for a, b in [(3, 4), (-10, 2), (0.1, 0.1)]))
check("연습 6.3.6(b): 두 사상은 서로의 역사상 (수치)",
      all(np.allclose([float(e) for e in GF.subs({xx: a, yy: b})], [a, b]) for a, b in pts_d)
      and all(np.allclose([float(e) for e in FG.subs({xx: a, yy: b})], [a, b]) for a, b in [(3, 4), (-10, 2), (0.1, 0.1)]))

summary()
