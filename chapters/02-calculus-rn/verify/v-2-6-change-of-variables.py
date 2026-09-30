"""2.6절 적분과 변수변환 공식: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/verify/v-2-6-change-of-variables.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

r, th, x, y, t = sp.symbols("r theta x y t", real=True)
a, eps = sp.symbols("a epsilon", positive=True)
rng = np.random.default_rng(0)

# 그림 2.6.1 / 예 2.6.10: 부채꼴 조각의 넓이 = 중심의 |det DP| vol Q
r1, r2, t1, t2 = sp.Integer(1), sp.Rational(3, 2), sp.pi / 8, 3 * sp.pi / 8
cell = sp.integrate(r, (r, r1, r2), (th, t1, t2))
check("그림 2.6.1: area P(Q) = 5π/32", sp.simplify(cell - 5 * sp.pi / 32) == 0)
check("예 2.6.10: area = (r2²-r1²)Δθ/2 = r̄ΔrΔθ", sp.simplify(cell - (r1 + r2) / 2 * (r2 - r1) * (t2 - t1)) == 0)

# 예 2.6.3(b): 원을 덮는 정사각형들의 넓이 합 → 0 (N조각)
M, d = 1.0, 2 * np.pi              # 단위원: |γ'| = 1, 구간 길이 2π
tot = [(2 * M * d / N) ** 2 * N for N in (10, 100, 1000)]
check("예 2.6.3(b): 원의 덮개 넓이 (2Md/N)^2 N → 0", tot[0] > tot[1] > tot[2] and tot[2] < 0.2)

# 예 2.6.10: 원판의 넓이
area_eps = sp.integrate(r, (r, eps, a), (th, -sp.pi + eps, sp.pi - eps))
sym_equal("예 2.6.10: vol P(D_ε) = (2π-2ε)(a²-ε²)/2", area_eps, (2 * sp.pi - 2 * eps) * (a ** 2 - eps ** 2) / 2)
check("예 2.6.10: ε → 0 에서 πa²", sp.limit(area_eps, eps, 0) == sp.pi * a ** 2)
check("예 2.6.10: 오차 상한 4ε²+2a² sin ε → 0", sp.limit(4 * eps ** 2 + 2 * a ** 2 * sp.sin(eps), eps, 0) == 0)
# 몬테카를로로 원판 넓이 교차검증 (a = 1)
pts = rng.uniform(-1, 1, (400000, 2))
close("예 2.6.10: 단위원판 넓이 (몬테카를로)", 4 * np.mean((pts ** 2).sum(1) <= 1), np.pi, tol=0.01)
# B ∖ P(D_ε) ⊆ 정사각형 ∪ 직사각형 (수치 표본)
aa, ee = 1.0, 0.1
ok = True
for q in rng.uniform(-1, 1, (20000, 2)):
    rr, tt = np.hypot(*q), np.arctan2(q[1], q[0])
    if rr > aa:
        continue
    inPD = (ee <= rr) and (abs(tt) <= np.pi - ee)
    inS = abs(q[0]) <= ee and abs(q[1]) <= ee
    inR = (-aa <= q[0] <= 0) and abs(q[1]) <= aa * np.sin(ee)
    ok &= inPD or inS or inR
check("예 2.6.10: B ⊆ P(D_ε) ∪ [-ε,ε]² ∪ [-a,0]×[-a sin ε, a sin ε] (표본)", ok)

# 명제 2.6.8: |det A| 배율 (몬테카를로, 단위정사각형의 상)
A = np.array([[2.0, 1.0], [0.5, -1.5]])
Ai = np.linalg.inv(A)
box = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ A.T
lo, hi = box.min(0), box.max(0)
samp = rng.uniform(lo, hi, (400000, 2))
pre = samp @ Ai.T
inside = np.all((pre >= 0) & (pre <= 1), axis=1)
close("명제 2.6.8: vol A([0,1]^2) = |det A| (몬테카를로)", np.mean(inside) * np.prod(hi - lo), abs(np.linalg.det(A)), tol=0.02)

# 예 2.6.11: 구면좌표
rho, t_, f_ = sp.symbols("rho theta phi", real=True)
G = sp.Matrix([rho * sp.sin(t_) * sp.cos(f_), rho * sp.sin(t_) * sp.sin(f_), rho * sp.cos(t_)])
JG = G.jacobian([rho, t_, f_])
check("예 2.6.11: det DG = ρ² sin θ", sp.simplify(JG.det() - rho ** 2 * sp.sin(t_)) == 0)
vol_ball = sp.integrate(rho ** 2 * sp.sin(t_), (f_, 0, 2 * sp.pi), (t_, 0, sp.pi), (rho, 0, a))
check("예 2.6.11: 공의 부피 = 4πa³/3", sp.simplify(vol_ball - sp.Rational(4, 3) * sp.pi * a ** 3) == 0)
sph = EXAMPLES["sphere"]
TH, PH = sph["coords"]
R = sph["params"][0]
Xs = sph["expr"]
check("예 2.6.11: ρ = r에서 ∂_θG, ∂_φG = x_θ, x_φ", sp.simplify(JG[:, 1:3].subs({rho: R, t_: TH, f_: PH}) - Xs.jacobian([TH, PH])) == sp.zeros(3, 2))
check("예 2.6.11: ∂_ρG = x/r (단위법벡터)", sp.simplify(JG[:, 0].subs({rho: R, t_: TH, f_: PH}) - Xs / R) == sp.zeros(3, 1))
cr = Xs.diff(TH).cross(Xs.diff(PH))
sym_equal("예 2.6.11: |x_θ × x_φ| = r² sin θ", sp.sqrt(sp.simplify(cr.dot(cr))), R ** 2 * sp.sin(TH), sph["domain"])

# 비예 2.6.12
check("비예 2.6.12(a): ∫_0^{4π}∫_0^1 r dr dθ = 2π", sp.integrate(r, (r, 0, 1), (th, 0, 4 * sp.pi)) == 2 * sp.pi)
check("비예 2.6.12(b): ∫_0^1 det DF = -1, vol F(D) = 1", sp.integrate(-1, (x, 0, 1)) == -1)

# 연습
check("연습 2.6.1: 타원 넓이 = |det diag(a,b)| π", sp.Matrix([[2, 0], [0, 3]]).det() * sp.pi == 6 * sp.pi)
check("연습 2.6.2: ∫_B (x²+y²) = πa⁴/2", sp.simplify(sp.integrate(r ** 3, (r, 0, a), (th, -sp.pi, sp.pi)) - sp.pi * a ** 4 / 2) == 0)
T = sp.Matrix([[1, 1], [1, -2]])
check("연습 2.6.3: det T = -3, 넓이 = 1", T.det() == -3 and sp.Rational(3, 1) / abs(T.det()) == 1)
samp = rng.uniform([-1, -1.5], [3, 1], (400000, 2))
s_, u_ = samp[:, 0] + samp[:, 1], samp[:, 0] - 2 * samp[:, 1]
close("연습 2.6.3: 넓이 1 (몬테카를로)", np.mean((s_ >= 0) & (s_ <= 1) & (u_ >= 0) & (u_ <= 3)) * 4 * 2.5, 1.0, tol=0.02)
sym_equal("연습 2.6.5: ∫_{B_a} e^{-(x²+y²)} = π(1 - e^{-a²})", sp.integrate(sp.exp(-r ** 2) * r, (r, 0, a), (th, -sp.pi, sp.pi)), sp.pi * (1 - sp.exp(-a ** 2)))
check("연습 2.6.5: ∫_R e^{-x²} = √π", sp.integrate(sp.exp(-x ** 2), (x, -sp.oo, sp.oo)) == sp.sqrt(sp.pi))
tor = EXAMPLES["torus"]
U, V = tor["coords"]
Rt, rt = tor["params"]
rh = sp.symbols("rho", positive=True)
Tm = tor["expr"].subs(rt, rh)
JT = Tm.jacobian([rh, U, V])
check("연습 2.6.6: det DT = -ρ(R + ρ cos u)", sp.simplify(JT.det() + rh * (Rt + rh * sp.cos(U))) == 0)
check("연습 2.6.6: ρ = r에서 T는 원환면의 기준 매개화", Tm.subs(rh, rt) == tor["expr"])
volT = sp.integrate(rh * (Rt + rh * sp.cos(U)), (V, 0, 2 * sp.pi), (U, 0, 2 * sp.pi), (rh, 0, rt))
check("연습 2.6.6: 부피 2π²Rr²", sp.simplify(volT - 2 * sp.pi ** 2 * Rt * rt ** 2) == 0)
c = sp.symbols("c", real=True)
a1, b1, a2, b2 = sp.symbols("a1 b1 a2 b2", real=True)
check("연습 2.6.7: 층밀림 영역 넓이 = (b1-a1)(b2-a2)",
      sp.simplify(sp.integrate(sp.integrate(1, (x, a1 + c * y, b1 + c * y)), (y, a2, b2)) - (b1 - a1) * (b2 - a2)) == 0)

# 정리 2.6.13 (그린 정리): 식 (2.6.4)의 좌변(조각별 선적분)과 우변을 따로 계산해 비교한다.
def line_integral(P, Q, pieces):
    """pieces: [(γ¹(t), γ²(t), t0, t1)] — 조각마다 매끄러운 곡선의 선적분 Σ∫ P(γ)γ¹' + Q(γ)γ²' dt."""
    tot = 0
    for g1, g2, t0, t1 in pieces:
        sub = {x: g1, y: g2}
        tot += sp.integrate(P.subs(sub) * sp.diff(g1, t) + Q.subs(sub) * sp.diff(g2, t), (t, t0, t1))
    return sp.simplify(tot)

circle = [(a * sp.cos(t), a * sp.sin(t), 0, 2 * sp.pi)]
# 본문: P = -y/2, Q = x/2 이면 넓이 = ½∮ x dy - y dx = πa² (예 2.6.10과 같다)
sym_equal("정리 2.6.13 본문: ½∮(x dy - y dx) = πa² (원)", line_integral(-y / 2, x / 2, circle), sp.pi * a ** 2)
# 본문: 원은 양의 방향. γ + εJγ' = (1-ε)γ
g = sp.Matrix([a * sp.cos(t), a * sp.sin(t)])
gp = g.diff(t)
Jgp = sp.Matrix([-gp[1], gp[0]])
check("정리 2.6.13 본문: 원에서 Jγ' = (-a cos t, -a sin t) = -γ, 따라서 γ + εJγ' = (1-ε)γ", sp.simplify(Jgp + g) == sp.zeros(2, 1))
# 검산 1: P = -y³, Q = x³ 원판 — 우변을 극좌표로(예 2.6.10의 공식)
rhs = sp.integrate((3 * x ** 2 + 3 * y ** 2).subs({x: r * sp.cos(th), y: r * sp.sin(th)}) * r, (r, 0, a), (th, -sp.pi, sp.pi))
lhs = line_integral(-y ** 3, x ** 3, circle)
sym_equal("정리 2.6.13 검산: P=-y³, Q=x³, 원판에서 양변 = 3πa⁴/2", lhs, rhs)
check("정리 2.6.13 검산: 값 3πa⁴/2", sp.simplify(rhs - 3 * sp.pi * a ** 4 / 2) == 0)
# 검산 2: 네 조각으로 된 단위정사각형(반시계), P = x²y, Q = xy² — 조각마다 매끄러운 경우
square = [(t, 0 * t, 0, 1), (1 + 0 * t, t, 0, 1), (1 - t, 1 + 0 * t, 0, 1), (0 * t, 1 - t, 0, 1)]
P2, Q2 = x ** 2 * y, x * y ** 2
rhs2 = sp.integrate(sp.diff(Q2, x) - sp.diff(P2, y), (x, 0, 1), (y, 0, 1))
sym_equal("정리 2.6.13 검산: 단위정사각형(4조각), P=x²y, Q=xy²", line_integral(P2, Q2, square), rhs2)
# 방향을 뒤집으면 부호가 바뀐다(양의 방향 가정이 필요)
circle_cw = [(a * sp.cos(-t), a * sp.sin(-t), 0, 2 * sp.pi)]
sym_equal("정리 2.6.13 검산: 시계 방향이면 -πa²", line_integral(-y / 2, x / 2, circle_cw), -sp.pi * a ** 2)

summary()
