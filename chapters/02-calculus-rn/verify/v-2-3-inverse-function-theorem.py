"""2.3절 역함수 정리: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/verify/v-2-3-inverse-function-theorem.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

r, th, x, y = sp.symbols("r theta x y", real=True)
rp = sp.symbols("r", positive=True)
Pm = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
DP = Pm.jacobian([r, th])

# 예 2.3.2(c): 공과 R^n의 미분동형사상 (n = 3)
X = sp.Matrix(sp.symbols("x1:4", real=True))
nx2 = X.dot(X)
Fb = X / sp.sqrt(1 - nx2)
Gb = lambda Y: Y / sp.sqrt(1 + Y.dot(Y))
sym_equal("예 2.3.2(c): 1 + |F(x)|^2 = 1/(1-|x|^2)", 1 + Fb.dot(Fb), 1 / (1 - nx2), {s: (-0.5, 0.5) for s in X})
sym_equal("예 2.3.2(c): G(F(x)) = x", Gb(Fb), X, {s: (-0.5, 0.5) for s in X})

# 비예 2.3.3: x^3의 역함수는 0에서 미분 불가
h = sp.symbols("h", positive=True)
check("비예 2.3.3: h^{1/3}/h → ∞", sp.limit(h ** sp.Rational(1, 3) / h, h, 0, "+") == sp.oo)

# 보조정리 2.3.6 / 예 2.3.7: 반복법
p = np.array([1.5, np.pi / 6])
Pn = lambda q: np.array([q[0] * np.cos(q[1]), q[0] * np.sin(q[1])])
A = np.array([[np.cos(p[1]), -p[0] * np.sin(p[1])], [np.sin(p[1]), p[0] * np.cos(p[1])]])
Ai = np.linalg.inv(A)
close("예 2.3.7: DP(p)^{-1} = [[√3/2, 1/2], [-1/3, √3/3]]", Ai, np.array([[np.sqrt(3) / 2, 0.5], [-1 / 3, np.sqrt(3) / 3]]), tol=1e-12)
check("예 2.3.7: det DP(p) = 3/2", abs(np.linalg.det(A) - 1.5) < 1e-12)
yv = Pn(p) + np.array([0.1, -0.1])
close("예 2.3.7: y ≈ (1.39904, 0.65)", yv, [1.39904, 0.65], tol=5e-6)
table = [((1.5, 0.52360), 0.1414), ((1.53660, 0.43253), 0.0071), ((1.54297, 0.43462), 0.00057),
         ((1.54271, 0.43496), 0.000059), ((1.54266, 0.43493), 0.0000045)]
xk = p.copy()
for k, (xs, err) in enumerate(table):
    close(f"예 2.3.7: x_{k} ≈ {xs}", xk, xs, tol=5e-6)
    e = np.linalg.norm(Pn(xk) - yv)
    close(f"예 2.3.7: |P(x_{k}) - y| ≈ {err}", e, err, tol=0.05 * err + 1e-9)
    xk = xk + Ai @ (yv - Pn(xk))
exact = np.array([np.hypot(*yv), 2 * np.arctan(yv[1] / (yv[0] + np.hypot(*yv)))])
close("예 2.3.7: 극한 = P^{-1}(y) ≈ (1.54266, 0.43493)", exact, [1.54266, 0.43493], tol=5e-6)
xk = p.copy()
for _ in range(40):
    xk = xk + Ai @ (yv - Pn(xk))
close("예 2.3.7: 반복의 극한 = 공식 (2.3.3)의 역사상 값", xk, exact, tol=1e-12)

# 정리 2.3.5 증명 스케치: 정규화한 H의 축소 성질 (극좌표 사상, p 근처 수치)
Ft = lambda u: Ai @ (Pn(p + u) - Pn(p))
Hn = lambda u: u - Ft(u)
rng = np.random.default_rng(0)
dl = 0.1
ok = True
for _ in range(500):
    u1, u2 = rng.uniform(-1, 1, (2, 2)) * dl * 2 / np.sqrt(2)
    ok &= np.linalg.norm(Hn(u1) - Hn(u2)) <= 0.5 * np.linalg.norm(u1 - u2) + 1e-15
check("정리 2.3.5 스케치: δ = 0.1에서 |H(x)-H(x')| ≤ |x-x'|/2 (극좌표, 수치)", ok)

# 예 2.3.10
check("예 2.3.10: det DP = r", sp.simplify(DP.det() - r) == 0)
check("예 2.3.10(a): P(r, θ+2π) = P(r, θ), P(-r, θ+π) = P(r, θ)",
      sp.simplify(Pm.subs(th, th + 2 * sp.pi) - Pm) == sp.zeros(2, 1)
      and sp.simplify(Pm.subs({r: -r, th: th + sp.pi}, simultaneous=True) - Pm) == sp.zeros(2, 1))
# (b) 역사상 공식
th_s = sp.symbols("theta", real=True)
inv_r = sp.sqrt(x ** 2 + y ** 2)
inv_t = 2 * sp.atan(y / (x + sp.sqrt(x ** 2 + y ** 2)))
vals = [(1.3, 0.4), (0.7, -2.9), (2.0, 3.0), (0.2, 1.5)]
for rr, tt in vals:
    xx, yy = rr * np.cos(tt), rr * np.sin(tt)
    close(f"예 2.3.10(b): 역사상 (r, θ) = ({rr}, {tt})",
          [float(inv_r.subs({x: xx, y: yy})), float(inv_t.subs({x: xx, y: yy}))], [rr, tt], tol=1e-12)
# (c) (2.3.3)
DPinv = sp.simplify(DP.inv())
check("예 2.3.10(c): DP^{-1} = (1/r)[[r cos, r sin], [-sin, cos]]",
      sp.simplify(DPinv - sp.Matrix([[sp.cos(th), sp.sin(th)], [-sp.sin(th) / r, sp.cos(th) / r]])) == sp.zeros(2, 2))
Jinv = sp.Matrix([inv_r, inv_t]).jacobian([x, y])
target = sp.Matrix([[x / inv_r, y / inv_r], [-y / (x ** 2 + y ** 2), x / (x ** 2 + y ** 2)]])
sym_equal("예 2.3.10(c): 역사상의 야코비 행렬 = (2.3.3) 오른쪽", Jinv, target, {x: (0.2, 2), y: (-2, 2)})
sym_equal("예 2.3.10(c): (2.3.3)의 두 표현이 같다",
          DPinv.subs(r, rp), target.subs({x: rp * sp.cos(th), y: rp * sp.sin(th)}), {rp: (0.3, 2), th: (-3, 3)})
# 예 1.4.11과 일치: 극좌표 기저 성분
v1, v2 = sp.symbols("v1 v2", real=True)
comp = DPinv * sp.Matrix([v1, v2])
sym_equal("예 2.3.10(c): 예 1.4.11의 성분식과 일치",
          comp, sp.Matrix([sp.cos(th) * v1 + sp.sin(th) * v2, (-sp.sin(th) * v1 + sp.cos(th) * v2) / r]), {r: (0.3, 2)})
check("예 2.3.10(d): DP(0, θ)의 계수 1", DP.subs(r, 0).rank() == 1)

# 연습 2.3.1
F1 = sp.Matrix([x ** 2 - y ** 2, 2 * x * y])
J1 = F1.jacobian([x, y])
check("연습 2.3.1: det DF = 4(x^2+y^2)", sp.simplify(J1.det() - 4 * (x ** 2 + y ** 2)) == 0)
check("연습 2.3.1: DF(1,1)^{-1} = [[1/4,1/4],[-1/4,1/4]]",
      J1.subs({x: 1, y: 1}).inv() == sp.Matrix([[sp.Rational(1, 4), sp.Rational(1, 4)], [-sp.Rational(1, 4), sp.Rational(1, 4)]]))
check("연습 2.3.1: F(1,1) = (0,2)", F1.subs({x: 1, y: 1}) == sp.Matrix([0, 2]))
# 연습 2.3.2
E = sp.Matrix([sp.exp(x) * sp.cos(y), sp.exp(x) * sp.sin(y)])
check("연습 2.3.2: det DE = e^{2x}", sp.simplify(E.jacobian([x, y]).det() - sp.exp(2 * x)) == 0)
# 연습 2.3.3: 구면좌표
rho, t_, f_ = sp.symbols("rho theta phi", real=True)
Gs = sp.Matrix([rho * sp.sin(t_) * sp.cos(f_), rho * sp.sin(t_) * sp.sin(f_), rho * sp.cos(t_)])
JG = Gs.jacobian([rho, t_, f_])
check("연습 2.3.3: det DG = ρ^2 sin θ", sp.simplify(JG.det() - rho ** 2 * sp.sin(t_)) == 0)
M31 = JG[0:2, 1:3].det()
M32 = sp.Matrix([[JG[0, 0], JG[0, 2]], [JG[1, 0], JG[1, 2]]]).det()
check("연습 2.3.3: 소행렬식 M31 = ρ^2 cos θ sin θ, M32 = ρ sin^2 θ",
      sp.simplify(M31 - rho ** 2 * sp.cos(t_) * sp.sin(t_)) == 0 and sp.simplify(M32 - rho * sp.sin(t_) ** 2) == 0)
sph = EXAMPLES["sphere"]
TH, PH = sph["coords"]
R = sph["params"][0]
Dx = sph["expr"].jacobian([TH, PH])
check("연습 2.3.3: ρ = r에서 ∂_θG, ∂_φG = 구면의 x_θ, x_φ",
      sp.simplify(JG[:, 1:3].subs({rho: R, t_: TH, f_: PH}) - Dx) == sp.zeros(3, 2))
# 연습 2.3.5
f5 = x + 2 * x ** 2 * sp.sin(1 / x)
d5 = sp.diff(f5, x)
kk = sp.symbols("k", integer=True, positive=True)
check("연습 2.3.5: f'(1/(2kπ)) = -1", all(sp.simplify(d5.subs(x, 1 / (2 * k * sp.pi)) + 1) == 0 for k in range(1, 6)))
check("연습 2.3.5: f'(1/((2k+1)π)) = 3", all(sp.simplify(d5.subs(x, 1 / ((2 * k + 1) * sp.pi)) - 3) == 0 for k in range(1, 6)))
check("연습 2.3.5: f'(0) = 1", sp.limit(f5.subs(x, h) / h, h, 0, "+") == 1)
# 연습 2.3.6: DS(A)H = AH + HA, |H^2| ≤ |H|^2
n = 3
Am = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{i}{j}"))
Hm = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"h{i}{j}"))
check("연습 2.3.6: (A+H)^2 - A^2 - (AH+HA) = H^2", sp.expand((Am + Hm) ** 2 - Am ** 2 - (Am * Hm + Hm * Am) - Hm ** 2) == sp.zeros(n, n))
ok = True
for _ in range(300):
    Hn_ = rng.normal(size=(4, 4))
    ok &= np.linalg.norm(Hn_ @ Hn_) <= np.linalg.norm(Hn_) ** 2 + 1e-12
check("연습 2.3.6: |H^2| ≤ |H|^2 (프로베니우스, 수치)", ok)
Bm = np.eye(3) + 0.05 * rng.normal(size=(3, 3))
Ak = np.eye(3)
for _ in range(60):
    Ak = Ak + 0.5 * (Bm - Ak @ Ak)          # 축소사상 반복 (DS(I) = 2I)
close("연습 2.3.6: I 근처 B의 제곱근 (반복법)", Ak @ Ak, Bm, tol=1e-12)
# 연습 2.3.7: g(x) = (sin(x2)/2, cos(x1)/3) 예시, F = id + g는 전단사
gfun = lambda z: np.array([np.sin(z[1]) / 2, np.cos(z[0]) / 3])
Dg = lambda z: np.array([[0, np.cos(z[1]) / 2], [-np.sin(z[0]) / 3, 0]])
check("연습 2.3.7: 예시 g의 ||Dg|| ≤ 1/2", max(np.linalg.norm(Dg(z), 2) for z in rng.uniform(-5, 5, (500, 2))) <= 0.5 + 1e-12)
target_y = np.array([3.0, -2.0])
z = np.zeros(2)
for _ in range(80):
    z = target_y - gfun(z)
close("연습 2.3.7: T_y 반복으로 F(x) = y 풀기", z + gfun(z), target_y, tol=1e-12)

summary()
