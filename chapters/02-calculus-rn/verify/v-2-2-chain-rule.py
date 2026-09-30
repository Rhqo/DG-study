"""2.2절 연쇄법칙과 고계 미분: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/verify/v-2-2-chain-rule.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

r, th, t, s = sp.symbols("r theta t s", real=True)
x, y, u, v = sp.symbols("x y u v", real=True)
rp = sp.symbols("r", positive=True)
Pm = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
DP = Pm.jacobian([r, th])
sub = {x: r * sp.cos(th), y: r * sp.sin(th)}


def via_chain(f):
    """(2.2.2): (∂_r g, ∂_θ g) = Df(P) DP."""
    Df = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)]]).subs(sub)
    return Df * DP


def direct(f):
    g = f.subs(sub)
    return sp.Matrix([[sp.diff(g, r), sp.diff(g, th)]])


# 예 2.2.3 / 정리 2.2.1: (2.2.2)가 직접 미분과 같다
for name, f in (("x^2+y^2", x ** 2 + y ** 2), ("xy", x * y), ("x^3 y + sin(xy)", x ** 3 * y + sp.sin(x * y))):
    sym_equal(f"예 2.2.3: (2.2.2) = 직접 미분 (f = {name})", via_chain(f), direct(f))
sym_equal("예 2.2.3: f = x^2+y^2 → ∂_r g = 2r, ∂_θ g = 0", direct(x ** 2 + y ** 2), sp.Matrix([[2 * r, 0]]))
sym_equal("예 2.2.3: f = xy → ∂_θ g = r^2 cos 2θ", direct(x * y)[1], r ** 2 * sp.cos(2 * th))

# 그림 2.2.1의 값: DP(p)v
p = {r: sp.Rational(3, 2), th: sp.pi / 6}
vv = sp.Matrix([sp.Rational(2, 5), sp.Rational(3, 5)])
check("그림 2.2.1: DP(p)v = (√3/5 - 9/20, 1/5 + 9√3/20)",
      sp.simplify(DP.subs(p) * vv - sp.Matrix([sp.sqrt(3) / 5 - sp.Rational(9, 20), sp.Rational(1, 5) + 9 * sp.sqrt(3) / 20])) == sp.zeros(2, 1))
# 따름정리 2.2.2: 직선과 포물선의 상의 속도가 같다
w = sp.Matrix([-sp.Rational(4, 5), sp.Rational(3, 5)])
p0 = sp.Matrix([sp.Rational(3, 2), sp.pi / 6])
for name, curve in (("직선", p0 + t * vv), ("포물선", p0 + t * vv + t ** 2 * w)):
    img = Pm.subs({r: curve[0], th: curve[1]}, simultaneous=True)
    vel = img.diff(t).subs(t, 0)
    check(f"따름정리 2.2.2: {name}의 상의 속도 = DP(p)v", sp.simplify(vel - DP.subs(p) * vv) == sp.zeros(2, 1))

# 예 2.2.4: 나선의 재매개화 빠르기 1
hel = EXAMPLES["helix"]
T = hel["coords"][0]
a, b = hel["params"]
c = sp.sqrt(a ** 2 + b ** 2)
rep = hel["expr"].subs(T, s / c)
speed = sp.sqrt(sp.simplify(rep.diff(s).dot(rep.diff(s))))
sym_equal("예 2.2.4: |(γ∘h)'| = 1", speed, 1, hel["domain"])

# 명제 2.2.5(c): <α,β>' 공식 (구체적 곡선)
al = sp.Matrix([sp.cos(t), t ** 2, sp.exp(t)])
be = sp.Matrix([t, sp.sin(t), 1 + t])
sym_equal("명제 2.2.5 (2.2.3): <α,β>' = <α',β> + <α,β'>", sp.diff(al.dot(be), t), al.diff(t).dot(be) + al.dot(be.diff(t)))

# 예 2.2.6: 위도원
sph = EXAMPLES["sphere"]
TH, PH = sph["coords"]
R = sph["params"][0]
th0 = sp.symbols("theta0", real=True)
lat = sph["expr"].subs({TH: th0, PH: t})
sym_equal("예 2.2.6: 위도원 <γ, γ'> = 0", lat.dot(lat.diff(t)), 0)

# 정의 2.2.7의 예: ||R_α|| = 1, diag(a,b)
al_ = sp.symbols("alpha", real=True)
v1, v2 = sp.symbols("v1 v2", real=True)
Ra = sp.Matrix([[sp.cos(al_), -sp.sin(al_)], [sp.sin(al_), sp.cos(al_)]])
sym_equal("정의 2.2.7: |R_α v|^2 = |v|^2", (Ra * sp.Matrix([v1, v2])).dot(Ra * sp.Matrix([v1, v2])), v1 ** 2 + v2 ** 2)
rng = np.random.default_rng(1)
A = np.diag([0.3, -2.0])
samples = rng.normal(size=(2000, 2))
samples /= np.linalg.norm(samples, axis=1)[:, None]
close("정의 2.2.7: ||diag(0.3, -2)|| = 2 (단위원 위 최댓값)", np.max(np.linalg.norm(samples @ A.T, axis=1)), 2.0, tol=1e-4)
check("정의 2.2.7: 항등사상에서 K = √2 > ||id|| = 1", np.sqrt(2) > 1)

# 명제 2.2.8: 평균값 부등식 수치 확인 (극좌표 사상, 선분 위)
Pn = sp.lambdify((r, th), Pm, "numpy")
DPn = sp.lambdify((r, th), DP, "numpy")
aa, bb = np.array([1.0, 0.2]), np.array([2.0, 1.4])
M = max(np.linalg.norm(np.array(DPn(*(aa + tt * (bb - aa))), dtype=float), 2) for tt in np.linspace(0, 1, 2001))
lhs = np.linalg.norm(Pn(*bb).ravel() - Pn(*aa).ravel())
check("명제 2.2.8: |P(b)-P(a)| ≤ sup||DP|| |b-a| (수치)", lhs <= M * np.linalg.norm(bb - aa) + 1e-12)
check("비예 2.2.9: γ(2π) - γ(0) = 0", sp.Matrix([sp.cos(2 * sp.pi) - 1, sp.sin(2 * sp.pi)]) == sp.zeros(2, 1))

# 비예 2.2.13: x|x|
f13 = x * sp.Abs(x)
check("비예 2.2.13: (x|x|)' = 2|x| (x ≠ 0 에서)", sp.simplify(sp.diff(f13, x).subs(x, 3)) == 6 and sp.simplify(sp.diff(f13, x).subs(x, -2)) == 4)
check("비예 2.2.13: f'(0) = 0", sp.limit(f13.subs(x, t) / t, t, 0) == 0)

# 정리 2.2.14: C^2 함수의 혼합 편미분 대칭 (예시)
fs = sp.exp(x * y) * sp.sin(x + y ** 2)
check("정리 2.2.14: 혼합 편미분 대칭 (예시 함수)", sp.simplify(sp.diff(fs, x, y) - sp.diff(fs, y, x)) == 0)

# 비예 2.2.15: Peano
fp = x * y * (x ** 2 - y ** 2) / (x ** 2 + y ** 2)
h = sp.symbols("h", real=True)
yy = sp.symbols("yy", real=True, nonzero=True)
fx0y = sp.limit(fp.subs({x: h, y: yy}) / h, h, 0)
fyx0 = sp.limit(fp.subs({x: yy, y: h}) / h, h, 0)
check("비예 2.2.15: ∂_x f(0, y) = -y", sp.simplify(fx0y + yy) == 0)
check("비예 2.2.15: ∂_y f(x, 0) = x", sp.simplify(fyx0 - yy) == 0)
check("비예 2.2.15: ∂_y∂_x f(0) = -1, ∂_x∂_y f(0) = 1", sp.diff(-y, y) == -1 and sp.diff(x, x) == 1)

# 보조정리 2.2.16 / 정리 2.2.17: 1변수 테일러 (적분 나머지) 구체 확인
g = sp.exp(2 * t) * sp.cos(t)
k = 3
taylor = sum(sp.diff(g, t, j).subs(t, 0) / sp.factorial(j) * t ** j for j in range(k + 1))
remainder = sp.integrate((t - s) ** k * sp.diff(g, t, k + 1).subs(t, s), (s, 0, t)) / sp.factorial(k)
sym_equal("보조정리 2.2.16: k=3, g = e^{2t} cos t", taylor + remainder, g)
# 정리 2.2.17: 2변수, k = 1의 적분 나머지 공식
F2 = sp.exp(x) * sp.sin(y) + x ** 3 * y
px, py, hx, hy = sp.Rational(1, 3), sp.Rational(-1, 2), sp.Rational(1, 4), sp.Rational(2, 5)
lin = F2.subs({x: px, y: py}) + sp.diff(F2, x).subs({x: px, y: py}) * hx + sp.diff(F2, y).subs({x: px, y: py}) * hy
H = sp.hessian(F2, (x, y))
integrand = (1 - t) * (sp.Matrix([[hx, hy]]) * H * sp.Matrix([hx, hy]))[0].subs({x: px + t * hx, y: py + t * hy})
R1 = sp.Integral(integrand, (t, 0, 1)).evalf(30)
close("정리 2.2.17: k=1 적분 나머지 공식 (수치)", float(lin + R1), float(F2.subs({x: px + hx, y: py + hy})), tol=1e-12)

# 예 2.2.18: 반구면
fh = sp.sqrt(rp ** 2 - x ** 2 - y ** 2)
at0 = {x: 0, y: 0}
check("예 2.2.18: f(0)=r, ∇f(0)=0", fh.subs(at0) == rp and sp.diff(fh, x).subs(at0) == 0 and sp.diff(fh, y).subs(at0) == 0)
check("예 2.2.18: 2계 편미분 (−1/r, 0, −1/r)",
      sp.simplify(sp.diff(fh, x, 2).subs(at0) + 1 / rp) == 0 and sp.diff(fh, x, y).subs(at0) == 0
      and sp.simplify(sp.diff(fh, y, 2).subs(at0) + 1 / rp) == 0)
sym_equal("예 2.2.18: ∂_x∂_x f = -1/f - x^2/f^3", sp.diff(fh, x, 2), -1 / fh - x ** 2 / fh ** 3, {x: (-0.3, 0.3), y: (-0.3, 0.3), rp: (1, 2)})
# 나머지가 |h|^3 정도
fhn = sp.lambdify((x, y), fh.subs(rp, 1), "numpy")
ratios = []
for eps in (1e-1, 5e-2, 2.5e-2):
    hv = eps * np.array([0.6, 0.8])
    rem = fhn(*hv) - (1 - (hv @ hv) / 2)
    ratios.append(abs(rem) / eps ** 3)
check("예 2.2.18: |R_2(h)|/|h|^3 유계 (r=1, 세 스케일)", max(ratios) < 1.0 and min(ratios) > 0)

# 따름정리 2.2.19: g_i 공식 (구체 함수)
f19 = sp.exp(x) * sp.cos(y) + x * y ** 2
pp = (sp.Rational(1, 2), sp.Rational(-1, 3))
g1 = sp.integrate(sp.diff(f19, x).subs({x: pp[0] + t * (x - pp[0]), y: pp[1] + t * (y - pp[1])}, simultaneous=True), (t, 0, 1))
g2 = sp.integrate(sp.diff(f19, y).subs({x: pp[0] + t * (x - pp[0]), y: pp[1] + t * (y - pp[1])}, simultaneous=True), (t, 0, 1))
sym_equal("따름정리 2.2.19: f = f(p) + Σ(x^i-p^i) g_i", f19.subs({x: pp[0], y: pp[1]}) + (x - pp[0]) * g1 + (y - pp[1]) * g2, f19,
          {x: (-1, 1), y: (-1, 1)})
sym_equal("따름정리 2.2.19: g_1(p) = ∂_1 f(p)", sp.limit(sp.limit(g1, x, pp[0]), y, pp[1]), sp.diff(f19, x).subs({x: pp[0], y: pp[1]}))

# 예 2.2.20: e^{-1/t}
tp = sp.symbols("t", positive=True)
S = sp.symbols("S")
f20 = sp.exp(-1 / tp)
q = sp.Integer(1)
for kk in range(1, 6):
    q = sp.expand(S ** 2 * (q - sp.diff(q, S)))
    sym_equal(f"예 2.2.20: f^({kk})(t) = q_{kk}(1/t) e^(-1/t)", sp.diff(f20, tp, kk), q.subs(S, 1 / tp) * sp.exp(-1 / tp), {tp: (0.2, 3)})
    check(f"예 2.2.20: q_{kk}(1/t)e^(-1/t)/t → 0 (t → 0+)", sp.limit(q.subs(S, 1 / tp) * sp.exp(-1 / tp) / tp, tp, 0, "+") == 0)

# 연습 2.2.1
sym_equal("연습 2.2.1: f = x^2 y", via_chain(x ** 2 * y),
          sp.Matrix([[3 * r ** 2 * sp.cos(th) ** 2 * sp.sin(th), r ** 3 * (-2 * sp.cos(th) * sp.sin(th) ** 2 + sp.cos(th) ** 3)]]))
# 연습 2.2.2: 나선
gam = hel["expr"]
sym_equal("연습 2.2.2: 나선 <γ', γ''> = 0", gam.diff(T).dot(gam.diff(T, 2)), 0)
# 연습 2.2.3
f3 = sp.exp(x) * sp.cos(y)
T2 = sum(sp.diff(f3, x, i, y, j).subs(at0) / (sp.factorial(i) * sp.factorial(j)) * x ** i * y ** j
         for i in range(3) for j in range(3) if i + j <= 2)
sym_equal("연습 2.2.3: 2차 테일러 다항식 = 1 + x + (x^2 - y^2)/2", T2, 1 + x + (x ** 2 - y ** 2) / 2)
check("연습 2.2.3: 혼합 편미분 = -e^x sin y", sp.simplify(sp.diff(f3, x, y) + sp.exp(x) * sp.sin(y)) == 0
      and sp.simplify(sp.diff(f3, y, x) + sp.exp(x) * sp.sin(y)) == 0)
# 연습 2.2.4
Fm = sp.Matrix([x + y, x * y])
Gf = lambda U, V: U ** 2 - 2 * V
DG = sp.Matrix([[2 * u, -2]]).subs({u: x + y})
check("연습 2.2.4: DG(F)DF = (2x, 2y)", sp.simplify(DG * Fm.jacobian([x, y]) - sp.Matrix([[2 * x, 2 * y]])) == sp.zeros(1, 2))
check("연습 2.2.4: G∘F = x^2 + y^2", sp.expand(Gf(x + y, x * y)) == x ** 2 + y ** 2)
# 연습 2.2.5(b): x^2 sin(1/x)
f5 = x ** 2 * sp.sin(1 / x)
d5 = sp.diff(f5, x)
check("연습 2.2.5(b): f'(1/(kπ)) = (-1)^{k+1}", all(sp.simplify(d5.subs(x, 1 / (kk * sp.pi)) - (-1) ** (kk + 1)) == 0 for kk in range(1, 6)))
# 연습 2.2.6: 오일러 항등식 (k = 3 예시)
f6 = x ** 3 + x * y ** 2 + y ** 4 / x
sym_equal("연습 2.2.6: 동차 3차 함수의 Σ x^i ∂_i f = 3f", x * sp.diff(f6, x) + y * sp.diff(f6, y), 3 * f6, {x: (0.3, 2), y: (0.3, 2)})
# 연습 2.2.7: 극좌표 라플라시안
fL = sp.Function("f")
gL = fL(r * sp.cos(th), r * sp.sin(th))
lap_polar = sp.diff(gL, r, 2) + sp.diff(gL, r) / r + sp.diff(gL, th, 2) / r ** 2
for name, ff in (("x^2+y^2", x ** 2 + y ** 2), ("x^3 y - e^x sin y", x ** 3 * y - sp.exp(x) * sp.sin(y)), ("x^4", x ** 4)):
    gg = ff.subs(sub)
    lp = sp.diff(gg, r, 2) + sp.diff(gg, r) / r + sp.diff(gg, th, 2) / r ** 2
    lap = (sp.diff(ff, x, 2) + sp.diff(ff, y, 2)).subs(sub)
    sym_equal(f"연습 2.2.7: 극좌표 라플라시안 (f = {name})", lp, lap, {r: (0.3, 2), th: (0, 6)})
check("연습 2.2.7: f = x^2+y^2 에서 4", sp.simplify((sp.diff(r ** 2, r, 2) + sp.diff(r ** 2, r) / r)) == 4)

summary()
