"""2.1절 전미분: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/verify/v-2-1-total-derivative.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

r, th, t, s = sp.symbols("r theta t s", real=True)
x, y, z = sp.symbols("x y z", real=True)
h1, h2 = sp.symbols("h1 h2", real=True)


def jac(F, vars_):
    return sp.Matrix(F).jacobian(sp.Matrix(vars_))


# 예 2.1.3(a): 아핀사상의 전미분 = A
A = sp.Matrix([[1, 2, 0], [3, -1, 4]])
b = sp.Matrix([5, -2])
X = sp.Matrix(sp.symbols("X1:4", real=True))
check("예 2.1.3(a): D(Ax+b) = A", jac(A * X + b, list(X)) == A)

# 예 2.1.4(a): |p+h|^2 - |p|^2 - 2<p,h> = |h|^2
P3 = sp.Matrix(sp.symbols("p1:4", real=True))
H3 = sp.Matrix(sp.symbols("k1:4", real=True))
sym_equal("예 2.1.4(a): 나머지 = |h|^2",
          (P3 + H3).dot(P3 + H3) - P3.dot(P3) - 2 * P3.dot(H3), H3.dot(H3))
# 예 2.1.4(b),(c): 쌍선형 나머지
Y, Z, K, L = (sp.Matrix(sp.symbols(f"{c}1:3", real=True)) for c in "yzkl")
sym_equal("예 2.1.4(b): 내적의 나머지 = <k,l>",
          (Y + K).dot(Z + L) - Y.dot(Z) - K.dot(Z) - Y.dot(L), K.dot(L))
sig, sc = sp.symbols("sigma s_", real=True)
sym_equal("예 2.1.4(c): 스칼라곱의 나머지 = σ l",
          (sc + sig) * (Z + L) - sc * Z - sig * Z - sc * L, sig * L)

# 비예 2.1.5: |x|의 한쪽 차분몫 (n = 1)
check("비예 2.1.5: |t|/t의 좌우 극한이 다르다",
      sp.limit(sp.Abs(t) / t, t, 0, "+") == 1 and sp.limit(sp.Abs(t) / t, t, 0, "-") == -1)

# 명제 1.5.7 (1장) 재확인: |Av| ≤ K|v| (무작위 수치 검사)
rng = np.random.default_rng(0)
ok = True
for _ in range(200):
    M = rng.normal(size=(3, 4))
    v = rng.normal(size=4)
    ok &= np.linalg.norm(M @ v) <= np.sqrt((M ** 2).sum()) * np.linalg.norm(v) + 1e-12
check("명제 1.5.7 (1장) 재확인: |Av| ≤ (Σ(A^i_j)^2)^{1/2}|v| (200개 무작위 검사)", ok)

# 비예 2.1.12(a): f = xy/(x^2+y^2)
f13 = x * y / (x ** 2 + y ** 2)
check("비예 2.1.12(a): 축 위에서 0", f13.subs(y, 0) == 0 and f13.subs(x, 0) == 0)
check("비예 2.1.12(a): 대각선 위에서 1/2", sp.simplify(f13.subs(y, x)) == sp.Rational(1, 2))
# 비예 2.1.12(b): g = x^3/(x^2+y^2), D_v g(0) = (v1)^3/|v|^2
v1, v2 = sp.symbols("v1 v2", real=True)
g13 = x ** 3 / (x ** 2 + y ** 2)
Dv = sp.limit(sp.simplify(g13.subs({x: t * v1, y: t * v2}) / t), t, 0)
sym_equal("비예 2.1.12(b): D_v g(0) = (v1)^3/|v|^2", Dv, v1 ** 3 / (v1 ** 2 + v2 ** 2))
check("비예 2.1.12(b): D_{e1}+D_{e2} = 1, D_{e1+e2} = 1/2",
      Dv.subs({v1: 1, v2: 0}) + Dv.subs({v1: 0, v2: 1}) == 1 and Dv.subs({v1: 1, v2: 1}) == sp.Rational(1, 2))

# 예 2.1.14: 극좌표 사상
Pm = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
DP = jac(Pm, [r, th])
check("예 2.1.14: DP = (2.1.5)", DP == sp.Matrix([[sp.cos(th), -r * sp.sin(th)], [sp.sin(th), r * sp.cos(th)]]))
sym_equal("예 2.1.14: det DP = r", DP.det(), r)
p = {r: sp.Rational(3, 2), th: sp.pi / 6}
check("예 2.1.14: P(p) = (3√3/4, 3/4)", sp.simplify(Pm.subs(p) - sp.Matrix([3 * sp.sqrt(3) / 4, sp.Rational(3, 4)])) == sp.zeros(2, 1))
check("예 2.1.14: DP(p)", sp.simplify(DP.subs(p) - sp.Matrix([[sp.sqrt(3) / 2, -sp.Rational(3, 4)],
                                                             [sp.Rational(1, 2), 3 * sp.sqrt(3) / 4]])) == sp.zeros(2, 2))
Pn = sp.lambdify((r, th), Pm, "numpy")
DPn = np.array(DP.subs(p).evalf(), dtype=float)
pn = np.array([1.5, np.pi / 6])
table = {0.1: (0.0126, 0.0891), 0.01: (0.000125, 0.0088), 0.001: (0.00000125, 0.0009)}
for tt, (rem, ratio) in table.items():
    hh = tt * np.array([1.0, 1.0])
    err = np.linalg.norm(Pn(*(pn + hh)).ravel() - Pn(*pn).ravel() - DPn @ hh)
    close(f"예 2.1.14: t = {tt} 나머지 ≈ {rem}", err, rem, tol=0.5 * 10 ** np.floor(np.log10(rem)) * 0.2 + 1e-9)
    close(f"예 2.1.14: t = {tt} 비 ≈ {ratio}", err / np.linalg.norm(hh), ratio, tol=6e-5)

# 예 2.1.15: 원과 나선 (EXAMPLES)
for key in ("circle", "helix"):
    ex = EXAMPLES[key]
    tt = ex["coords"][0]
    gam = ex["expr"]
    vel = gam.diff(tt)
    if key == "circle":
        rr = ex["params"][0]
        check("예 2.1.15: 원의 γ' = (-r sin t, r cos t, 0)", vel == sp.Matrix([-rr * sp.sin(tt), rr * sp.cos(tt), 0]))
        sym_equal("예 2.1.15: 원의 |γ'| = r", sp.sqrt(vel.dot(vel)), rr)
    else:
        aa, bb = ex["params"]
        check("예 2.1.15: 나선의 γ' = (-a sin t, a cos t, b)", vel == sp.Matrix([-aa * sp.sin(tt), aa * sp.cos(tt), bb]))
        sym_equal("예 2.1.15: 나선의 |γ'| = √(a²+b²)", sp.simplify(vel.dot(vel)), aa ** 2 + bb ** 2)

# 예 2.1.16: 구면의 기준 매개화 (EXAMPLES)
sph = EXAMPLES["sphere"]
TH, PH = sph["coords"]
R = sph["params"][0]
Xs = sph["expr"]
Dx = jac(Xs, [TH, PH])
check("예 2.1.16: Dx = (2.1.6)", Dx == sp.Matrix([
    [R * sp.cos(TH) * sp.cos(PH), -R * sp.sin(TH) * sp.sin(PH)],
    [R * sp.cos(TH) * sp.sin(PH), R * sp.sin(TH) * sp.cos(PH)],
    [-R * sp.sin(TH), 0]]))
gram = sp.simplify((Dx.T * Dx).det())
sym_equal("예 2.1.16: det(Dx^T Dx) = r^4 sin^2 θ (> 0이면 계수 2)", gram, R ** 4 * sp.sin(TH) ** 2, sph["domain"])
check("예 2.1.16: θ = 0에서 둘째 열 = 0", Dx[:, 1].subs(TH, 0) == sp.zeros(3, 1))

# 연습 2.1.1
F1 = sp.Matrix([x ** 2 - y ** 2, 2 * x * y])
DF1 = jac(F1, [x, y])
check("연습 2.1.1: DF = [[2x,-2y],[2y,2x]]", DF1 == sp.Matrix([[2 * x, -2 * y], [2 * y, 2 * x]]))
check("연습 2.1.1: det DF(1,1) = 8", DF1.subs({x: 1, y: 1}).det() == 8)

# 연습 2.1.2: 원기둥 (EXAMPLES)
cyl = EXAMPLES["cylinder"]
U_, V_ = cyl["coords"]
Rc = cyl["params"][0]
Dc = jac(cyl["expr"], [U_, V_])
check("연습 2.1.2: 원기둥 Dx", Dc == sp.Matrix([[-Rc * sp.sin(U_), 0], [Rc * sp.cos(U_), 0], [0, 1]]))
sym_equal("연습 2.1.2: det(Dx^T Dx) = r^2 > 0", sp.simplify((Dc.T * Dc).det()), Rc ** 2)

# 연습 2.1.3: f = <x, Ax>
A3 = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"a{i}{j}", real=True))
fq = lambda w: w.dot(A3 * w)
rem = sp.expand(fq(P3 + H3) - fq(P3) - (H3.dot(A3 * P3) + P3.dot(A3 * H3)))
sym_equal("연습 2.1.3: 나머지 = <h, Ah>", rem, sp.expand(H3.dot(A3 * H3)))

# 연습 2.1.4: <x, x_θ> = <x, x_φ> = 0
sym_equal("연습 2.1.4: <x, x_θ> = 0", Xs.dot(Dx[:, 0]), 0)
sym_equal("연습 2.1.4: <x, x_φ> = 0", Xs.dot(Dx[:, 1]), 0)

# 연습 2.1.5(c): f = x^3 y/(x^4 + y^2)
f5 = x ** 3 * y / (x ** 4 + y ** 2)
Dv5 = sp.limit(sp.simplify(f5.subs({x: t * v1, y: t * v2}) / t), t, 0)
check("연습 2.1.5(c): D_v f(0) = 0 (v2 ≠ 0)", sp.simplify(Dv5) == 0)
check("연습 2.1.5(c): f(t, t^2) = t/2", sp.simplify(f5.subs({x: t, y: t ** 2}) - t / 2) == 0)
sym_equal("연습 2.1.5(c): x^4 + y^2 - 2x^2|y| = (x^2 - |y|)^2",
          x ** 4 + y ** 2 - 2 * x ** 2 * sp.Abs(y), (x ** 2 - sp.Abs(y)) ** 2)
sym_equal("연습 2.1.5(c): |f(h)|/|h| on (t,t^2) = 1/(2√(1+t^2)) (t>0)",
          (t / 2) / sp.sqrt(t ** 2 + t ** 4), 1 / (2 * sp.sqrt(1 + t ** 2)), {t: (0.01, 1)})

# 연습 2.1.6: sqrt|xy| 대각선 위의 비 = 1/√2
tp = sp.symbols("tp", positive=True)
g6 = sp.sqrt(sp.Abs(x * y))
check("연습 2.1.6: g(t,t)/|(t,t)| = 1/√2", sp.simplify(g6.subs({x: tp, y: tp}) / sp.sqrt(2 * tp ** 2)) == 1 / sp.sqrt(2))

# 연습 2.1.7: ∂_1 f(t, 0) = 2t sin(1/t) - cos(1/t), t_k = 1/(kπ)에서 ±1
rho = sp.sqrt(x ** 2 + y ** 2)
f7 = rho ** 2 * sp.sin(1 / rho)
d1 = sp.diff(f7, x)
sym_equal("연습 2.1.7: ∂_1 f 공식", d1, 2 * x * sp.sin(1 / rho) - x / rho * sp.cos(1 / rho), {x: (0.2, 1.5), y: (0.2, 1.5)})
d1axis = sp.simplify(d1.subs(y, 0).subs(x, tp))
sym_equal("연습 2.1.7: ∂_1 f(t,0) = 2t sin(1/t) - cos(1/t)", d1axis, 2 * tp * sp.sin(1 / tp) - sp.cos(1 / tp))
vals = [float(d1axis.subs(tp, 1 / (k * sp.pi))) for k in range(10, 14)]
close("연습 2.1.7: ∂_1 f(1/(kπ), 0) = (-1)^{k+1}", vals, [(-1) ** (k + 1) for k in range(10, 14)], tol=1e-9)

summary()
