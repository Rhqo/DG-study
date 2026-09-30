"""2.4절 음함수 정리와 등위집합: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/verify/v-2-4-implicit-function-theorem.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

x, y, z, t = sp.symbols("x y z t", real=True)
rp = sp.symbols("r", positive=True)

# 비예 2.4.3(a): x^2 - y^2 = 0은 두 직선
f = x ** 2 - y ** 2
check("비예 2.4.3(a): x^2 - y^2 = (x - y)(x + y)", sp.factor(f) == (x - y) * (x + y))
check("비예 2.4.3: 원점에서 Df = 0", [sp.diff(f, v).subs({x: 0, y: 0}) for v in (x, y)] == [0, 0])

# 정리 2.4.4 증명: DΦ(h, l) = (h, D_xF h + D_yF l)의 가역성 (구체 예)
F2 = sp.Matrix([x ** 2 + y ** 2 + z ** 2 - 1, x * y + z])        # n = 1 (x), k = 2 (y, z)
Phi = sp.Matrix([x, F2[0], F2[1]])
pt = {x: sp.Rational(1, 3), y: sp.Rational(2, 3), z: -sp.Rational(2, 9)}
DyF = F2.jacobian([y, z]).subs(pt)
check("정리 2.4.4: D_yF 가역이면 DΦ 가역 (예시)", DyF.det() != 0 and Phi.jacobian([x, y, z]).subs(pt).det() == DyF.det())

# 예 2.4.5: 원
F = x ** 2 + y ** 2
g = sp.sqrt(rp ** 2 - x ** 2)
sym_equal("예 2.4.5: g' = -x/y (g = √(r²-x²))", sp.diff(g, x), -x / g, {x: (-0.5, 0.5), rp: (1, 2)})
sym_equal("예 2.4.5: 공식 (2.4.2) = -F_x/F_y", -sp.diff(F, x) / sp.diff(F, y), -x / y, {x: (0.1, 1), y: (0.1, 1)})
check("예 2.4.5: (r, 0)에서 D_yF = 0, D_xF = 2r", sp.diff(F, y).subs({x: rp, y: 0}) == 0 and sp.diff(F, x).subs({x: rp, y: 0}) == 2 * rp)

# 비예 2.4.6: y^3 - x
G6 = y ** 3 - x
check("비예 2.4.6: D_yF(0,0) = 0, D_xF = -1", sp.diff(G6, y).subs({x: 0, y: 0}) == 0 and sp.diff(G6, x) == -1)
xp = sp.symbols("xp", positive=True)
sym_equal("비예 2.4.6: g' = 1/(3 x^{2/3})", sp.diff(xp ** sp.Rational(1, 3), xp), 1 / (3 * xp ** sp.Rational(2, 3)), {xp: (0.1, 2)})
sym_equal("비예 2.4.6: -F_x/F_y = 1/(3y^2) on y = x^{1/3}",
          (-sp.diff(G6, x) / sp.diff(G6, y)).subs(y, xp ** sp.Rational(1, 3)), 1 / (3 * xp ** sp.Rational(2, 3)), {xp: (0.1, 2)})

# (2.4.3) 기울기와 방향미분, 비예 2.1.12(b)
v1, v2 = sp.symbols("v1 v2", real=True)
fs = sp.exp(x) * sp.sin(y) + x * y ** 3
Dv = sp.diff(fs.subs({x: x + t * v1, y: y + t * v2}), t).subs(t, 0)
sym_equal("(2.4.3): D_v f = <grad f, v>", Dv, sp.diff(fs, x) * v1 + sp.diff(fs, y) * v2)
gnd = x ** 3 / (x ** 2 + y ** 2)
check("비예 2.1.12(b) 재확인: D_(1,1) g(0) = 1/2 ≠ <(1,0),(1,1)>",
      sp.limit(gnd.subs({x: t, y: t}) / t, t, 0) == sp.Rational(1, 2))
# 코시-슈바르츠: 단위벡터 v에 대해 D_v f ≤ |grad f|, 등호는 v = grad/|grad|
rng = np.random.default_rng(0)
gv = np.array([2.0, -1.0, 0.5])
us = rng.normal(size=(2000, 3)); us /= np.linalg.norm(us, axis=1)[:, None]
check("(2.4.3) 뒤: 단위벡터 v에서 <grad, v> ≤ |grad|", np.max(us @ gv) <= np.linalg.norm(gv) + 1e-12)
close("(2.4.3) 뒤: v = grad/|grad|에서 등호", gv @ (gv / np.linalg.norm(gv)), np.linalg.norm(gv))

# 예 2.4.9: 구면 (EXAMPLES)
sph = EXAMPLES["sphere"]
TH, PH = sph["coords"]
R = sph["params"][0]
Fs = x ** 2 + y ** 2 + z ** 2
Xs = sph["expr"]
check("예 2.4.9: 구면은 F = x²+y²+z²의 등위집합 F^{-1}(r²)", sp.simplify(Fs.subs({x: Xs[0], y: Xs[1], z: Xs[2]}) - R ** 2) == 0)
gz = sp.sqrt(rp ** 2 - x ** 2 - y ** 2)
Dg = -(1 / sp.diff(Fs, z)) * sp.Matrix([[sp.diff(Fs, x), sp.diff(Fs, y)]])
sym_equal("예 2.4.9: (2.4.2)의 Dg = (-x/z, -y/z) = 예 2.2.18의 편미분",
          Dg.subs(z, gz), sp.Matrix([[sp.diff(gz, x), sp.diff(gz, y)]]), {x: (-0.4, 0.4), y: (-0.4, 0.4), rp: (1, 2)})
# 명제 2.4.10: 구면의 x_θ, x_φ는 ker DF에
gradF = sp.Matrix([2 * Xs[0], 2 * Xs[1], 2 * Xs[2]])
for i, name in ((0, "x_θ"), (1, "x_φ")):
    sym_equal(f"명제 2.4.10: 구면에서 {name} ∈ ker DF", gradF.dot(Xs.diff([TH, PH][i])), 0)

# 명제 2.4.10(b)의 구성: 곡선 α(t) = (a + tu, g(a + tu))의 속도 = v
a0 = sp.Rational(3, 5)
gcirc = sp.sqrt(1 - x ** 2)
u = sp.Rational(2, 1)
alpha = sp.Matrix([a0 + t * u, gcirc.subs(x, a0 + t * u)])
vel = alpha.diff(t).subs(t, 0)
check("명제 2.4.10(b): α'(0) ∈ ker DF(p) (원, p = (3/5, 4/5))",
      sp.simplify(sp.Matrix([[2 * a0, 2 * gcirc.subs(x, a0)]]) * vel) == sp.zeros(1, 1))

# 예 2.4.11: 쌍곡선
q = {x: sp.sqrt(2), y: 1}
check("예 2.4.11: f(q) = 1, grad f(q) = (2√2, -2)", f.subs(q) == 1 and [sp.diff(f, v).subs(q) for v in (x, y)] == [2 * sp.sqrt(2), -2])
h = sp.sqrt(x ** 2 - 1)
check("예 2.4.11: h'(√2) = √2", sp.simplify(sp.diff(h, x).subs(x, sp.sqrt(2)) - sp.sqrt(2)) == 0)
check("예 2.4.11: -f_x/f_y = √2", sp.simplify((-sp.diff(f, x) / sp.diff(f, y)).subs(q) - sp.sqrt(2)) == 0)
ht = sp.sqrt(1 + y ** 2)
check("예 2.4.11: h̃'(1) = 1/√2", sp.simplify(sp.diff(ht, y).subs(y, 1) - 1 / sp.sqrt(2)) == 0)
check("예 2.4.11: <grad f(q), (1, √2)> = 0", sp.simplify(2 * sp.sqrt(2) * 1 - 2 * sp.sqrt(2)) == 0)

# 연습 2.4.1: 엽선
Ff = x ** 3 + y ** 3 - 3 * x * y
sols = sp.solve([Ff, sp.diff(Ff, y)], [x, y], dict=True)
real_sols = {(sp.nsimplify(s[x]), sp.nsimplify(s[y])) for s in sols if s[x].is_real and s[y].is_real}
check("연습 2.4.1: ∂_yF = 0인 곡선 위의 점 = (0,0), (2^{2/3}, 2^{1/3})",
      real_sols == {(0, 0), (2 ** sp.Rational(2, 3), 2 ** sp.Rational(1, 3))})
pq = {x: sp.Rational(3, 2), y: sp.Rational(3, 2)}
check("연습 2.4.1: (3/2,3/2)는 곡선 위, g' = -1", Ff.subs(pq) == 0 and (-sp.diff(Ff, x) / sp.diff(Ff, y)).subs(pq) == -1)
check("연습 2.4.1: ∂_xF = ∂_yF = 9/4", sp.diff(Ff, x).subs(pq) == sp.Rational(9, 4) and sp.diff(Ff, y).subs(pq) == sp.Rational(9, 4))
# 연습 2.4.2: 위도원
Fl = sp.Matrix([x ** 2 + y ** 2 + z ** 2, z])
J = Fl.jacobian([x, y, z])
hh = sp.symbols("h", real=True)
check("연습 2.4.2: (-y, x, 0) ∈ ker DF", sp.simplify(J * sp.Matrix([-y, x, 0])) == sp.zeros(2, 1))
check("연습 2.4.2: x ≠ 0이면 계수 2", J.subs({x: 1, y: 0, z: hh}).rank() == 2 and J.subs({x: 0, y: 1, z: hh}).rank() == 2)
# 연습 2.4.3(a)
f3 = (x - y) ** 3
check("연습 2.4.3(a): (x-y)^3의 Df는 y = x 위에서 0", [sp.diff(f3, v).subs(y, x) for v in (x, y)] == [0, 0])
# 연습 2.4.5: 원뿔
cone = x ** 2 + y ** 2 - z ** 2
check("연습 2.4.5: 원뿔에서 (t,0,±t) ∈ C, grad(0) = 0",
      cone.subs({x: t, y: 0, z: t}) == 0 and cone.subs({x: t, y: 0, z: -t}) == 0
      and [sp.diff(cone, v).subs({x: 0, y: 0, z: 0}) for v in (x, y, z)] == [0, 0, 0])
# 연습 2.4.6: 라그랑주
lam = sp.symbols("lambda", real=True)
sol = sp.solve([1 - 2 * lam * x, 1 - 2 * lam * y, x ** 2 + y ** 2 - 1], [x, y, lam], dict=True)
vals = sorted(float(s[x] + s[y]) for s in sol)
close("연습 2.4.6: x+y의 임계값 = ±√2", vals, [-np.sqrt(2), np.sqrt(2)], tol=1e-12)
th_ = np.linspace(0, 2 * np.pi, 100001)
close("연습 2.4.6: 원 위 x+y의 최댓값 (수치) = √2", np.max(np.cos(th_) + np.sin(th_)), np.sqrt(2), tol=1e-8)
# 연습 2.4.7: SL(2)
a, b, c, d = sp.symbols("a b c d", real=True)
det = a * d - b * c
gd = [sp.diff(det, v) for v in (a, b, c, d)]
check("연습 2.4.7: grad det = (d, -c, -b, a)", gd == [d, -c, -b, a])
check("연습 2.4.7: d = (1+bc)/a는 det = 1의 해", sp.simplify(det.subs(d, (1 + b * c) / a) - 1) == 0)
Hm = sp.symbols("h11 h12 h21 h22", real=True)
Ddet = sum(gdi.subs({a: 1, b: 0, c: 0, d: 1}) * hi for gdi, hi in zip(gd, Hm))
check("연습 2.4.7: D det(I)(H) = tr H", sp.simplify(Ddet - (Hm[0] + Hm[3])) == 0)

summary()
