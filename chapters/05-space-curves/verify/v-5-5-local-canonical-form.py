"""5.5절 국소 표준형: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

국소 표준형 x = h - κ²h³/6, y = κh²/2 + κ'h³/6, z = κτh³/6 (+ O(h⁴))을 나선(dgsym.EXAMPLES["helix"])의 정확한 식,
꼬인 삼차곡선(호의 길이의 역함수 급수), 수치 나머지로 확인한다. 부호는 b' = -τn 규약(§6.2)이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/verify/v-5-5-local-canonical-form.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
s, t, h = sp.symbols("s t h", real=True)

# 정리 5.5.2 증명: γ''' = -κ²t + κ'n + κτb (프레네-세레 공식에서, 기호적인 κ(s), τ(s))
kf, tf = sp.Function("kappa")(s), sp.Function("tau")(s)
# 틀을 성분 벡터로 두고 프레네-세레로 미분하는 연산
def d(c):   # c = (ct, cn, cb): c_t t + c_n n + c_b b의 도함수의 성분
    ct, cn, cb = c
    return (sp.diff(ct, s) - kf * cn, sp.diff(cn, s) + kf * ct - tf * cb, sp.diff(cb, s) + tf * cn)
g1 = (sp.Integer(1), sp.Integer(0), sp.Integer(0))     # γ' = t
g2 = d(g1)
g3 = d(g2)
check("정리 5.5.2 증명: γ'' = κn", all(sp.simplify(x - y) == 0 for x, y in zip(g2, (0, kf, 0))))
check("정리 5.5.2 증명: γ''' = -κ²t + κ'n + κτb",
      all(sp.simplify(x - y) == 0 for x, y in zip(g3, (-kf ** 2, sp.diff(kf, s), kf * tf))))

# 예 5.5.3: 나선의 정확한 프레네 좌표 ---------------------------------------------------------------
eh = EX["helix"]
a, b = eh["params"]
c = sp.sqrt(a ** 2 + b ** 2)
D = {a: (0.5, 2), b: (-1.5, 1.5), s: (-3, 3)}
beta = sp.Matrix([a * sp.cos(s / c), a * sp.sin(s / c), b * s / c])
T = beta.diff(s)
N = sp.simplify(beta.diff(s, 2) / sp.sqrt(sp.simplify(beta.diff(s, 2).dot(beta.diff(s, 2)))))
B = sp.simplify(T.cross(N))
dv = beta - beta.subs(s, 0)
X, Y, Z = (sp.simplify(dv.dot(v.subs(s, 0))) for v in (T, N, B))
sym_equal("예 5.5.3: x = (a²/c) sin(s/c) + b²s/c²", X, (a ** 2 / c) * sp.sin(s / c) + b ** 2 * s / c ** 2, D)
sym_equal("예 5.5.3: y = a(1 - cos(s/c))", Y, a * (1 - sp.cos(s / c)), D)
sym_equal("예 5.5.3: z = (ab/c)(s/c - sin(s/c))", Z, (a * b / c) * (s / c - sp.sin(s / c)), D)
k, tau = eh["expected"]["kappa"], eh["expected"]["tau"]
for name, expr, target in (("x", X, s - k ** 2 * s ** 3 / 6), ("y", Y, k * s ** 2 / 2), ("z", Z, k * tau * s ** 3 / 6)):
    ser = sp.series(expr, s, 0, 4).removeO()
    sym_equal(f"정리 5.5.2/예 5.5.3: 나선의 {name}의 3차 테일러 다항식", ser, target, {a: (0.5, 2), b: (-1.5, 1.5), s: (-1, 1)})
# 나머지가 O(h⁴): a = 1, b = 0.3에서 |오차|/s⁴ 유계
num = {a: 1, b: sp.Rational(3, 10)}
err = sp.lambdify(s, [ (X - (s - k ** 2 * s ** 3 / 6)).subs(num), (Y - k * s ** 2 / 2).subs(num), (Z - k * tau * s ** 3 / 6).subs(num)], "numpy")
for hh in (0.2, 0.1, 0.05):
    ratio = np.linalg.norm(np.array(err(hh), float)) / hh ** 4
    check(f"정리 5.5.2: 나머지/h⁴ 유계 (h = {hh}, 비 {ratio:.4f})", ratio < 0.1)

# 예 5.5.4: 세 평면에 비친 모양 --------------------------------------------------------------------
kk, tt = sp.symbols("kappa tau", positive=True)
xa, ya, za = h, kk * h ** 2 / 2, kk * tt * h ** 3 / 6
check("예 5.5.4: 접촉평면: y = (κ/2)x² (주요항)", sp.simplify(ya - kk / 2 * xa ** 2) == 0)
check("예 5.5.4: 전직평면: z = (κτ/6)x³ (주요항)", sp.simplify(za - kk * tt / 6 * xa ** 3) == 0)
check("예 5.5.4: 법평면: z² = (2τ²/(9κ)) y³ (주요항)", sp.simplify(za ** 2 - 2 * tt ** 2 / (9 * kk) * ya ** 3) == 0)

# 따름정리 5.5.5: 부호 (나선, a = 1, b = 0.3: τ > 0) --------------------------------------------------
Yf = sp.lambdify(s, Y.subs(num), "numpy")
Zf = sp.lambdify(s, Z.subs(num), "numpy")
hs = np.array([-0.3, -0.1, -0.01, 0.01, 0.1, 0.3])
check("따름정리 5.5.5(a): 0 < |h|에서 y > 0", np.all(Yf(hs) > 0))
check("따름정리 5.5.5(b): τ > 0이면 z의 부호 = h의 부호", np.all(np.sign(Zf(hs)) == np.sign(hs)))
Zl = sp.lambdify(s, Z.subs({a: 1, b: -sp.Rational(3, 10)}), "numpy")
check("따름정리 5.5.5(b): τ < 0(왼손 나선)이면 z의 부호 = -h의 부호", np.all(np.sign(Zl(hs)) == -np.sign(hs)))

# 비예 5.5.6: (t, t², t⁴)는 τ(0) = 0이고 접촉평면을 가로지르지 않는다 ------------------------------------------
q = sp.Matrix([t, t ** 2, t ** 4])
kq, tq = dgsym.curvature_torsion(q, t)
check("비예 5.5.6: κ(0) = 2", sp.simplify(kq.subs(t, 0) - 2) == 0)
check("비예 5.5.6: τ(0) = 0", sp.simplify(tq.subs(t, 0)) == 0)
sym_equal("비예 5.5.6: τ(t) = 48t/|γ'×γ''|²", tq, 48 * t / (256 * t ** 6 + 144 * t ** 4 + 4), {t: (-1, 1)})
Tq, Nq, Bq = dgsym.frenet_frame(q, t)
check("비예 5.5.6: t = 0에서 틀 = (e1, e2, e3)", sp.simplify(sp.Matrix.hstack(Tq, Nq, Bq).subs(t, 0) - sp.eye(3)) == sp.zeros(3, 3))

# 예 5.5.7: 꼬인 삼차곡선 (호의 길이 급수) --------------------------------------------------------------
tw = sp.Matrix([t, t ** 2, t ** 3])
speed = sp.sqrt(1 + 4 * t ** 2 + 9 * t ** 4)
s_of_t = sp.integrate(sp.series(speed, t, 0, 6).removeO(), (t, 0, t))
check("예 5.5.7: s = t + (2/3)t³ + O(t⁵)", sp.expand(s_of_t - (t + sp.Rational(2, 3) * t ** 3)).as_poly(t).degree() >= 5)
# t = s - (2/3)s³ + O(s⁵): 역급수
t_of_s = s - sp.Rational(2, 3) * s ** 3
check("예 5.5.7: 역급수 t(s) = s - (2/3)s³ + O(s⁵)",
      sp.series(s_of_t.subs(t, t_of_s) - s, s, 0, 5).removeO() == 0)
comp = [sp.series(e.subs(t, t_of_s), s, 0, 4).removeO() for e in tw]
kt, tt_ = dgsym.curvature_torsion(tw, t)
k0, tau0 = kt.subs(t, 0), tt_.subs(t, 0)
kprime0 = sp.limit(sp.diff(kt, t), t, 0)                       # κ는 t의 짝함수라 κ'(0) = 0
check("예 5.5.7: κ(0) = 2, τ(0) = 3, κ'(0) = 0", sp.simplify(k0 - 2) == 0 and sp.simplify(tau0 - 3) == 0 and kprime0 == 0)
check("예 5.5.7: x = s - κ²s³/6 = s - (2/3)s³", sp.expand(comp[0] - (s - k0 ** 2 * s ** 3 / 6)) == 0)
check("예 5.5.7: y = κs²/2 = s² (3차까지)", sp.expand(comp[1] - k0 * s ** 2 / 2) == 0)
check("예 5.5.7: z = κτs³/6 = s³", sp.expand(comp[2] - k0 * tau0 * s ** 3 / 6) == 0)

# 연습문제 --------------------------------------------------------------------------------
X1, Y1, Z1 = (e.subs({a: 1, b: 1}) for e in (X, Y, Z))
sym_equal("연습 5.5.1: a = b = 1이면 x = (1/√2) sin(s/√2) + s/2", X1, sp.sin(s / sp.sqrt(2)) / sp.sqrt(2) + s / 2, {s: (-2, 2)})
sym_equal("연습 5.5.1: a = b = 1이면 κ = τ = 1/2, z ≈ s³/24", sp.series(Z1, s, 0, 4).removeO(), s ** 3 / 24, {s: (-1, 1)})
# 연습 5.5.2: 극한
check("연습 5.5.2: lim 2y/h² = κ (나선)", sp.simplify(sp.limit(2 * Y / s ** 2, s, 0) - k) == 0)
check("연습 5.5.2: lim 6z/(κh³) = τ (나선)", sp.simplify(sp.limit(6 * Z / (k * s ** 3), s, 0) - tau) == 0)
# 연습 5.5.5: 접선과 γ(s0 + h)를 지나는 평면의 법선 ∝ (0, -z, y) → (0, 0, 1)
ratio = sp.limit(Z / Y, s, 0)
check("연습 5.5.5: z(h)/y(h) → 0 (나선)", sp.simplify(ratio) == 0)
# 연습 5.5.7: 접촉구면 (꼬인 삼차곡선, t = 1)
Tw, Nw, Bw = dgsym.frenet_frame(tw, t)
kw, tw_ = dgsym.curvature_torsion(tw, t)
v = sp.sqrt(tw.diff(t).dot(tw.diff(t)))
rho = 1 / kw
rho_s = sp.diff(rho, t) / v
cen = tw + rho * Nw + (rho_s / tw_) * Bw
t1 = sp.Integer(1)
cen1 = sp.simplify(cen.subs(t, t1))
R2 = sp.simplify((rho ** 2 + (rho_s / tw_) ** 2).subs(t, t1))
f = (tw - cen1).dot(tw - cen1) - R2
# 호의 길이에 대한 도함수 = (1/v) d/dt
D1 = sp.diff(f, t) / v
D2 = sp.diff(D1, t) / v
D3 = sp.diff(D2, t) / v
for name, e in (("f", f), ("f'", D1), ("f''", D2), ("f'''", D3)):
    val = float(sp.N(e.subs(t, t1), 30))
    close(f"연습 5.5.7: 접촉구면에서 {name}(s0) = 0 (꼬인 삼차곡선, t = 1)", val, 0.0, tol=1e-12)
D4 = sp.diff(D3, t) / v
check("연습 5.5.7: f''''(s0) ≠ 0 (일반적으로 4차에서 벗어난다)", abs(float(sp.N(D4.subs(t, t1), 30))) > 1e-6)

summary()
