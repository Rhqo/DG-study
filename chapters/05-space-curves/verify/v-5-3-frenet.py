"""5.3절 프레네-세레 공식: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

프레네-세레 공식은 t' = κn, n' = -κt + τb, b' = -τn (§6.2). 기준 예제(나선, 구면)는 dgsym.EXAMPLES에서 가져오고,
일반 매개변수 공식은 dgsym.curvature_torsion·frenet_frame과 비교한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/verify/v-5-3-frenet.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
s, t = sp.symbols("s t", real=True)


def norm(v):
    return sp.sqrt(sp.simplify(v.dot(v)))


def general_formulas(g, var, positive=()):
    """명제 5.3.4: t, n, b, κ, τ (일반 매개변수)."""
    g = sp.Matrix(g)
    d1, d2, d3 = (g.diff(var, j) for j in (1, 2, 3))
    cr = d1.cross(d2)
    T = d1 / norm(d1)
    B = cr / norm(cr)
    N = B.cross(T)
    kap = norm(cr) / norm(d1) ** 3
    tau = cr.dot(d3) / cr.dot(cr)
    return [dgsym.simp(x, positive) for x in (T, N, B, kap, tau)] + [norm(d1)]


# 정리 5.3.1과 예 5.3.3: 나선 ------------------------------------------------------------------
eh = EX["helix"]
(th,) = eh["coords"]
a, b = eh["params"]
c = sp.sqrt(a ** 2 + b ** 2)
D = {a: (0.5, 2), b: (-1.5, 1.5), s: (-6, 6)}
beta = sp.Matrix([a * sp.cos(s / c), a * sp.sin(s / c), b * s / c])
T = beta.diff(s)
N = sp.simplify(beta.diff(s, 2) / norm(beta.diff(s, 2)))
B = sp.simplify(T.cross(N))
k, tau = eh["expected"]["kappa"], eh["expected"]["tau"]
sym_equal("정리 5.3.1/예 5.3.3: 나선 t' = κn", T.diff(s), k * N, D)
sym_equal("정리 5.3.1/예 5.3.3: 나선 n' = -κt + τb", N.diff(s), -k * T + tau * B, D)
sym_equal("정리 5.3.1/예 5.3.3: 나선 b' = -τn", B.diff(s), -tau * N, D)
Kmat = sp.Matrix([[0, -k, 0], [k, 0, -tau], [0, tau, 0]])
F = sp.Matrix.hstack(T, N, B)
sym_equal("정리 5.3.1: 행렬 꼴 (t n b)' = (t n b) K", F.diff(s), F * Kmat, D)
check("정리 5.3.1: K는 반대칭", sp.simplify(Kmat + Kmat.T) == sp.zeros(3, 3))

# 따름정리 5.3.2: 다르부 벡터 --------------------------------------------------------------------
om = sp.simplify(tau * T + k * B)
sym_equal("예 5.3.3: 나선의 다르부 벡터 ω = (0, 0, 1/c)", om, sp.Matrix([0, 0, 1 / c]), D)
for name, v in (("t", T), ("n", N), ("b", B)):
    sym_equal(f"따름정리 5.3.2: {name}' = ω × {name} (나선)", v.diff(s), om.cross(v), D)
sym_equal("연습 5.3.3: |ω|² = κ² + τ² = 1/c²", om.dot(om), k ** 2 + tau ** 2, D)
sym_equal("연습 5.3.3: |ω| = 1/c", sp.simplify(k ** 2 + tau ** 2), 1 / c ** 2, D)
# 따름정리 5.3.2의 항등식 자체 (기호적인 κ, τ와 임의의 양의 정규직교기저: e1, e2, e3)
kk, tt = sp.symbols("kappa tau", real=True)
e1, e2, e3 = sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])
w = tt * e1 + kk * e3
check("따름정리 5.3.2: ω × t = κn", sp.simplify(w.cross(e1) - kk * e2) == sp.zeros(3, 1))
check("따름정리 5.3.2: ω × n = -κt + τb", sp.simplify(w.cross(e2) - (-kk * e1 + tt * e3)) == sp.zeros(3, 1))
check("따름정리 5.3.2: ω × b = -τn", sp.simplify(w.cross(e3) + tt * e2) == sp.zeros(3, 1))

# 명제 5.3.4, 예 5.3.5: 일반 매개변수 공식 (나선) ---------------------------------------------------------
Tg, Ng, Bg, kg, taug, vg = general_formulas(eh["expr"], th)
DH = {a: (0.5, 2), b: (-1.5, 1.5), th: (-4, 4)}
sym_equal("예 5.3.5: 나선 κ = |γ'×γ''|/|γ'|³ = a/(a²+b²)", kg, k, DH)
sym_equal("예 5.3.5: 나선 τ = <γ'×γ'',γ'''>/|γ'×γ''|² = b/(a²+b²)", taug, tau, DH)
sym_equal("예 5.3.5: 나선 |γ'×γ''|² = a²(a²+b²)", sp.simplify(eh["expr"].diff(th).cross(eh["expr"].diff(th, 2)).dot(
    eh["expr"].diff(th).cross(eh["expr"].diff(th, 2)))), a ** 2 * (a ** 2 + b ** 2))
sym_equal("예 5.3.5: 나선 <γ'×γ'', γ'''> = a²b", eh["expr"].diff(th).cross(eh["expr"].diff(th, 2)).dot(eh["expr"].diff(th, 3)),
          a ** 2 * b)
Td, Nd, Bd = dgsym.frenet_frame(eh["expr"], th)
sym_equal("명제 5.3.4: 틀 공식 = dgsym.frenet_frame", sp.Matrix.hstack(Tg, Ng, Bg), sp.Matrix.hstack(Td, Nd, Bd), DH)
# (c) d/dt 공식
sym_equal("명제 5.3.4(c): dt/dt = vκn", Tg.diff(th), vg * kg * Ng, DH)
sym_equal("명제 5.3.4(c): dn/dt = v(-κt + τb)", Ng.diff(th), vg * (-kg * Tg + taug * Bg), DH)
sym_equal("명제 5.3.4(c): db/dt = -vτn", Bg.diff(th), -vg * taug * Ng, DH)

# 증명의 전개 γ''' = (v'' - κ²v³)t + (3vv'κ + v²κ')n + v³κτ b (일반 곡선: 수치로 한 곡선에서)
g1 = sp.Matrix([t, t ** 2, t ** 3])
T1, N1, B1, k1, tau1, v1 = general_formulas(g1, t)
lhs = g1.diff(t, 3)
rhs = (v1.diff(t, 2) - k1 ** 2 * v1 ** 3) * T1 + (3 * v1 * v1.diff(t) * k1 + v1 ** 2 * k1.diff(t)) * N1 + v1 ** 3 * k1 * tau1 * B1
sym_equal("명제 5.3.4 증명: γ'''의 프레네 틀 전개 (꼬인 삼차곡선)", lhs, rhs, {t: (-1.5, 1.5)})
sym_equal("명제 5.3.4 증명: γ'' = v't + v²κn (꼬인 삼차곡선)", g1.diff(t, 2), v1.diff(t) * T1 + v1 ** 2 * k1 * N1, {t: (-1.5, 1.5)})

# 예 5.3.6: 꼬인 삼차곡선 ------------------------------------------------------------------------
sym_equal("예 5.3.6: τ = 3/(9t⁴+9t²+1)", tau1, 3 / (9 * t ** 4 + 9 * t ** 2 + 1), {t: (-2, 2)})
sym_equal("예 5.3.6: κ = 2√(9t⁴+9t²+1)/(1+4t²+9t⁴)^{3/2}", k1,
          2 * sp.sqrt(9 * t ** 4 + 9 * t ** 2 + 1) / (1 + 4 * t ** 2 + 9 * t ** 4) ** sp.Rational(3, 2), {t: (-2, 2)})
check("예 5.3.6: t = 0에서 κ = 2, τ = 3", sp.simplify(k1.subs(t, 0) - 2) == 0 and sp.simplify(tau1.subs(t, 0) - 3) == 0)
check("예 5.3.6: t = 0에서 (t, n, b) = (e1, e2, e3)",
      sp.simplify(sp.Matrix.hstack(T1, N1, B1).subs(t, 0) - sp.eye(3)) == sp.zeros(3, 3))
kd1, td1 = dgsym.curvature_torsion(g1, t)
sym_equal("예 5.3.6: dgsym과 일치 (κ)", kd1, k1, {t: (-2, 2)})
sym_equal("예 5.3.6: dgsym과 일치 (τ)", td1, tau1, {t: (-2, 2)})

# 예 5.3.7: 구면 위의 원 ------------------------------------------------------------------------
es = EX["sphere"]
tht, ph = es["coords"]
(r,) = es["params"]
th0 = sp.symbols("theta0", positive=True)
DS = {r: (0.5, 2), th0: (0.2, 2.9), ph: (0.1, 6.2)}
lat = es["expr"].subs(tht, th0)
Tl, Nl, Bl, kl, taul, vl = general_formulas(lat, ph, (sp.sin(th0),))
sym_equal("예 5.3.7(a): 위도원 t = (-sin φ, cos φ, 0)", Tl, sp.Matrix([-sp.sin(ph), sp.cos(ph), 0]), DS)
sym_equal("예 5.3.7(a): 위도원 n = -(cos φ, sin φ, 0)", Nl, -sp.Matrix([sp.cos(ph), sp.sin(ph), 0]), DS)
sym_equal("예 5.3.7(a): 위도원 b = e3", Bl, sp.Matrix([0, 0, 1]), DS)
sym_equal("예 5.3.7(a): 위도원 κ = 1/(r sin θ0)", kl, 1 / (r * sp.sin(th0)), DS)
check("예 5.3.7(a): 위도원 τ = 0", sp.simplify(taul) == 0)
Nsph = es["expected"]["normal"].subs(tht, th0)
sym_equal("예 5.3.7(a): N = -sin θ0 n + cos θ0 b", Nsph, -sp.sin(th0) * Nl + sp.cos(th0) * Bl, DS)
sym_equal("예 5.3.7(a): 곡률벡터의 N 성분 κ<n, N> = -1/r", kl * Nl.dot(Nsph), -1 / r, DS)
tang = sp.simplify(kl * Nl - (kl * Nl.dot(Nsph)) * Nsph)
sym_equal("예 5.3.7(a): 곡률벡터의 접평면 성분 크기² = cot²θ0/r²", sp.simplify(tang.dot(tang)), sp.cot(th0) ** 2 / r ** 2, DS)
mer = es["expr"].subs(ph, sp.Symbol("phi0", real=True))
ph0 = sp.Symbol("phi0", real=True)
Tm, Nm, Bm, km, taum, vm = general_formulas(mer, tht, (sp.sin(tht),))
DM = {r: (0.5, 2), tht: (0.2, 2.9), ph0: (-3, 3)}
sym_equal("예 5.3.7(b): 대원 n = -N", Nm, -es["expected"]["normal"].subs(ph, ph0), DM)
sym_equal("예 5.3.7(b): 대원 b = (-sin φ0, cos φ0, 0) (상수)", Bm, sp.Matrix([-sp.sin(ph0), sp.cos(ph0), 0]), DM)
sym_equal("예 5.3.7(b): 대원 κ = 1/r", km, 1 / r, DM)
check("예 5.3.7(b): 대원 τ = 0", sp.simplify(taum) == 0)
# (c) 구면 위 단위속력 곡선: <γ'', N> = -1/r (구면 위의 아무 곡선: θ(s), φ(s)를 임의 함수로)
thf, phf = sp.Function("th")(s), sp.Function("ph")(s)
gs = es["expr"].subs({tht: thf, ph: phf})
Ns = gs / r
# <γ'', γ> = -|γ'|² (|γ|² = r²을 두 번 미분); 단위속력이면 -1
sym_equal("예 5.3.7(c): 구면 위 곡선에서 <γ'', γ> = -|γ'|²", sp.simplify(gs.diff(s, 2).dot(gs) + gs.diff(s).dot(gs.diff(s))), 0)

# 연습문제 --------------------------------------------------------------------------------
ch = sp.Matrix([sp.cosh(t), sp.sinh(t), t])
_, _, _, kch, tch, _ = general_formulas(ch, t)
sym_equal("연습 5.3.1: κ = 1/(2cosh² t)", kch, 1 / (2 * sp.cosh(t) ** 2), {t: (-2, 2)})
sym_equal("연습 5.3.1: τ = 1/(2cosh² t)", tch, 1 / (2 * sp.cosh(t) ** 2), {t: (-2, 2)})
check("연습 5.3.1: γ'×γ'' = (-sinh, cosh, -1)", sp.simplify(ch.diff(t).cross(ch.diff(t, 2)) - sp.Matrix([-sp.sinh(t), sp.cosh(t), -1])) == sp.zeros(3, 1))
check("연습 5.3.1: <γ'×γ'', γ'''> = 1", sp.simplify(ch.diff(t).cross(ch.diff(t, 2)).dot(ch.diff(t, 3)) - 1) == 0)
sym_equal("연습 5.3.2: t(1) = (1,2,3)/√14", T1.subs(t, 1), sp.Matrix([1, 2, 3]) / sp.sqrt(14))
sym_equal("연습 5.3.2: b(1) = (3,-3,1)/√19", B1.subs(t, 1), sp.Matrix([3, -3, 1]) / sp.sqrt(19))
sym_equal("연습 5.3.2: n(1) = (-11,-8,9)/√266", N1.subs(t, 1), sp.Matrix([-11, -8, 9]) / sp.sqrt(266))
sym_equal("연습 5.3.2: κ(1) = 2√19/14^{3/2}", k1.subs(t, 1), 2 * sp.sqrt(19) / 14 ** sp.Rational(3, 2))
sym_equal("연습 5.3.2: τ(1) = 3/19", tau1.subs(t, 1), sp.Rational(3, 19))
sym_equal("연습 5.3.4(d): |n'|² = κ² + τ² (나선)", N.diff(s).dot(N.diff(s)), k ** 2 + tau ** 2, D)
# 연습 5.3.6: 구면 곡선 R² = ρ² + (ρ'/τ)² — 구면 위의 곡선(위도원은 τ = 0이라 제외)로 (t, ...)를 수치 확인
# 단위구면 위의 곡선 γ(t) = (cos t cos(t/2)... ) 대신: 원점 중심 반지름 1인 구면 위 곡선 (sin u cos 2u, sin u sin 2u, cos u)
u = sp.symbols("u", real=True)
sc = sp.Matrix([sp.sin(u) * sp.cos(2 * u), sp.sin(u) * sp.sin(2 * u), sp.cos(u)])
check("연습 5.3.6: 시험 곡선은 단위구면 위", sp.simplify(sc.dot(sc) - 1) == 0)
ksc, tsc = dgsym.curvature_torsion(sc, u)
vsc = norm(sc.diff(u))
rho = 1 / ksc
rho_s = sp.diff(rho, u) / vsc                     # 호의 길이에 대한 도함수
expr = rho ** 2 + (rho_s / tsc) ** 2
fexpr = sp.lambdify(u, expr, "numpy")
for uv in (0.4, 0.9, 1.3, 2.2):
    close(f"연습 5.3.6: ρ² + (ρ'/τ)² = 1 (u = {uv})", float(fexpr(uv)), 1.0, tol=1e-9)
# 연습 5.3.7 (랑크레): τ/κ가 상수이면 u = (c t + b)/√(1+c²)는 상수 — (cosh, sinh, t)에서
Tc, Nc, Bc, kc, tc_, vc = general_formulas(ch, t)
cst = sp.simplify(tc_ / kc)
check("연습 5.3.7: (cosh, sinh, t)에서 τ/κ = 1", cst == 1)
uvec = sp.simplify((cst * Tc + Bc) / sp.sqrt(1 + cst ** 2))
check("연습 5.3.7: u = (t + b)/√2는 상수", sp.simplify(uvec.diff(t)) == sp.zeros(3, 1))
sym_equal("연습 5.3.7: u = e2 (축 방향), <t, u> = 1/√2", uvec, sp.Matrix([0, 1, 0]), {t: (-2, 2)})

summary()
