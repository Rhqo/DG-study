"""4.5절 접촉원과 곡률중심: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

구면의 위도원·대원은 dgsym.EXAMPLES["sphere"]의 매개화에서 가져온다. 부호곡률은 정의(명제 4.4.5의 공식)로
계산하고, 공간 속 평면곡선의 곡률 |γ''| (명제 4.5.9)는 dgsym.curvature_torsion의 κ와 교차검증한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/verify/v-4-5-osculating-circle.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
t, s, u = sp.symbols("t s u", real=True)
Jm = sp.Matrix([[0, -1], [1, 0]])


def kappa_s(g, var):
    g = sp.Matrix(g)
    d1, d2 = g.diff(var), g.diff(var, 2)
    return d2.dot(Jm * d1) / sp.sqrt(d1.dot(d1)) ** 3


def center(g, var):
    """정의 4.5.1: c = γ + n_s/κ_s (일반 매개변수, n_s = Jγ'/|γ'|)."""
    g = sp.Matrix(g)
    d1 = g.diff(var)
    n = Jm * d1 / sp.sqrt(d1.dot(d1))
    return g + n / kappa_s(g, var)


# 정의 4.5.1: c = γ + γ''/|γ''|² (단위속력) — 일반 단위속력 곡선: t = (cos θ, sin θ)
th = sp.Function("theta")(s)
tt = sp.Matrix([sp.cos(th), sp.sin(th)])
gpp = tt.diff(s)                                    # γ'' = θ' J t
ks = th.diff(s)                                     # κ_s = θ' (정리 4.4.9)
sym_equal("정의 4.5.1: n_s/κ_s = γ''/|γ''|²", (Jm * tt) / ks, gpp / gpp.dot(gpp))

# 예 4.5.2 ---------------------------------------------------------------------------
e = EX["circle"]
(tc,) = e["coords"]
(r,) = e["params"]
circ = e["expr"][:2, 0]
sym_equal("예 4.5.2(a): 원의 곡률중심은 원점", center(circ, tc), sp.zeros(2, 1), e["domain"])
par = sp.Matrix([t, t ** 2 / 2])
sym_equal("예 4.5.2(b): y = x²/2의 곡률중심 (−x³, 1 + 3x²/2)", center(par, t), sp.Matrix([-t ** 3, 1 + sp.Rational(3, 2) * t ** 2]))
x0 = sp.Rational(1, 2)
sym_equal("그림 4.5.1: x0 = 1/2에서 c = (−1/8, 11/8)", center(par, t).subs(t, x0), sp.Matrix([-sp.Rational(1, 8), sp.Rational(11, 8)]))
sym_equal("그림 4.5.1: 곡률반지름 (5/4)^{3/2}", 1 / kappa_s(par, t).subs(t, x0), sp.Rational(5, 4) ** sp.Rational(3, 2))
A_, B_ = sp.symbols("A B", positive=True)
ell = sp.Matrix([A_ * sp.cos(t), B_ * sp.sin(t)])
c_ell = center(ell, t)
sym_equal("예 4.5.12: 타원의 축폐선 ((A²−B²)/A cos³t, −(A²−B²)/B sin³t)", c_ell,
          sp.Matrix([(A_ ** 2 - B_ ** 2) / A_ * sp.cos(t) ** 3, -(A_ ** 2 - B_ ** 2) / B_ * sp.sin(t) ** 3]),
          {A_: (1.5, 3), B_: (0.5, 1.4)})
sym_equal("예 4.5.2(c)/연습 4.5.2: (A,0)에서 반지름 B²/A", 1 / kappa_s(ell, t).subs(t, 0), B_ ** 2 / A_)
sym_equal("예 4.5.2(c)/연습 4.5.2: (0,B)에서 반지름 A²/B", 1 / kappa_s(ell, t).subs(t, sp.pi / 2), A_ ** 2 / B_)
sym_equal("연습 4.5.2: (0,B)에서 곡률중심 (0, B − A²/B)", c_ell.subs(t, sp.pi / 2), sp.Matrix([0, B_ - A_ ** 2 / B_]))

# 비예 4.5.3 ------------------------------------------------------------------------
cub = sp.Matrix([t, t ** 3])
check("비예 4.5.3(a): y = x³의 κ_s(0) = 0", sp.simplify(kappa_s(cub, t).subs(t, 0)) == 0)
# (b) 포물선 꼭짓점에 접하는 반지름 2인 원 y = 2 − √(4 − x²) ≈ x²/4 (곡선은 x²/2)
xx = sp.symbols("x", real=True)
circ2 = 2 - sp.sqrt(4 - xx ** 2)
check("비예 4.5.3(b): 반지름 2인 접하는 원은 x²/4 + O(x⁴)", sp.series(circ2, xx, 0, 4).removeO() == xx ** 2 / 4)
circ1 = 1 - sp.sqrt(1 - xx ** 2)
check("비예 4.5.3(b): 접촉원 y = 1 − √(1 − x²) = x²/2 + x⁴/8 + …",
      sp.series(circ1, xx, 0, 6).removeO() == xx ** 2 / 2 + xx ** 4 / 8)

# 정리 4.5.5: f_C의 도함수 (일반 단위속력 곡선) ------------------------------------------------
g = sp.Matrix([sp.Function("x")(s), sp.Function("y")(s)])
cvec = sp.Matrix(sp.symbols("c1 c2", real=True))
rho = sp.symbols("rho", positive=True)
fC = (g - cvec).dot(g - cvec) - rho ** 2
sym_equal("정리 4.5.5: f' = 2<γ', γ − c>", sp.diff(fC, s), 2 * g.diff(s).dot(g - cvec))
sym_equal("정리 4.5.5: f'' = 2<γ'', γ − c> + 2|γ'|²", sp.diff(fC, s, 2), 2 * g.diff(s, 2).dot(g - cvec) + 2 * g.diff(s).dot(g.diff(s)))
# 접하는 원 c = γ0 + λ n0 에서 f''(s0) = 2 − 2κλ (단위속력, γ'' = κ n)
kap, lam = sp.symbols("kappa lambda", real=True)
n0 = sp.Matrix([0, 1])
t0 = sp.Matrix([1, 0])
f2 = 2 * (kap * n0).dot(-lam * n0) + 2
sym_equal("정리 4.5.5: f''(s0) = 2(1 − κλ)", f2, 2 * (1 - kap * lam))
check("정리 4.5.5: f''(s0) = 0 ⟺ λ = 1/κ", sp.solve(sp.Eq(f2, 0), lam) == [1 / kap])

# 비고 4.5.8: 접촉원에서 f'''(s0) = −2κ'/κ (단위속력 곡선 t = (cos θ, sin θ)) -------------------------
gam = sp.Matrix([sp.Function("X")(s), sp.Function("Y")(s)])
s0 = sp.symbols("s0", real=True)
# 단위속력 곡선을 θ(s)로 만들고, s0에서의 접촉원 중심으로 f를 만든다
thf = sp.Function("theta")
Tv = sp.Matrix([sp.cos(thf(s)), sp.sin(thf(s))])
# γ의 도함수는 Tv, Tv', Tv''로 표현된다. f''' = 2<γ''', γ − c> + 6<γ'', γ'>. <γ'', γ'> = 0.
gp, gpp_, gppp = Tv, Tv.diff(s), Tv.diff(s, 2)
k_ = thf(s).diff(s)
c_minus_gamma = (Jm * Tv) / k_                 # c − γ(s0) (s0에서 계산)
f3 = 2 * gppp.dot(-c_minus_gamma) + 6 * gpp_.dot(gp)
sym_equal("비고 4.5.8: f'''(s0) = −2κ_s'/κ_s", f3, -2 * k_.diff(s) / k_)

# 그림 4.5.2 / 연습 4.5.5: 타원 A = 2, B = 1에서 t = π/6은 꼭짓점이 아니다
k21 = kappa_s(sp.Matrix([2 * sp.cos(t), sp.sin(t)]), t)
sym_equal("연습 4.5.5: A=2, B=1 타원 κ_s = 2/(1 + 3 sin²t)^{3/2}", k21, 2 / (1 + 3 * sp.sin(t) ** 2) ** sp.Rational(3, 2))
check("연습 4.5.5: dκ/dt(π/6) ≠ 0", sp.simplify(sp.diff(k21, t).subs(t, sp.pi / 6)) != 0)
sym_equal("연습 4.5.5: dκ/dt = −18 sin t cos t (1 + 3 sin²t)^{−5/2}", sp.diff(k21, t),
          -18 * sp.sin(t) * sp.cos(t) * (1 + 3 * sp.sin(t) ** 2) ** sp.Rational(-5, 2))

# 명제 4.5.7: 세 점 원의 중심 → 곡률중심 (수치) -----------------------------------------------
def gam_num(x):
    return np.array([2 * np.cos(x), np.sin(x)])


def circum(p1, p2, p3):
    M = 2 * np.array([p2 - p1, p3 - p1])
    b = np.array([p2 @ p2 - p1 @ p1, p3 @ p3 - p1 @ p1])
    return np.linalg.solve(M, b)


cT = np.array([1.5 * np.cos(np.pi / 6) ** 3, -3 * np.sin(np.pi / 6) ** 3])
errs = [np.linalg.norm(circum(gam_num(np.pi / 6 - 2 * h), gam_num(np.pi / 6 + 0.5 * h), gam_num(np.pi / 6 + h)) - cT)
        for h in (0.2, 0.05, 0.01, 0.002)]
check("명제 4.5.7: 비대칭인 세 점에서도 중심이 곡률중심으로 수렴 (타원, t0 = π/6)", errs[-1] < 1e-2 and all(a > b for a, b in zip(errs, errs[1:])))
# D(s0, s0) = det[t; κ n] = κ
tv = sp.Matrix([sp.cos(sp.Symbol("a")), sp.sin(sp.Symbol("a"))])
check("명제 4.5.7: det[t; Jt] = 1", sp.simplify(sp.Matrix.hstack(tv, Jm * tv).T.det() - 1) == 0)

# 명제 4.5.9: 평면 안의 공간곡선 — 무작위 정규직교기저로 독립성 확인 (수치) ---------------------------
rng = np.random.default_rng(3)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
a1, a2 = Q[:, 0], Q[:, 1]
p0 = rng.normal(size=3)
ss = np.linspace(0, 1, 5)
# 평면 곡선 (x, y) = 타원을 호의 길이가 아닌 매개변수로 두고, 곡률중심을 두 방식으로 비교
tvals = np.linspace(0.3, 2.5, 7)
ok = True
for tv_ in tvals:
    xy = gam_num(tv_)
    d1 = np.array([-2 * np.sin(tv_), np.cos(tv_)])
    d2 = np.array([-2 * np.cos(tv_), -np.sin(tv_)])
    k = (d1[0] * d2[1] - d2[0] * d1[1]) / np.linalg.norm(d1) ** 3
    c_hat = xy + np.array([-d1[1], d1[0]]) / np.linalg.norm(d1) / k
    G = p0 + xy[0] * a1 + xy[1] * a2
    D1 = d1[0] * a1 + d1[1] * a2
    D2 = d2[0] * a1 + d2[1] * a2
    # 단위속력으로 바꾼 가속도: γ''_s = (D2 − <D2, T>T)/|D1|², T = D1/|D1|
    T = D1 / np.linalg.norm(D1)
    acc = (D2 - (D2 @ T) * T) / (D1 @ D1)
    ok &= abs(np.linalg.norm(acc) - abs(k)) < 1e-12
    ok &= np.allclose(G + acc / (acc @ acc), p0 + c_hat[0] * a1 + c_hat[1] * a2)
check("명제 4.5.9: |γ''| = |κ_s|, c = γ + γ''/|γ''|² (무작위 평면, 타원)", ok)

# 예 4.5.10, 4.5.11: 구면의 대원과 위도원 (EXAMPLES["sphere"]) --------------------------------------
e = EX["sphere"]
tht, phi = e["coords"]
(rs,) = e["params"]
X = e["expr"]
th0 = sp.symbols("theta0", positive=True)
ph0 = sp.symbols("phi0", real=True)
dom = {th0: (0.1, 3.0), rs: (0.5, 2), ph0: (0, 6)}
# 위도원: 호의 길이 매개화 φ = s/(r sin θ0)
lat = X.subs(tht, th0)
rho0 = rs * sp.sin(th0)
lat_s = lat.subs(phi, s / rho0)
sym_equal("예 4.5.11: 위도원의 호의 길이 매개화는 단위속력", sp.simplify(lat_s.diff(s).dot(lat_s.diff(s))), 1, dom)
acc = lat_s.diff(s, 2)
sym_equal("예 4.5.11: |γ''|² = 1/(r sin θ0)²", sp.simplify(acc.dot(acc)), 1 / (rs * sp.sin(th0)) ** 2, dom)
c_lat = sp.simplify(lat_s + acc / acc.dot(acc))
sym_equal("예 4.5.11: 곡률중심 (0, 0, r cos θ0)", c_lat, sp.Matrix([0, 0, rs * sp.cos(th0)]), dom)
k_dg, tau_dg = dgsym.curvature_torsion(lat, phi, (sp.sin(th0),))
sym_equal("예 4.5.11: dgsym κ (공간곡선 공식) = 1/(r sin θ0)", k_dg, 1 / (rs * sp.sin(th0)), dom)
sym_equal("예 4.5.11: 위도원은 평면곡선 (dgsym τ = 0)", tau_dg, 0, dom)
Nout = lat_s / rs                              # 바깥쪽 단위법벡터 N = x/r (§7)
cosang = sp.simplify((acc / sp.sqrt(acc.dot(acc))).dot(-Nout))
sym_equal("예 4.5.11: 곡률벡터와 안쪽 법선 사이 각의 cos = sin θ0", cosang, sp.sin(th0), dom)
# 앞으로 볼 것 상자: 법선 성분 1/r, 접선 성분의 크기 |cot θ0|/r, 길이 × 접선 성분 = 2π cos θ0
normal_comp = sp.simplify(acc.dot(-Nout))
sym_equal("상자(8·9장 예고): 법선 성분 = 1/r", normal_comp, 1 / rs, dom)
tang = sp.simplify(acc - normal_comp * (-Nout))
sym_equal("상자(8·9장 예고): 접선 성분의 크기² = cot²θ0/r²", sp.simplify(tang.dot(tang)), sp.cot(th0) ** 2 / rs ** 2, dom)
sym_equal("상자(9·10장 예고): 2πr sin θ0 · cot θ0/r = 2π cos θ0", 2 * sp.pi * rs * sp.sin(th0) * sp.cot(th0) / rs,
          2 * sp.pi * sp.cos(th0), dom)
sym_equal("상자(10장 예고): 2π cos θ0 + (1/r²)·2πr²(1 − cos θ0) = 2π",
          2 * sp.pi * sp.cos(th0) + 2 * sp.pi * (1 - sp.cos(th0)), 2 * sp.pi)
# 대원: 경선 φ = φ0, 호의 길이 θ = s/r
mer_s = X.subs(phi, ph0).subs(tht, s / rs)
accm = mer_s.diff(s, 2)
sym_equal("예 4.5.10: 경선의 |γ''| = 1/r", sp.simplify(accm.dot(accm)), 1 / rs ** 2, {rs: (0.5, 2), s: (0.1, 3), ph0: (0, 6)})
sym_equal("예 4.5.10: 대원의 곡률중심 = O", sp.simplify(mer_s + accm / accm.dot(accm)), sp.zeros(3, 1), {rs: (0.5, 2), s: (0.1, 3), ph0: (0, 6)})
k_mer, _ = dgsym.curvature_torsion(X.subs(phi, ph0), tht, (sp.sin(tht),))
sym_equal("예 4.5.10: dgsym κ(경선) = 1/r (§7의 대원)", k_mer, 1 / rs, {rs: (0.5, 2), tht: (0.1, 3), ph0: (0, 6)})
check("예 4.5.11: θ0 = π/2(적도)일 때만 κ = 1/r", sp.solve(sp.Eq(1 / (rs * sp.sin(th0)), 1 / rs), th0) == [sp.pi / 2])

# 연습 4.5.1: y = x² ------------------------------------------------------------------
sym_equal("연습 4.5.1: y = x²의 축폐선 (−4x³, 3x² + 1/2)", center(sp.Matrix([t, t ** 2]), t), sp.Matrix([-4 * t ** 3, 3 * t ** 2 + sp.Rational(1, 2)]))
# 연습 4.5.3: 곡률 2/r인 위도원
check("연습 4.5.3: sin θ0 = 1/2 → θ0 = π/6, 5π/6", set(sp.solve(sp.Eq(sp.sin(th0), sp.Rational(1, 2)), th0)) == {sp.pi / 6, 5 * sp.pi / 6})

# 연습 4.5.6: c' = (1/κ_s)' n_s (단위속력 곡선) -------------------------------------------------
Tn = sp.Matrix([sp.cos(thf(s)), sp.sin(thf(s))])
gamma_s = sp.Matrix([sp.Integral(sp.cos(thf(u)), (u, 0, s)), sp.Integral(sp.sin(thf(u)), (u, 0, s))])
c_s = gamma_s + (Jm * Tn) / thf(s).diff(s)
sym_equal("연습 4.5.6: c' = (1/κ_s)' n_s", sp.simplify(c_s.diff(s)), sp.diff(1 / thf(s).diff(s), s) * (Jm * Tn))

# 연습 4.5.7: 구면 위 단위속력 곡선에서 <γ, γ''> = −1 ------------------------------------------------
check("연습 4.5.7: 위도원에서 <γ, γ''> = −1", sp.simplify(lat_s.dot(acc) + 1) == 0)

# 연습 4.5.8: 이웃한 법선의 교점 → 곡률중심 (타원, 수치) --------------------------------------------
def unit_speed_frame(tv_):
    d1 = np.array([-2 * np.sin(tv_), np.cos(tv_)])
    T = d1 / np.linalg.norm(d1)
    return gam_num(tv_), T, np.array([-T[1], T[0]])


g0, T0, N0 = unit_speed_frame(np.pi / 6)
errs = []
for h in (0.1, 0.01, 0.001):
    g1, T1, N1 = unit_speed_frame(np.pi / 6 + h)
    lamb, mu = np.linalg.solve(np.array([N0, -N1]).T, g1 - g0)
    errs.append(np.linalg.norm(g0 + lamb * N0 - cT))
check("연습 4.5.8: 법선의 교점이 곡률중심으로 수렴", errs[-1] < 1e-2 and errs[0] > errs[1] > errs[2])

summary()
