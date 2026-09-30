"""5.1절 공간곡선의 곡률: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

기준 예제(원, 나선, 구면)는 dgsym.EXAMPLES에서 가져오고, 곡률은 dgsym.curvature_torsion·frenet_frame과
정의(호의 길이 매개화의 |γ''|)의 두 경로로 비교한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/verify/v-5-1-curvature.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
s, t, u, h = sp.symbols("s t u h", real=True)


def norm(v):
    return sp.sqrt(sp.simplify(v.dot(v)))


def kappa_def(g, var):
    """명제 5.1.2(a): κ = |dt/dt| / |γ'| (일반 매개변수)."""
    g = sp.Matrix(g)
    d1 = g.diff(var)
    T = d1 / norm(d1)
    return sp.simplify(norm(sp.simplify(T.diff(var))) / norm(d1))


# 예 5.1.3: 직선과 원 ---------------------------------------------------------------
e = EX["circle"]
(tc,) = e["coords"]
(r,) = e["params"]
circ_s = sp.Matrix([r * sp.cos(s / r), r * sp.sin(s / r), 0])
sym_equal("예 5.1.3: 원의 호의 길이 매개화는 단위속력", circ_s.diff(s).dot(circ_s.diff(s)), 1)
sym_equal("예 5.1.3: 원의 κ = |γ''| = 1/r", norm(circ_s.diff(s, 2)), 1 / r, {r: (0.5, 2)})
k_c, _ = dgsym.curvature_torsion(e["expr"], tc)
sym_equal("예 5.1.3: dgsym의 원 κ = §7 기대값 1/r", k_c, e["expected"]["kappa"], e["domain"])
v = sp.Matrix(sp.symbols("v1:4", real=True))
p0 = sp.Matrix(sp.symbols("p1:4", real=True))
check("예 5.1.3: 직선 p + sv는 γ'' = 0", (p0 + s * v).diff(s, 2) == sp.zeros(3, 1))

# 명제 5.1.2: 일반 매개변수 공식과 재매개화 불변성 ----------------------------------------------
eh = EX["helix"]
(th,) = eh["coords"]
a, b = eh["params"]
dom = eh["domain"]
sym_equal("명제 5.1.2(a): 나선에서 |dt/dt|/|γ'| = a/(a²+b²)", kappa_def(eh["expr"], th), eh["expected"]["kappa"], dom)
# (b) 방향을 뒤집는 재매개화 h(u) = -2u + 1 (나선)
hel_rev = eh["expr"].subs(th, -2 * u + 1)
sym_equal("명제 5.1.2(b): 방향을 뒤집는 재매개화에서 κ^β(u) = κ(h(u))", kappa_def(hel_rev, u),
          eh["expected"]["kappa"], {a: (0.5, 2), b: (0.3, 1.5), u: (-2, 2)})
hel_cub = eh["expr"].subs(th, u ** 3 + u)                     # h' = 3u² + 1 > 0
sym_equal("명제 5.1.2(b): 재매개화 h(u) = u³ + u에서도 κ는 같다", kappa_def(hel_cub, u),
          eh["expected"]["kappa"], {a: (0.5, 2), b: (0.3, 1.5), u: (-1.5, 1.5)})

# 비예 5.1.4: |γ''(t)|은 곡률이 아니다 --------------------------------------------------------
sym_equal("비예 5.1.4: 나선의 |γ''(t)| = a", norm(eh["expr"].diff(th, 2)), a, dom)
check("비예 5.1.4: a ≠ a/(a²+b²) (a = 1, b = 0.3)", abs(1.0 - 1 / 1.09) > 0.05)

# 명제 5.1.5: 평면곡선에서 κ = |κ_s| ---------------------------------------------------------
A_, B_ = sp.symbols("A B", positive=True)
ell = sp.Matrix([A_ * sp.cos(t), B_ * sp.sin(t), 0])
k_e, _ = dgsym.curvature_torsion(ell, t)
sym_equal("명제 5.1.5: 타원에서 κ = |κ_s|", k_e, sp.Abs(dgsym.signed_curvature(ell, t)), {A_: (1.2, 3), B_: (0.3, 1.1), t: (0, 6.2)})
sine = sp.Matrix([t, sp.sin(t), 0])
k_sn, _ = dgsym.curvature_torsion(sine, t)
sym_equal("명제 5.1.5: 사인 곡선에서 κ = |κ_s|", k_sn, sp.Abs(dgsym.signed_curvature(sine, t)), {t: (-3, 3)})

# 예 5.1.7: 나선의 곡률 (호의 길이 매개화 (4.3.2)) ---------------------------------------------
c = sp.sqrt(a ** 2 + b ** 2)
beta = sp.Matrix([a * sp.cos(s / c), a * sp.sin(s / c), b * s / c])
sym_equal("예 5.1.7: (4.3.2)는 단위속력", beta.diff(s).dot(beta.diff(s)), 1)
sym_equal("예 5.1.7: β'' = -(a/c²)(cos(s/c), sin(s/c), 0)", beta.diff(s, 2),
          -(a / c ** 2) * sp.Matrix([sp.cos(s / c), sp.sin(s / c), 0]))
sym_equal("예 5.1.7: κ = a/(a²+b²) = §7 기대값", norm(beta.diff(s, 2)), eh["expected"]["kappa"], {a: (0.5, 2), b: (-1.5, 1.5)})
k_h, _ = dgsym.curvature_torsion(eh["expr"], th)
sym_equal("예 5.1.7: dgsym.curvature_torsion과 일치", k_h, eh["expected"]["kappa"], dom)
sym_equal("예 5.1.7: 곡률반지름 (a²+b²)/a인 원과 같은 곡률", 1 / eh["expected"]["kappa"], a + b ** 2 / a)
# 곡률이 같은 원과 나선의 수치 예: a = b = 1이면 κ = 1/2 = 반지름 2인 원
close("예 5.1.7: a = b = 1이면 κ = 1/2", float(eh["expected"]["kappa"].subs({a: 1, b: 1})), 0.5)

# 예 5.1.10: 주법선벡터 ----------------------------------------------------------------------
n_beta = beta.diff(s, 2) / norm(beta.diff(s, 2))
sym_equal("예 5.1.10: 나선의 n = -(cos(s/c), sin(s/c), 0)", n_beta, -sp.Matrix([sp.cos(s / c), sp.sin(s / c), 0]),
          {a: (0.5, 2), b: (-1.5, 1.5), s: (-5, 5)})
sym_equal("예 5.1.10: 주법선 직선은 λ = a에서 z축의 점 (0, 0, bs/c)를 지난다", beta + a * n_beta, sp.Matrix([0, 0, b * s / c]),
          {a: (0.5, 2), b: (-1.5, 1.5), s: (-5, 5)})
check("예 5.1.10: n은 수평(<n, e3> = 0)", sp.simplify(n_beta[2]) == 0)
Tf, Nf, Bf = dgsym.frenet_frame(eh["expr"], th)
sym_equal("예 5.1.10: dgsym.frenet_frame의 n과 일치", Nf, -sp.Matrix([sp.cos(th), sp.sin(th), 0]), dom)
sym_equal("예 5.1.10: 원의 n = -γ/r", circ_s.diff(s, 2) / norm(circ_s.diff(s, 2)), -circ_s / r, {r: (0.5, 2), s: (-5, 5)})

# 비예 5.1.11: (t, t³, 0)의 주법선은 t = 0에서 뒤집힌다 ---------------------------------------------
cub = sp.Matrix([t, t ** 3, 0])
_, Nc, _ = dgsym.frenet_frame(cub, t)
Ncf = sp.lambdify(t, list(Nc), "numpy")
close("비예 5.1.11: t → 0+에서 n → (0, 1, 0)", np.array(Ncf(1e-7), float), np.array([0, 1, 0.0]), tol=1e-6)
close("비예 5.1.11: t → 0-에서 n → (0, -1, 0)", np.array(Ncf(-1e-7), float), np.array([0, -1, 0.0]), tol=1e-6)
k_cub, _ = dgsym.curvature_torsion(cub, t)
check("비예 5.1.11: κ(0) = 0", sp.simplify(k_cub.subs(t, 0)) == 0)
sym_equal("비예 5.1.11: t > 0에서 n = (-3t², 1, 0)/√(1+9t⁴)", Nc, sp.Matrix([-3 * t ** 2, 1, 0]) / sp.sqrt(1 + 9 * t ** 4), {t: (0.05, 2)})

# 정의 5.1.12와 연습 5.1.4: 곡률중심 --------------------------------------------------------------
cen = beta + beta.diff(s, 2) / beta.diff(s, 2).dot(beta.diff(s, 2))
sym_equal("정의 5.1.12/연습 5.1.4: 나선의 곡률중심 (-(b²/a)cos, -(b²/a)sin, bs/c)", cen,
          sp.Matrix([-(b ** 2 / a) * sp.cos(s / c), -(b ** 2 / a) * sp.sin(s / c), b * s / c]),
          {a: (0.5, 2), b: (-1.5, 1.5), s: (-5, 5)})
cen_t = cen.subs(s, c * u)                      # u = s/c
sym_equal("연습 5.1.4: 곡률중심의 자취는 반지름 b²/a, 높이 증가율 b인 나선", cen_t,
          sp.Matrix([(b ** 2 / a) * sp.cos(u + sp.pi), (b ** 2 / a) * sp.sin(u + sp.pi), b * u]),
          {a: (0.5, 2), b: (-1.5, 1.5), u: (-5, 5)})

# 정리 5.1.14: 접촉평면과 접촉원의 근사 차수 (나선, s0 = 0) ------------------------------------------------
t0 = beta.diff(s).subs(s, 0)
n0 = n_beta.subs(s, 0)
nu_osc = t0.cross(n0)
k0 = eh["expected"]["kappa"]
d_osc = sp.series(sp.simplify((beta - beta.subs(s, 0)).dot(nu_osc)), s, 0, 4).removeO()
check("정리 5.1.14(a): 접촉평면까지의 부호 거리는 s³부터 시작",
      sp.simplify(d_osc.coeff(s, 0)) == 0 and sp.simplify(d_osc.coeff(s, 1)) == 0 and sp.simplify(d_osc.coeff(s, 2)) == 0)
d_n = sp.series(sp.simplify((beta - beta.subs(s, 0)).dot(n0)), s, 0, 3).removeO()
sym_equal("정리 5.1.14(a): 다른 평면(법선 n0)은 d ~ κ s²/2", d_n.coeff(s, 2), k0 / 2, {a: (0.5, 2), b: (0.3, 1.5)})
sig = (beta.subs(s, 0) + n0 / k0) + (1 / k0) * (-sp.cos(k0 * s) * n0 + sp.sin(k0 * s) * t0)
sym_equal("정리 5.1.14(b): σ(s0) = γ(s0)", sig.subs(s, 0), beta.subs(s, 0), {a: (0.5, 2), b: (0.3, 1.5)})
sym_equal("정리 5.1.14(b): σ'(s0) = t0", sig.diff(s).subs(s, 0), t0, {a: (0.5, 2), b: (0.3, 1.5)})
sym_equal("정리 5.1.14(b): σ''(s0) = κ n0 = γ''(s0)", sig.diff(s, 2).subs(s, 0), beta.diff(s, 2).subs(s, 0), {a: (0.5, 2), b: (0.3, 1.5)})
sym_equal("정리 5.1.14(b): σ는 단위속력", sp.simplify(sig.diff(s).dot(sig.diff(s))), 1, {a: (0.5, 2), b: (0.3, 1.5), s: (-3, 3)})
sym_equal("정리 5.1.14(b): |σ - c0| = 1/κ", sp.simplify((sig - beta.subs(s, 0) - n0 / k0).dot(sig - beta.subs(s, 0) - n0 / k0)),
          1 / k0 ** 2, {a: (0.5, 2), b: (0.3, 1.5), s: (-3, 3)})
num = {a: 1, b: sp.Rational(3, 10)}
err = (beta - sig).subs(num)
errf = sp.lambdify(s, list(err), "numpy")
for hh in (0.1, 0.05, 0.025):
    ratio = np.linalg.norm(np.array(errf(hh), float)) / hh ** 3
    check(f"정리 5.1.14(b): |γ - σ|/h³ 유계 (h = {hh}, 비 {ratio:.4f})", ratio < 0.2)

# 연습 5.1.1: (cosh t, sinh t, t) ---------------------------------------------------------
ch = sp.Matrix([sp.cosh(t), sp.sinh(t), t])
T_ch = ch.diff(t) / norm(ch.diff(t))
sym_equal("연습 5.1.1: t = (tanh t, 1, sech t)/√2", T_ch, sp.Matrix([sp.tanh(t), 1, 1 / sp.cosh(t)]) / sp.sqrt(2), {t: (-2, 2)})
sym_equal("연습 5.1.1: κ = 1/(2cosh² t)", kappa_def(ch, t), 1 / (2 * sp.cosh(t) ** 2), {t: (-2, 2)})
k_ch, _ = dgsym.curvature_torsion(ch, t)
sym_equal("연습 5.1.1: dgsym과 일치", k_ch, 1 / (2 * sp.cosh(t) ** 2), {t: (-2, 2)})

# 연습 5.1.2(d): (e^t, 0, 0) ---------------------------------------------------------------
ex_line = sp.Matrix([sp.exp(t), 0, 0])
check("연습 5.1.2(d): (e^t,0,0)은 γ'' ≠ 0", ex_line.diff(t, 2) != sp.zeros(3, 1))
check("연습 5.1.2(d): 그러나 dt/dt = 0이므로 κ = 0", sp.simplify((ex_line.diff(t) / norm(ex_line.diff(t))).diff(t)) == sp.zeros(3, 1))

# 연습 5.1.3: 구면 위 곡선 κ ≥ 1/r (위도원에서 확인) -------------------------------------------------
es = EX["sphere"]
tht, ph = es["coords"]
(rs,) = es["params"]
th0 = sp.symbols("theta0", positive=True)
lat = es["expr"].subs(tht, th0)
rho0 = rs * sp.sin(th0)
lat_s = lat.subs(ph, s / rho0)
sym_equal("연습 5.1.3: 위도원의 호의 길이 매개화는 단위속력", sp.simplify(lat_s.diff(s).dot(lat_s.diff(s))), 1, {rs: (0.5, 2), th0: (0.2, 3.0)})
sym_equal("연습 5.1.3: <γ, γ''> = -1", sp.simplify(lat_s.dot(lat_s.diff(s, 2))), -1, {rs: (0.5, 2), th0: (0.2, 3.0), s: (-3, 3)})
k_lat, _ = dgsym.curvature_torsion(lat, ph, (sp.sin(th0),))
sym_equal("연습 5.1.3: 위도원 κ = 1/(r sin θ0) (예 4.5.12)", k_lat, 1 / (rs * sp.sin(th0)), {rs: (0.5, 2), th0: (0.2, 3.0)})
for th_val in (0.3, 0.9, np.pi / 2, 2.4):
    check(f"연습 5.1.3: θ0 = {th_val:.3f}에서 κ ≥ 1/r", 1 / np.sin(th_val) >= 1 - 1e-15)

# 연습 5.1.5: 접선까지의 거리 --------------------------------------------------------------
dvec = beta - beta.subs(s, 0)
dist2 = sp.simplify(dvec.dot(dvec) - dvec.dot(t0) ** 2)
ser = sp.series(dist2, s, 0, 5).removeO()
sym_equal("연습 5.1.5: dist² = κ² s⁴/4 + O(s⁵) (나선)", ser.coeff(s, 4), k0 ** 2 / 4, {a: (0.5, 2), b: (0.3, 1.5)})
check("연습 5.1.5: dist²의 s⁰–s³ 계수는 0", all(sp.simplify(ser.coeff(s, j)) == 0 for j in range(4)))

# 연습 5.1.6: 접벡터 지시곡선과 전곡률 ------------------------------------------------------------
Tb = beta.diff(s)
sym_equal("연습 5.1.6: |t'| = κ", norm(Tb.diff(s)), k0, {a: (0.5, 2), b: (-1.5, 1.5)})
sym_equal("연습 5.1.6: 한 바퀴(길이 2πc)의 전곡률 = 2πa/√(a²+b²)", k0 * 2 * sp.pi * c, 2 * sp.pi * a / c)
check("연습 5.1.6: 나선의 t는 높이 b/c, 반지름 a/c인 원을 그린다",
      sp.simplify(Tb[2] - b / c) == 0 and sp.simplify(Tb[0] ** 2 + Tb[1] ** 2 - a ** 2 / c ** 2) == 0)

# 연습 5.1.7: 단위속력 원의 σ'' = -(σ - q)/ρ² --------------------------------------------------
rho, phi0 = sp.symbols("rho phi0", positive=True)
E1 = sp.Matrix([1, 0, 0]); F1 = sp.Matrix([0, sp.Rational(3, 5), sp.Rational(4, 5)])
q = sp.Matrix([1, 2, 3])
sc = q + rho * sp.cos(s / rho + phi0) * E1 + rho * sp.sin(s / rho + phi0) * F1
sym_equal("연습 5.1.7: 단위속력 원 σ'' = -(σ - q)/ρ²", sc.diff(s, 2), -(sc - q) / rho ** 2, {rho: (0.3, 2), phi0: (0.1, 3)})
sym_equal("연습 5.1.7: 단위속력", sp.simplify(sc.diff(s).dot(sc.diff(s))), 1, {rho: (0.3, 2), phi0: (0.1, 3)})

summary()
