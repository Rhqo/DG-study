"""4.4절 평면곡선의 부호곡률: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

부호곡률은 dgsym에 아직 구현되어 있지 않다(최종 보고에서 dgsym.signed_curvature 추가를 제안).
그래서 이 스크립트는 κ_s를 **정의** (호의 길이 매개화에서 t' = κ_s n_s, n_s = Jt)로부터 계산하고,
명제 4.4.5의 공식과 비교하며, 그 절댓값을 dgsym.curvature_torsion의 κ와 교차검증한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/verify/v-4-4-signed-curvature.py``
"""

import numpy as np
import sympy as sp

import dgnum
import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
t, s, u = sp.symbols("t s u", real=True)
Jm = sp.Matrix([[0, -1], [1, 0]])


def J(v):
    return Jm * sp.Matrix(v)


def kappa_s_formula(g, var):
    """명제 4.4.5: κ_s = <γ'', Jγ'>/|γ'|³ (평면곡선, 2×1)."""
    g = sp.Matrix(g)
    d1, d2 = g.diff(var), g.diff(var, 2)
    return d2.dot(J(d1)) / sp.sqrt(d1.dot(d1)) ** 3


def kappa_s_definition_unit_speed(g, var):
    """정의 4.4.2: 단위속력 곡선에서 κ_s = <t', Jt>, t = γ'."""
    g = sp.Matrix(g)
    tt = g.diff(var)
    return tt.diff(var).dot(J(tt))


def kappa_unsigned_dgsym(g2, var, positive=()):
    """dgsym.curvature_torsion의 κ (평면곡선을 z = 0으로 올려서)."""
    g3 = sp.Matrix([g2[0], g2[1], 0])
    k, _ = dgsym.curvature_torsion(g3, var, positive)
    return k


# 예 4.4.3: 원 (기준 예제, EXAMPLES["circle"]) ----------------------------------------------
e = EX["circle"]
(tc,) = e["coords"]
(r,) = e["params"]
circ = e["expr"][:2, 0]
circ_s = circ.subs(tc, s / r)                                  # 호의 길이 매개화 (예 4.3.8)
sym_equal("예 4.4.3: 반시계 방향 원 κ_s = 1/r (정의로)", kappa_s_definition_unit_speed(circ_s, s), 1 / r, e["domain"])
sym_equal("예 4.4.3: n_s = Jt는 중심을 향한다: γ + r n_s = 0", circ_s + r * J(circ_s.diff(s)), sp.zeros(2, 1))
circ_cw = circ_s.subs(s, -s)
sym_equal("예 4.4.3: 시계 방향 원 κ_s = −1/r", kappa_s_definition_unit_speed(circ_cw, s), -1 / r, e["domain"])
sym_equal("예 4.4.3: §7 κ = 1/r (dgsym으로 교차검증)", kappa_unsigned_dgsym(circ, tc), e["expected"]["kappa"], e["domain"])
line = sp.Matrix([1 + s / sp.sqrt(2), 2 - s / sp.sqrt(2)])
check("예 4.4.3: 직선 κ_s = 0", sp.simplify(kappa_s_definition_unit_speed(line, s)) == 0)

# 비예 4.4.4: t에 대해 미분하면 틀린다 --------------------------------------------------------------
tv = circ.diff(tc) / r                                          # 단위접벡터 (r > 0)
sym_equal("비예 4.4.4: dt/dt = n_s (r과 무관)", tv.diff(tc), J(tv), e["domain"])
sym_equal("비예 4.4.4: dt/ds = (1/r) n_s", tv.diff(tc) / r, J(tv) / r, e["domain"])

# 명제 4.4.5: 일반 매개변수 공식 = 호의 길이 재매개화에서 계산한 값 (일반 함수) ----------------------------
x = sp.Function("x")(t)
y = sp.Function("y")(t)
G = sp.Matrix([x, y])
formula = kappa_s_formula(G, t)
sym_equal("명제 4.4.5: <γ'', Jγ'> = x'y'' − x''y'", sp.Matrix(G.diff(t, 2)).dot(J(G.diff(t))),
          x.diff(t) * y.diff(t, 2) - x.diff(t, 2) * y.diff(t))
# 연쇄법칙 검증: β = γ∘h, h' = 1/|γ'(h)|, h'' = d/ds(1/|γ'(h)|) = −<γ',γ''>/|γ'|⁴ · h'
# β' = h'γ', β'' = h''γ' + h'²γ''. <β'', Jβ'> = h'³ <γ'', Jγ'> (<γ', Jγ'> = 0).
hp = 1 / sp.sqrt(G.diff(t).dot(G.diff(t)))
hpp = sp.diff(hp, t) * hp
bp = hp * G.diff(t)
bpp = hpp * G.diff(t) + hp ** 2 * G.diff(t, 2)
sym_equal("명제 4.4.5: <β'', Jβ'> (연쇄법칙) = 공식", bpp.dot(J(bp)), formula)
sym_equal("명제 4.4.5: |β'| = 1", bp.dot(bp), 1)
sym_equal("명제 4.4.5: <v, Jv> = 0", G.diff(t).dot(J(G.diff(t))), 0)
# dt/dt = |γ'| κ_s n_s
Tt = G.diff(t) / sp.sqrt(G.diff(t).dot(G.diff(t)))
sym_equal("명제 4.4.5: dt/dt = |γ'| κ_s n_s", Tt.diff(t), sp.sqrt(G.diff(t).dot(G.diff(t))) * formula * J(Tt))

# 예 4.4.6 ---------------------------------------------------------------------------
f = sp.Function("f")(t)
graph = sp.Matrix([t, f])
sym_equal("예 4.4.6(a): 그래프 κ_s = f''/(1 + f'²)^{3/2}", kappa_s_formula(graph, t),
          f.diff(t, 2) / (1 + f.diff(t) ** 2) ** sp.Rational(3, 2))
par = sp.Matrix([t, t ** 2 / 2])
sym_equal("예 4.4.6(b): 포물선 y = x²/2의 κ_s = (1 + x²)^{−3/2}", kappa_s_formula(par, t), (1 + t ** 2) ** sp.Rational(-3, 2))
tp = sp.symbols("t", positive=True)
cusp = sp.Matrix([tp ** 2, tp ** 3])
sym_equal("예 4.4.6(c): 뾰족점 곡선 (t > 0) κ_s = 6/(t(4 + 9t²)^{3/2})", kappa_s_formula(cusp, tp),
          6 / (tp * (4 + 9 * tp ** 2) ** sp.Rational(3, 2)))
tn = sp.symbols("t", negative=True)
cusp_n = sp.Matrix([tn ** 2, tn ** 3])
sym_equal("예 4.4.6(c): 뾰족점 곡선 (t < 0) κ_s = 6/(|t|(4 + 9t²)^{3/2}) > 0", kappa_s_formula(cusp_n, tn),
          6 / (-tn * (4 + 9 * tn ** 2) ** sp.Rational(3, 2)))
sine = sp.Matrix([t, sp.sin(t)])
sym_equal("예 4.4.6(d): y = sin x의 κ_s = −sin x/(1 + cos²x)^{3/2}", kappa_s_formula(sine, t),
          -sp.sin(t) / (1 + sp.cos(t) ** 2) ** sp.Rational(3, 2))
sym_equal("예 4.4.6: |κ_s| = dgsym의 κ (포물선)", kappa_unsigned_dgsym(par, t), (1 + t ** 2) ** sp.Rational(-3, 2))

# 명제 4.4.7: 재매개화, 강체운동, 반사 -------------------------------------------------------------
# κ^β(u) = sgn(h') κ^γ(h(u)): h' > 0 인 경우와 h' < 0 인 경우를 구체적 h로
for name, hfun, sgn in (("h = u³ + u (보존)", u ** 3 + u, 1), ("h = −u³ − u (뒤집음)", -u ** 3 - u, -1)):
    circ_h = circ.subs(tc, hfun)
    sym_equal(f"명제 4.4.7(a): 원, {name}: κ_s = {sgn:+d}/r", kappa_s_formula(circ_h, u), sgn / r, {u: (-1, 1), r: (0.5, 2)})
al = sp.symbols("alpha", real=True)
R = sp.Matrix([[sp.cos(al), -sp.sin(al)], [sp.sin(al), sp.cos(al)]])
c1, c2 = sp.symbols("c1 c2", real=True)
psiG = R * G + sp.Matrix([c1, c2])
sym_equal("명제 4.4.7(b): RJ = JR", R * Jm, Jm * R)
sym_equal("명제 4.4.7(b): <Rv, Rw> = <v, w>", (R * sp.Matrix([x, y])).dot(R * sp.Matrix([f, t])), x * f + y * t)
sym_equal("명제 4.4.7(b): 강체운동은 κ_s를 보존", kappa_s_formula(psiG, t), formula)
rho = sp.Matrix([[1, 0], [0, -1]])
sym_equal("명제 4.4.7(c): ρJ = −Jρ", rho * Jm, -Jm * rho)
sym_equal("명제 4.4.7(c): 반사는 κ_s의 부호를 바꾼다", kappa_s_formula(rho * G, t), -formula)

# 보조정리 4.4.8 / 정리 4.4.9: 접선각 -------------------------------------------------------------
th = sp.Function("theta")(s)
Tth = sp.Matrix([sp.cos(th), sp.sin(th)])
sym_equal("정리 4.4.9: t = (cos θ, sin θ)이면 t' = θ' Jt", Tth.diff(s), th.diff(s) * J(Tth))
sym_equal("정리 4.4.9: <t', Jt> = θ'", Tth.diff(s).dot(J(Tth)), th.diff(s))
# 보조정리의 P, Q가 상수: a' = −θ'b, b' = θ'a 이면 P' = Q' = 0
a_, b_ = sp.Function("a")(s), sp.Function("b")(s)
subs_ab = {a_.diff(s): -th.diff(s) * b_, b_.diff(s): th.diff(s) * a_}
Pf = a_ * sp.cos(th) + b_ * sp.sin(th)
Qf = b_ * sp.cos(th) - a_ * sp.sin(th)
sym_equal("보조정리 4.4.8: P' = 0", sp.diff(Pf, s).subs(subs_ab), 0)
sym_equal("보조정리 4.4.8: Q' = 0", sp.diff(Qf, s).subs(subs_ab), 0)
sym_equal("보조정리 4.4.8: a = P cos θ − Q sin θ", Pf * sp.cos(th) - Qf * sp.sin(th), a_)
sym_equal("보조정리 4.4.8: b = P sin θ + Q cos θ", Pf * sp.sin(th) + Qf * sp.cos(th), b_)
# 원: 한 바퀴의 전체 곡률 2π
check("정리 4.4.9: 원 한 바퀴 ∫κ_s ds = (1/r)(2πr) = 2π", sp.simplify(sp.integrate(1 / r, (s, 0, 2 * sp.pi * r)) - 2 * sp.pi) == 0)

# 정리 4.4.10 / 예 4.4.12 ------------------------------------------------------------------
th1 = sp.integrate(sp.Integer(1), (u, 0, s))
g1 = sp.Matrix([sp.integrate(sp.cos(u), (u, 0, s)), sp.integrate(sp.sin(u), (u, 0, s))])
sym_equal("예 4.4.12(a): κ_s = 1 → γ(s) = (sin s, 1 − cos s)", g1, sp.Matrix([sp.sin(s), 1 - sp.cos(s)]))
sym_equal("예 4.4.12(a): 중심 (0, 1), 반지름 1", (g1 - sp.Matrix([0, 1])).dot(g1 - sp.Matrix([0, 1])), 1)
sym_equal("정리 4.4.10: 재구성한 곡선의 κ_s = 1 (정의로)", kappa_s_definition_unit_speed(g1, s), 1)
# 클로소이드 (수치)
ss = np.linspace(0, 8, 16001)
Y = dgnum.rk4(lambda sv, yv: np.array([sv, np.cos(yv[0]), np.sin(yv[0])]), [0, 0, 0], ss)
close("예 4.4.12(b): 클로소이드 θ(s) = s²/2", Y[:, 0], ss ** 2 / 2, 1e-9)
lim = np.sqrt(np.pi) / 2
check("예 4.4.12(b): γ(8)이 극한점 (√π/2, √π/2)에서 1/8 이내", np.hypot(Y[-1, 1] - lim, Y[-1, 2] - lim) < 1 / 8)
# 연습 4.4.7: 접선이 π/2 돌 때 s = √π
check("연습 4.4.7: θ(√π) = π/2", sp.simplify((s ** 2 / 2).subs(s, sp.sqrt(sp.pi)) - sp.pi / 2) == 0)
# 대칭 γ(−s) = −γ(s): 기호적으로
w = sp.symbols("w", real=True)
integrand = sp.cos(w ** 2 / 2)
check("연습 4.4.7: cos(u²/2), sin(u²/2)는 짝함수 → γ는 홀함수",
      sp.simplify(integrand.subs(w, -w) - integrand) == 0 and sp.simplify(sp.sin(w ** 2 / 2).subs(w, -w) - sp.sin(w ** 2 / 2)) == 0)

# 연습 4.4.1: 타원 ------------------------------------------------------------------------
A_, B_ = sp.symbols("A B", positive=True)
ell = sp.Matrix([A_ * sp.cos(t), B_ * sp.sin(t)])
k_ell = sp.simplify(kappa_s_formula(ell, t))
sym_equal("연습 4.4.1: 타원 κ_s = AB/(A² sin²t + B² cos²t)^{3/2}", k_ell,
          A_ * B_ / (A_ ** 2 * sp.sin(t) ** 2 + B_ ** 2 * sp.cos(t) ** 2) ** sp.Rational(3, 2))
sym_equal("연습 4.4.1: t = 0에서 κ_s = A/B²", k_ell.subs(t, 0), A_ / B_ ** 2)
sym_equal("연습 4.4.1: t = π/2에서 κ_s = B/A²", k_ell.subs(t, sp.pi / 2), B_ / A_ ** 2)

# 연습 4.4.2: y = e^x ----------------------------------------------------------------------
kexp = kappa_s_formula(sp.Matrix([t, sp.exp(t)]), t)
sym_equal("연습 4.4.2: κ_s = e^x/(1 + e^{2x})^{3/2}", kexp, sp.exp(t) / (1 + sp.exp(2 * t)) ** sp.Rational(3, 2))
xm = -sp.log(2) / 2
check("연습 4.4.2: κ_s'(−ln2/2) = 0", sp.simplify(sp.diff(kexp, t).subs(t, xm)) == 0)
sym_equal("연습 4.4.2: 최댓값 2/(3√3)", kexp.subs(t, xm), 2 / (3 * sp.sqrt(3)))

# 연습 4.4.4: 로그나선 ----------------------------------------------------------------------
lsp = sp.Matrix([sp.exp(t) * sp.cos(t), sp.exp(t) * sp.sin(t)])
sym_equal("연습 4.4.4: κ_s = e^{−t}/√2", kappa_s_formula(lsp, t), sp.exp(-t) / sp.sqrt(2))
d1 = lsp.diff(t)
sym_equal("연습 4.4.4: cos(γ, γ') = 1/√2", lsp.dot(d1) / sp.sqrt(lsp.dot(lsp) * d1.dot(d1)), 1 / sp.sqrt(2))

# 연습 4.4.5: κ_s ≡ c → γ + n_s/c 는 상수 ------------------------------------------------------
cc = sp.symbols("c", nonzero=True, real=True)
Tc = sp.Matrix([sp.cos(cc * s), sp.sin(cc * s)])     # θ' = c 인 단위접벡터
sym_equal("연습 4.4.5: (γ + n_s/c)' = t + (Jt)'/c = 0", Tc + J(Tc).diff(s) / cc, sp.zeros(2, 1))

# 연습 4.4.6: 그래프의 접선각 arctan f' ------------------------------------------------------------
fp = f.diff(t)
theta_g = sp.atan(fp)
sym_equal("연습 4.4.6: (cos θ, sin θ) = (1, f')/√(1 + f'²)", sp.Matrix([sp.cos(theta_g), sp.sin(theta_g)]),
          sp.Matrix([1, fp]) / sp.sqrt(1 + fp ** 2))
sym_equal("연습 4.4.6: (dθ/dx)/(ds/dx) = 그래프 공식", sp.diff(theta_g, t) / sp.sqrt(1 + fp ** 2),
          f.diff(t, 2) / (1 + fp ** 2) ** sp.Rational(3, 2))

# 연습 4.4.8: 리마송 ρ = 1 + 2cos t 의 전체 곡률 4π --------------------------------------------------
rho_ = 1 + 2 * sp.cos(t)
lim_c = rho_ * sp.Matrix([sp.cos(t), sp.sin(t)])
d1, d2 = lim_c.diff(t), lim_c.diff(t, 2)
sym_equal("연습 4.4.8: <γ'', Jγ'> = ρ² + 2ρ'² − ρρ'' = 9 + 6 cos t", d2.dot(J(d1)), 9 + 6 * sp.cos(t))
sym_equal("연습 4.4.8: |γ'|² = ρ² + ρ'² = 5 + 4 cos t", d1.dot(d1), 5 + 4 * sp.cos(t))
check("연습 4.4.8: ∫_0^{2π} dt/(5 + 4cos t) = 2π/3",
      sp.simplify(sp.integrate(1 / (5 + 4 * sp.cos(t)), (t, 0, 2 * sp.pi)) - 2 * sp.pi / 3) == 0)
tot = sp.Integral((9 + 6 * sp.cos(t)) / (5 + 4 * sp.cos(t)), (t, 0, 2 * sp.pi)).evalf(15)
close("연습 4.4.8: 전체 곡률 ∫κ_s|γ'|dt = 4π", float(tot), 4 * np.pi, 1e-10)

summary()
