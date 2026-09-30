"""5.2절 비틀림률: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

규약은 b' = -τ n (GUIDELINES.md §6.2). 나선·원은 dgsym.EXAMPLES에서 가져오고, 비틀림률은
정의(-<b', n>, 호의 길이 매개화), 명제 5.2.7의 공식, dgsym.curvature_torsion의 세 경로로 비교한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/verify/v-5-2-torsion.py``
"""

import math

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
s, t, u = sp.symbols("s t u", real=True)


def norm(v):
    return sp.sqrt(sp.simplify(v.dot(v)))


def frame_unit(g, var):
    """단위속력 곡선의 (t, n, b) (정의 5.1.8, 정의 5.2.1)."""
    T = g.diff(var)
    k = norm(g.diff(var, 2))
    N = sp.simplify(g.diff(var, 2) / k)
    B = sp.simplify(T.cross(N))
    return T, N, B, k


def tau_formula(g, var):
    """명제 5.2.7: 단위속력 곡선에서 τ = <γ'×γ'', γ'''>/κ²."""
    d1, d2, d3 = (g.diff(var, j) for j in (1, 2, 3))
    return sp.simplify(d1.cross(d2).dot(d3) / d2.dot(d2))


eh = EX["helix"]
(th,) = eh["coords"]
a, b = eh["params"]
c = sp.sqrt(a ** 2 + b ** 2)
beta = sp.Matrix([a * sp.cos(s / c), a * sp.sin(s / c), b * s / c])
D = {a: (0.5, 2), b: (-1.5, 1.5), s: (-6, 6)}
T, N, B, k = frame_unit(beta, s)

# 명제 5.2.2: 프레네 틀은 양의 정규직교기저 --------------------------------------------------
M = sp.Matrix.hstack(T, N, B)
sym_equal("명제 5.2.2: 나선의 (t, n, b)는 정규직교 (MᵀM = I)", sp.simplify(M.T * M), sp.eye(3), D)
sym_equal("명제 5.2.2: det(t, n, b) = 1", sp.simplify(M.det()), 1, D)
sym_equal("명제 5.2.2: n × b = t", sp.simplify(N.cross(B)), T, D)
sym_equal("명제 5.2.2: b × t = n", sp.simplify(B.cross(T)), N, D)

# 예 5.2.3: 종법선벡터 -----------------------------------------------------------------------
sym_equal("예 5.2.3: 나선 b = ((b/c) sin, -(b/c) cos, a/c)", B,
          sp.Matrix([(b / c) * sp.sin(s / c), -(b / c) * sp.cos(s / c), a / c]), D)
_, _, Bd = dgsym.frenet_frame(eh["expr"], th)
sym_equal("예 5.2.3: dgsym.frenet_frame의 b와 일치 (s = ct)", Bd, B.subs(s, c * th), {a: (0.5, 2), b: (-1.5, 1.5), th: (-3, 3)})
ec = EX["circle"]
(r,) = ec["params"]
circ = sp.Matrix([r * sp.cos(s / r), r * sp.sin(s / r), 0])
Tc, Nc, Bc, kc = frame_unit(circ, s)
check("예 5.2.3: 원의 b = e3", sp.simplify(Bc - sp.Matrix([0, 0, 1])) == sp.zeros(3, 1))

# 명제 5.2.4, 정의 5.2.5, 예 5.2.6 -------------------------------------------------------------
dB = sp.simplify(B.diff(s))
check("명제 5.2.4: <b', t> = 0", sp.simplify(dB.dot(T)) == 0)
check("명제 5.2.4: <b', b> = 0", sp.simplify(dB.dot(B)) == 0)
tau_def = sp.simplify(-dB.dot(N))
sym_equal("예 5.2.6: 나선 τ = -<b', n> = b/(a²+b²) = §7 기대값", tau_def, eh["expected"]["tau"], D)
sym_equal("예 5.2.6: b' = -τ n", dB, -tau_def * N, D)
check("예 5.2.6: 원의 b' = 0, τ = 0", sp.simplify(Bc.diff(s)) == sp.zeros(3, 1))
k_h, tau_h = dgsym.curvature_torsion(eh["expr"], th)
sym_equal("예 5.2.6: dgsym.curvature_torsion의 τ와 일치", tau_h, eh["expected"]["tau"], eh["domain"])
check("예 5.2.6: b = 0.3 > 0이면 τ > 0 (오른손 나선)", float(eh["expected"]["tau"].subs({a: 1, b: 0.3})) > 0)
close("그림 5.2.1: b = ±0.6이면 τ = ±0.441", float(eh["expected"]["tau"].subs({a: 1, b: 0.6})), 0.6 / 1.36)

# 명제 5.2.7: 단위속력 공식 --------------------------------------------------------------------
sym_equal("명제 5.2.7: 나선에서 공식 = 정의", tau_formula(beta, s), tau_def, D)
# 일반적인 단위속력 곡선에서 정의와 공식이 같음을 수치로 (구면 위 곡선을 호의 길이로)
ch = sp.Matrix([sp.cosh(t), sp.sinh(t), t])            # s = √2 sinh t
ch_s = ch.subs(t, sp.asinh(s / sp.sqrt(2)))
sym_equal("명제 5.2.7: (cosh, sinh, t)의 호의 길이 매개화는 단위속력", sp.simplify(ch_s.diff(s).dot(ch_s.diff(s))), 1, {s: (-3, 3)})
Tq, Nq, Bq, kq = frame_unit(ch_s, s)
sym_equal("명제 5.2.7: (cosh, sinh, t)에서 -<b', n> = 공식", sp.simplify(-Bq.diff(s).dot(Nq)), tau_formula(ch_s, s), {s: (-3, 3)})
sym_equal("명제 5.2.7: (cosh, sinh, t)의 τ = 1/(2+s²)", tau_formula(ch_s, s), 1 / (2 + s ** 2), {s: (-3, 3)})

# 명제 5.2.8: 재매개화와 반사 -----------------------------------------------------------------
rev = beta.subs(s, -s + 1)
sym_equal("명제 5.2.8(a): 방향을 뒤집어도 τ는 같다", tau_formula(rev, s), tau_def.subs(s, -s + 1), D)
_, _, Brev, _ = frame_unit(rev, s)
sym_equal("명제 5.2.8(a): 방향을 뒤집으면 b는 -b", Brev, -B.subs(s, -s + 1), D)
rho = sp.diag(1, 1, -1)
refl = rho * beta
sym_equal("명제 5.2.8(b): 반사 (x,y,-z)는 τ의 부호를 바꾼다", tau_formula(refl, s), -tau_def, D)
sym_equal("명제 5.2.8(b): 반사된 나선 = b를 -b로 바꾼 나선", refl, beta.subs(b, -b), D)
check("명제 5.2.8(b): det ρ = -1", rho.det() == -1)

# 명제 5.2.9: τ ≡ 0 ⟺ 평면 (κ > 0) ---------------------------------------------------------
es = EX["sphere"]
tht, ph = es["coords"]
(rs,) = es["params"]
th0 = sp.symbols("theta0", positive=True)
lat = es["expr"].subs(tht, th0)
_, tau_lat = dgsym.curvature_torsion(lat, ph, (sp.sin(th0),))
check("명제 5.2.9: 위도원(평면곡선)의 τ = 0", sp.simplify(tau_lat) == 0)
A_, B_ = sp.symbols("A B", positive=True)
ell = sp.Matrix([A_ * sp.cos(t), B_ * sp.sin(t), 0])
_, tau_e = dgsym.curvature_torsion(ell, t)
check("명제 5.2.9: 타원의 τ = 0", sp.simplify(tau_e) == 0)

# 비예 5.2.10 ------------------------------------------------------------------------------
fpos = sp.exp(-1 / t)
right = sp.Matrix([t, 0, fpos])
left = sp.Matrix([t, fpos.subs(t, -t), 0])
kR, tR = dgsym.curvature_torsion(right, t)
kL, tL = dgsym.curvature_torsion(left, t)
fpp = sp.simplify(sp.diff(fpos, t, 2))
sym_equal("비예 5.2.10: f''(t) = e^{-1/t}(1 - 2t)/t⁴ (t > 0)", fpp, sp.exp(-1 / t) * (1 - 2 * t) / t ** 4, {t: (0.05, 3)})
check("비예 5.2.10: t = 1/2에서 κ = 0 (오른쪽 조각)", sp.simplify(kR.subs(t, sp.Rational(1, 2))) == 0)
check("비예 5.2.10: t = -1/2에서 κ = 0 (왼쪽 조각)", sp.simplify(kL.subs(t, -sp.Rational(1, 2))) == 0)
for v in (0.2, 0.8, 1.7):
    check(f"비예 5.2.10: t = ±{v}에서 τ = 0", abs(float(tR.subs(t, v))) < 1e-12 and abs(float(tL.subs(t, -v))) < 1e-12)
f = lambda x: math.exp(-1 / x) if x > 0 else 0.0
P0, P1, P2, Q = (np.array([x, f(-x), f(x)]) for x in (0.0, -1.0, -2.0, 1.0))
check("비예 5.2.10: γ(0), γ(-1), γ(-2)는 한 직선 위에 있지 않다", np.linalg.norm(np.cross(P1 - P0, P2 - P0)) > 1e-3)
check("비예 5.2.10: γ(1)은 z > 0이므로 xy평면 밖", Q[2] > 0)
check("비예 5.2.10: 네 점이 한 평면 위에 있지 않다 (det ≠ 0)", abs(np.linalg.det(np.stack([P1, P2, Q]))) > 1e-2)
# t < 0 쪽 접촉평면 xy, t > 0 쪽 xz: 종법선이 ±e3, ±e2
_, _, BR = dgsym.frenet_frame(right, t)
_, _, BL = dgsym.frenet_frame(left, t)
close("비예 5.2.10: t > 0 (t < 1/2)에서 b = ±e2", abs(np.array(sp.lambdify(t, list(BR), "numpy")(0.3), float)), np.array([0, 1, 0.0]), tol=1e-12)
close("비예 5.2.10: t < 0 (t > -1/2)에서 b = ±e3", abs(np.array(sp.lambdify(t, list(BL), "numpy")(-0.3), float)), np.array([0, 0, 1.0]), tol=1e-12)

# 비고 5.2.11: 접촉평면이 도는 각 ------------------------------------------------------------
num = {a: 1, b: sp.Rational(3, 10)}
Bf = sp.lambdify(s, list(B.subs(num)), "numpy")
tau_num = float(tau_def.subs(num))
b0 = np.array(Bf(0.0), float)
for hh in (1e-2, 1e-3):
    ang = math.acos(max(-1.0, min(1.0, float(np.dot(b0, np.array(Bf(hh), float))))))
    close(f"비고 5.2.11: 각/거리 → |τ| (h = {hh})", ang / hh, abs(tau_num), tol=1e-3)

# 연습문제 --------------------------------------------------------------------------------
tau_h_expr = eh["expected"]["tau"]
sol = sp.solve(sp.diff(tau_h_expr, b), b)
check("연습 5.2.1: dτ/db = 0 ⇔ b = ±a", set(sol) == {a, -a})
sym_equal("연습 5.2.1: b = a에서 τ = 1/(2a) = κ", tau_h_expr.subs(b, a), 1 / (2 * a))
sym_equal("연습 5.2.1: b = a에서 κ = 1/(2a)", eh["expected"]["kappa"].subs(b, a), 1 / (2 * a))
sym_equal("연습 5.2.2: <t, e3> = b/c", T[2], b / c, D)
sym_equal("연습 5.2.2: <b, e3> = a/c", B[2], a / c, D)
sp3 = sp.Matrix([sp.Rational(3, 5) * sp.sin(s), sp.Rational(4, 5) * sp.sin(s), 2 + sp.cos(s)])
sym_equal("연습 5.2.3: 단위속력", sp.simplify(sp3.diff(s).dot(sp3.diff(s))), 1)
T3, N3, B3, k3 = frame_unit(sp3, s)
check("연습 5.2.3: κ = 1", sp.simplify(k3 - 1) == 0)
check("연습 5.2.3: τ = 0 (공식)", tau_formula(sp3, s) == 0)
sym_equal("연습 5.2.3: b = (-4/5, 3/5, 0)", B3, sp.Matrix([-sp.Rational(4, 5), sp.Rational(3, 5), 0]), {s: (-3, 3)})
check("연습 5.2.3: 곡률중심 = (0, 0, 2)", sp.simplify(sp3 + sp3.diff(s, 2)) == sp.Matrix([0, 0, 2]))
# 연습 5.2.5: 일반화된 나선
al = sp.symbols("alpha", positive=True)
thf = sp.Function("theta")
bp = sp.Matrix([sp.sin(al) * sp.cos(thf(s * sp.sin(al))), sp.sin(al) * sp.sin(thf(s * sp.sin(al))), sp.cos(al)])
b2, b3 = bp.diff(s), bp.diff(s, 2)
kk = sp.Symbol("k")
kap2 = sp.simplify(b2.dot(b2))
tau5 = sp.simplify(bp.cross(b2).dot(b3) / kap2)
dth = sp.Subs(sp.Derivative(thf(sp.Symbol("xi")), sp.Symbol("xi")), sp.Symbol("xi"), s * sp.sin(al)).doit()
sym_equal("연습 5.2.5: κ² = k² sin⁴α", kap2, (sp.diff(thf(s * sp.sin(al)), s) / sp.sin(al)) ** 2 * sp.sin(al) ** 4)
sym_equal("연습 5.2.5: τ = k sin α cos α", tau5, (sp.diff(thf(s * sp.sin(al)), s) / sp.sin(al)) * sp.sin(al) * sp.cos(al))
check("연습 5.2.5: 단위속력", sp.simplify(bp.dot(bp)) == 1)
# 연습 5.2.6: -γ는 b가 같고 τ가 반대
negb = -beta
_, _, Bn, _ = frame_unit(negb, s)
sym_equal("연습 5.2.6: -γ의 종법선 = γ의 종법선", Bn, B, D)
sym_equal("연습 5.2.6: -γ의 τ = -τ", tau_formula(negb, s), -tau_def, D)
sym_equal("연습 5.2.6: |b'| = |τ|", sp.simplify(dB.dot(dB)), tau_def ** 2, D)
# 연습 5.2.7: z축 둘레 π 회전은 τ 보존
Rz = sp.diag(-1, -1, 1)
sym_equal("연습 5.2.7: z축 둘레 π 회전 (det = 1)은 τ를 보존", tau_formula(Rz * beta, s), tau_def, D)
sym_equal("연습 5.2.7: 점대칭 -I_3 (det = -1)는 τ를 뒤집음", tau_formula(-beta, s), -tau_def, D)
# 연습 5.2.7 풀이: Aγ(t) = γ(t + π) - (0, 0, πb) (매개변수 이동 + z축 방향 평행이동)
_tt = sp.symbols("t", real=True)
_hel = sp.Matrix([sp.Symbol("a", positive=True) * sp.cos(_tt), sp.Symbol("a", positive=True) * sp.sin(_tt), sp.Symbol("b", real=True) * _tt])
check("연습 5.2.7 풀이: Aγ(t) = γ(t + π) - (0, 0, πb)", sp.simplify(Rz * _hel - _hel.subs(_tt, _tt + sp.pi) + sp.Matrix([0, 0, sp.pi * sp.Symbol("b", real=True)])) == sp.zeros(3, 1))

summary()
