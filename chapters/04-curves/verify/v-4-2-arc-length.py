"""4.2절 정칙곡선과 호의 길이: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/verify/v-4-2-arc-length.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
t = sp.symbols("t", real=True)


def speed(g, var):
    d = sp.Matrix(g).diff(var)
    return sp.sqrt(sp.expand(d.dot(d)))


# 예 4.2.2 / 비예 4.2.3: 정칙성 ---------------------------------------------------------
e = EX["circle"]
(tc,) = e["coords"]
(r,) = e["params"]
sym_equal("예 4.2.2: 원의 속력 r > 0 (정칙)", speed(e["expr"], tc), r, e["domain"])
e = EX["helix"]
(th,) = e["coords"]
a, b = e["params"]
sym_equal("예 4.2.2: 나선의 속력 √(a²+b²) > 0 (정칙)", speed(e["expr"], th), sp.sqrt(a ** 2 + b ** 2), e["domain"])
g33 = sp.Matrix([t ** 3, t ** 3, 0])
check("비예 4.2.3(b): (t³, t³)은 t = 0에서 특이", g33.diff(t).subs(t, 0) == sp.zeros(3, 1))
check("비예 4.2.3(b): (t³, t³)의 자취는 직선 y = x 위", sp.simplify(g33[1] - g33[0]) == 0)

# 명제 4.2.6 뒤의 설명: 뾰족점 곡선은 y축 위의 그래프 x = |y|^{2/3}이지만 그 함수는 0에서 미분불가
y = sp.symbols("y", positive=True)
check("명제 4.2.6 뒤: x = y^{2/3}의 도함수는 y → 0+에서 발산",
      sp.limit(sp.diff(y ** sp.Rational(2, 3), y), y, 0, "+") == sp.oo)

# 보조정리 4.2.5의 예: (f^{-1})' = 1/f'(f^{-1})
# f(u) = u³ + u (f' = 3u² + 1 > 0): 역함수를 수치로 구해 (f⁻¹)'(y) = 1/f'(f⁻¹(y))를 확인
def finv(yv):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid ** 3 + mid < yv else (lo, mid)
    return (lo + hi) / 2
y0, hstep = 2.0, 1e-5
num = (finv(y0 + hstep) - finv(y0 - hstep)) / (2 * hstep)
close("보조정리 4.2.5 확인 (f = u³ + u, y = 2): (f⁻¹)'(y) = 1/f'(f⁻¹(y))", num, 1 / (3 * finv(y0) ** 2 + 1), 1e-7)

# 예 4.2.8: 원, 나선, 뾰족점 곡선의 길이 ------------------------------------------------------
e = EX["circle"]
check("예 4.2.8(a): 원 [0, 2π]의 길이 2πr", sp.simplify(sp.integrate(r, (tc, 0, 2 * sp.pi)) - 2 * sp.pi * r) == 0)
check("예 4.2.8(a): 원 [0, 4π]의 길이 4πr", sp.simplify(sp.integrate(r, (tc, 0, 4 * sp.pi)) - 4 * sp.pi * r) == 0)
Lh = sp.integrate(sp.sqrt(a ** 2 + b ** 2), (th, 0, 2 * sp.pi))
sym_equal("예 4.2.8(b): 나선 한 바퀴의 길이 2π√(a²+b²)", Lh, 2 * sp.pi * sp.sqrt(a ** 2 + b ** 2), EX["helix"]["domain"])
# 펼친 원기둥: (a cos t, a sin t, z) ↦ (at, z)에서 나선은 (at, bt), 길이 = √((2πa)² + (2πb)²)
sym_equal("예 4.2.8(b): 펼친 직사각형의 대각선 길이", sp.sqrt((2 * sp.pi * a) ** 2 + (2 * sp.pi * b) ** 2),
          2 * sp.pi * sp.sqrt(a ** 2 + b ** 2), EX["helix"]["domain"])
tp = sp.symbols("t", positive=True)
cusp = sp.Matrix([tp ** 2, tp ** 3])
sp_c = sp.simplify(speed(cusp, tp))
sym_equal("예 4.2.8(c): 뾰족점 곡선의 속력 (t > 0) = t√(4 + 9t²)", sp_c, tp * sp.sqrt(4 + 9 * tp ** 2))
Lc = sp.integrate(tp * sp.sqrt(4 + 9 * tp ** 2), (tp, 0, 1))
sym_equal("예 4.2.8(c): ∫_0^1 t√(4+9t²) dt = (13√13 − 8)/27", Lc, (13 * sp.sqrt(13) - 8) / 27)
close("예 4.2.8(c): [−1, 1]의 길이 ≈ 2.879", float(2 * Lc), 2.8794, tol=5e-4)

# 정리 4.2.11: 그림 4.2.1의 포물선 (t, t²), [−1, 1] ---------------------------------------------
Lp = sp.integrate(sp.sqrt(1 + 4 * t ** 2), (t, -1, 1))
sym_equal("그림 4.2.1: L = √5 + asinh(2)/2", Lp, sp.sqrt(5) + sp.asinh(2) / 2)
Lpf = float(Lp)
ok_le, ok_bound = True, True
for m in range(1, 41):
    uu = np.linspace(-1, 1, m + 1)
    pts = np.stack([uu, uu ** 2], 1)
    ell = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    ok_le &= ell <= Lpf + 1e-12
    ok_bound &= (Lpf - ell) <= 2 * 2 * 2 * (2 / m) + 1e-12       # 2M(t2 − t1)|P|, M = 2
check("정리 4.2.11: ℓ(P) ≤ L (포물선, m = 1..40)", ok_le)
check("정리 4.2.11: L − ℓ(P) ≤ 2M(t2−t1)|P| (포물선, m = 1..40)", ok_bound)
close("그림 4.2.1: ℓ(P_2) = 2√2", 2 * np.sqrt(2), np.sum(np.linalg.norm(np.diff(np.array([[-1, 1], [0, 0], [1, 1]]), axis=0), axis=1)))
# 불균등한 무작위 분할에서도 상한 확인
rng = np.random.default_rng(0)
ok_rand = True
for _ in range(200):
    k = rng.integers(2, 30)
    uu = np.sort(np.concatenate([[-1, 1], rng.uniform(-1, 1, k)]))
    pts = np.stack([uu, uu ** 2], 1)
    ell = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    mesh = np.max(np.diff(uu))
    ok_rand &= (0 <= Lpf - ell + 1e-12) and (Lpf - ell <= 2 * 2 * 2 * mesh + 1e-12)
check("정리 4.2.11: 무작위 분할 200개에서 0 ≤ L − ℓ(P) ≤ 2M(t2−t1)|P|", ok_rand)

# 연습 4.2.1: 별모양 곡선 (cos³t, sin³t) ------------------------------------------------------
ast = sp.Matrix([sp.cos(t) ** 3, sp.sin(t) ** 3])
d = ast.diff(t)
sym_equal("연습 4.2.1: |γ'|² = 9 sin²t cos²t", sp.simplify(d.dot(d)), 9 * sp.sin(t) ** 2 * sp.cos(t) ** 2)
Last = 4 * sp.integrate(3 * sp.sin(t) * sp.cos(t), (t, 0, sp.pi / 2))
check("연습 4.2.1: 전체 길이 6", sp.simplify(Last - 6) == 0)
close("연습 4.2.1: 수치적분으로도 6", float(sp.Integral(3 * sp.Abs(sp.sin(t) * sp.cos(t)), (t, 0, 2 * sp.pi)).evalf()), 6.0, 1e-9)

# 연습 4.2.2: 현수선 ------------------------------------------------------------------
c = sp.symbols("c", positive=True)
cat = sp.Matrix([t, sp.cosh(t)])
sym_equal("연습 4.2.2: 현수선의 속력 cosh t", sp.simplify(speed(cat, t)), sp.cosh(t), {t: (-3, 3)})
check("연습 4.2.2: [−c, c]의 길이 2 sinh c", sp.simplify(sp.integrate(sp.cosh(t), (t, -c, c)) - 2 * sp.sinh(c)) == 0)

# 연습 4.2.3: (2t, t², t³/3) ----------------------------------------------------------
g3 = sp.Matrix([2 * t, t ** 2, t ** 3 / 3])
sym_equal("연습 4.2.3: |γ'|² = (t² + 2)²", sp.expand(g3.diff(t).dot(g3.diff(t))), sp.expand((t ** 2 + 2) ** 2))
check("연습 4.2.3: [0, 3]의 길이 15", sp.integrate(t ** 2 + 2, (t, 0, 3)) == 15)

# 연습 4.2.5: 로그나선 ------------------------------------------------------------------
ls = sp.Matrix([sp.exp(-t) * sp.cos(t), sp.exp(-t) * sp.sin(t)])
sym_equal("연습 4.2.5: 로그나선의 속력 √2 e^{−t}", sp.simplify(speed(ls, t)), sp.sqrt(2) * sp.exp(-t))
check("연습 4.2.5: [0, ∞)의 길이 √2", sp.integrate(sp.sqrt(2) * sp.exp(-t), (t, 0, sp.oo)) == sp.sqrt(2))

# 연습 4.2.7: 원에 내접하는 정m각형 --------------------------------------------------------
m = sp.symbols("m", positive=True, integer=True)
chord = sp.sqrt(((r * sp.cos(2 * sp.pi / m) - r) ** 2 + (r * sp.sin(2 * sp.pi / m)) ** 2))
sym_equal("연습 4.2.7: 현의 길이 = 2r sin(π/m)", sp.simplify(chord.subs(m, 7)), 2 * r * sp.sin(sp.pi / 7), {r: (0.5, 2)})
ok = True
for mm in range(3, 200):
    ell = 2 * mm * np.sin(np.pi / mm)
    err = 2 * np.pi - ell
    ok &= (0 <= err <= np.pi ** 3 / (3 * mm ** 2) + 1e-15) and (err <= 8 * np.pi ** 2 / mm)
check("연습 4.2.7: 0 ≤ 2πr − ℓ ≤ π³r/(3m²) ≤ 8π²r/m (r = 1, m = 3..199)", ok)
close("연습 4.2.7: m²·(2π − ℓ) → π³/3", 1000 ** 2 * (2 * np.pi - 2 * 1000 * np.sin(np.pi / 1000)), np.pi ** 3 / 3, 1e-4)

summary()
