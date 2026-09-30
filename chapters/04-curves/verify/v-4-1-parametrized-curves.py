"""4.1절 매개곡선과 속도벡터: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

기준 예제(원, 나선, 구면)는 dgsym.EXAMPLES에서 가져온다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/verify/v-4-1-parametrized-curves.py``
"""

import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
t = sp.symbols("t", real=True)


def ip(u, v):
    return sp.expand(sp.Matrix(u).dot(sp.Matrix(v)))


# 예 4.1.3 / 예 4.1.8: 원 -------------------------------------------------------
e = EX["circle"]
(tc,) = e["coords"]
(r,) = e["params"]
g = e["expr"]
dg = g.diff(tc)
sym_equal("예 4.1.8: 원의 속도 γ' = (−r sin t, r cos t, 0)", dg, sp.Matrix([-r * sp.sin(tc), r * sp.cos(tc), 0]))
sym_equal("예 4.1.8: 원의 속력 |γ'|² = r²", dg.dot(dg), r ** 2)
sym_equal("예 4.1.8: <γ, γ'> = 0", g.dot(dg), 0)
beta = g.subs(tc, 2 * tc)
sym_equal("예 4.1.3/4.1.8: β(t) = γ(2t)의 속도 = 2γ'(2t)", beta.diff(tc), 2 * dg.subs(tc, 2 * tc))
sym_equal("예 4.1.3: β의 자취도 원 x² + y² = r²", beta[0] ** 2 + beta[1] ** 2, r ** 2)
cw = g.subs(tc, -tc)
sym_equal("예 4.1.3: 시계 방향 원 (r cos t, −r sin t, 0) = γ(−t)", cw, sp.Matrix([r * sp.cos(tc), -r * sp.sin(tc), 0]))

# 예 4.1.4 / 4.1.8: 나선 ---------------------------------------------------------
e = EX["helix"]
(th,) = e["coords"]
a, b = e["params"]
h = e["expr"]
dh = h.diff(th)
sym_equal("예 4.1.8: 나선의 속도 (−a sin t, a cos t, b)", dh, sp.Matrix([-a * sp.sin(th), a * sp.cos(th), b]))
sym_equal("예 4.1.8: 나선의 속력² = a² + b²", dh.dot(dh), a ** 2 + b ** 2)
sym_equal("예 4.1.4: 나선은 원기둥 x² + y² = a² 위에 있다", h[0] ** 2 + h[1] ** 2, a ** 2)
sym_equal("예 4.1.4: 한 바퀴에 2πb 올라간다", h.subs(th, th + 2 * sp.pi) - h, sp.Matrix([0, 0, 2 * sp.pi * b]))
# 연습 4.1.3: 속도와 z축 사이의 각
cosang = dh.dot(sp.Matrix([0, 0, 1])) / sp.sqrt(dh.dot(dh))
sym_equal("연습 4.1.3: cos(속도, e3) = b/√(a²+b²)", cosang, b / sp.sqrt(a ** 2 + b ** 2), e["domain"])

# 비예 4.1.9: 뾰족점 --------------------------------------------------------------
cusp = sp.Matrix([t ** 2, t ** 3, 0])
dcusp = cusp.diff(t)
check("비예 4.1.9: γ'(0) = 0", dcusp.subs(t, 0) == sp.zeros(3, 1))
sym_equal("비예 4.1.9: 자취는 y² = x³", cusp[1] ** 2 - cusp[0] ** 3, 0)
# 속도의 방향: t → 0± 에서 ±(1, 0)
unit = dcusp / sp.sqrt(dcusp.dot(dcusp))
lim_p = [sp.limit(unit[i], t, 0, "+") for i in range(2)]
lim_m = [sp.limit(unit[i], t, 0, "-") for i in range(2)]
check("비예 4.1.9: t → 0+에서 속도 방향 → (1, 0)", lim_p == [1, 0])
check("비예 4.1.9: t → 0−에서 속도 방향 → (−1, 0)", lim_m == [-1, 0])

# 예 4.1.10: 스스로 만나는 곡선 -------------------------------------------------------
node = sp.Matrix([t ** 2 - 1, t ** 3 - t, 0])
dnode = node.diff(t)
check("예 4.1.10: γ(1) = γ(−1) = 0", node.subs(t, 1) == sp.zeros(3, 1) and node.subs(t, -1) == sp.zeros(3, 1))
check("예 4.1.10: γ'(1) = (2, 2, 0), γ'(−1) = (−2, 2, 0)",
      dnode.subs(t, 1) == sp.Matrix([2, 2, 0]) and dnode.subs(t, -1) == sp.Matrix([-2, 2, 0]))
sym_equal("예 4.1.10: 자취는 y² = x²(x + 1)", node[1] ** 2 - node[0] ** 2 * (node[0] + 1), 0)
check("예 4.1.10: γ'은 0이 되지 않는다 (2t = 0이면 3t² − 1 = −1)",
      sp.solve([sp.Eq(dnode[0], 0), sp.Eq(dnode[1], 0)], t) == [])

# 명제 4.1.11 / 따름정리 4.1.12: 곱의 미분법 (일반 함수) ---------------------------------
A = sp.Matrix([sp.Function(f"a{i}")(t) for i in range(1, 4)])
Bv = sp.Matrix([sp.Function(f"b{i}")(t) for i in range(1, 4)])
f = sp.Function("f")(t)
sym_equal("명제 4.1.11(b): <α,β>' = <α',β> + <α,β'>", sp.diff(A.dot(Bv), t), A.diff(t).dot(Bv) + A.dot(Bv.diff(t)))
sym_equal("명제 4.1.11(a): (fα)' = f'α + fα'", (f * A).diff(t), f.diff(t) * A + f * A.diff(t))
u = sp.symbols("u", real=True)
hh = sp.Function("h")(u)
sym_equal("명제 4.1.11(c): (α∘h)' = h'·α'(h)", A.subs(t, hh).diff(u),
          hh.diff(u) * A.diff(t).subs(t, hh))
sym_equal("따름정리 4.1.12: (|α|²)' = 2<α,α'>", sp.diff(A.dot(A), t), 2 * A.dot(A.diff(t)))

# 예 4.1.13: 구면 위의 위도원과 경선 ----------------------------------------------------
e = EX["sphere"]
tht, phi = e["coords"]
(rs,) = e["params"]
X = e["expr"]
th0, ph0 = sp.symbols("theta0 phi0", real=True)
lat = X.subs(tht, th0)            # 위도원: θ = θ0, 매개변수 φ
mer = X.subs(phi, ph0)            # 경선: φ = φ0, 매개변수 θ
sym_equal("예 4.1.13: 위도원은 구면 위 (|γ|² = r²)", lat.dot(lat), rs ** 2)
sym_equal("예 4.1.13: 위도원의 속도 ⊥ 위치", lat.dot(lat.diff(phi)), 0)
sym_equal("예 4.1.13: 위도원의 속도 = (−r sin θ0 sin φ, r sin θ0 cos φ, 0)", lat.diff(phi),
          sp.Matrix([-rs * sp.sin(th0) * sp.sin(phi), rs * sp.sin(th0) * sp.cos(phi), 0]))
check("예 4.1.13: 위도원은 평면 z = r cos θ0 위", sp.simplify(lat[2] - rs * sp.cos(th0)) == 0)
sym_equal("예 4.1.13: 위도원의 반지름 r sin θ0", lat[0] ** 2 + lat[1] ** 2, rs ** 2 * sp.sin(th0) ** 2)
sym_equal("예 4.1.13: 경선의 속도 ⊥ 위치", mer.dot(mer.diff(tht)), 0)
sym_equal("예 4.1.13: 경선의 속력 = r", mer.diff(tht).dot(mer.diff(tht)), rs ** 2)
nrm = sp.Matrix([-sp.sin(ph0), sp.cos(ph0), 0])
sym_equal("예 4.1.13: 경선은 원점을 지나는 평면 <x, (−sin φ0, cos φ0, 0)> = 0 위", mer.dot(nrm), 0)

# 연습 4.1.1: (t, t², t³) --------------------------------------------------------
g1 = sp.Matrix([t, t ** 2, t ** 3])
d1 = g1.diff(t)
sym_equal("연습 4.1.1: 속력² = 1 + 4t² + 9t⁴", d1.dot(d1), 1 + 4 * t ** 2 + 9 * t ** 4)
check("연습 4.1.1: 속도가 xy평면과 평행 (z' = 0) ⟺ t = 0", sp.solve(sp.Eq(d1[2], 0), t) == [0])
check("연습 4.1.1: 가속도 (0, 2, 6t)", g1.diff(t, 2) == sp.Matrix([0, 2, 6 * t]))

# 연습 4.1.2: 타원의 속력 ----------------------------------------------------------
A_, B_ = sp.symbols("A B", positive=True)
ell = sp.Matrix([A_ * sp.cos(t), B_ * sp.sin(t)])
sp2 = sp.simplify(ell.diff(t).dot(ell.diff(t)))
sym_equal("연습 4.1.2: |γ'|² = a² sin²t + b² cos²t", sp2, A_ ** 2 * sp.sin(t) ** 2 + B_ ** 2 * sp.cos(t) ** 2)
sym_equal("연습 4.1.2: |γ'|² = b² + (a² − b²) sin²t", sp2, B_ ** 2 + (A_ ** 2 - B_ ** 2) * sp.sin(t) ** 2)

# 연습 4.1.5: 포물선 위에서 (0, 1)에 가장 가까운 점 --------------------------------------
par = sp.Matrix([t, t ** 2])
p0 = sp.Matrix([0, 1])
dist2 = sp.expand((par - p0).dot(par - p0))
crit = sp.solve(sp.diff(dist2, t), t)
check("연습 4.1.5: 임계점 t = 0, ±1/√2", set(crit) == {0, 1 / sp.sqrt(2), -1 / sp.sqrt(2)})
check("연습 4.1.5: 최소 거리² = 3/4 (t = ±1/√2)", sp.simplify(dist2.subs(t, 1 / sp.sqrt(2)) - sp.Rational(3, 4)) == 0)
check("연습 4.1.5: t = 0에서 거리² = 1 (극대)", dist2.subs(t, 0) == 1)
tt0 = 1 / sp.sqrt(2)
check("연습 4.1.5: γ(t0) − p ⊥ γ'(t0)", sp.simplify((par - p0).dot(par.diff(t)).subs(t, tt0)) == 0)

# 연습 4.1.7: f(t) = e^{−1/t} (예 2.2.20)는 (0, ∞)에서 (0, 1) 위로 가는 증가함수, f'(t) → 0 (t → 0+)
tp = sp.symbols("t", positive=True)
fexp = sp.exp(-1 / tp)
check("연습 4.1.7: f'(t) = e^{−1/t}/t² > 0", sp.simplify(sp.diff(fexp, tp) - sp.exp(-1 / tp) / tp ** 2) == 0)
check("연습 4.1.7: lim_{t→0+} f = 0, lim_{t→∞} f = 1", sp.limit(fexp, tp, 0, "+") == 0 and sp.limit(fexp, tp, sp.oo) == 1)
check("연습 4.1.7: lim_{t→0+} f'(t) = 0 (γ'(0) = 0과 일치)", sp.limit(sp.diff(fexp, tp), tp, 0, "+") == 0)

summary()
