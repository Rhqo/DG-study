"""1.1절 벡터공간과 기저: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/verify/v-1-1-vector-spaces-bases.py``
"""

import sympy as sp

from dgcheck import check, sym_equal, summary
from dgsym import EXAMPLES

x1, x2, x3, t = sp.symbols("x1 x2 x3 t", real=True)

# ---------------------------------------------------------------------------
# 이 장의 기준 기저 B = (b1, b2) (예 1.1.15)
# ---------------------------------------------------------------------------
b1 = sp.Matrix([2, 1])
b2 = sp.Matrix([-1, 1])
w = sp.Matrix([0, 1])
B = sp.Matrix.hstack(b1, b2)

check("예 1.1.10(a): (b1, b2) 일차독립 (det B = 3 ≠ 0)", B.det() == 3)
check("예 1.1.10(c): b1 + 2 b2 - 3 w = 0", b1 + 2 * b2 - 3 * w == sp.zeros(2, 1))

comp = B.solve(sp.Matrix([x1, x2]))
sym_equal("식 (1.1.4): v^1 = (x^1 + x^2)/3", comp[0], (x1 + x2) / 3)
sym_equal("식 (1.1.4): v^2 = (-x^1 + 2x^2)/3", comp[1], (-x1 + 2 * x2) / 3)
check("예 1.1.15: [(1,2)]_B = (1, 1)", B.solve(sp.Matrix([1, 2])) == sp.Matrix([1, 1]))
check("예 1.1.15: [(3,0)]_B = (1, -1)", B.solve(sp.Matrix([3, 0])) == sp.Matrix([1, -1]))

v = sp.Matrix([1, 2])
check("비예 1.1.16(b): v = 1 b1 + 1 b2 + 0 w", b1 + b2 == v)
check("비예 1.1.16(b): v = 0 b1 - 1 b2 + 3 w", -b2 + 3 * w == v)
check("비예 1.1.16(b): 두 표현의 차 (1, 2, -3)", (1 * b1 + 1 * b2 + 0 * w) - (0 * b1 - 1 * b2 + 3 * w)
      == 1 * b1 + 2 * b2 - 3 * w)
# (b1, -b1)의 span에 v가 없다: [b1 v]의 행렬식 ≠ 0
check("비예 1.1.16(a): v ∉ span(b1)", sp.Matrix.hstack(b1, v).det() != 0)

# ---------------------------------------------------------------------------
# 예 1.1.22: 평면 x^1 + x^2 + x^3 = 0
# ---------------------------------------------------------------------------
c1 = sp.Matrix([1, -1, 0])
c2 = sp.Matrix([0, 1, -1])
check("예 1.1.22: c1, c2 ∈ W", sum(c1) == 0 and sum(c2) == 0)
check("예 1.1.22: (c1, c2) 일차독립 (rank 2)", sp.Matrix.hstack(c1, c2).rank() == 2)
xw = sp.Matrix([x1, x2, -x1 - x2])
sym_equal("예 1.1.22: x = x^1 c1 + (x^1 + x^2) c2", xw, x1 * c1 + (x1 + x2) * c2)

# ---------------------------------------------------------------------------
# 연습 1.1.2
# ---------------------------------------------------------------------------
M = sp.Matrix.hstack(sp.Matrix([1, 1, 0]), sp.Matrix([0, 1, 1]), sp.Matrix([1, 0, 1]))
check("연습 1.1.2: 기저 (det = 2 ≠ 0)", M.det() == 2)
check("연습 1.1.2: (2,3,1)의 성분 = (2, 1, 0)", M.solve(sp.Matrix([2, 3, 1])) == sp.Matrix([2, 1, 0]))
s = (x1 + x2 + x3) / 2
sym_equal("연습 1.1.2: 일반해 (s - x^3, s - x^1, s - x^2)", M.solve(sp.Matrix([x1, x2, x3])),
          sp.Matrix([s - x3, s - x1, s - x2]))

# ---------------------------------------------------------------------------
# 연습 1.1.3: P_2의 기저 (1, t-1, (t-1)^2)
# ---------------------------------------------------------------------------
c0, c1s, c2s = sp.symbols("c0 c1 c2")
sym_equal("연습 1.1.3: t^2 = (t-1)^2 + 2(t-1) + 1", t ** 2, (t - 1) ** 2 + 2 * (t - 1) + 1)
p = c0 + c1s * t + c2s * t ** 2
sym_equal("연습 1.1.3: 일반 다항식의 전개", p,
          (c0 + c1s + c2s) + (c1s + 2 * c2s) * (t - 1) + c2s * (t - 1) ** 2)
sym_equal("연습 1.1.3: 성분 = (p(1), p'(1), p''(1)/2)",
          sp.Matrix([c0 + c1s + c2s, c1s + 2 * c2s, c2s]),
          sp.Matrix([p.subs(t, 1), sp.diff(p, t).subs(t, 1), sp.diff(p, t, 2).subs(t, 1) / 2]))

# ---------------------------------------------------------------------------
# 연습 1.1.5: 구면의 두 벡터 (dgsym.EXAMPLES["sphere"]의 편미분)
# ---------------------------------------------------------------------------
sph = EXAMPLES["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
X = sph["expr"]
u = X.diff(th)
wv = X.diff(ph)
sym_equal("연습 1.1.5: u = x_θ", u, sp.Matrix([r * sp.cos(th) * sp.cos(ph), r * sp.cos(th) * sp.sin(ph), -r * sp.sin(th)]))
sym_equal("연습 1.1.5: w = x_φ", wv, sp.Matrix([-r * sp.sin(th) * sp.sin(ph), r * sp.sin(th) * sp.cos(ph), 0]))
# 2×2 소행렬식의 제곱합 = r^4 sin^2θ > 0 이면 rank 2 (일차독립)
UW = sp.Matrix.hstack(u, wv)
minors = [UW.extract([i, j], [0, 1]).det() for i, j in ((0, 1), (0, 2), (1, 2))]
sym_equal("연습 1.1.5: 소행렬식 제곱합 = r^4 sin^2 θ", sum(m ** 2 for m in minors), r ** 4 * sp.sin(th) ** 2,
          sph["domain"])
check("연습 1.1.5: θ = 0이면 w = 0", sp.simplify(wv.subs(th, 0)) == sp.zeros(3, 1))

# ---------------------------------------------------------------------------
# 연습 1.1.6: (1, t, ..., t^m)의 일차독립 — t=0에서의 미분 행렬이 diag(k!)
# ---------------------------------------------------------------------------
m = 6
D = sp.Matrix(m + 1, m + 1, lambda k, j: sp.diff(t ** j, t, k).subs(t, 0))
check("연습 1.1.6: t=0 미분 행렬 = diag(k!) (가역)", D == sp.diag(*[sp.factorial(k) for k in range(m + 1)]))

# ---------------------------------------------------------------------------
# 연습 1.1.7: dim(U+W) = dim U + dim W - dim(U∩W) (구체적 예)
# ---------------------------------------------------------------------------
Ub = sp.Matrix.hstack(sp.Matrix([1, 0, 0, 0]), sp.Matrix([0, 1, 0, 0]), sp.Matrix([0, 0, 1, 0]))
Wb = sp.Matrix.hstack(sp.Matrix([0, 1, 1, 0]), sp.Matrix([0, 0, 1, 1]))
dimU, dimW = Ub.rank(), Wb.rank()
dimSum = sp.Matrix.hstack(Ub, Wb).rank()
# U∩W: Ua = Wb c 의 해공간 차원 (U, W의 열이 각각 일차독립이므로 해공간 차원 = dim(U∩W))
dimCap = len(sp.Matrix.hstack(Ub, -Wb).nullspace())
check("연습 1.1.7: 예 U=span(e1,e2,e3), W=span((0,1,1,0),(0,0,1,1)) ⊆ R^4",
      dimSum == dimU + dimW - dimCap and (dimU, dimW, dimCap, dimSum) == (3, 2, 1, 4))

summary()
