"""1.2절 선형사상과 행렬: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/verify/v-1-2-linear-maps.py``
"""

import sympy as sp

from dgcheck import check, sym_equal, summary

t, al, be = sp.symbols("t alpha beta", real=True)
x1, x2 = sp.symbols("x1 x2", real=True)


def matrix_of_poly_map(D, src_deg, dst_deg):
    """P_src → P_dst 선형사상 D의 기저 (1, t, ..., t^deg)에 대한 행렬 (열 = D t^j의 성분)."""
    cols = []
    for j in range(src_deg + 1):
        img = sp.Poly(sp.expand(D(t ** j)), t)
        cols.append([img.coeff_monomial(t ** i) for i in range(dst_deg + 1)])
    return sp.Matrix(cols).T


# 예 1.2.7(a): P
P = sp.Rational(1, 2) * sp.Matrix([[1, 1], [1, 1]])
check("예 1.2.7(a): ker P = span((1,-1))", P.nullspace() == [sp.Matrix([-1, 1])] or
      sp.Matrix.hstack(*P.nullspace(), sp.Matrix([1, -1])).rank() == 1)
check("예 1.2.7(a): im P = span((1,1)), rank 1", P.rank() == 1 and
      sp.Matrix.hstack(P.columnspace()[0], sp.Matrix([1, 1])).rank() == 1)
check("예 1.2.7: P∘P = P", P * P == P)

# 예 1.2.15: D: P_2 → P_1
D = lambda p: sp.diff(p, t)
MD = matrix_of_poly_map(D, 2, 1)
check("예 1.2.15: [D] = [[0,1,0],[0,0,2]]", MD == sp.Matrix([[0, 1, 0], [0, 0, 2]]))
pvec = sp.Matrix([3, -1, 5])
check("예 1.2.15: [D][p] = (-1, 10) (p = 3 - t + 5t^2)", MD * pvec == sp.Matrix([-1, 10]))
check("예 1.2.15: p' = -1 + 10t", sp.expand(D(3 - t + 5 * t ** 2)) == -1 + 10 * t)
check("예 1.2.15: rank D = 2, dim ker = 1", MD.rank() == 2 and len(MD.nullspace()) == 1)

# 예 1.2.18: B와 B^{-1}
B = sp.Matrix([[2, -1], [1, 1]])
Binv = sp.Rational(1, 3) * sp.Matrix([[1, 1], [-1, 2]])
check("예 1.2.18: B^{-1} = (1/3)[[1,1],[-1,2]]", B.inv() == Binv)
check("예 1.2.18: B B^{-1} = I", B * Binv == sp.eye(2))
sym_equal("예 1.2.18: B^{-1} x = 식 (1.1.4)", Binv * sp.Matrix([x1, x2]),
          sp.Matrix([(x1 + x2) / 3, (-x1 + 2 * x2) / 3]))
check("그림 1.2.1(a): T(1,1) = (1,2)", B * sp.Matrix([1, 1]) == sp.Matrix([1, 2]))

# 정리 1.2.16의 구체적 확인: 성분식 Σ_k S^i_k T^k_j = (ST)^i_j
S = sp.Matrix(2, 3, lambda i, j: sp.Symbol(f"S{i}{j}"))
T = sp.Matrix(3, 2, lambda i, j: sp.Symbol(f"T{i}{j}"))
comp = sp.Matrix(2, 2, lambda i, j: sum(S[i, k] * T[k, j] for k in range(3)))
check("정리 1.2.16: 성분식 = 행렬곱", sp.simplify(comp - S * T) == sp.zeros(2, 2))

# 연습 1.2.1
Ta = sp.Matrix([[1, 2], [3, 0]])
check("연습 1.2.1(a): 행렬 [[1,2],[3,0]]", Ta * sp.Matrix([x1, x2]) == sp.Matrix([x1 + 2 * x2, 3 * x1]))
Tc = lambda x, y: sp.Matrix([x * y, x])
check("연습 1.2.1(c): T(2,2) ≠ 2T(1,1)", Tc(2, 2) != 2 * Tc(1, 1))

# 연습 1.2.2: 회전
R = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])
sym_equal("연습 1.2.2: R_α R_β = R_{α+β}", R(al) * R(be), R(al + be))

# 연습 1.2.3
T3 = sp.Matrix([[1, 1, 1], [1, 0, -1]])
ns = T3.nullspace()
check("연습 1.2.3: ker T = span((1,-2,1))", len(ns) == 1 and sp.Matrix.hstack(ns[0], sp.Matrix([1, -2, 1])).rank() == 1)
check("연습 1.2.3: rank T = 2, 1 + 2 = 3", T3.rank() == 2 and len(ns) + T3.rank() == 3)

# 연습 1.2.5(c)
check("연습 1.2.5(c): P e1 = P e2", P * sp.Matrix([1, 0]) == P * sp.Matrix([0, 1]))

# 연습 1.2.6: D on P_3
N = matrix_of_poly_map(D, 3, 3)
check("연습 1.2.6: [D] on P_3", N == sp.Matrix([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]]))
N2, N3 = N ** 2, N ** 3
check("연습 1.2.6: N^2의 0 아닌 성분 (1,3)=2, (2,4)=6",
      N2 == sp.Matrix([[0, 0, 2, 0], [0, 0, 0, 6], [0, 0, 0, 0], [0, 0, 0, 0]]))
check("연습 1.2.6: N^3의 0 아닌 성분 (1,4)=6", N3 == sp.Matrix([[0, 0, 0, 6], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
check("연습 1.2.6: N^4 = 0", N ** 4 == sp.zeros(4, 4))
check("연습 1.2.6: rank 3, dim ker 1", N.rank() == 3 and len(N.nullspace()) == 1)

# 연습 1.2.7: P에 대한 (2,0)의 분해
v = sp.Matrix([2, 0])
k, c = v - P * v, P * v
check("연습 1.2.7: (2,0) = (1,-1) + (1,1)", k == sp.Matrix([1, -1]) and c == sp.Matrix([1, 1]))
check("연습 1.2.7: k ∈ ker P", P * k == sp.zeros(2, 1))

summary()
