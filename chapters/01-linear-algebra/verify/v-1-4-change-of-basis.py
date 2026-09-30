"""1.4절 기저변환과 성분변환: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

벡터 성분은 열벡터, 여벡터 성분은 행벡터로 나타낸다.
실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/verify/v-1-4-change-of-basis.py``
"""

import random

import sympy as sp

from dgcheck import check, sym_equal, summary

r = sp.symbols("r", positive=True)
th = sp.symbols("theta", real=True)
v1, v2, w1, w2 = sp.symbols("v1 v2 w1 w2", real=True)


def new_components(B, v_old):
    """정리 1.4.5: [v]_new = B^{-1} [v]_old."""
    return sp.simplify(B.inv()) * v_old   # 역행렬을 먼저 단순화 (특수값 대입 시 0/0 방지)


def new_cov(B, om_old):
    """정리 1.4.6: (ω̃) = (ω) B."""
    return om_old * B


# ---------------------------------------------------------------------------
# 일반 정리의 기호 확인 (n = 2, 3의 일반 행렬)
# ---------------------------------------------------------------------------
for n in (2, 3):
    Bs = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"B{i}{j}"))
    vt = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"vt{i}"))
    om = sp.Matrix(1, n, lambda i, j: sp.Symbol(f"om{j}"))
    # 옛 기저 = 표준기저라 두면 새 기저벡터는 B의 열. v = Σ ṽ^j b̃_j 의 옛 성분 = B ṽ
    v_old = sum((vt[j] * Bs.col(j) for j in range(n)), sp.zeros(n, 1))
    check(f"정리 1.4.5 (n={n}): v^i = Σ_j B^i_j ṽ^j", sp.simplify(v_old - Bs * vt) == sp.zeros(n, 1))
    # 명제 1.4.8: (ωB)(B^{-1}v) = ωv
    vv = Bs * vt
    lhs = (new_cov(Bs, om) * new_components(Bs, vv))[0, 0]
    check(f"명제 1.4.8 (n={n}): Σ ω̃_j ṽ^j = Σ ω_i v^i", sp.simplify(lhs - (om * vv)[0, 0]) == 0)
    # 정리 1.4.6 둘째 식: β̃^j = Σ_i (B^{-1})^j_i e^i 이 새 기저의 쌍대기저
    Binv = Bs.inv()
    check(f"정리 1.4.6 (n={n}): β̃^j(b̃_k) = δ^j_k", sp.simplify(Binv * Bs) == sp.eye(n))

# 명제 1.4.4(b): 합성 BC (무작위 정수 가역행렬)
random.seed(4)
def rand_inv(n):
    while True:
        M = sp.Matrix(n, n, lambda i, j: random.randint(-3, 3))
        if M.det() != 0:
            return M
B, Cm = rand_inv(3), rand_inv(3)
Bt = B                 # 옛 기저 = 표준, 새 기저 = B의 열
Bh = B * Cm            # 새 기저에서 C로 한 번 더: 표준 성분은 B·(C의 열)
check("명제 1.4.4(b): E → B̂ 의 행렬 = BC", Bh == B * Cm and (B.inv() * Bh) == Cm)

# ---------------------------------------------------------------------------
# 예 1.4.10: E → B (기준 기저)
# ---------------------------------------------------------------------------
B0 = sp.Matrix([[2, -1], [1, 1]])
v = sp.Matrix([1, 2])
omega = sp.Matrix([[1, 1]])
check("예 1.4.10: [v]_B = (1, 1)", new_components(B0, v) == sp.Matrix([1, 1]))
check("예 1.4.10: ω의 새 성분 = (3, 0)", new_cov(B0, omega) == sp.Matrix([[3, 0]]))
check("예 1.4.10: 불변량 = 3", (new_cov(B0, omega) * new_components(B0, v))[0, 0] == 3 == (omega * v)[0, 0])
check("예 1.4.10: 쌍대기저 = B^{-1}의 행", B0.inv() == sp.Matrix([[1, 1], [-1, 2]]) / 3)

# 그림 1.4.1(b): 늘인 기저
Bs2 = sp.diag(2, 1)
check("그림 1.4.1(b): [v] = (1/2, 2), ω̃ = (2, 1)",
      new_components(Bs2, v) == sp.Matrix([sp.Rational(1, 2), 2]) and new_cov(Bs2, omega) == sp.Matrix([[2, 1]]))

# 비예 1.4.3(b)
check("비예 1.4.3(b): (b1, 2b1)의 형식적 행렬은 비가역", sp.Matrix([[2, 4], [1, 2]]).det() == 0)

# ---------------------------------------------------------------------------
# 예 1.4.11: 극좌표 기저
# ---------------------------------------------------------------------------
Bp = sp.Matrix([[sp.cos(th), -r * sp.sin(th)], [sp.sin(th), r * sp.cos(th)]])
vt = sp.simplify(new_components(Bp, sp.Matrix([v1, v2])))
sym_equal("예 1.4.11: ṽ^1 = cosθ v^1 + sinθ v^2", vt[0], sp.cos(th) * v1 + sp.sin(th) * v2)
sym_equal("예 1.4.11: ṽ^2 = (-sinθ v^1 + cosθ v^2)/r", vt[1], (-sp.sin(th) * v1 + sp.cos(th) * v2) / r)
vt_num = sp.simplify(new_components(Bp, sp.Matrix([1, 0])).subs({r: 2, th: sp.pi / 2}))
check("예 1.4.11: r=2, θ=π/2, v=e_1 → ṽ = (0, -1/2)", vt_num == sp.Matrix([0, -sp.Rational(1, 2)]))
x = r * sp.cos(th)
y = r * sp.sin(th)
sym_equal("예 1.4.11: e^2의 새 성분 = (∂y/∂r, ∂y/∂θ)", new_cov(Bp, sp.Matrix([[0, 1]])),
          sp.Matrix([[sp.diff(y, r), sp.diff(y, th)]]))
sym_equal("예 1.4.11: e^1의 새 성분 = (∂x/∂r, ∂x/∂θ)", new_cov(Bp, sp.Matrix([[1, 0]])),
          sp.Matrix([[sp.diff(x, r), sp.diff(x, th)]]))

# ---------------------------------------------------------------------------
# 정리 1.4.12, 따름정리 1.4.13, 예 1.4.14
# ---------------------------------------------------------------------------
A = sp.Matrix(2, 3, lambda i, j: sp.Symbol(f"A{i}{j}"))
Bv = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"b{i}{j}"))
Cw = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"c{i}{j}"))
# 표준행렬 M = A (옛 기저 = 표준). 새 기저에서 [T] = C^{-1} M B 인지 직접: T(b̃_j)의 새 성분
M = A
new = sp.Matrix.hstack(*[Cw.inv() * (M * Bv.col(j)) for j in range(3)])
check("정리 1.4.12: [T]~ = C^{-1} A B (기호)", sp.simplify(new - Cw.inv() * A * Bv) == sp.zeros(2, 3))
X = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"x{i}{j}"))
Y = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"y{i}{j}"))
check("따름정리 1.4.13: tr(XY) = tr(YX)", sp.expand((X * Y).trace() - (Y * X).trace()) == 0)
Bi = rand_inv(3)
check("따름정리 1.4.13: tr(B^{-1} X B) = tr X", sp.simplify((Bi.inv() * X * Bi).trace() - X.trace()) == 0)

P = sp.Rational(1, 2) * sp.Matrix([[1, 1], [1, 1]])
Bd = sp.Matrix([[1, 1], [1, -1]])
check("예 1.4.14: B^2 = 2I", Bd ** 2 == 2 * sp.eye(2))
check("예 1.4.14: AB = [[1,0],[1,0]]", P * Bd == sp.Matrix([[1, 0], [1, 0]]))
check("예 1.4.14: B^{-1}AB = diag(1, 0)", Bd.inv() * P * Bd == sp.diag(1, 0))
check("예 1.4.14: 대각합 1 = 1", P.trace() == (Bd.inv() * P * Bd).trace() == 1)

# ---------------------------------------------------------------------------
# 연습문제
# ---------------------------------------------------------------------------
check("연습 1.4.1: [v] = (2, 1)", new_components(Bd, sp.Matrix([3, 1])) == sp.Matrix([2, 1]))
check("연습 1.4.1: ω̃ = (1, 3)", new_cov(Bd, sp.Matrix([[2, -1]])) == sp.Matrix([[1, 3]]))
check("연습 1.4.1: 불변량 5", (sp.Matrix([[1, 3]]) * sp.Matrix([2, 1]))[0, 0] == 5 ==
      (sp.Matrix([[2, -1]]) * sp.Matrix([3, 1]))[0, 0])
lam = sp.symbols("l1 l2 l3", nonzero=True)
D = sp.diag(*lam)
vv = sp.Matrix(sp.symbols("u1 u2 u3"))
check("연습 1.4.2: ṽ^j = v^j/λ_j", sp.simplify(new_components(D, vv) - sp.Matrix([vv[i] / lam[i] for i in range(3)]))
      == sp.zeros(3, 1))
e1 = sp.Matrix([1, 0])
check("연습 1.4.3(b): Σ v^i w^i 는 기저에 의존 (1 → 1/4)",
      (e1.T * e1)[0, 0] == 1 and (new_components(Bs2, e1).T * new_components(Bs2, e1))[0, 0] == sp.Rational(1, 4))
check("연습 1.4.3(e): 성분 총합 2 → 1", sum(P) == 2 and sum(Bd.inv() * P * Bd) == 1)
vals = {r: 2, th: sp.pi / 3}
vt5 = sp.simplify(new_components(Bp, e1).subs(vals))
ot5 = sp.simplify(new_cov(Bp, sp.Matrix([[1, 0]])).subs(vals))
check("연습 1.4.5: ṽ = (1/2, -√3/4)", vt5 == sp.Matrix([sp.Rational(1, 2), -sp.sqrt(3) / 4]))
check("연습 1.4.5: ω̃ = (1/2, -√3)", ot5 == sp.Matrix([[sp.Rational(1, 2), -sp.sqrt(3)]]))
check("연습 1.4.5: ω(v) = 1", sp.simplify((ot5 * vt5)[0, 0]) == 1)
M6 = B0.inv() * Bd
check("연습 1.4.6: B → B̃ 행렬 = (1/3)[[2,0],[1,-3]]", M6 == sp.Matrix([[2, 0], [1, -3]]) / 3)
check("연습 1.4.6: (1.4.1) 확인", B0 * M6 == Bd)
X7 = sp.Matrix([[1, 0], [0, 0]]); Y7 = sp.Matrix([[0, 1], [0, 0]]); Z7 = sp.Matrix([[0, 0], [1, 0]])
check("연습 1.4.7: tr(XYZ) = 1, tr(YXZ) = 0", (X7 * Y7 * Z7).trace() == 1 and (Y7 * X7 * Z7).trace() == 0)
Xs, Ys, Zs = (sp.Matrix(3, 3, lambda i, j, s=s: sp.Symbol(f"{s}{i}{j}")) for s in "pqs")
check("연습 1.4.7: tr(XYZ) = tr(YZX)", sp.expand((Xs * Ys * Zs).trace() - (Ys * Zs * Xs).trace()) == 0)
# 연습 1.4.8: 모든 가역행렬과 교환하면 스칼라배 — diag(…2…)와 I+E로 충분
A8 = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"a{i}{j}"))
eqs = []
for k in range(3):
    Dk = sp.eye(3); Dk[k, k] = 2
    eqs += list(A8 * Dk - Dk * A8)
for k in (1, 2):
    E = sp.zeros(3); E[0, k] = 1
    eqs += list(A8 * (sp.eye(3) + E) - (sp.eye(3) + E) * A8)
sol = sp.solve(eqs, list(A8), dict=True)[0]
check("연습 1.4.8: 해는 A = cI", sp.simplify(A8.subs(sol) - A8[0, 0].subs(sol) * sp.eye(3)) == sp.zeros(3))

summary()
