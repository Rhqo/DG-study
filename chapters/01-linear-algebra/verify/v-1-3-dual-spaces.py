"""1.3절 쌍대공간과 쌍대기저: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

여벡터는 행벡터(1×n sympy Matrix)로, 벡터는 열벡터로 나타낸다.
실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/verify/v-1-3-dual-spaces.py``
"""

import random

import sympy as sp

from dgcheck import check, sym_equal, summary

x1, x2, t = sp.symbols("x1 x2 t", real=True)
r = sp.symbols("r", positive=True)
th = sp.symbols("theta", real=True)


def dual_basis(B):
    """명제 1.3.8: 열이 기저인 행렬 B의 쌍대기저 = B^{-1}의 행."""
    Bi = B.inv()
    return [Bi.row(i) for i in range(B.shape[0])]


def is_dual(betas, B):
    n = B.shape[0]
    return all(sp.simplify((betas[i] * B.col(j))[0, 0] - (1 if i == j else 0)) == 0
               for i in range(n) for j in range(n))


# 예 1.3.6: 기준 기저
B = sp.Matrix([[2, -1], [1, 1]])
be = dual_basis(B)
check("예 1.3.6: β^1 = (1/3)(1, 1)", be[0] == sp.Matrix([[sp.Rational(1, 3), sp.Rational(1, 3)]]))
check("예 1.3.6: β^2 = (1/3)(-1, 2)", be[1] == sp.Matrix([[-sp.Rational(1, 3), sp.Rational(2, 3)]]))
check("예 1.3.6: β^i(b_j) = δ^i_j", is_dual(be, B))
e1 = sp.Matrix([[1, 0]])
comp = [(e1 * B.col(j))[0, 0] for j in range(2)]
check("예 1.3.6: e^1의 성분 = (2, -1)", comp == [2, -1])
check("예 1.3.6: e^1 = 2β^1 - β^2", 2 * be[0] - be[1] == e1)
check("그림 1.3.1(a): ω(v) = 3, ω(b_2) = 0 (ω = (1,1))",
      (sp.Matrix([[1, 1]]) * sp.Matrix([1, 2]))[0, 0] == 3 and (sp.Matrix([[1, 1]]) * B.col(1))[0, 0] == 0)

# 식 (1.3.4): ω(v) = Σ ω_i v^i (임의의 ω, v, 기저 B)
w1, w2 = sp.symbols("w1 w2")
om = sp.Matrix([[w1, w2]])
vv = sp.Matrix([x1, x2])
om_comp = sp.Matrix([[(om * B.col(j))[0, 0] for j in range(2)]])   # ω_i = ω(b_i)
v_comp = B.inv() * vv                                                # v^i = β^i(v)
sym_equal("식 (1.3.4): ω(v) = Σ ω_i v^i", (om * vv)[0, 0], (om_comp * v_comp)[0, 0])

# 비예 1.3.7
eta = B.col(0).T
check("비예 1.3.7: η = b_1^T, η(b_1) = 5, η(b_2) = -1",
      (eta * B.col(0))[0, 0] == 5 and (eta * B.col(1))[0, 0] == -1)

# 주의 상자: b_2를 (0,1)로 바꾸면 β'^1 = (1/2, 0)
Bp = sp.Matrix([[2, 0], [1, 1]])
check("주의: b_2' = (0,1)일 때 β'^1 = (1/2) e^1", dual_basis(Bp)[0] == sp.Matrix([[sp.Rational(1, 2), 0]]))

# 예 1.3.9: 극좌표 기저
Bp_ = sp.Matrix([[sp.cos(th), -r * sp.sin(th)], [sp.sin(th), r * sp.cos(th)]])
pol = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
sym_equal("예 1.3.9: b_1 = ∂p/∂r, b_2 = ∂p/∂θ", sp.Matrix.hstack(pol.diff(r), pol.diff(th)), Bp_)
Binv_expected = sp.Matrix([[sp.cos(th), sp.sin(th)], [-sp.sin(th) / r, sp.cos(th) / r]])
sym_equal("식 (1.3.8): B^{-1}", sp.simplify(Bp_.inv()), Binv_expected)
check("예 1.3.9: β^i(b_j) = δ^i_j", is_dual([Binv_expected.row(0), Binv_expected.row(1)], Bp_))
sym_equal("예 1.3.9: β^2 = b_2^T / r^2", Binv_expected.row(1), Bp_.col(1).T / r ** 2)

# 명제 1.3.11(a): [T^*] = A^T, (T^*ω)_j = Σ_i ω_i A^i_j (무작위 정수 행렬)
random.seed(1)
A = sp.Matrix(3, 2, lambda i, j: random.randint(-4, 4))
Bv = sp.Matrix([[1, 2], [0, 1]])            # V = R^2의 기저 (열)
Cw = sp.Matrix([[1, 0, 1], [1, 1, 0], [0, 1, 1]])  # W = R^3의 기저 (열)
T = Cw * A * Bv.inv()                        # [T]^C_B = A가 되도록 한 표준행렬
check("명제 1.3.11: [T]^C_B = A (구성 확인)", Cw.inv() * T * Bv == A)
gam = dual_basis(Cw)
bet = dual_basis(Bv)
ok = True
for i in range(3):
    Tstar_gi = gam[i] * T                   # T^*γ^i = γ^i ∘ T (행벡터)
    expected = sum((A[i, j] * bet[j] for j in range(2)), sp.zeros(1, 2))
    ok = ok and Tstar_gi == expected
check("명제 1.3.11(a): T^*γ^i = Σ_j A^i_j β^j", ok)

# 정리 1.3.12 / 따름정리 1.3.13: rank A^T = rank A (무작위 행렬 몇 개)
ok = True
for k in range(20):
    m, n = random.randint(1, 5), random.randint(1, 5)
    M = sp.Matrix(m, n, lambda i, j: random.randint(-2, 2))
    if random.random() < 0.5 and m > 1:
        M[m - 1, :] = M[0, :] + M[1 % m, :]   # 종속인 행을 섞는다
    ok = ok and M.rank() == M.T.rank()
check("따름정리 1.3.13: 무작위 행렬 20개에서 rank A^T = rank A", ok)

# 비고 1.3.16: R에서 기저 (2)가 주는 동형은 1 ↦ (x ↦ x/4)
beta_2 = sp.Rational(1, 2)     # 기저 (2)의 쌍대기저: x ↦ x/2
check("비고 1.3.16: 1 = (1/2)·2 ↦ (1/2)·β = x/4", sp.Rational(1, 2) * beta_2 == sp.Rational(1, 4))

# 연습 1.3.1
M3 = sp.Matrix([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
b3 = dual_basis(M3)
check("연습 1.3.1: 쌍대기저", b3 == [sp.Matrix([[1, 1, -1]]) / 2, sp.Matrix([[-1, 1, 1]]) / 2,
                                 sp.Matrix([[1, -1, 1]]) / 2])
check("연습 1.3.1: (2,3,1)의 성분 (2,1,0)", [(b * sp.Matrix([2, 3, 1]))[0, 0] for b in b3] == [2, 1, 0])

# 연습 1.3.2
basis = [sp.Integer(1), t, t ** 2]
fs = [lambda p: p.subs(t, 0), lambda p: sp.diff(p, t).subs(t, 0), lambda p: sp.diff(p, t, 2).subs(t, 0) / 2]
check("연습 1.3.2: 쌍대기저 (p(0), p'(0), p''(0)/2)",
      all(sp.simplify(fs[i](basis[j]) - (1 if i == j else 0)) == 0 for i in range(3) for j in range(3)))
check("연습 1.3.2: p ↦ p(1)의 성분 (1,1,1)", [b.subs(t, 1) for b in basis] == [1, 1, 1])
check("연습 1.3.2: p ↦ ∫_0^1 p의 성분 (1, 1/2, 1/3)",
      [sp.integrate(b, (t, 0, 1)) for b in basis] == [1, sp.Rational(1, 2), sp.Rational(1, 3)])

# 연습 1.3.4
u = sp.Matrix([3, 0])
check("연습 1.3.4: β^1(u) = 1, β^2(u) = -1, ω(u) = 3",
      [(b * u)[0, 0] for b in be] == [1, -1] and (sp.Matrix([[1, 1]]) * u)[0, 0] == 3)

# 연습 1.3.6
A6 = sp.Matrix([[1, 0], [0, 1], [1, 1]])
std3 = [sp.eye(3).row(i) for i in range(3)]
check("연습 1.3.6: T^*e^3 = e^1 + e^2", std3[2] * A6 == sp.Matrix([[1, 1]]))
check("연습 1.3.6: [T^*] = A^T", sp.Matrix.hstack(*[(e * A6).T for e in std3]) == A6.T)

summary()
