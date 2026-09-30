"""11.4절 여러 가지 예: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

ℝℙⁿ의 차트는 §7 (3장 식 (3.4.5), (3.4.6)) 그대로다. 원환면 매개화는 dgsym.EXAMPLES["torus"]에서 가져온다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/verify/v-11-4-more-examples.py``
"""

import itertools

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

rng = np.random.default_rng(114)


def phi(i, x):
    """φ_i[x] = (x^1/x^i, …, (x^i/x^i 빼기), …, x^{n+1}/x^i), i는 1부터 (3.4.5)."""
    x = list(x)
    return sp.Matrix([x[k] / x[i - 1] for k in range(len(x)) if k != i - 1])


def lift(i, u):
    """j_i(u) = (u^1, …, u^{i-1}, 1, u^i, …, u^n): φ_i^{-1}(u)의 대표 (3.4.6)."""
    u = list(u)
    return sp.Matrix(u[: i - 1] + [1] + u[i - 1:])


# ---------------------------------------------------------------- 명제 11.4.1: φ_j∘φ_i^{-1} = h_j∘j_i
for n in (1, 2, 3):
    u = sp.symbols(f"u1:{n + 1}", real=True)
    dom = {s: (0.3, 1.7) for s in u}
    ok = True
    for i, j in itertools.permutations(range(1, n + 2), 2):
        T = phi(j, lift(i, u))                       # φ_j∘φ_i^{-1}
        # 정의역: j_i(u)의 j번째 성분 ≠ 0. 그 성분은 u^j (j < i) 또는 u^{j-1} (j > i)
        comp = lift(i, u)[j - 1]
        expected_comp = u[j - 1] if j < i else u[j - 2]
        ok &= sp.simplify(comp - expected_comp) == 0
        # 역: φ_i∘φ_j^{-1}∘φ_j∘φ_i^{-1} = id
        back = phi(i, lift(j, list(T)))
        sym_equal(f"명제 11.4.1: φ_{i}∘φ_{j}^{{-1}}∘φ_{j}∘φ_{i}^{{-1}} = id (n = {n})", back, sp.Matrix(u), dom)
    check(f"명제 11.4.1: φ_i(U_i ∩ U_j) = {{u^j ≠ 0}} (j < i) 또는 {{u^(j-1) ≠ 0}} (j > i) (n = {n})", ok)

# ---------------------------------------------------------------- 예 11.4.2: ℝℙ²
u1, u2 = sp.symbols("u1 u2", real=True)
U = [u1, u2]
T21 = phi(2, lift(1, U)); T31 = phi(3, lift(1, U)); T32 = phi(3, lift(2, U))
sym_equal("예 11.4.2: φ_2∘φ_1^{-1}(u) = (1/u^1, u^2/u^1)", T21, sp.Matrix([1 / u1, u2 / u1]))
sym_equal("예 11.4.2: φ_3∘φ_1^{-1}(u) = (1/u^2, u^1/u^2)", T31, sp.Matrix([1 / u2, u1 / u2]))
sym_equal("예 11.4.2: φ_3∘φ_2^{-1}(u) = (u^1/u^2, 1/u^2)", T32, sp.Matrix([u1 / u2, 1 / u2]))
sym_equal("예 11.4.2: φ_1∘φ_2^{-1} = φ_2∘φ_1^{-1} (같은 식)", phi(1, lift(2, U)), T21)
sym_equal("예 11.4.2: (φ_2∘φ_1^{-1})∘(φ_2∘φ_1^{-1}) = id", T21.subs({u1: T21[0], u2: T21[1]}, simultaneous=True), sp.Matrix(U))
# 연습 11.4.1: φ_1∘φ_3^{-1}(u) = (u^2/u^1, 1/u^1)
sym_equal("연습 11.4.1: φ_1∘φ_3^{-1}(u) = (u^2/u^1, 1/u^1)", phi(1, lift(3, U)), sp.Matrix([u2 / u1, 1 / u1]))
# 연습 11.4.5: det D(φ_2∘φ_1^{-1}) = -1/(u^1)^3
J = T21.jacobian(U)
sym_equal("연습 11.4.5: D(φ_2∘φ_1^{-1}) = [[-1/u1², 0], [-u2/u1², 1/u1]]", J, sp.Matrix([[-1 / u1 ** 2, 0], [-u2 / u1 ** 2, 1 / u1]]))
sym_equal("연습 11.4.5: det = -1/(u^1)^3", J.det(), -1 / u1 ** 3)

# 그림 11.4.1: d = (0.6, 0.5, 0.9)
d = [sp.Rational(3, 5), sp.Rational(1, 2), sp.Rational(9, 10)]
check("그림 11.4.1: φ_3[d] = (2/3, 5/9), φ_1[d] = (5/6, 3/2)",
      phi(3, d) == sp.Matrix([sp.Rational(2, 3), sp.Rational(5, 9)]) and phi(1, d) == sp.Matrix([sp.Rational(5, 6), sp.Rational(3, 2)]))
check("그림 11.4.1: φ_1∘φ_3^{-1}(2/3, 5/9) = (5/6, 3/2)", phi(1, lift(3, list(phi(3, d)))) == phi(1, d))
# 대표에 무관 (λ ≠ 0)
lam = sp.Rational(-7, 3)
check("예 11.4.2: φ_i[λx] = φ_i[x] (대표 무관)", all(phi(i, [lam * c for c in d]) == phi(i, d) for i in (1, 2, 3)))

# ---------------------------------------------------------------- 예 11.4.7: 벡터공간의 기저 차트
Bm = sp.Matrix([[1, 1], [1, -1]])               # 기저변환 행렬 (열 = 새 기저의 옛 성분)
v = sp.Matrix(sp.symbols("v1 v2", real=True))
new = Bm.inv() * v                               # [v]_new = B^{-1}[v]_old (1.4.2)
check("예 11.4.7: 좌표변환 [v]_old ↦ B^{-1}[v]_old는 선형, 야코비 행렬 = B^{-1}", new.jacobian(list(v)) == Bm.inv())

# ---------------------------------------------------------------- 예 11.4.8 / 연습 11.4.3: GL(n), det > 0
for n in (2, 3):
    A = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{i}{j}"))
    check(f"예 11.4.8: det는 n^2 = {n * n}개 성분의 {n}차 다항식 (n = {n})", sp.Poly(A.det(), *A).total_degree() == n)
M = rng.normal(size=(200, 2, 2))
dets = np.linalg.det(M)
check("연습 11.4.3: 가역인 2×2 표본은 det > 0과 det < 0 두 부류 (둘 다 존재)", bool(np.any(dets > 0) and np.any(dets < 0)))

# ---------------------------------------------------------------- 연습 11.4.6: 계수가 m인 m×n 행렬 (m ≤ n)
Mm = sp.Matrix(2, 3, lambda i, j: sp.Symbol(f"b{i}{j}"))
minors = [Mm[:, list(c)].det() for c in itertools.combinations(range(3), 2)]
check("연습 11.4.6: 2×3 행렬의 2×2 소행렬식 3개는 성분의 다항식", all(sp.Poly(mm, *Mm).total_degree() == 2 for mm in minors))
ok = True
for _ in range(300):
    Z = rng.normal(size=(2, 3))
    rk = np.linalg.matrix_rank(Z)
    some = any(abs(np.linalg.det(Z[:, list(c)])) > 1e-12 for c in itertools.combinations(range(3), 2))
    ok &= (rk == 2) == some
check("연습 11.4.6: 계수 = 2 ⇔ 0이 아닌 2×2 소행렬식이 있다 (수치)", ok)

# ---------------------------------------------------------------- 예 11.4.12 / 그림 11.4.2: 원환면의 각 차트
ex = EXAMPLES["torus"]
Rs, rs = ex["params"]
us, vs = ex["coords"]
a = sp.Matrix([sp.cos(us), sp.sin(us)]); b = sp.Matrix([sp.cos(vs), sp.sin(vs)])
Phi = sp.Matrix([(Rs + rs * a[0]) * b[0], (Rs + rs * a[0]) * b[1], rs * a[1]])       # (3.3.3)
sym_equal("예 11.4.12: Φ∘(α×α)^{-1}(u, v) = x(u, v) (§7 원환면)", Phi, ex["expr"], ex["domain"])
# (α×α)∘(σ×σ)^{-1}의 성분별 도함수 (예 11.3.7의 역함수): d/ds [α∘σ^{-1}](s) = 2/(1 + s^2)
s = sp.symbols("s", real=True)
sig_inv = sp.Matrix([2 * s, s ** 2 - 1]) / (s ** 2 + 1)
alpha_of = 2 * sp.atan(sig_inv[1] / (1 + sig_inv[0]))
ss = np.linspace(-0.9, 3.0, 50)
num = np.array([float(sp.diff(alpha_of, s).subs(s, t)) for t in ss])
close("예 11.4.12: d/ds α(σ^{-1}(s)) = 2/(1 + s^2) (α∘σ^{-1}의 도함수, s ∈ (-0.9, 3), 수치)", num, 2 / (1 + ss ** 2), tol=1e-9)

# ---------------------------------------------------------------- 연습 11.4.7: 원기둥좌표
r, t, z = sp.symbols("r t z", real=True)
Cyl = sp.Matrix([r * sp.cos(t), r * sp.sin(t), z])
sym_equal("연습 11.4.7: det D(원기둥좌표) = r", Cyl.jacobian([r, t, z]).det(), r)

summary()
