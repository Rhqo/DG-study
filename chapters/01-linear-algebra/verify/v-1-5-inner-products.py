"""1.5절 내적과 그람 행렬: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/verify/v-1-5-inner-products.py``
"""

import random

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

x1, x2, t = sp.symbols("x1 x2 t", real=True)
r = sp.symbols("r", positive=True)
th = sp.symbols("theta", real=True)


def gram(vectors, G=None):
    """벡터 목록의 그람 행렬 (내적 x^T G y, 기본은 유클리드)."""
    n = len(vectors)
    G = sp.eye(len(vectors[0])) if G is None else G
    return sp.Matrix(n, n, lambda i, j: (vectors[i].T * G * vectors[j])[0, 0])


def gram_schmidt(vectors, G=None):
    G = sp.eye(len(vectors[0])) if G is None else G
    ip = lambda a, b: (a.T * G * b)[0, 0]
    us = []
    for b in vectors:
        w = b - sum((ip(b, u) * u for u in us), sp.zeros(len(b), 1))
        us.append(sp.simplify(w / sp.sqrt(ip(w, w))))
    return us


# 예 1.5.2(b): G = [[2,1],[1,2]] 양의 정부호
G = sp.Matrix([[2, 1], [1, 2]])
q = (sp.Matrix([[x1, x2]]) * G * sp.Matrix([x1, x2]))[0, 0]
sym_equal("예 1.5.2(b): g(x,x) = (x1+x2)^2 + x1^2 + x2^2", q, (x1 + x2) ** 2 + x1 ** 2 + x2 ** 2)
check("예 1.5.2(b): G의 고윳값 1, 3 > 0", sorted(G.eigenvals().keys()) == [1, 3])

# 비예 1.5.3
check("비예 1.5.3(c): k((1,-1),(1,-1)) = -2",
      (sp.Matrix([[1, -1]]) * sp.Matrix([[1, 2], [2, 1]]) * sp.Matrix([1, -1]))[0, 0] == -2)

# 그림 1.5.1(b)의 g-정규직교기저와 유클리드 내적
e1, e2 = sp.Matrix([1, 0]), sp.Matrix([0, 1])
f1, f2 = gram_schmidt([e1, e2], G)
check("그림 1.5.1(b): u_1 = e_1/√2", f1 == e1 / sp.sqrt(2))
sym_equal("그림 1.5.1(b): u_2 = (-1/2, 1)/√(3/2)", f2, sp.Matrix([-sp.Rational(1, 2), 1]) / sp.sqrt(sp.Rational(3, 2)))
check("그림 1.5.1(b): g-정규직교", gram([f1, f2], G) == sp.eye(2))
sym_equal("본문: <u_1, u_2> = -1/(2√3)", (f1.T * f2)[0, 0], -1 / (2 * sp.sqrt(3)))

# 정리 1.5.5 수치 확인 (무작위)
rng = np.random.default_rng(0)
ok = True
for _ in range(200):
    A = rng.normal(size=(3, 3)); Gn = A @ A.T + 0.1 * np.eye(3)
    v, w = rng.normal(size=3), rng.normal(size=3)
    ok = ok and abs(v @ Gn @ w) <= np.sqrt(v @ Gn @ v) * np.sqrt(w @ Gn @ w) + 1e-12
check("정리 1.5.5: 무작위 내적·벡터 200쌍에서 코시-슈바르츠", ok)

# 명제 1.5.7: |Tx| ≤ C|x|
ok = True
for _ in range(200):
    A = rng.normal(size=(2, 3)); x = rng.normal(size=3)
    ok = ok and np.linalg.norm(A @ x) <= np.sqrt((A ** 2).sum()) * np.linalg.norm(x) + 1e-12
check("명제 1.5.7: 무작위 200개에서 |Ax| ≤ ||A||_F |x|", ok)

# 예 1.5.11: 기준 기저의 그람-슈미트
b1, b2 = sp.Matrix([2, 1]), sp.Matrix([-1, 1])
u1, u2 = gram_schmidt([b1, b2])
w2 = b2 - (b2.T * u1)[0, 0] * u1
check("식 (1.5.4): w_2 = (-3/5, 6/5)", sp.simplify(w2) == sp.Matrix([-sp.Rational(3, 5), sp.Rational(6, 5)]))
sym_equal("예 1.5.11: |w_2| = 3/√5", sp.sqrt((w2.T * w2)[0, 0]), 3 / sp.sqrt(5))
check("예 1.5.11: u_2 = (-1, 2)/√5", sp.simplify(u2 - sp.Matrix([-1, 2]) / sp.sqrt(5)) == sp.zeros(2, 1))
check("예 1.5.11: 정규직교", sp.simplify(gram([u1, u2])) == sp.eye(2))

# 예 1.5.14(a): 기준 기저의 그람 행렬
B = sp.Matrix.hstack(b1, b2)
Gb = gram([b1, b2])
check("예 1.5.14(a): G = [[5,-1],[-1,2]]", Gb == sp.Matrix([[5, -1], [-1, 2]]))
check("예 1.5.14(a): G = B^T B (명제 1.5.13(c))", Gb == B.T * B)
vb = sp.Matrix([1, 1])
check("예 1.5.14(a): |v|^2 = [v]^T G [v] = 5", (vb.T * Gb * vb)[0, 0] == 5 == (B * vb).dot(B * vb))

# 명제 1.5.13(c): 일반 기호 확인
Gs = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"g{min(i, j)}{max(i, j)}"))
Bs = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"B{i}{j}"))
comp = sp.Matrix(2, 2, lambda k, l: sum(Bs[i, k] * Bs[j, l] * Gs[i, j] for i in range(2) for j in range(2)))
check("명제 1.5.13(c): g̃_kl = Σ B^i_k B^j_l g_ij = (B^T G B)_kl", sp.expand(comp - Bs.T * Gs * Bs) == sp.zeros(2, 2))
# 불변성: [v]~^T G~ [w]~ = [v]^T G [w]
vv, ww = sp.Matrix(sp.symbols("p1 p2")), sp.Matrix(sp.symbols("q1 q2"))
lhs = ((Bs.inv() * vv).T * (Bs.T * Gs * Bs) * (Bs.inv() * ww))[0, 0]
check("본문: Σ g̃_kl ṽ^k w̃^l = Σ g_ij v^i w^j", sp.simplify(lhs - (vv.T * Gs * ww)[0, 0]) == 0)

# 예 1.5.14(b): 극좌표 기저
pb1 = sp.Matrix([sp.cos(th), sp.sin(th)])
pb2 = sp.Matrix([-r * sp.sin(th), r * sp.cos(th)])
sym_equal("식 (1.5.6): 극좌표 기저의 G = diag(1, r^2)", gram([pb1, pb2]), sp.diag(1, r ** 2))

# 명제 1.5.15 / 예 1.5.16
beta = B.inv()      # 행 = 쌍대기저 (명제 1.3.8)
ghat_b1 = sum((Gb[i, 0] * beta.row(i) for i in range(2)), sp.zeros(1, 2))
check("예 1.5.16(a): ĝ(b_1) = 5β^1 - β^2 = (2, 1) = b_1^T", ghat_b1 == b1.T)
Gp = sp.diag(1, r ** 2)
check("예 1.5.16(b): G^{-1} = diag(1, 1/r^2)", Gp.inv() == sp.diag(1, 1 / r ** 2))
# β^2 ↔ b_2/r^2: ĝ(b_2/r^2)(b_j) = δ^2_j
check("예 1.5.16(b): ĝ(b_2/r^2) = β^2",
      [sp.simplify((pb2 / r ** 2).dot(bj)) for bj in (pb1, pb2)] == [0, 1])
# (1.5.7): Σ_j g^{ij} g_{jk} = δ^i_k (기준 기저)
check("본문: G^{-1} G = I (g^{ij} g_{jk} = δ^i_k)", Gb.inv() * Gb == sp.eye(2))
# 일반: v_ω의 성분 = G^{-1} ω 이면 g(v_ω, b_k) = ω_k
om = sp.Matrix(sp.symbols("o1 o2"))
vom = Gb.inv() * om                      # [v_ω]_B
check("명제 1.5.15: g(v_ω, b_k) = ω_k", sp.simplify((vom.T * Gb) - om.T) == sp.zeros(1, 2))

# 연습 1.5.1
check("연습 1.5.1(b): (1,-1)에서 -2", (sp.Matrix([[1, -1]]) * sp.Matrix([[0, 1], [1, 0]]) * sp.Matrix([1, -1]))[0, 0] == -2)
Gc = sp.Matrix([[1, 2], [2, 5]])
sym_equal("연습 1.5.1(c): 제곱 완성", (sp.Matrix([[x1, x2]]) * Gc * sp.Matrix([x1, x2]))[0, 0],
          (x1 + 2 * x2) ** 2 + x2 ** 2)

# 연습 1.5.2: 구면의 두 벡터 (dgsym.EXAMPLES["sphere"]), §7의 E, F, G와 일치
sph = EXAMPLES["sphere"]
sth, sph_ph = sph["coords"]
(sr,) = sph["params"]
X = sph["expr"]
Gs2 = gram([X.diff(sth), X.diff(sph_ph)])
exp = sph["expected"]
sym_equal("연습 1.5.2: 그람 행렬 = diag(r^2, r^2 sin^2θ) = (E F; F G) (§7)", Gs2,
          sp.Matrix([[exp["E"], exp["F"]], [exp["F"], exp["G"]]]), sph["domain"])

# 연습 1.5.3
vs = [sp.Matrix([1, 1, 0]), sp.Matrix([0, 1, 1]), sp.Matrix([1, 0, 1])]
us = gram_schmidt(vs)
expected = [sp.Matrix([1, 1, 0]) / sp.sqrt(2), sp.Matrix([-1, 1, 2]) / sp.sqrt(6), sp.Matrix([1, -1, 1]) / sp.sqrt(3)]
check("연습 1.5.3: u_1, u_2, u_3", all(sp.simplify(a - b) == sp.zeros(3, 1) for a, b in zip(us, expected)))
check("연습 1.5.3: 정규직교", sp.simplify(gram(us)) == sp.eye(3))

# 연습 1.5.5: 편극 항등식, 평행사변형 법칙 (기호)
vv3, ww3 = sp.Matrix(sp.symbols("a1 a2")), sp.Matrix(sp.symbols("c1 c2"))
nrm2 = lambda z: (z.T * Gs * z)[0, 0]
check("연습 1.5.5(a): 편극 항등식", sp.expand((nrm2(vv3 + ww3) - nrm2(vv3 - ww3)) / 4 - (vv3.T * Gs * ww3)[0, 0]) == 0)
check("연습 1.5.5(b): 평행사변형 법칙", sp.expand(nrm2(vv3 + ww3) + nrm2(vv3 - ww3) - 2 * nrm2(vv3) - 2 * nrm2(ww3)) == 0)

# 연습 1.5.6(b): 정규직교기저에서 A^i_j = g(u_i, T u_j)
Tm = sp.Matrix([[1, 2], [3, 4]])
U = sp.Matrix.hstack(u1, u2)             # 유클리드 정규직교기저 (예 1.5.11)
A_U = sp.simplify(U.inv() * Tm * U)      # [T]_U (정리 1.4.12)
A_ip = sp.Matrix(2, 2, lambda i, j: sp.simplify((U.col(i).T * Tm * U.col(j))[0, 0]))
check("연습 1.5.6(b): [T]_U 의 성분 = <u_i, T u_j>", sp.simplify(A_U - A_ip) == sp.zeros(2, 2))

summary()
