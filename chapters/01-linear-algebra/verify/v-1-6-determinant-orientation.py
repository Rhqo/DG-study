"""1.6절 행렬식, 방향, 외적: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

행렬식은 본문의 첫째 행 전개 (1.6.2)를 그대로 구현한 det_rec로 다시 계산해 sympy의 det과 비교한다.
실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/verify/v-1-6-determinant-orientation.py``
"""

import itertools
import random

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

t = sp.symbols("t", real=True)


def det_rec(A):
    """(1.6.2): 첫째 행을 따라 전개한 행렬식."""
    A = sp.Matrix(A)
    n = A.shape[0]
    if n == 1:
        return A[0, 0]
    return sum((-1) ** j * A[0, j] * det_rec(A.extract(list(range(1, n)), [c for c in range(n) if c != j]))
               for j in range(n))


def cross(u, v):
    """(1.6.9)."""
    return sp.Matrix([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]])


# ---------------------------------------------------------------------------
# 정리 1.6.5: det_rec는 다중선형·교대·정규화이고 sympy det과 같다 (n = 2, 3, 4, 기호 행렬)
# ---------------------------------------------------------------------------
for n in (2, 3, 4):
    A = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{i}{j}"))
    check(f"정리 1.6.5 (n={n}): (1.6.2) = sympy det", sp.expand(det_rec(A) - A.det()) == 0)
    check(f"정리 1.6.5 (n={n}): det(I) = 1", det_rec(sp.eye(n)) == 1)
    # 교대: 두 열을 맞바꾸면 부호가 바뀐다
    S = A.copy(); S[:, 0], S[:, n - 1] = A[:, n - 1], A[:, 0]
    check(f"정리 1.6.5 (n={n}): 열 교환 → 부호 반전", sp.expand(det_rec(S) + det_rec(A)) == 0)
    # 다중선형: 첫 열에 대해 선형
    c = sp.Symbol("c")
    x = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"x{i}"))
    Ax = A.copy(); Ax[:, 0] = c * A[:, 0] + x
    Ax2 = A.copy(); Ax2[:, 0] = x
    check(f"정리 1.6.5 (n={n}): 첫 열에 대해 선형", sp.expand(det_rec(Ax) - c * det_rec(A) - det_rec(Ax2)) == 0)

# (1.6.1): 임의의 교대 3-다중선형 D는 D(e)·det. 예: D(u,v,w) = det(Mu, Mv, Mw)
M = sp.Matrix([[1, 2, 0], [0, 1, 3], [4, 0, 1]])
U = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"u{i}{j}"))
D = lambda X: (M * X).det()
check("식 (1.6.1): D(v) = D(e) det(v)  (D = det∘M, 정리 1.6.8(a)의 증명)", sp.expand(D(U) - D(sp.eye(3)) * U.det()) == 0)

# 예 1.6.7
a, b, c_, d = sp.symbols("a b c d")
check("예 1.6.7(a): det 2x2 = ad - bc", det_rec(sp.Matrix([[a, b], [c_, d]])) == a * d - b * c_)
B = sp.Matrix([[2, -1], [1, 1]])
check("예 1.6.7(c): det B = 3", det_rec(B) == 3)
r = sp.symbols("r", positive=True)
th = sp.symbols("theta", real=True)
Bp = sp.Matrix([[sp.cos(th), -r * sp.sin(th)], [sp.sin(th), r * sp.cos(th)]])
check("예 1.6.7(c): 극좌표 기저 det = r", sp.simplify(det_rec(Bp)) == r)

# 정리 1.6.8 (무작위 정수 행렬)
random.seed(6)
ok_mul = ok_T = ok_inv = True
for _ in range(30):
    n = random.randint(1, 4)
    X = sp.Matrix(n, n, lambda i, j: random.randint(-3, 3))
    Y = sp.Matrix(n, n, lambda i, j: random.randint(-3, 3))
    ok_mul &= det_rec(X * Y) == det_rec(X) * det_rec(Y)
    ok_T &= det_rec(X.T) == det_rec(X)
    ok_inv &= (det_rec(X) != 0) == (X.rank() == n)
check("정리 1.6.8(a): det(AB) = det A det B (30개)", ok_mul)
check("정리 1.6.8(b): det ≠ 0 ⇔ 가역 (30개)", ok_inv)
check("정리 1.6.8(c): det A^T = det A (30개)", ok_T)

# (1.6.5) 라이프니츠 공식과 s(σ^{-1}) = s(σ)
n = 4
A = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{i}{j}"))
def P(sig):
    return sp.Matrix.hstack(*[sp.eye(n).col(sig[j]) for j in range(n)])
perms = list(itertools.permutations(range(n)))
leib = sum(P(s).det() * sp.prod([A[s[j], j] for j in range(n)]) for s in perms)
check("식 (1.6.5): 라이프니츠 공식 = det", sp.expand(leib - A.det()) == 0)
inv = lambda s: tuple(sorted(range(n), key=lambda i: s[i]))
check("정리 1.6.8(c) 증명: s(σ^{-1}) = s(σ)", all(P(s).det() == P(inv(s)).det() for s in perms))
check("정리 1.6.8(c) 증명: P_σ P_τ = P_{σ∘τ}",
      all(P(s) * P(q) == P(tuple(s[q[j]] for j in range(n))) for s in perms[:6] for q in perms[:6]))

# 따름정리 1.6.9
V = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"v{i}{j}"))
check("따름정리 1.6.9: det(V^T V) = (det V)^2", sp.expand((V.T * V).det() - V.det() ** 2) == 0)

# 명제 1.6.10: det(B^{-1} A B) = det A
Bi = sp.Matrix([[1, 2, 0], [0, 1, 1], [1, 0, 1]])
check("명제 1.6.10: det(B^{-1}AB) = det A", sp.simplify((Bi.inv() * U * Bi).det() - U.det()) == 0)
al = sp.symbols("alpha", real=True)
check("본문: det R_α = 1", sp.simplify(sp.Matrix([[sp.cos(al), -sp.sin(al)], [sp.sin(al), sp.cos(al)]]).det()) == 1)
check("본문: det P = 0", (sp.Rational(1, 2) * sp.Matrix([[1, 1], [1, 1]])).det() == 0)

# 예 1.6.13
check("예 1.6.13(a): (b_2, b_1)의 행렬 det = -3", sp.Matrix([[-1, 2], [1, 1]]).det() == -3)
check("예 1.6.13(b): (e1,e3,e2) det -1, (e2,e3,e1) det +1",
      sp.Matrix.hstack(*[sp.eye(3).col(k) for k in (0, 2, 1)]).det() == -1 and
      sp.Matrix.hstack(*[sp.eye(3).col(k) for k in (1, 2, 0)]).det() == 1)

# ---------------------------------------------------------------------------
# 외적 (정의 1.6.15, 정리 1.6.16, 명제 1.6.17)
# ---------------------------------------------------------------------------
u = sp.Matrix(sp.symbols("u1 u2 u3"))
v = sp.Matrix(sp.symbols("v1 v2 v3"))
w = sp.Matrix(sp.symbols("w1 w2 w3"))
uv = cross(u, v)
check("식 (1.6.9) = sympy cross", uv == u.cross(v))
check("식 (1.6.7): <u×v, w> = det(u, v, w)", sp.expand(uv.dot(w) - sp.Matrix.hstack(u, v, w).det()) == 0)
check("식 (1.6.8): det(u, v, e_i) = (u×v)^i",
      all(sp.expand(det_rec(sp.Matrix.hstack(u, v, sp.eye(3).col(i))) - uv[i]) == 0 for i in range(3)))
check("정리 1.6.16(a): v×u = -u×v", cross(v, u) == -uv)
check("정리 1.6.16(b): <u×v, u> = <u×v, v> = 0", sp.expand(uv.dot(u)) == 0 and sp.expand(uv.dot(v)) == 0)
check("정리 1.6.16(d): |u×v|^2 = |u|^2|v|^2 - <u,v>^2", sp.expand(uv.dot(uv) - u.dot(u) * v.dot(v) + u.dot(v) ** 2) == 0)
check("정리 1.6.16(e): det(u, v, u×v) = |u×v|^2", sp.expand(sp.Matrix.hstack(u, v, uv).det() - uv.dot(uv)) == 0)
e = [sp.eye(3).col(k) for k in range(3)]
check("명제 1.6.17: e1×e2 = e3, e2×e3 = e1, e3×e1 = e2",
      cross(e[0], e[1]) == e[2] and cross(e[1], e[2]) == e[0] and cross(e[2], e[0]) == e[1])
# 무작위 양의 정규직교기저 (QR로 만든 뒤 det 부호를 맞춤)
rng = np.random.default_rng(1)
ok = True
for _ in range(20):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    q1, q2, q3 = Q.T
    ok &= np.allclose(np.cross(q1, q2), q3) and np.allclose(np.cross(q2, q3), q1) and np.allclose(np.cross(q3, q1), q2)
check("명제 1.6.17: 무작위 양의 정규직교기저 20개", ok)

# 예 1.6.18
ua, va = sp.Matrix([1, 2, 3]), sp.Matrix([-1, 0, 2])
check("예 1.6.18(a): (1,2,3)×(-1,0,2) = (4,-5,2)", cross(ua, va) == sp.Matrix([4, -5, 2]))
check("예 1.6.18(a): |u×v|^2 = 45 = 14·5 - 25", cross(ua, va).dot(cross(ua, va)) == 45 ==
      ua.dot(ua) * va.dot(va) - ua.dot(va) ** 2)
check("예 1.6.18(b): b_1×b_2 = (0,0,3)", cross(sp.Matrix([2, 1, 0]), sp.Matrix([-1, 1, 0])) == sp.Matrix([0, 0, 3]))
# 본문: 곱의 미분법과 비결합성
f = sp.Matrix([sp.Function(f"f{k}")(t) for k in range(3)])
g = sp.Matrix([sp.Function(f"g{k}")(t) for k in range(3)])
check("본문: (u×v)' = u'×v + u×v'", sp.simplify(cross(f, g).diff(t) - cross(f.diff(t), g) - cross(f, g.diff(t))) == sp.zeros(3, 1))
check("본문: (e1×e1)×e2 = 0, e1×(e1×e2) = -e2", cross(cross(e[0], e[0]), e[1]) == sp.zeros(3, 1) and
      cross(e[0], cross(e[0], e[1])) == -e[1])

# ---------------------------------------------------------------------------
# 연습문제
# ---------------------------------------------------------------------------
check("연습 1.6.1(a): det = -5", sp.Matrix([[1, 3], [2, 1]]).det() == -5)
check("연습 1.6.1(b): det = 2", det_rec(sp.Matrix([[1, 0, 1], [1, 1, 0], [0, 1, 1]])) == 2)

hel = EXAMPLES["helix"]
(tt,) = hel["coords"]
ha, hb = hel["params"]
g1, g2 = hel["expr"].diff(tt), hel["expr"].diff(tt, 2)
cc = cross(g1, g2)
sym_equal("연습 1.6.2: γ'×γ'' = (ab sin t, -ab cos t, a^2)", cc, sp.Matrix([ha * hb * sp.sin(tt), -ha * hb * sp.cos(tt), ha ** 2]))
sym_equal("연습 1.6.2: |γ'×γ''| = a√(a²+b²)", sp.sqrt(sp.simplify(cc.dot(cc))), ha * sp.sqrt(ha ** 2 + hb ** 2), hel["domain"])
check("연습 1.6.2: γ'×γ'' ⟂ γ', γ''", sp.simplify(cc.dot(g1)) == 0 and sp.simplify(cc.dot(g2)) == 0)
kappa = sp.sqrt(sp.simplify(cc.dot(cc))) / sp.sqrt(sp.simplify(g1.dot(g1))) ** 3
sym_equal("연습 1.6.2: κ = a/(a²+b²) (§7, 5장에서 사용)", kappa, hel["expected"]["kappa"], hel["domain"])

sph = EXAMPLES["sphere"]
sth, sph_ph = sph["coords"]
(sr,) = sph["params"]
X = sph["expr"]
uw = cross(X.diff(sth), X.diff(sph_ph))
sym_equal("연습 1.6.3: u×w = r² sinθ (sinθcosφ, sinθsinφ, cosθ)", uw,
          sr ** 2 * sp.sin(sth) * sp.Matrix([sp.sin(sth) * sp.cos(sph_ph), sp.sin(sth) * sp.sin(sph_ph), sp.cos(sth)]))
sym_equal("연습 1.6.3: |u×w|² = det(그람) = r⁴ sin²θ", sp.simplify(uw.dot(uw)), sr ** 4 * sp.sin(sth) ** 2, sph["domain"])
Nunit = sp.simplify(uw / (sr ** 2 * sp.sin(sth)))
sym_equal("연습 1.6.3: 단위벡터 = x/r = §7의 바깥쪽 N", Nunit, sph["expected"]["normal"], sph["domain"])

check("연습 1.6.4(a): det(I+I) = 4 ≠ 2", (2 * sp.eye(2)).det() == 4)
cs = sp.Symbol("c")
check("연습 1.6.4(b): det(cA) = c^n det A (n=3)", sp.expand((cs * U).det() - cs ** 3 * U.det()) == 0)
ts = sp.Symbol("t")
vv2, ww2 = sp.Matrix(sp.symbols("p1 p2")), sp.Matrix(sp.symbols("q1 q2"))
check("연습 1.6.5: det(v + t w, w) = det(v, w)", sp.expand(sp.Matrix.hstack(vv2 + ts * ww2, ww2).det() - sp.Matrix.hstack(vv2, ww2).det()) == 0)
check("연습 1.6.6: (u×v)×w = <u,w>v - <v,w>u", sp.expand(cross(uv, w) - (u.dot(w) * v - v.dot(w) * u)) == sp.zeros(3, 1))
Cm = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"C{i}{j}"))
blk = sp.diag(Cm, 1)
check("연습 1.6.7(b): det [[C,0],[0,1]] = det C", sp.expand(det_rec(blk) - Cm.det()) == 0)
check("연습 1.6.7(a): 반사의 det = -1", sp.diag(1, -1).det() == -1)
# 연습 1.6.8: 그람 행렬식 ≥ 0, 종속이면 0
ok = True
for _ in range(100):
    k, n = rng.integers(1, 4), 4
    Mn = rng.normal(size=(n, k))
    ok &= np.linalg.det(Mn.T @ Mn) > -1e-12
    Md = Mn.copy(); Md[:, -1] = Md[:, 0] * 2.0 if k > 1 else 0.0
    ok &= abs(np.linalg.det(Md.T @ Md)) < 1e-9
check("연습 1.6.8: 무작위 100개에서 det G ≥ 0, 종속이면 0", ok)
uu, vv = np.array([1.2, -0.3, 0.5]), np.array([0.4, 1.1, -0.7])
close("연습 1.6.8: k=2, n=3에서 √det G = |u×v|", np.sqrt(np.linalg.det(np.array([[uu @ uu, uu @ vv], [uu @ vv, vv @ vv]]))),
      np.linalg.norm(np.cross(uu, vv)), tol=1e-12)

summary()
