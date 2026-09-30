"""2.5절 계수정리: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/verify/v-2-5-rank-theorem.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

x, y, z, u, v, t = sp.symbols("x y z u v t", real=True)
rng = np.random.default_rng(0)


def normal_form(A):
    """보조정리 2.5.1의 증명대로 P, Q를 만든다 (sympy 정확 계산)."""
    A = sp.Matrix(A)
    m, n = A.shape
    ker = A.nullspace()
    r = n - len(ker)
    # 핵의 기저를 R^n의 기저로 확장: 표준기저에서 필요한 만큼 앞에 붙인다
    vs = []
    for j in range(n):
        cand = sp.Matrix.hstack(*(vs + ker + [sp.eye(n)[:, j]])) if (vs or ker) else sp.eye(n)[:, j]
        if cand.rank() > len(vs) + len(ker):
            vs.append(sp.eye(n)[:, j])
        if len(vs) == r:
            break
    Q = sp.Matrix.hstack(*(vs + ker))
    cs = [A * vv for vv in vs]
    for i in range(m):
        if len(cs) == m:
            break
        cand = sp.Matrix.hstack(*(cs + [sp.eye(m)[:, i]]))
        if cand.rank() > len(cs):
            cs.append(sp.eye(m)[:, i])
    C = sp.Matrix.hstack(*cs)
    return C.inv(), Q, r


def J(m, n, r):
    M = sp.zeros(m, n)
    for i in range(r):
        M[i, i] = 1
    return M


# 보조정리 2.5.1: 여러 행렬에서 PAQ = J_r
for A in ([[1, 2], [2, 4]], [[1, 0, 2], [0, 1, 1], [1, 1, 3]], [[0, 0], [0, 0], [1, 2]], [[2, 1, 0, 1], [4, 2, 0, 2]]):
    P, Q, r = normal_form(A)
    check(f"보조정리 2.5.1: PAQ = J_r (A = {A})", P * sp.Matrix(A) * Q == J(*sp.Matrix(A).shape, r) and P.det() != 0 and Q.det() != 0)

# 보조정리 2.5.2: 그람 행렬식이 0이 아니면 열이 일차독립 (수치) / 계수의 반연속성 예
M = sp.Matrix([[x, 1], [y, x], [1, y]])
G = M.T * M
check("보조정리 2.5.2: det(M^T M) = 0 ⇔ 계수 < 2 (예시 점)", all(
    (sp.N(G.subs({x: a, y: b}).det()) > 1e-12) == (M.subs({x: a, y: b}).rank() == 2)
    for a, b in [(0.3, 0.7), (1, 1), (2, -1), (0, 0)]))
check("보조정리 2.5.2: x ↦ x^2의 계수는 0에서 0, 근처에서 1", sp.diff(x ** 2, x).subs(x, 0) == 0 and sp.diff(x ** 2, x).subs(x, 0.1) != 0)

# 정리 2.5.3 증명의 구성을 구체 예에서 따라가기: F(x, y) = (x + y + (x+y)^2, sin(x+y)) — 계수 1, DF(0)을 J_1로 정규화
Fex = sp.Matrix([x + y + (x + y) ** 2, sp.sin(x + y)])
DFex = Fex.jacobian([x, y])
check("정리 2.5.3 예시: 모든 점에서 계수 1", all(DFex.subs({x: a, y: b}).rank() == 1 for a, b in rng.uniform(-0.3, 0.3, (6, 2))))
P, Q, r = normal_form(DFex.subs({x: 0, y: 0}))
check("정리 2.5.3 예시: 0단계 PDF(0)Q = J_1", P * DFex.subs({x: 0, y: 0}) * Q == J(2, 2, 1) and r == 1)
X1, X2 = sp.symbols("X1 X2", real=True)
Ft = P * Fex.subs({x: (Q * sp.Matrix([X1, X2]))[0], y: (Q * sp.Matrix([X1, X2]))[1]}, simultaneous=True)
check("정리 2.5.3 예시: D F~(0) = J_1", sp.simplify(Ft.jacobian([X1, X2]).subs({X1: 0, X2: 0})) == J(2, 2, 1))
# 1단계: φ(X) = (F~^1(X), X2); φ^{-1}를 구해 F~∘φ^{-1} = (z1, R(z1, z2))이고 R이 z2에 무관함을 확인
z1, z2 = sp.symbols("z1 z2", real=True)
phi1 = Ft[0]
sols = sp.solve(sp.Eq(phi1.subs(X2, z2), z1), X1)
ok = False
for sol in sols:
    if sp.simplify(sol.subs({z1: 0, z2: 0})) == 0:
        R = sp.simplify(Ft[1].subs({X1: sol, X2: z2}))
        ok = sp.simplify(sp.diff(R, z2)) == 0
        Rfun = R
check("정리 2.5.3 예시: 2단계 ∂R/∂z'' = 0", ok)

# 예 2.5.5: 원에 감는 사상
Fc = sp.Matrix([sp.cos(x + y), sp.sin(x + y)])
Dc = Fc.jacobian([x, y])
check("예 2.5.5: 두 열이 같고 0이 아님", Dc[:, 0] == Dc[:, 1] and sp.simplify(Dc[:, 0].dot(Dc[:, 0])) == 1)
check("예 2.5.5: F∘φ^{-1}(u, v) = (cos u, sin u)", sp.simplify(Fc.subs({x: u - v, y: v}, simultaneous=True) - sp.Matrix([sp.cos(u), sp.sin(u)])) == sp.zeros(2, 1))
s0 = 0.5
for uu in np.linspace(s0 - 3.0, s0 + 3.0, 7):
    pt = np.array([np.cos(uu), np.sin(uu)])
    ang = s0 + np.arctan2(*(np.array([[np.cos(s0), np.sin(s0)], [-np.sin(s0), np.cos(s0)]]) @ pt)[::-1])
    close(f"예 2.5.5: ψ(P(1, u)) = (u, 0) (u = {uu:.2f})", [ang, np.hypot(*pt) - 1], [uu, 0.0], tol=1e-12)

# 예 2.5.6: 그래프 곡면 (EXAMPLES)
gr = EXAMPLES["graph"]
U, V = gr["coords"]
fF = gr["functions"][0]
Xg = gr["expr"]
Dg = Xg.jacobian([U, V])
check("예 2.5.6: Dx의 위쪽 2×2 = I", Dg[0:2, :] == sp.eye(2))
psi = lambda P3: sp.Matrix([P3[0], P3[1], P3[2] - fF(P3[0], P3[1])])
check("예 2.5.6: ψ∘x(u, v) = (u, v, 0)", sp.simplify(psi(Xg) - sp.Matrix([U, V, 0])) == sp.zeros(3, 1))
w = sp.symbols("w", real=True)
psi_inv = sp.Matrix([x, y, w + fF(x, y)])
check("예 2.5.6: ψ∘ψ^{-1} = id", sp.simplify(psi(psi_inv) - sp.Matrix([x, y, w])) == sp.zeros(3, 1))

# 비예 2.5.7(b): 극좌표의 계수
r_, th = sp.symbols("r theta", real=True)
DP = sp.Matrix([r_ * sp.cos(th), r_ * sp.sin(th)]).jacobian([r_, th])
check("비예 2.5.7(b): r = 0에서 계수 1, r ≠ 0에서 계수 2", DP.subs(r_, 0).rank() == 1 and DP.subs({r_: 0.5, th: 1}).rank() == 2)

# 주의 상자: 8자 곡선의 β' ≠ 0
tt = np.linspace(-np.pi, np.pi, 200001)
speed = np.hypot(2 * np.cos(2 * tt), np.cos(tt))
check("주의: 8자 곡선 β'(t) ≠ 0 (최솟값 > 0)", speed.min() > 0.5)

# 연습 2.5.1(a): 원환면 (EXAMPLES)
tor = EXAMPLES["torus"]
Ut, Vt = tor["coords"]
Rt, rt = tor["params"]
Dt = tor["expr"].jacobian([Ut, Vt])
gram = sp.simplify((Dt.T * Dt).det())
sym_equal("연습 2.5.1(a): 그람 행렬식 = r^2 (R + r cos u)^2", gram, rt ** 2 * (Rt + rt * sp.cos(Ut)) ** 2, tor["domain"])
sym_equal("연습 2.5.1(a): <x_u, x_v> = 0", Dt[:, 0].dot(Dt[:, 1]), 0)
# 연습 2.5.1(b)
Fb = sp.Matrix([x * y, x ** 2 + y ** 2])
Db = Fb.jacobian([x, y])
check("연습 2.5.1(b): det DF = 2(y^2 - x^2)", sp.simplify(Db.det() - 2 * (y ** 2 - x ** 2)) == 0)
check("연습 2.5.1(b): 계수 2/1/0", Db.subs({x: 1, y: 0}).rank() == 2 and Db.subs({x: 1, y: 1}).rank() == 1 and Db.subs({x: 0, y: 0}).rank() == 0)
# 연습 2.5.3
A3 = sp.Matrix([[1, 2], [2, 4]])
P3 = sp.Matrix([[1, 0], [-2, 1]])
Q3 = sp.Matrix([[1, -2], [0, 1]])
check("연습 2.5.3: AQ = [[1,0],[2,0]], PAQ = J_1", A3 * Q3 == sp.Matrix([[1, 0], [2, 0]]) and P3 * A3 * Q3 == J(2, 2, 1))
# 연습 2.5.5
F5 = sp.Matrix([x ** 2 + y ** 2, 2 * (x ** 2 + y ** 2)])
check("연습 2.5.5: 계수 1 (원점 밖)", all(F5.jacobian([x, y]).subs({x: a, y: b}).rank() == 1 for a, b in [(1, 0), (0.3, -2), (-1, 1)]))
up = sp.symbols("u", positive=True)
phi_inv = sp.Matrix([sp.sqrt(u - v ** 2), v])
sym_equal("연습 2.5.5: φ∘φ^{-1} = id", sp.Matrix([phi_inv[0] ** 2 + phi_inv[1] ** 2, phi_inv[1]]), sp.Matrix([u, v]), {u: (1, 2), v: (-0.5, 0.5)})
comp = F5.subs({x: phi_inv[0], y: phi_inv[1]}, simultaneous=True)
check("연습 2.5.5: F∘φ^{-1} = (u, 2u), ψ 후 (u, 0)",
      sp.simplify(comp - sp.Matrix([u, 2 * u])) == sp.zeros(2, 1) and sp.simplify(sp.Matrix([comp[0], comp[1] - 2 * comp[0]]) - sp.Matrix([u, 0])) == sp.zeros(2, 1))
check("연습 2.5.5: det Dφ = 2x", sp.simplify(sp.Matrix([x ** 2 + y ** 2, y]).jacobian([x, y]).det() - 2 * x) == 0)

summary()
