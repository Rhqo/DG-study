"""2.7절 상미분방정식: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

수치 적분은 tools/dgnum.py의 RK4를 쓴다 (scipy 사용 불가, 부록 E).
실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/verify/v-2-7-ode.py``
"""

import numpy as np
import sympy as sp

import dgnum
from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

t, s, x, y, z = sp.symbols("t s x y z", real=True)
x0, y0, z0, b = sp.symbols("x0 y0 z0 b", real=True)

# 예 2.7.2: 원과 나선 (EXAMPLES)
circ = EXAMPLES["circle"]
T = circ["coords"][0]
r = circ["params"][0]
g = circ["expr"]
check("예 2.7.2(a): 원은 V = (-y, x)의 해", sp.simplify(g.diff(T)[:2, 0] - sp.Matrix([-g[1], g[0]])) == sp.zeros(2, 1))
hel = EXAMPLES["helix"]
Th = hel["coords"][0]
aa, bb = hel["params"]
h = hel["expr"]
check("예 2.7.2(b): 나선은 V = (-y, x, b)의 해", sp.simplify(h.diff(Th) - sp.Matrix([-h[1], h[0], bb])) == sp.zeros(3, 1))

# 비예 2.7.4: x' = 3|x|^{2/3}의 두 해
check("비예 2.7.4: t^3은 해", sp.simplify(sp.diff(t ** 3, t) - 3 * sp.Abs(t ** 3) ** sp.Rational(2, 3)).subs(t, 2) == 0
      and sp.simplify(sp.diff(t ** 3, t) - 3 * sp.Abs(t ** 3) ** sp.Rational(2, 3)).subs(t, -1.5) == 0)
tt = np.linspace(-2, 2, 401)
close("비예 2.7.4: (t^3)' = 3|t^3|^{2/3} (수치)", 3 * tt ** 2, 3 * np.abs(tt ** 3) ** (2 / 3), tol=1e-9)
# 비예 2.7.5
check("비예 2.7.5: 1/(1-t)는 x' = x^2의 해", sp.simplify(sp.diff(1 / (1 - t), t) - (1 / (1 - t)) ** 2) == 0)

# 정리 2.7.7(c) / 연습 2.7.3: x' = x^2의 흐름
theta = lambda tt_, xx: xx / (1 - tt_ * xx)
sym_equal("연습 2.7.3: θ(t, θ(s, x)) = θ(t+s, x)", theta(t, theta(s, x)), theta(t + s, x), {t: (0.1, 0.3), s: (0.1, 0.3), x: (0.2, 1)})
check("연습 2.7.1(b): x0/(1 - x0 t)는 해", sp.simplify(sp.diff(theta(t, x0), t) - theta(t, x0) ** 2) == 0)
check("연습 2.7.1(a): x0 e^t는 x' = x의 해", sp.simplify(sp.diff(x0 * sp.exp(t), t) - x0 * sp.exp(t)) == 0)

# RK4로 흐름의 성질과 초기값에 대한 매끄러운 의존성 확인 (진자)
def Vp(tt_, Y):
    return np.array([Y[1], -np.sin(Y[0])])


def flow(Y0, Tf, n=2000):
    return dgnum.rk4(Vp, Y0, np.linspace(0, Tf, n + 1))[-1]


Y0 = np.array([-1.9, 0.0])
close("정리 2.7.7(c): θ_1∘θ_2 = θ_3 (진자, RK4)", flow(flow(Y0, 2.0), 1.0), flow(Y0, 3.0), tol=1e-9)
close("정리 2.7.7: θ_{-3}∘θ_3 = id (진자, RK4)", dgnum.rk4(lambda tt_, Y: -Vp(tt_, Y), flow(Y0, 3.0), np.linspace(0, 3.0, 2001))[-1], Y0, tol=1e-9)
# 매끄러운 의존성: 유한차분 야코비안이 수렴한다
J = []
for hstep in (1e-3, 1e-4):
    J.append(np.column_stack([(flow(Y0 + hstep * e, 3.0) - flow(Y0 - hstep * e, 3.0)) / (2 * hstep) for e in np.eye(2)]))
close("정리 2.7.6(c): D_x θ_3의 중심차분이 수렴 (진자)", J[0], J[1], tol=1e-5)
close("예 2.7.10: det D_x θ_3 = 1 (넓이 보존, 수치)", np.linalg.det(J[1]), 1.0, tol=1e-6)

# 예 2.7.10: 에너지 보존
xs, vs = sp.symbols("x v", real=True)
E = vs ** 2 / 2 - sp.cos(xs)
check("예 2.7.10: dE/dt = E_x v + E_v (-sin x) = 0", sp.simplify(sp.diff(E, xs) * vs + sp.diff(E, vs) * (-sp.sin(xs))) == 0)
traj = dgnum.rk4(Vp, np.array([0.5, 1.8]), np.linspace(0, 20, 20001))
Evals = 0.5 * traj[:, 1] ** 2 - np.cos(traj[:, 0])
check("예 2.7.10: RK4 궤도에서 E 보존 (|ΔE| < 1e-9)", np.max(np.abs(Evals - Evals[0])) < 1e-9)
C_ = np.sqrt(2 * (Evals[0] + 1))
check("예 2.7.10: |v(t)| ≤ √(2(E0+1))", np.max(np.abs(traj[:, 1])) <= C_ + 1e-9)

# 명제 2.7.11(a): 시간 의존 방정식의 자율화 (예: x' = t x)
fx = lambda tt_, xx: tt_ * xx
Vt = lambda ss, Y: np.array([1.0, fx(Y[0], Y[1])])
sol = dgnum.rk4(Vt, np.array([0.5, 2.0]), np.linspace(0, 1.0, 2001))[-1]
close("명제 2.7.11(a): 자율화한 해 = 2 exp((t²-t0²)/2)", sol, [1.5, 2.0 * np.exp((1.5 ** 2 - 0.5 ** 2) / 2)], tol=1e-9)

# 정리 2.7.12: 선형 방정식의 해 (프레네형 계수, 구간 전체) + 그뢴월 상한
def lin(tt_, X):
    k, tau = 1 + 0.5 * np.sin(tt_), 0.3 * np.cos(tt_)
    A = np.array([[0, k, 0], [-k, 0, tau], [0, -tau, 0]])
    return A @ X


X = dgnum.rk4(lin, np.array([1.0, 0.0, 0.0]), np.linspace(0, 30, 30001))
close("정리 2.7.12: 반대칭 계수 선형계는 |X| 보존, 긴 구간에서 유계", np.linalg.norm(X, axis=1).max(), 1.0, tol=1e-8)
a_, c_ = 2.0, 1.0
Xg = dgnum.rk4(lambda tt_, Z: np.array([[0.5, 1.0], [-1.0, 0.3]]) @ Z + np.array([np.sin(tt_), 1.0]), np.array([1.0, -1.0]), np.linspace(0, 2, 4001))
k_ = 2 * np.linalg.norm(np.array([[0.5, 1.0], [-1.0, 0.3]]), 2) + 1
bound = (np.sum(Xg[0] ** 2) + np.sqrt(2) ** 2 / k_) * np.exp(k_ * np.linspace(0, 2, 4001))
check("정리 2.7.12: 그뢴월 상한 |x|² ≤ (|x0|² + c²/k) e^{k t}", np.all(np.sum(Xg ** 2, axis=1) <= bound + 1e-9))

# 연습 2.7.2
gam = sp.Matrix([x0 * sp.cos(t) - y0 * sp.sin(t), x0 * sp.sin(t) + y0 * sp.cos(t), z0 + b * t])
check("연습 2.7.2: V = (-y, x, b)의 해", sp.simplify(gam.diff(t) - sp.Matrix([-gam[1], gam[0], b])) == sp.zeros(3, 1) and gam.subs(t, 0) == sp.Matrix([x0, y0, z0]))
# 연습 2.7.6: 조화진동자
th_ = sp.Matrix([x0 * sp.cos(t) + y0 * sp.sin(t), -x0 * sp.sin(t) + y0 * sp.cos(t)])
check("연습 2.7.6: (x,v)' = (v, -x)의 해", sp.simplify(th_.diff(t) - sp.Matrix([th_[1], -th_[0]])) == sp.zeros(2, 1))
Rm = lambda a: sp.Matrix([[sp.cos(a), sp.sin(a)], [-sp.sin(a), sp.cos(a)]])
sym_equal("연습 2.7.6: θ_t θ_s = θ_{t+s}", Rm(t) * Rm(s), Rm(t + s))
# 연습 2.7.7(b): 선형 증가 V의 전역 존재 (수치 예: V(x) = (x2, -x1 + sin x1))
Vg = lambda tt_, Y: np.array([Y[1], -Y[0] + np.sin(Y[0])])
Yg = dgnum.rk4(Vg, np.array([1.0, 0.5]), np.linspace(0, 20, 20001))
check("연습 2.7.7(b): |V(x)| ≤ 2|x| 인 계의 해가 긴 시간에도 유계", np.all(np.isfinite(Yg)) and np.max(np.abs(Yg)) < 1e3)

summary()
