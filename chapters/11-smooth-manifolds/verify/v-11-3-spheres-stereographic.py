"""11.3절 구면과 입체사영: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

입체사영 σ, σ̃의 식은 §7 그대로다(dgsym.EXAMPLES에는 아직 항목이 없다).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/verify/v-11-3-spheres-stereographic.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(113)


def sigma(x):          # §7: σ(x) = (x^1, …, x^n)/(1 - x^{n+1})
    return sp.Matrix(x[:-1]) / (1 - x[-1])


def sigma_t(x):        # §7: σ̃(x) = (x^1, …, x^n)/(1 + x^{n+1})
    return sp.Matrix(x[:-1]) / (1 + x[-1])


def sigma_inv(u):      # (3.2.4)
    n2 = sum(ui ** 2 for ui in u)
    return sp.Matrix(list(2 * sp.Matrix(u)) + [n2 - 1]) / (n2 + 1)


def sigma_t_inv(u):    # (11.1.1)
    n2 = sum(ui ** 2 for ui in u)
    return sp.Matrix(list(2 * sp.Matrix(u)) + [1 - n2]) / (n2 + 1)


# ---------------------------------------------------------------- 명제 11.3.1: σ̃∘σ^{-1}(u) = u/|u|^2
for n in (1, 2, 3):
    u = sp.symbols(f"u1:{n + 1}", real=True)
    U = sp.Matrix(u)
    n2 = sum(ui ** 2 for ui in u)
    sym_equal(f"명제 11.3.1: σ̃(σ^{{-1}}(u)) = u/|u|^2 (n = {n})", sigma_t(sigma_inv(u)), U / n2)
    sym_equal(f"명제 11.3.1: σ(σ̃^{{-1}}(v)) = v/|v|^2 (n = {n})", sigma(sigma_t_inv(u)), U / n2)
    sym_equal(f"명제 11.3.1 (11.3.3): 1 + x^{{n+1}} = 2|u|^2/(|u|^2 + 1) (n = {n})", 1 + sigma_inv(u)[-1], 2 * n2 / (n2 + 1))
    I = U / n2
    II = I / sum(c ** 2 for c in I)
    sym_equal(f"명제 11.3.1: I∘I = id (n = {n})", II, U)
    # 예 11.3.3: DI = (E - 2uu^T/|u|^2)/|u|^2, det DI = -|u|^{-2n}
    DI = I.jacobian(u)
    sym_equal(f"예 11.3.3 (11.3.5): DI(u) = (E - 2uu^T/|u|^2)/|u|^2 (n = {n})", DI, (sp.eye(n) - 2 * U * U.T / n2) / n2)
    sym_equal(f"예 11.3.3: det DI = -|u|^(-2n) (n = {n})", DI.det(), -1 / n2 ** n)
    # σ(S) = 0, σ̃(N) = 0
    S = sp.Matrix([0] * n + [-1]); N = sp.Matrix([0] * n + [1])
    check(f"명제 11.3.1: σ(S) = 0, σ̃(N) = 0 (n = {n})", sigma(S) == sp.zeros(n, 1) and sigma_t(N) == sp.zeros(n, 1))

# n = 2 행렬식의 손 계산 과정 (예 11.3.3)
u1, u2 = sp.symbols("u1 u2", real=True)
rho = u1 ** 2 + u2 ** 2
sym_equal("예 11.3.3: ∂(u^1/ρ)/∂u^1 = ((u^2)^2 - (u^1)^2)/ρ^2", sp.diff(u1 / rho, u1), (u2 ** 2 - u1 ** 2) / rho ** 2)
sym_equal("예 11.3.3: ∂(u^1/ρ)/∂u^2 = -2u^1u^2/ρ^2", sp.diff(u1 / rho, u2), -2 * u1 * u2 / rho ** 2)
sym_equal("예 11.3.3: ((u1²-u2²)² + 4u1²u2²) = ρ²", (u1 ** 2 - u2 ** 2) ** 2 + 4 * u1 ** 2 * u2 ** 2, rho ** 2)

# ---------------------------------------------------------------- 예 11.3.4, 명제 11.3.5: 반구 차트
def graph_inv(w, i, sign):
    """(φ_i^±)^{-1}(w) = (w^1, …, w^{i-1}, ±√(1-|w|²), w^i, …, w^n), i는 1부터."""
    w = list(w)
    h = sign * sp.sqrt(1 - sum(c ** 2 for c in w))
    return sp.Matrix(w[: i - 1] + [h] + w[i - 1:])


def graph_chart(x, i):
    x = list(x)
    return sp.Matrix(x[: i - 1] + x[i:])


for n in (1, 2):
    w = sp.symbols(f"w1:{n + 1}", real=True)
    dom = {s: (-0.5 / n, 0.5 / n) for s in w}
    for i in range(1, n + 2):
        for sg in (1, -1):
            P = graph_inv(w, i, sg)
            sym_equal(f"예 11.3.4: |(φ_{i}^{'+' if sg > 0 else '-'})^{{-1}}(w)|^2 = 1 (n = {n})", (P.T * P)[0], 1, dom)
            sym_equal(f"예 11.3.4: φ_{i}^± ∘ (φ_{i}^±)^{{-1}} = id (n = {n}, 부호 {sg})", graph_chart(P, i), sp.Matrix(w), dom)

# 정리 11.3.6 (11.3.9): σ∘(φ_i^±)^{-1}
n = 2
w = sp.symbols("w1:3", real=True)
nw2 = sum(c ** 2 for c in w)
dom = {s: (-0.5, 0.5) for s in w}
for sg in (1, -1):
    # i ≤ n: (w^1, …, w^{i-1}, ±√(1-|w|²), w^i, …, w^{n-1})/(1 - w^n)
    for i in (1, 2):
        lhs = sigma(graph_inv(w, i, sg))
        full = list(w[: i - 1]) + [sg * sp.sqrt(1 - nw2)] + list(w[i - 1:])
        rhs = sp.Matrix(full[:n]) / (1 - w[n - 1])
        sym_equal(f"정리 11.3.6 (11.3.9): σ∘(φ_{i}^{'+' if sg > 0 else '-'})^{{-1}} (n = 2)", lhs, rhs, dom)
    lhs = sigma(graph_inv(w, 3, sg))
    rhs = sp.Matrix(w) / (1 - sg * sp.sqrt(1 - nw2))
    sym_equal(f"정리 11.3.6 (11.3.9): σ∘(φ_3^{'+' if sg > 0 else '-'})^{{-1}}(w) = w/(1 ∓ √(1-|w|²)) (n = 2)", lhs, rhs, dom)
check("정리 11.3.6: φ_3^+(N) = 0 (그래서 σ∘(φ_3^+)^{-1}의 정의역은 B^n \\ {0})", graph_chart(sp.Matrix([0, 0, 1]), 3) == sp.zeros(2, 1))
# 대척사상과 그래프 차트: φ_i^±(-y) = -φ_i^∓(y)
y = sp.Matrix(sp.symbols("y1:4", real=True))
check("정리 11.3.6: φ_i^±(a(y)) = -φ_i^∓(y) (좌표를 빼는 사상은 선형)", all(graph_chart(-y, i) == -graph_chart(y, i) for i in (1, 2, 3)))

# ---------------------------------------------------------------- 예 11.3.7: S^1의 각 차트
t = sp.symbols("t", real=True)
f_sig = sp.cos(t) / (1 - sp.sin(t))
f_sigt = sp.cos(t) / (1 + sp.sin(t))
sym_equal("예 11.3.7 (11.3.11): d/dt [cos t/(1 - sin t)] = 1/(1 - sin t)", sp.diff(f_sig, t), 1 / (1 - sp.sin(t)), {t: (-1.4, 1.4)})
sym_equal("예 11.3.7: d/dt [cos t/(1 + sin t)] = -1/(1 + sin t)", sp.diff(f_sigt, t), -1 / (1 + sp.sin(t)), {t: (-1.4, 1.4)})
sym_equal("예 11.3.7: cos t/(1 - sin t) = (1 + sin t)/cos t (cos t ≠ 0)", f_sig, (1 + sp.sin(t)) / sp.cos(t), {t: (-1.4, 1.4)})
ts = rng.uniform(-np.pi + 1e-6, np.pi - 1e-6, 2000)
alpha = 2 * np.arctan(np.sin(ts) / (1 + np.cos(ts)))
close("예 11.3.7: α(cos t, sin t) = 2 arctan(sin t/(1 + cos t)) = t (t ∈ (-π, π))", alpha, ts, tol=1e-9)

# ---------------------------------------------------------------- 비예 11.3.9: c(u) = |u|^2 u
for n in (1, 2, 3):
    vv = rng.normal(size=(500, n))
    c = np.sum(vv ** 2, axis=1, keepdims=True) * vv
    back = c / np.linalg.norm(c, axis=1, keepdims=True) ** (2 / 3)
    check(f"비예 11.3.9: c^{{-1}}(v) = v/|v|^(2/3) (n = {n}, 수치)", bool(np.allclose(back, vv)))
h = sp.symbols("h", positive=True)
check("비예 11.3.9: c^{-1}의 0에서 방향 e_1의 차분몫 h^{1/3}/h → ∞", sp.limit(h / h ** sp.Rational(2, 3) / h, h, 0, "+") == sp.oo)

# ---------------------------------------------------------------- 그림 11.3.1, 연습 11.3.1, 11.3.2
x1 = sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)])
z1 = sp.Matrix([-sp.Rational(4, 5), -sp.Rational(3, 5)])
check("그림 11.3.1 / 연습 11.3.1: σ(x) = 3, σ̃(x) = 1/3", sigma(x1)[0] == 3 and sigma_t(x1)[0] == sp.Rational(1, 3))
check("그림 11.3.1 / 연습 11.3.1: σ(z) = -1/2, σ̃(z) = -2", sigma(z1)[0] == -sp.Rational(1, 2) and sigma_t(z1)[0] == -2)
x2 = sp.Matrix([sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3)])
check("연습 11.3.2: |x| = 1", (x2.T * x2)[0] == 1)
check("연습 11.3.2: σ(x) = (2, 1), σ̃(x) = (2/5, 1/5) = σ/|σ|^2",
      sigma(x2) == sp.Matrix([2, 1]) and sigma_t(x2) == sp.Matrix([sp.Rational(2, 5), sp.Rational(1, 5)]))

# ---------------------------------------------------------------- 연습 11.3.3: 그래프 차트끼리의 좌표변환
ww = sp.symbols("w", real=True)
sym_equal("연습 11.3.3: S^1에서 φ_2^+∘(φ_1^+)^{-1}(w) = √(1-w^2)", graph_chart(graph_inv([ww], 1, 1), 2)[0], sp.sqrt(1 - ww ** 2), {ww: (0.1, 0.9)})
w = sp.symbols("w1:3", real=True)
sym_equal("연습 11.3.3: S^2에서 φ_2^+∘(φ_1^+)^{-1}(w) = (√(1-|w|²), w^2)", graph_chart(graph_inv(w, 1, 1), 2),
          sp.Matrix([sp.sqrt(1 - w[0] ** 2 - w[1] ** 2), w[1]]), {w[0]: (0.1, 0.5), w[1]: (-0.4, 0.4)})

# ---------------------------------------------------------------- 연습 11.3.5: 두 각 차트
ok = True
for t0 in rng.uniform(0, 2 * np.pi, 2000):
    p = np.array([np.cos(t0), np.sin(t0)])
    if abs(t0 - np.pi) < 1e-6:
        continue
    a_val = 2 * np.arctan(p[1] / (1 + p[0]))              # α ∈ (-π, π)
    expected = t0 if t0 < np.pi else t0 - 2 * np.pi
    ok &= bool(np.isclose(a_val, expected))
check("연습 11.3.5: α∘β^{-1}(t) = t (0 < t < π), t - 2π (π < t < 2π)", ok)
# β = τ∘α∘a (τ(t) = t + π)이고 -σ^{-1}(u) = σ̃^{-1}(-u)이므로 β와 σ, σ̃의 좌표변환은 α의 좌표변환으로 쓰인다
uu = sp.symbols("u", real=True)
sym_equal("연습 11.3.5: -σ^{-1}(u) = σ̃^{-1}(-u) (n = 1)", -sigma_inv([uu]), sigma_t_inv([-uu]))


def alpha_np(p):       # (11.3.10)
    return 2 * np.arctan(p[1] / (1 + p[0]))


def beta_np(p):        # (cos t, sin t) ↦ t ∈ (0, 2π)
    return np.arctan2(p[1], p[0]) % (2 * np.pi)


def s_inv_np(u):
    return np.array([2 * u, u * u - 1]) / (u * u + 1)


def st_inv_np(v):
    return np.array([2 * v, 1 - v * v]) / (v * v + 1)


ok = True
for s0 in rng.uniform(-6, 6, 2000):
    if abs(s0) < 1e-3 or abs(s0 - 1) < 1e-3 or abs(s0 + 1) < 1e-3:
        continue
    ok &= bool(np.isclose(beta_np(s_inv_np(s0)), alpha_np(st_inv_np(-s0)) + np.pi))   # β∘σ^{-1}(u) = α∘σ̃^{-1}(-u) + π
    ok &= bool(np.isclose(beta_np(st_inv_np(s0)), alpha_np(s_inv_np(-s0)) + np.pi))   # β∘σ̃^{-1}(v) = α∘σ^{-1}(-v) + π
for t0 in rng.uniform(0.01, 2 * np.pi - 0.01, 2000):
    e = np.array([np.cos(t0), np.sin(t0)])
    ea = np.array([np.cos(t0 - np.pi), np.sin(t0 - np.pi)])                            # α^{-1}(t - π)
    if abs(e[1] - 1) > 1e-6:
        ok &= bool(np.isclose(e[0] / (1 - e[1]), -ea[0] / (1 + ea[1])))               # σ∘β^{-1}(t) = -σ̃(α^{-1}(t - π))
    if abs(e[1] + 1) > 1e-6:
        ok &= bool(np.isclose(e[0] / (1 + e[1]), -ea[0] / (1 - ea[1])))               # σ̃∘β^{-1}(t) = -σ(α^{-1}(t - π))
check("연습 11.3.5: β와 σ, σ̃의 네 좌표변환 = α의 좌표변환과 평행이동·(-id)의 합성 (수치)", ok)

# ---------------------------------------------------------------- 연습 11.3.7: 임의의 점에서의 입체사영 σ∘A
for n in (1, 2):
    u = sp.symbols(f"u1:{n + 1}", real=True)
    a = sp.Rational(3, 5); b = sp.Rational(4, 5)
    A = sp.eye(n + 1)
    A[0, 0], A[0, n], A[n, 0], A[n, n] = a, -b, b, a        # (x^1, x^{n+1}) 평면의 회전
    check(f"연습 11.3.7: A는 직교행렬 (n = {n})", sp.simplify(A.T * A) == sp.eye(n + 1))
    X = A * sigma_inv(u)
    T = sigma(X)                                          # σ∘A∘σ^{-1}
    den = sp.simplify(1 - X[-1])
    # 분모가 0인 점 = A σ^{-1}(u) = N인 점 = σ(A^{-1}N)
    q = sigma(A.T * sp.Matrix([0] * n + [1]))
    check(f"연습 11.3.7: σ∘A∘σ^{{-1}}의 분모는 u = σ(A^{{-1}}N)에서만 0 (n = {n})",
          sp.simplify(den.subs({u[i]: q[i] for i in range(n)})) == 0)
    # 좌표변환과 그 역(같은 꼴, A 대신 A^T)이 서로 역
    T2 = sigma(A.T * sigma_inv(list(T)))
    vals = {u[i]: sp.Rational(1, 7 + i) for i in range(n)}
    check(f"연습 11.3.7: (σ∘A^T∘σ^{{-1}})∘(σ∘A∘σ^{{-1}}) = id (n = {n}, 유리점 대입)",
          sp.simplify(T2.subs(vals) - sp.Matrix([vals[s] for s in u])) == sp.zeros(n, 1))

summary()
