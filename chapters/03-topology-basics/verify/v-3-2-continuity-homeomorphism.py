"""3.2절 연속사상과 위상동형사상: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/verify/v-3-2-continuity-homeomorphism.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(32)

# 그림 3.2.1 캡션: p = 0.5, V = (f(p) - 0.25, f(p) + 0.25), 조각 (arcsin 0.063, arcsin 0.896)
fp = 1 + 0.6 * np.sin(0.5)
lo, hi = (fp - 0.25 - 1) / 0.6, (fp + 0.25 - 1) / 0.6
close("그림 3.2.1: arcsin 인자 0.063, 0.896", [lo, hi], [0.063, 0.896], tol=6e-4)
close("그림 3.2.1: 조각의 끝점 (0.063, 1.111)", [np.arcsin(lo), np.arcsin(hi)], [0.063, 1.111], tol=6e-4)

# 예 3.2.8(c): det은 성분의 다항식 (n = 2, 3), GL(n)은 det ≠ 0
for n in (2, 3):
    A = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{i}{j}"))
    check(f"예 3.2.8(c): det은 성분의 다항식 (n = {n})", A.det().is_polynomial(*A))
M = rng.normal(size=(3, 3))
check("예 3.2.8(c): 가역행렬 근처는 가역 (작은 섭동)",
      all(abs(np.linalg.det(M + 1e-3 * rng.normal(size=(3, 3)))) > 0 for _ in range(100)) and abs(np.linalg.det(M)) > 1e-2)

# 예 3.2.11(a): 쌍곡선 xy = 1의 사영은 R \ {0} (0에 수렴하는 x 값, 0은 제외)
xs = np.array([10.0 ** (-k) for k in range(1, 8)])
check("예 3.2.11(a): (x, 1/x) ∈ C이고 x → 0, 0 ∉ pr_1(C)", bool(np.allclose(xs * (1 / xs), 1)) and not np.any(xs == 0))

# 예 3.2.14(b): F(x) = x/sqrt(1 - |x|^2), G(y) = y/sqrt(1 + |y|^2) 서로 역 (n = 3, 기호)
x = sp.Matrix(sp.symbols("x1:4", real=True))
y = sp.Matrix(sp.symbols("y1:4", real=True))
nx2 = (x.T * x)[0]
ny2 = (y.T * y)[0]
Fx = x / sp.sqrt(1 - nx2)
Gy = y / sp.sqrt(1 + ny2)
# G(F(x)): |F(x)|^2 = |x|^2/(1 - |x|^2)
nF2 = sp.simplify((Fx.T * Fx)[0])
sym_equal("예 3.2.14(b): 1 + |F(x)|^2 = 1/(1 - |x|^2)", 1 + nF2, 1 / (1 - nx2), {s: (-0.5, 0.5) for s in x})
GF = Fx / sp.sqrt(1 + nF2)
sym_equal("예 3.2.14(b): G(F(x)) = x", GF, x, {s: (-0.5, 0.5) for s in x})
nG2 = sp.simplify((Gy.T * Gy)[0])
FG = Gy / sp.sqrt(1 - nG2)
sym_equal("예 3.2.14(b): F(G(y)) = y", FG, y, {s: (-3, 3) for s in y})
Ys = rng.normal(size=(1000, 3)) * 5
check("예 3.2.14(b): |G(y)| < 1", bool(np.all(np.linalg.norm(Ys / np.sqrt(1 + np.sum(Ys ** 2, 1, keepdims=True)), axis=1) < 1)))

# 예 3.2.14(c): 입체사영 (n = 2 기호, n = 1..4 수치)
u = sp.Matrix(sp.symbols("u1:3", real=True))
nu2 = (u.T * u)[0]
sinv = sp.Matrix([2 * u[0], 2 * u[1], nu2 - 1]) / (nu2 + 1)
sym_equal("예 3.2.14(c): |σ^{-1}(u)|^2 = 1", (sinv.T * sinv)[0], 1)
sig = lambda X: sp.Matrix([X[0], X[1]]) / (1 - X[2])
sym_equal("예 3.2.14(c): σ(σ^{-1}(u)) = u", sig(sinv), u)
check("예 3.2.14(c): σ^{-1}(u)의 마지막 성분 < 1", sp.simplify(1 - sinv[2]) == 2 / (nu2 + 1))


def sigma(X):
    return X[..., :-1] / (1 - X[..., -1:])


def sigma_inv(U):
    n2 = np.sum(U ** 2, axis=-1, keepdims=True)
    return np.concatenate([2 * U, n2 - 1], axis=-1) / (n2 + 1)


ok = True
for n in range(1, 5):
    X = rng.normal(size=(2000, n + 1)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X[X[:, -1] < 0.999]
    ok &= bool(np.allclose(sigma_inv(sigma(X)), X, atol=1e-9))
    U = rng.normal(size=(2000, n)) * 3
    ok &= bool(np.allclose(sigma(sigma_inv(U)), U, atol=1e-9))
    # |σ(x)|^2 = (1 + x^{n+1})/(1 - x^{n+1})
    ok &= bool(np.allclose(np.sum(sigma(X) ** 2, 1), (1 + X[:, -1]) / (1 - X[:, -1])))
check("예 3.2.14(c): σ∘σ^{-1} = id, σ^{-1}∘σ = id, |σ|^2 공식 (n = 1..4, 수치)", ok)

# 비예 3.2.15 / 그림 3.2.2(a): e(2π - δ) ∈ B_ε((1,0)), 2π - δ ∉ [0, 1)
e = lambda t: np.array([np.cos(t), np.sin(t)])
eps = 0.35
check("비예 3.2.15: 작은 δ에서 e(2π-δ) ∈ B_ε((1,0)) (ε = 0.35, δ ≤ 0.3)",
      all(np.linalg.norm(e(2 * np.pi - d) - [1, 0]) < eps for d in np.linspace(1e-4, 0.3, 50)))
# e^{-1}이 (1,0)에서 불연속: e(2π - δ) → (1,0)이지만 e^{-1}(e(2π - δ)) = 2π - δ → 2π ≠ 0 = e^{-1}(1,0)
check("비예 3.2.15: e^{-1}(e(2π-δ)) → 2π ≠ 0", abs((2 * np.pi - 1e-6) - 0) > 6)

# 그림 3.2.2(b): σ(3/5, 4/5) = 3, σ(y) for y = (-sin 0.9, -cos 0.9) ≈ -0.44
close("그림 3.2.2(b): σ(y) ≈ -0.48", sigma(np.array([-np.sin(0.9), -np.cos(0.9)]))[0], -0.48, tol=5e-3)

# 연습 3.2.3
close("연습 3.2.3(a): σ(3/5, 4/5) = 3", sigma(np.array([0.6, 0.8]))[0], 3.0)
close("연습 3.2.3(a): σ^{-1}(3) = (3/5, 4/5)", sigma_inv(np.array([3.0])), [0.6, 0.8])
close("연습 3.2.3(a): σ(0, -1) = 0", sigma(np.array([0.0, -1.0]))[0], 0.0)
close("연습 3.2.3(b): σ(0,0,-1) = (0,0)", sigma(np.array([0.0, 0.0, -1.0])), [0, 0])
close("연습 3.2.3(b): σ^{-1}(1,1) = (2,2,1)/3", sigma_inv(np.array([1.0, 1.0])), np.array([2, 2, 1]) / 3)

# 연습 3.2.6: A^T A = I이면 성분 제곱합 = n (무작위 직교행렬)
ok = True
for n in (2, 3, 5):
    Qm, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ok &= bool(np.allclose(Qm.T @ Qm, np.eye(n)) and abs(np.sum(Qm ** 2) - n) < 1e-10)
check("연습 3.2.6: O(n)의 원소는 노름 sqrt(n)", ok)


# 연습 3.2.7: h(x) = (|x|_inf/|x|) x, k(y) = (|y|/|y|_inf) y
def h(X):
    ni, n2 = np.max(np.abs(X), 1, keepdims=True), np.linalg.norm(X, axis=1, keepdims=True)
    return ni / n2 * X


def k(Y):
    ni, n2 = np.max(np.abs(Y), 1, keepdims=True), np.linalg.norm(Y, axis=1, keepdims=True)
    return n2 / ni * Y


Xq = rng.uniform(-1, 1, size=(5000, 2))
Yd = rng.normal(size=(5000, 2)); Yd *= (rng.uniform(0, 1, size=(5000, 1)) / np.linalg.norm(Yd, axis=1, keepdims=True))
check("연습 3.2.7: h(Q) ⊆ B_1(0), k(B_1(0)) ⊆ Q",
      bool(np.all(np.linalg.norm(h(Xq), axis=1) < 1) and np.all(np.max(np.abs(k(Yd)), 1) < 1)))
check("연습 3.2.7: k∘h = id, h∘k = id", bool(np.allclose(k(h(Xq)), Xq) and np.allclose(h(k(Yd)), Yd)))
check("연습 3.2.7: |h(x)| = |x|_inf <= |x|, |k(y)| <= sqrt2 |y|",
      bool(np.allclose(np.linalg.norm(h(Xq), axis=1), np.max(np.abs(Xq), 1))
           and np.all(np.linalg.norm(k(Yd), axis=1) <= np.sqrt(2) * np.linalg.norm(Yd, axis=1) + 1e-12)))

summary()
