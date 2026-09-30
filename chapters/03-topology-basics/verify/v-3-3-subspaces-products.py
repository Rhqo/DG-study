"""3.3절 부분공간과 곱공간: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

원환면의 매개화는 tools/dgsym.py의 EXAMPLES["torus"]에서 가져온다 (§3.6).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/verify/v-3-3-subspaces-products.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

rng = np.random.default_rng(33)

# 예 3.3.2(d), 연습 3.3.3(d): {k} = Z ∩ (k - 1/2, k + 1/2); {1/k} = S ∩ (1/(k+1), 1/(k-1))
Z = np.arange(-20, 21)
check("예 3.3.2(d): Z ∩ (k-1/2, k+1/2) = {k}",
      all(set(Z[(Z > k - 0.5) & (Z < k + 0.5)]) == {k} for k in Z))
S = np.array([1 / j for j in range(1, 400)])
ok = True
for k in range(1, 300):
    hi = 2.0 if k == 1 else 1 / (k - 1)
    sel = S[(S > 1 / (k + 1)) & (S < hi)]
    ok &= len(sel) == 1 and abs(sel[0] - 1 / k) < 1e-15
check("연습 3.3.3(d): {1/k}는 부분공간에서 열려 있다", ok)

# 비예 3.3.11: 8자 곡선 β(t) = (sin 2t, sin t)
t, s = sp.symbols("t s", real=True)
beta = sp.Matrix([sp.sin(2 * t), sp.sin(t)])
x, y = beta
sym_equal("비예 3.3.11 / 연습 3.3.6: x^2 = 4y^2(1-y^2) on β", x ** 2, 4 * y ** 2 * (1 - y ** 2))
sym_equal("비예 3.3.11: s = π - t이면 sin 2s = -sin 2t", sp.sin(2 * (sp.pi - t)), -sp.sin(2 * t))
sym_equal("비예 3.3.11: s = -π - t이면 sin 2s = -sin 2t", sp.sin(2 * (-sp.pi - t)), -sp.sin(2 * t))
check("비예 3.3.11: β(t) → 0 (t → ±π)",
      sp.limit(beta[0], t, sp.pi) == 0 and sp.limit(beta[1], t, sp.pi) == 0
      and sp.limit(beta[0], t, -sp.pi) == 0 and sp.limit(beta[1], t, -sp.pi) == 0)
# 단사성 수치 확인: 격자 위 서로 다른 t의 상 사이 최소 거리 > 0 (가까운 t 쌍 제외)
bt = lambda tt: np.stack([np.sin(2 * tt), np.sin(tt)], -1)
T = np.linspace(-np.pi, np.pi, 1601)[1:-1]
P = bt(T)
Dm = np.linalg.norm(P[:, None] - P[None], axis=2)
far = np.abs(T[:, None] - T[None]) > 0.05
check("비예 3.3.11: 단사 (표본, |t - s| > 0.05이면 β(t) ≠ β(s))", bool(np.min(Dm[far]) > 1e-4))
# 그림 3.3.2: t ∈ (π - 0.12, π)의 상은 B_0.3(0) 안
tt = np.linspace(np.pi - 0.12, np.pi - 1e-6, 200)
check("그림 3.3.2: β((π-0.12, π)) ⊆ B_0.3(0)", bool(np.all(np.linalg.norm(bt(tt), axis=1) < 0.3)))

# 연습 3.3.6: L ⊆ E. y ∈ [-1,1]마다 두 t에서 x = ±2y sqrt(1-y^2)
ok = True
for yy in np.linspace(-1, 1, 401):
    t0 = np.arcsin(yy)
    t1 = np.pi - t0 if yy >= 0 else -np.pi - t0
    xs = 2 * yy * np.sqrt(max(0.0, 1 - yy ** 2))
    ok &= np.allclose(bt(t0), [xs, yy], atol=1e-12)
    if -np.pi < t1 < np.pi:
        ok &= np.allclose(bt(t1), [-xs, yy], atol=1e-12)
    else:
        ok &= abs(yy) < 1e-12
check("연습 3.3.6: L의 모든 점이 β의 상", bool(ok))

# 예 3.3.13, 식 (3.3.2): B^inf_r((x,y)) = B^inf_r(x) × B^inf_r(y)
ok = True
for _ in range(200):
    m, n = rng.integers(1, 4, size=2)
    c = rng.normal(size=m + n); r = rng.uniform(0.1, 2)
    Q = c + rng.uniform(-2.5, 2.5, size=(500, m + n))
    lhs = np.max(np.abs(Q - c), 1) < r
    rhs = (np.max(np.abs(Q[:, :m] - c[:m]), 1) < r) & (np.max(np.abs(Q[:, m:] - c[m:]), 1) < r)
    ok &= bool(np.array_equal(lhs, rhs))
check("예 3.3.13: (3.3.2) 정육면체 공 = 공의 곱", ok)

# 예 3.3.17: Φ, Ψ (원환면). 매개화는 EXAMPLES["torus"]
ex = EXAMPLES["torus"]
u, v = ex["coords"]
R, r = ex["params"]
X = ex["expr"]
a = sp.Matrix([sp.cos(u), sp.sin(u)])
b = sp.Matrix([sp.cos(v), sp.sin(v)])
Phi = sp.Matrix([(R + r * a[0]) * b[0], (R + r * a[0]) * b[1], r * a[1]])
sym_equal("예 3.3.17: Φ((cos u, sin u), (cos v, sin v)) = x(u, v)", Phi, X, ex["domain"])
rho = (R + r * sp.cos(u))  # sqrt((p1)^2 + (p2)^2)를 R + r cos u > 0으로 벗긴 값
sym_equal("예 3.3.17: (p^1)^2 + (p^2)^2 = (R + r cos u)^2", X[0] ** 2 + X[1] ** 2, rho ** 2, ex["domain"])
Psi = sp.Matrix([(rho - R) / r, X[2] / r, X[0] / rho, X[1] / rho])
sym_equal("예 3.3.17: Ψ(x(u,v)) = ((cos u, sin u), (cos v, sin v))", Psi, sp.Matrix([a[0], a[1], b[0], b[1]]), ex["domain"])
# 수치: 무작위 (R, r), 무작위 T^2 점에서 Ψ∘Φ = id, ρ >= R - r > 0
ok = True
for _ in range(300):
    Rn = rng.uniform(1.1, 3); rn = rng.uniform(0.1, 0.95) * Rn
    uu, vv = rng.uniform(0, 2 * np.pi, 2)
    A = np.array([np.cos(uu), np.sin(uu)]); Bv = np.array([np.cos(vv), np.sin(vv)])
    p = np.array([(Rn + rn * A[0]) * Bv[0], (Rn + rn * A[0]) * Bv[1], rn * A[1]])
    rh = np.hypot(p[0], p[1])
    back = np.array([(rh - Rn) / rn, p[2] / rn, p[0] / rh, p[1] / rh])
    ok &= bool(rh >= Rn - rn - 1e-12 and np.allclose(back, np.r_[A, Bv]))
check("예 3.3.17: Ψ∘Φ = id (수치), ρ >= R - r > 0", ok)

# 연습 3.3.2: 반구 그래프 사상 Γ(u) = (u, sqrt(1-|u|^2))
u1, u2 = sp.symbols("u1 u2", real=True)
G = sp.Matrix([u1, u2, sp.sqrt(1 - u1 ** 2 - u2 ** 2)])
sym_equal("연습 3.3.2: |Γ(u)|^2 = 1", (G.T * G)[0], 1, {u1: (-0.6, 0.6), u2: (-0.6, 0.6)})

summary()
