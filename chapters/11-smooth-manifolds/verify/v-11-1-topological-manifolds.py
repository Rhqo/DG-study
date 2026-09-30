"""11.1절 위상다양체: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

입체사영 σ와 남극 입체사영 σ̃는 §7의 식을 그대로 쓴다(dgsym.EXAMPLES에는 아직 항목이 없다).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/verify/v-11-1-topological-manifolds.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(111)

# ---------------------------------------------------------------- 예 11.1.6: 남극 입체사영
# §7: σ(x) = (x^1, …, x^n)/(1 - x^{n+1}),  σ̃(x) = -σ(-x) = (x^1, …, x^n)/(1 + x^{n+1})
n = 3
X = sp.Matrix(sp.symbols("x1:5", real=True))


def sigma(x):
    return sp.Matrix(x[:-1]) / (1 - x[-1])


sym_equal("예 11.1.6: -σ(-x) = x'/(1 + x^{n+1}) (n = 3)", -sigma(-X), sp.Matrix(X[:-1]) / (1 + X[-1]),
          {s: (-0.4, 0.4) for s in X})

U = sp.Matrix(sp.symbols("u1:4", real=True))
nu2 = (U.T * U)[0]
sig_inv = sp.Matrix([2 * U[0], 2 * U[1], 2 * U[2], nu2 - 1]) / (nu2 + 1)          # (3.2.4)
sigt_inv = sp.Matrix([2 * U[0], 2 * U[1], 2 * U[2], 1 - nu2]) / (nu2 + 1)         # (11.1.1)
sym_equal("예 11.1.6: σ̃^{-1}(u) = -σ^{-1}(-u)", sigt_inv, -sig_inv.subs({U[i]: -U[i] for i in range(3)}, simultaneous=True))
sym_equal("예 11.1.6: |σ̃^{-1}(u)|^2 = 1", (sigt_inv.T * sigt_inv)[0], 1)
sym_equal("예 11.1.6: σ̃(σ̃^{-1}(u)) = u", sp.Matrix(sigt_inv[:-1]) / (1 + sigt_inv[-1]), U)
check("예 11.1.6: σ̃^{-1}(u)의 마지막 성분 > -1", sp.simplify(1 + sigt_inv[-1]) == 2 / (nu2 + 1))

ok = True
for m in range(1, 5):
    P = rng.normal(size=(2000, m + 1))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    P = P[P[:, -1] > -0.999]
    st = P[:, :-1] / (1 + P[:, -1:])
    n2 = np.sum(st ** 2, axis=1, keepdims=True)
    back = np.concatenate([2 * st, 1 - n2], axis=1) / (n2 + 1)
    ok &= bool(np.allclose(back, P, atol=1e-9))
check("예 11.1.6: σ̃^{-1}∘σ̃ = id (n = 1..4, 수치)", ok)

# ---------------------------------------------------------------- 예 11.1.8: 그래프와 원뿔
f = lambda x, y: np.sqrt(x ** 2 + y ** 2)
pts = rng.uniform(-2, 2, size=(1000, 2))
G = np.column_stack([pts, f(*pts.T)])
check("예 11.1.8 / 연습 11.1.2: π∘(x ↦ (x, f(x))) = id, 원뿔 위의 점", bool(np.allclose(G[:, :2], pts)) and bool(np.allclose(G[:, 2] ** 2, G[:, 0] ** 2 + G[:, 1] ** 2)))

# ---------------------------------------------------------------- 비예 11.1.11: 공에서 한 점을 빼도 경로연결 (n ≥ 2)
ok = True
for n_ in (2, 3):
    for _ in range(300):
        c = rng.normal(size=n_)
        delta = 1.0
        # p, q, c가 한 직선 위에 있고 선분 [p, q]가 c를 지나는 나쁜 경우
        d = rng.normal(size=n_); d /= np.linalg.norm(d)
        p = c - 0.6 * delta * d * rng.uniform(0.2, 1)
        q = c + 0.6 * delta * d * rng.uniform(0.2, 1)
        e = rng.normal(size=n_); e -= e.dot(d) * d; e /= np.linalg.norm(e)
        r = c + 0.5 * delta * e                                  # 직선 밖, 공 안
        ss = np.linspace(0, 1, 201)[:, None]
        seg1 = (1 - ss) * p + ss * r
        seg2 = (1 - ss) * r + ss * q
        for seg in (seg1, seg2):
            ok &= bool(np.all(np.linalg.norm(seg - c, axis=1) < delta))       # 공 안 (볼록성)
            ok &= bool(np.min(np.linalg.norm(seg - c, axis=1)) > 1e-6)         # c를 지나지 않음
check("비예 11.1.11: B_δ(c) \\ {c}의 두 점을 우회 경로로 잇기 (n = 2, 3, 수치)", ok)

# ---------------------------------------------------------------- 명제 11.1.15(b): r < ε/3, |q - x| < r이면 닫힌공 B̄_r(q) ⊆ B_ε(x)
ok = True
for _ in range(2000):
    n_ = rng.integers(1, 5)
    x = rng.normal(size=n_)
    eps = rng.uniform(0.1, 2)
    r = rng.uniform(0.01, 1) * eps / 3
    v = rng.normal(size=n_); v /= np.linalg.norm(v)
    q = x + rng.uniform(0, 0.999) * r * v
    w = rng.normal(size=n_); w /= np.linalg.norm(w)
    y = q + rng.uniform(0, 1) * r * w                  # |y - q| ≤ r
    ok &= bool(np.linalg.norm(x - q) < r and np.linalg.norm(y - x) < eps)
check("명제 11.1.15(b): B̄_r(q) ⊆ B_ε(x) (수치)", ok)

# ---------------------------------------------------------------- 연습 11.1.6: RP^1 ≅ S^1, 제곱 사상
a, b = sp.symbols("a b", real=True)
sq = sp.Matrix([a ** 2 - b ** 2, 2 * a * b])
sym_equal("연습 11.1.6: |s(a, b)|^2 = (a^2 + b^2)^2", (sq.T * sq)[0], (a ** 2 + b ** 2) ** 2)
sym_equal("연습 11.1.6: s(-z) = s(z)", sq.subs({a: -a, b: -b}, simultaneous=True), sq)
t_, s_ = sp.symbols("t s", real=True)
sym_equal("연습 11.1.6: s(cos t, sin t) = (cos 2t, sin 2t)", sq.subs({a: sp.cos(t_), b: sp.sin(t_)}), sp.Matrix([sp.cos(2 * t_), sp.sin(2 * t_)]))
ok = True
for _ in range(500):
    t1, t2 = rng.uniform(0, 2 * np.pi, 2)
    same = np.allclose([np.cos(2 * t1), np.sin(2 * t1)], [np.cos(2 * t2), np.sin(2 * t2)], atol=1e-12)
    pm = np.isclose(np.cos(t1 - t2) ** 2, 1, atol=1e-9)
    ok &= bool(same == pm) or not same
check("연습 11.1.6: s(z) = s(w)이면 w = ±z (수치)", ok)
th = np.linspace(0, np.pi, 7)[:-1]
ok = all(not np.allclose([np.cos(2 * t1), np.sin(2 * t1)], [np.cos(2 * t2), np.sin(2 * t2)]) for i, t1 in enumerate(th) for t2 in th[i + 1:])
check("연습 11.1.6: 반원 [0, π)의 서로 다른 각은 서로 다른 상 (단사)", ok)

# ---------------------------------------------------------------- 연습 11.1.7: S^n \ {N, S} ≅ S^{n-1} × R
ok = True
for m in (1, 2, 3):
    P = rng.normal(size=(2000, m + 1)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    P = P[np.abs(P[:, -1]) < 0.999]
    uu = P[:, :-1] / (1 - P[:, -1:])
    ru = np.linalg.norm(uu, axis=1, keepdims=True)
    v, t = uu / ru, np.log(ru)
    # 역사상: (v, t) ↦ σ^{-1}(e^t v)
    w = np.exp(t) * v
    n2 = np.sum(w ** 2, axis=1, keepdims=True)
    back = np.concatenate([2 * w, n2 - 1], axis=1) / (n2 + 1)
    ok &= bool(np.allclose(back, P, atol=1e-9)) and bool(np.allclose(np.linalg.norm(v, axis=1), 1))
check("연습 11.1.7: (σ/|σ|, log|σ|)와 (v, t) ↦ σ^{-1}(e^t v)는 서로 역 (n = 1, 2, 3, 수치)", ok)

summary()
