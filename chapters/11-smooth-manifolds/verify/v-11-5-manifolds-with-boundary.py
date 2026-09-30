"""11.5절 경계를 갖는 다양체: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

ℍⁿ = {x ∈ ℝⁿ : x^n ≥ 0} (§6.4).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/verify/v-11-5-manifolds-with-boundary.py``
"""

import itertools

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(115)

# ---------------------------------------------------------------- 예 11.5.2 / 연습 11.5.1: [0, 1]
s = sp.symbols("s", real=True)
psi0 = lambda x: x          # [0, 1) → [0, 1) ⊆ ℍ¹
psi1 = lambda x: 1 - x      # (0, 1] → [0, 1) ⊆ ℍ¹
check("예 11.5.2: ψ_1∘ψ_0^{-1}(s) = 1 - s, 역도 같은 꼴", sp.simplify(psi1(psi0(s)) - (1 - s)) == 0 and sp.simplify(psi0(psi1(psi1(s))) - s) == 0)
check("예 11.5.2: ψ_0(0) = 0, ψ_1(1) = 0 ∈ ∂ℍ¹", psi0(0) == 0 and psi1(1) == 0)


# ---------------------------------------------------------------- 예 11.5.13: 닫힌 공의 경계 차트
def bchart(x, i, sign):
    """ψ_i^±(x) = (x에서 i번째 성분을 뺀 y, √(1 - |y|²) ∓ x^i), i는 1부터."""
    x = list(x)
    y = x[: i - 1] + x[i:]
    return sp.Matrix(y + [sp.sqrt(1 - sum(c ** 2 for c in y)) - sign * x[i - 1]])


def bchart_inv(w, i, sign):
    w = list(w)
    y, t = w[:-1], w[-1]
    xi = sign * (sp.sqrt(1 - sum(c ** 2 for c in y)) - t)
    return sp.Matrix(y[: i - 1] + [xi] + y[i - 1:])


for n in (1, 2, 3):
    X = sp.symbols(f"x1:{n + 1}", real=True)
    W = sp.symbols(f"w1:{n + 1}", real=True)
    for i in range(1, n + 1):
        for sg in (1, -1):
            dom = {c: (-0.3 / n, 0.3 / n) for c in W[:-1]}
            dom[W[-1]] = (0.05, 0.3)
            sym_equal(f"예 11.5.13: ψ_{i}^± ∘ (ψ_{i}^±)^{{-1}} = id (n = {n}, 부호 {sg})", bchart(bchart_inv(W, i, sg), i, sg), sp.Matrix(W), dom)
            P = bchart_inv(W, i, sg)
            # t = 0이면 |x| = 1 (경계 구면 위)
            on_sphere = sp.simplify(sum(c ** 2 for c in P).subs(W[-1], 0))
            check(f"예 11.5.13: t = 0 ⇔ |x| = 1: |(ψ_{i}^±)^{{-1}}(y, 0)|^2 = 1 (n = {n}, 부호 {sg})", on_sphere == 1)

# 수치: ψ_i^±는 V_i^± = {±x^i > 0} ∩ B̄ⁿ을 {|y| < 1, 0 ≤ t < √(1 - |y|²)}로 보낸다
ok = True
for n in (2, 3):
    Xs = rng.uniform(-1, 1, size=(20000, n))
    Xs = Xs[np.linalg.norm(Xs, axis=1) <= 1]
    for i in range(n):
        for sg in (1, -1):
            V = Xs[sg * Xs[:, i] > 0]
            y = np.delete(V, i, axis=1)
            h = np.sqrt(1 - np.sum(y ** 2, axis=1))
            t = h - sg * V[:, i]
            ok &= bool(np.all(np.linalg.norm(y, axis=1) < 1) and np.all(t >= -1e-12) and np.all(t < h + 1e-12))
check("예 11.5.13: 치역 조건 |y| < 1, 0 ≤ t < √(1 - |y|²) (n = 2, 3, 수치)", ok)

# 유도된 경계 차트 = 반구 차트 (11.3.7): ψ̂_i^±(x) = x에서 i번째 성분을 뺀 것
n = 3
X = sp.symbols("x1:4", real=True)
for i in (1, 2, 3):
    check(f"예 11.5.13: 경계 차트를 ∂ℍ³로 제한한 앞 두 성분 = 반구 차트 φ_{i}^± (x를 빼기)", list(bchart(X, i, 1))[:-1] == [X[k] for k in range(3) if k != i - 1])

# 경계 차트끼리의 좌표변환: n = 2, ψ_2^+ → ψ_1^+ (정의역: w^1 > 0 부분)
W2 = sp.symbols("w1 w2", real=True)
T = bchart(bchart_inv(W2, 2, 1), 1, 1)
# 공식: x = (w1, √(1-w1²) - w2), ψ_1^+(x) = (x2, √(1 - x2²) - x1)
x1, x2 = W2[0], sp.sqrt(1 - W2[0] ** 2) - W2[1]
sym_equal("예 11.5.13: ψ_1^+∘(ψ_2^+)^{-1} = (√(1-w1²) - w2, √(1-x2²) - w1)", T, sp.Matrix([x2, sp.sqrt(1 - x2 ** 2) - x1]),
          {W2[0]: (0.2, 0.6), W2[1]: (0.0, 0.2)})
# 경계에서 경계로: w2 = 0이면 결과의 둘째 성분 = 0
check("예 11.5.13: ψ_1^+∘(ψ_2^+)^{-1}은 ∂ℍ²를 ∂ℍ²로 (w1 > 0)", sp.simplify(T[1].subs(W2[1], 0).subs(W2[0], sp.Rational(3, 5))) == 0)

# 그림 11.5.1
p = np.array([0.3, 0.45]); q = np.array([-0.6, 0.8])
close("그림 11.5.1: ψ_2^+(p) = (0.3, √0.91 - 0.45)", [p[0], np.sqrt(1 - p[0] ** 2) - p[1]], [0.3, np.sqrt(0.91) - 0.45], tol=1e-12)
close("그림 11.5.1: ψ_2^+(q) = (-0.6, 0)", [q[0], np.sqrt(1 - q[0] ** 2) - q[1]], [-0.6, 0.0], tol=1e-12)

# ---------------------------------------------------------------- 정리 11.5.11: 역함수 정리 논증의 핵심 단계 (수치 예)
# η = ψ∘φ^{-1}가 경계점으로 가면 열린 근방의 상이 ℍⁿ 밖으로 나간다: 예로 η = id를 쓰면 b = 0 ∈ ∂ℍ²의 근방에 x^2 < 0인 점이 있다
b = np.array([0.2, 0.0])
check("정리 11.5.11: ∂ℍⁿ의 점 b의 모든 공은 x^n < 0인 점 b - (ε/2)e_n을 포함", all((b - np.array([0, e / 2]))[1] < 0 and np.linalg.norm(np.array([0, e / 2])) < e for e in (1.0, 0.1, 1e-3)))

# ---------------------------------------------------------------- 비고 11.5.14: 경계점에서의 전미분은 확장에 무관
# F(x) = e^x (x ≥ 0)의 두 매끄러운 확장: F1 = e^x, F2 = e^x + b(x), b(x) = e^{-1/x^2} (x < 0), 0 (x ≥ 0)
x = sp.symbols("x", real=True)
xn = sp.symbols("xn", negative=True)
b_left = sp.exp(-1 / xn ** 2)
check("비고 11.5.14: b의 0에서의 왼쪽 도함수 극한 = 0 (b'(x) = 2x^{-3} e^{-1/x^2} → 0)", sp.limit(sp.diff(b_left, xn), xn, 0, "-") == 0)
check("비고 11.5.14: b(x)/x → 0 (x → 0-)이므로 F2'(0) = F1'(0) = 1", sp.limit(b_left / xn, xn, 0, "-") == 0 and sp.diff(sp.exp(x), x).subs(x, 0) == 1)

# ---------------------------------------------------------------- 연습 11.5.4: √x는 [0, ∞)에서 매끄럽지 않다
h = sp.symbols("h", positive=True)
check("연습 11.5.4: √h/h → ∞ (h → 0+)", sp.limit(sp.sqrt(h) / h, h, 0, "+") == sp.oo)

# ---------------------------------------------------------------- 연습 11.5.7: 닫힌 윗반구
ok = True
for n in (1, 2, 3):
    Xs = rng.normal(size=(3000, n + 1)); Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    Xs = Xs[Xs[:, -1] >= 0]
    pr = Xs[:, :-1]
    back = np.column_stack([pr, np.sqrt(np.clip(1 - np.sum(pr ** 2, axis=1), 0, None))])
    ok &= bool(np.allclose(back, Xs)) and bool(np.all(np.linalg.norm(pr, axis=1) <= 1 + 1e-12))
check("연습 11.5.7: 닫힌 윗반구 → B̄ⁿ, x ↦ x'과 역 w ↦ (w, √(1 - |w|²)) (n = 1, 2, 3, 수치)", ok)

# ---------------------------------------------------------------- 비예 11.5.3(b): 반공 B_δ(c) ∩ ℍⁿ에서 한 점을 빼도 경로연결 (n ≥ 2)
ok = True
for n_ in (2, 3):
    for _ in range(300):
        c = np.zeros(n_); c[:-1] = rng.normal(size=n_ - 1)       # c ∈ ∂ℍⁿ (가장 나쁜 경우)
        delta = 1.0
        d = rng.normal(size=n_); d[-1] = abs(d[-1]); d /= np.linalg.norm(d)
        p = c + 0.5 * d * rng.uniform(0.2, 1)                     # 반공 안
        q = c - 0.5 * d * rng.uniform(0.2, 1)
        q[-1] = max(q[-1], 0.0)
        # r: 반공의 내부, p, q, c를 지나는 평면(직선) 밖의 점
        r = c + np.eye(n_)[-1] * 0.5
        if abs(np.linalg.det(np.vstack([p - c, r - c])[:, :2])) < 1e-9 and n_ == 2:
            r = c + np.array([0.3, 0.4])
        ss = np.linspace(0, 1, 201)[:, None]
        for a_, b_ in ((p, r), (r, q)):
            seg = (1 - ss) * a_ + ss * b_
            inside = np.all(np.linalg.norm(seg - c, axis=1) < delta) and np.all(seg[:, -1] >= -1e-12)
            avoid = np.min(np.linalg.norm(seg - c, axis=1)) > 1e-9 or np.allclose(a_, c) or np.allclose(b_, c)
            ok &= bool(inside)
            # 선분이 c를 지나지 않는지는 r이 직선 밖에 있으면 성립; 수치로 확인 (p, q가 c와 한 직선 위일 때)
            if abs(np.cross(np.append(p - c, 0)[:3], np.append(q - c, 0)[:3])).max() < 1e-9:
                ok &= bool(avoid)
check("비예 11.5.3(b): 반공 ∩ ℍⁿ 안의 우회 경로 (볼록성, n = 2, 3, 수치)", ok)

summary()
