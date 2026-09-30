"""3.5절 하우스도르프 조건과 제2가산 조건: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/verify/v-3-5-hausdorff-second-countable.py``
"""

from fractions import Fraction
from itertools import combinations

import numpy as np

from dgcheck import check, close, summary

rng = np.random.default_rng(35)

# 예 3.5.2(a): r = d(p,q)/2이면 B_r(p) ∩ B_r(q) = ∅ (표본)
ok = True
for _ in range(300):
    p, q = rng.normal(size=(2, 3))
    r = np.linalg.norm(p - q) / 2
    Z = rng.normal(size=(3000, 3)) * 2 + (p + q) / 2
    ok &= not np.any((np.linalg.norm(Z - p, axis=1) < r) & (np.linalg.norm(Z - q, axis=1) < r))
check("예 3.5.2(a): 반지름 d/2인 두 공은 서로소 (표본)", bool(ok))

# 명제 3.5.7: y ≠ ±x이면 ε = min(|x-y|, |x+y|)/2에 대해 네 공 B_ε(±x), B_ε(±y) 중 x쪽과 y쪽이 서로소
ok = True
for n in (1, 2, 3):
    for _ in range(500):
        x = rng.normal(size=n + 1); x /= np.linalg.norm(x)
        y = rng.normal(size=n + 1); y /= np.linalg.norm(y)
        eps = 0.5 * min(np.linalg.norm(x - y), np.linalg.norm(x + y))
        # 중심 거리 >= 2ε이면 열린공은 서로소
        ok &= np.linalg.norm(x - y) >= 2 * eps - 1e-15 and np.linalg.norm(x + y) >= 2 * eps - 1e-15
        # 직접 표본: W_x의 점 a와 W_y의 점 b가 ±로 같지 않음
        A = x + eps * 0.999 * rng.uniform(-1, 1, size=(400, n + 1)) / np.sqrt(n + 1)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        A = A[np.linalg.norm(A - x, axis=1) < eps]
        ok &= not np.any(np.linalg.norm(A - y, axis=1) < eps)      # b = a 불가
        ok &= not np.any(np.linalg.norm(-A - y, axis=1) < eps)     # b = -a 불가
check("명제 3.5.7: RP^n의 두 점을 분리하는 ε (n = 1,2,3, 표본)", bool(ok))

# 명제 3.5.9 / 그림 3.5.2: r = 1/5, q = (3/10, 1/5), p = (0.43, 0.27), ε ≈ 0.4248
p = np.array([0.43, 0.27]); q = np.array([0.3, 0.2]); r = 0.2; eps = 0.4248
check("그림 3.5.2: |q - p| < r, |q - p| + r < ε", np.linalg.norm(q - p) < r and np.linalg.norm(q - p) + r < eps)
# 명제 3.5.9의 일반 논증: |q^i - p^i| < r/sqrt(n)이면 |q - p| < r, 그리고 B_r(q) ⊆ B_2r(p)
ok = True
for n in range(1, 6):
    for _ in range(300):
        p = rng.normal(size=n); e = rng.uniform(0.01, 1)
        r = Fraction(int(e / 2 * 1000) - 1, 1000) if int(e / 2 * 1000) > 1 else Fraction(1, 1000)
        rf = float(r)
        qn = np.array([float(Fraction(round(pi * 10 ** 6), 10 ** 6)) for pi in p])   # 유리점
        ok &= bool(np.all(np.abs(qn - p) < rf / np.sqrt(n)) and np.linalg.norm(qn - p) < rf and 2 * rf < e)
        Zs = qn + rng.normal(size=(200, n)) * rf
        Zs = Zs[np.linalg.norm(Zs - qn, axis=1) < rf]
        ok &= bool(np.all(np.linalg.norm(Zs - p, axis=1) < e))
check("명제 3.5.9: 유리점 q, 유리 반지름 r로 p ∈ B_r(q) ⊆ B_ε(p) (표본)", bool(ok))

# 연습 3.5.1: {a,b} 위의 위상 네 개, 하우스도르프는 이산위상뿐
X = frozenset("ab")
subsets = [frozenset(), frozenset("a"), frozenset("b"), X]


def is_top(T):
    return frozenset() in T and X in T and all(u | v in T and u & v in T for u in T for v in T)


tops = [set(T) for k in range(len(subsets) + 1) for T in combinations(subsets, k) if is_top(set(T))]
haus = [T for T in tops if any("a" in U and "b" in V and not (U & V) for U in T for V in T)]
check("연습 3.5.1: {a,b} 위의 위상은 4개", len(tops) == 4)
check("연습 3.5.1: 하우스도르프는 이산위상 하나", len(haus) == 1 and set(haus[0]) == set(subsets))

# 연습 3.5.7: x, y ≠ 0이 일차종속 ⇔ 모든 2×2 소행렬식 = 0
ok = True
for _ in range(2000):
    n1 = rng.integers(2, 5)
    x = rng.normal(size=n1)
    y = rng.normal() * x if rng.uniform() < 0.5 else rng.normal(size=n1)
    dep = np.linalg.matrix_rank(np.column_stack([x, y]), tol=1e-9) <= 1
    minors = [x[i] * y[j] - x[j] * y[i] for i in range(n1) for j in range(i + 1, n1)]
    ok &= dep == bool(np.allclose(minors, 0, atol=1e-9))
check("연습 3.5.7: 일차종속 ⇔ 소행렬식 모두 0 (표본)", bool(ok))

summary()
