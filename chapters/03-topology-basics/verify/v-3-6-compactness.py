"""3.6절 컴팩트성: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

원환면의 매개화는 tools/dgsym.py의 EXAMPLES["torus"]에서 가져온다 (§3.6).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/verify/v-3-6-compactness.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

rng = np.random.default_rng(36)

# 비예 3.6.3(b): 유한개 (1/k, 1)의 합집합은 1/(2K)를 덮지 못한다
check("비예 3.6.3(b): 유한 부분덮개 없음 (K = 2..200)",
      all(not any(1 / k < 1 / (2 * K) < 1 for k in range(2, K + 1)) for K in range(2, 201)))


# 정리 3.6.4: [a,b]의 무작위 열린구간 덮개에서 유한 부분덮개를 탐욕적으로 찾는다 (sup 논증의 알고리즘판)
def finite_subcover(cover, a, b):
    """cover: 열린구간 목록. [a, b]를 덮는 유한 부분덮개를 왼쪽부터 찾는다 (없으면 None)."""
    reach, chosen = a, []
    while True:
        cands = [(lo, hi) for lo, hi in cover if lo < reach < hi]
        if not cands:
            return None
        lo, hi = max(cands, key=lambda c: c[1])
        chosen.append((lo, hi))
        if hi > b:
            return chosen
        reach = hi


ok = True
for _ in range(200):
    a, b = 0.0, 1.0
    # 모든 점을 덮는 무작위 덮개: 격자 점마다 작은 구간
    centers = np.linspace(a, b, 400)
    cover = [(c - w, c + w) for c, w in zip(centers, rng.uniform(0.003, 0.05, centers.size))]
    sub = finite_subcover(cover, a, b)
    ok &= sub is not None and len(sub) <= len(cover)
    if sub:
        xs = np.linspace(a, b, 5001)
        ok &= all(any(lo < x < hi for lo, hi in sub) for x in xs[::50])
check("정리 3.6.4: 무작위 덮개에서 유한 부분덮개 (표본)", bool(ok))

# 예 3.6.8(c): 원환면 방정식 (sqrt(x^2+y^2) - R)^2 + z^2 = r^2 과 유계성 |p| <= R + r
ex = EXAMPLES["torus"]
u, v = ex["coords"]; R, r = ex["params"]; X = ex["expr"]
rho = R + r * sp.cos(u)
sym_equal("예 3.6.8(c): (ρ - R)^2 + z^2 = r^2 (ρ = R + r cos u > 0)", (rho - R) ** 2 + X[2] ** 2, r ** 2, ex["domain"])
sym_equal("연습 3.6.2: |x(u,v)|^2 = R^2 + 2Rr cos u + r^2", (X.T * X)[0], R ** 2 + 2 * R * r * sp.cos(u) + r ** 2, ex["domain"])
fz = sp.lambdify((u, v, R, r), X[2], "numpy"); fn = sp.lambdify((u, v, R, r), sp.sqrt((X.T * X)[0]), "numpy")
U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 721), np.linspace(0, 2 * np.pi, 5))
Rv, rv = 2.0, 0.8
close("연습 3.6.2: max z = r, min z = -r", [fz(U, V, Rv, rv).max(), fz(U, V, Rv, rv).min()], [rv, -rv], tol=1e-6)
close("연습 3.6.2: max |p| = R + r, min |p| = R - r", [fn(U, V, Rv, rv).max(), fn(U, V, Rv, rv).min()], [Rv + rv, Rv - rv], tol=1e-6)

# 예 3.6.11(a): f = x^1 + 2x^2 on S^1, (3.6.2), max √5 at (1,2)/√5
th = sp.symbols("theta", real=True)
th0 = sp.atan2(2, 1)
sym_equal("예 3.6.11(a): cosθ + 2 sinθ = √5 cos(θ - θ0)", sp.cos(th) + 2 * sp.sin(th), sp.sqrt(5) * sp.cos(th - th0))
T = np.linspace(0, 2 * np.pi, 200001)
F = np.cos(T) + 2 * np.sin(T)
close("예 3.6.11(a): max = √5, min = -√5", [F.max(), F.min()], [np.sqrt(5), -np.sqrt(5)], tol=1e-9)
imax = np.argmax(F)
close("예 3.6.11(a): 최대점 (1,2)/√5", [np.cos(T[imax]), np.sin(T[imax])], np.array([1, 2]) / np.sqrt(5), tol=1e-4)

# 보조정리 3.6.17 / 예 3.6.18: δ = 0.05가 르베그 수, 더 크면 x = 0.35에서 실패
cover = [(-0.1, 0.4), (0.3, 0.75), (0.6, 1.1)]


def ok_delta(d, xs=np.linspace(0, 1, 20001)):
    return all(any(a <= max(0, x - d) and min(1, x + d) <= b for a, b in cover)   # 열린 J ⊆ 열린 U: 끝점 비교는 <=
               for x in xs)


check("예 3.6.18: δ = 0.05는 르베그 수", ok_delta(0.05))
check("예 3.6.18: δ = 0.051은 실패 (x = 0.35)", not ok_delta(0.051) and
      not any(a <= 0.35 - 0.051 and 0.35 + 0.051 <= b for a, b in cover))

# 따름정리 3.6.19 대비: 컴팩트가 아니면 균등연속이 아닐 수 있다 — 1/x on (0,1)
xs = np.array([10.0 ** (-k) for k in range(2, 8)])
check("따름정리 3.6.19 대비: |1/x - 1/(2x)| = 1/(2x) → ∞ while |x - 2x| → 0",
      bool(np.all(np.diff(1 / (2 * xs)) > 0) and np.all(np.diff(xs) < 0)))

# 명제 3.6.16: 유계 수열의 수렴하는 부분수열 (수치 예: x_k = (cos k, sin k) ∈ S^1, 부분수열이 (1,0)에 가까워짐)
k = np.arange(1, 200000)
pts = np.stack([np.cos(k), np.sin(k)], 1)
d = np.linalg.norm(pts - [1, 0], axis=1)
best = np.minimum.accumulate(d)
check("명제 3.6.16: (cos k, sin k)의 부분수열이 (1,0)에 다가간다 (최소 거리 감소)", best[-1] < 1e-4)

# 연습 3.6.5: kI의 노름 = k sqrt(n), O(n) 노름 sqrt(n)
check("연습 3.6.5: ||kI|| = k sqrt n", all(np.isclose(np.linalg.norm(k_ * np.eye(n)), k_ * np.sqrt(n)) for k_ in (1, 5, 10) for n in (1, 2, 3)))

summary()
