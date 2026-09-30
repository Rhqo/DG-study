"""3.7절 연결성: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/verify/v-3-7-connectedness.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, summary

rng = np.random.default_rng(37)

# 그림 3.7.1(a): 두 원판은 서로소이고 x^1 = 0.05가 가른다
c1, r1 = np.array([-0.7, 0.0]), 0.55
c2, r2 = np.array([0.8, 0.1]), 0.45
check("비예 3.7.3(c) / 그림 3.7.1(a): D_1 ⊆ {x^1 < 0.05}, D_2 ⊆ {x^1 > 0.05}",
      c1[0] + r1 < 0.05 < c2[0] - r2 and np.linalg.norm(c1 - c2) > r1 + r2)
# 그림 3.7.1(b): 경로가 고리 안
t = np.linspace(0, 1, 10001)
rho = 0.75 + 0.1 * np.sin(3 * np.pi * t)
check("그림 3.7.1(b): 1/2 < ρ(t) < 1", bool(np.all((rho > 0.5) & (rho < 1))))

# 예 3.7.10(b): p, q ≠ 0이 원점을 지나는 직선 위에 있으면 직선 밖의 r로 우회; 선분 [p,r], [r,q]는 원점을 피한다
def seg_hits_origin(a, b):
    """선분 [a, b]가 원점을 지나는가 (a, b ≠ 0)."""
    # 원점 = (1-s)a + s b ⇔ a, b가 반대 방향으로 평행
    M = np.column_stack([a, b])
    if np.linalg.matrix_rank(M, tol=1e-12) == 2:
        return False
    return float(a @ b) < 0


ok = True
for n in (2, 3, 4):
    for _ in range(300):
        p = rng.normal(size=n); q = -rng.uniform(0.2, 3) * p           # 원점을 사이에 둔 두 점
        assert seg_hits_origin(p, q)
        e = rng.normal(size=n); e -= (e @ p) / (p @ p) * p              # 직선 밖의 방향
        r = p + e
        ok &= not seg_hits_origin(p, r) and not seg_hits_origin(r, q)
check("예 3.7.10(b): 직선 밖의 점 r로 우회하면 원점을 피한다 (n = 2,3,4)", bool(ok))

# 예 3.7.10(e): det I = 1 > 0, det diag(-1, 1, ..., 1) = -1 < 0
check("예 3.7.10(e): det의 부호가 둘 다 나타난다",
      all(np.isclose(np.linalg.det(np.eye(n)), 1) and np.isclose(np.linalg.det(np.diag([-1] + [1] * (n - 1))), -1)
          for n in range(1, 6)))

# 비예 3.7.11: 사인 곡선의 폐포 — (0, y0)에 수렴하는 점 x_k = 1/(arcsin y0 + 2πk)
ok = True
for y0 in np.linspace(-1, 1, 21):
    xk = 1 / (np.arcsin(y0) + 2 * np.pi * np.arange(1, 2000))
    ok &= bool(np.allclose(np.sin(1 / xk), y0, atol=1e-9) and xk[-1] < 1e-3 and np.all(xk > 0))
check("비예 3.7.11: (x_k, y0) → (0, y0), sin(1/x_k) = y0", ok)
k = np.arange(1, 50)
check("비예 3.7.11 / 그림 3.7.2: sin(1/x_k) = 1, sin(1/x'_k) = -1",
      bool(np.allclose(np.sin(np.pi / 2 + 2 * np.pi * k), 1) and np.allclose(np.sin(3 * np.pi / 2 + 2 * np.pi * k), -1)))

# 연습 3.7.2: x^3 - 3x + 1의 부호와 근
x = sp.symbols("x", real=True)
p3 = x ** 3 - 3 * x + 1
check("연습 3.7.2: p(0)=1, p(1)=-1, p(2)=3", (p3.subs(x, 0), p3.subs(x, 1), p3.subs(x, 2)) == (1, -1, 3))
roots = sorted(float(rt) for rt in sp.real_roots(sp.Poly(p3, x)))
check("연습 3.7.2: (0,1)과 (1,2)에 근이 하나씩", sum(0 < rt < 1 for rt in roots) == 1 and sum(1 < rt < 2 for rt in roots) == 1)

# 연습 3.7.5: g(x) = f(x) - f(-x)는 홀함수이고 0이 되는 점이 있다 (예: f(θ) = cos θ + sin 2θ + 0.3 sin 3θ)
th = np.linspace(0, 2 * np.pi, 100001)
f = lambda a: np.cos(a) + np.sin(2 * a) + 0.3 * np.sin(3 * a) + 0.5 * np.cos(a) ** 2
g = f(th) - f(th + np.pi)
check("연습 3.7.5: g(θ + π) = -g(θ)", bool(np.allclose(f(th + np.pi) - f(th + 2 * np.pi), -g)))
check("연습 3.7.5: g가 부호를 바꾼다 (0이 되는 점 존재)", bool(np.any(np.sign(g[:-1]) != np.sign(g[1:]))))

summary()
