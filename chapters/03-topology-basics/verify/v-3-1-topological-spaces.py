"""3.1절 열린집합과 위상공간: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/verify/v-3-1-topological-spaces.py``
"""

from itertools import combinations

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(31)


def d2(x, y):
    return np.linalg.norm(x - y, axis=-1)


def dinf(x, y):
    return np.max(np.abs(x - y), axis=-1)


def d1(x, y):
    return np.sum(np.abs(x - y), axis=-1)


# 예 3.1.2: 세 거리의 삼각부등식 (무작위 표본, n = 1..5)
ok = True
for n in range(1, 6):
    X, Y, Z = (rng.normal(size=(4000, n)) * rng.uniform(0.01, 5, size=(4000, 1)) for _ in range(3))
    for d in (d2, dinf, d1):
        ok &= bool(np.all(d(X, Z) <= d(X, Y) + d(Y, Z) + 1e-12))
check("예 3.1.2: d, d_inf, d_1의 삼각부등식 (표본)", ok)

# 비예 3.1.3: rho = (x - y)^2는 삼각부등식을 어긴다
rho = lambda x, y: (x - y) ** 2
check("비예 3.1.3: rho(0,2) = 4 > rho(0,1) + rho(1,2) = 2", rho(0, 2) == 4 and rho(0, 1) + rho(1, 2) == 2)

# 예 3.1.4(b): 이산거리의 삼각부등식 (모든 경우)
dd = lambda x, y: 0 if x == y else 1
pts = ["a", "b", "c"]
check("예 3.1.4(b): 이산거리 삼각부등식", all(dd(x, z) <= dd(x, y) + dd(y, z) for x in pts for y in pts for z in pts))

# 보조정리 3.1.6: q ∈ B_r(p), s = r - d(p,q)이면 B_s(q) ⊆ B_r(p) (표본)
ok = True
for d in (d2, dinf, d1):
    for _ in range(200):
        p = rng.normal(size=3); r = rng.uniform(0.1, 2)
        dirn = rng.normal(size=3); dirn /= d(dirn, 0 * dirn)
        q = p + rng.uniform(0, 0.999) * r * dirn          # d(p, q) < r
        s = r - d(p, q)
        pts_ = q + rng.normal(size=(500, 3))
        pts_ = pts_[d(pts_, q) < s]
        ok &= bool(np.all(d(pts_, p) < r))
check("보조정리 3.1.6: B_s(q) ⊆ B_r(p) (세 거리, 표본)", ok)

# 비예 3.1.8(b): 경계점 q = (1,0)의 공은 D 밖의 점 (1 + eps/2, 0)을 포함
check("비예 3.1.8(b)", all(abs(1 + e / 2) > 1 and abs(e / 2) < e for e in (0.36, 0.2, 1e-3)))

# 비예 3.1.10: ⋂ (-1/k, 1/k) = {0} (x ≠ 0이면 1/k < |x|인 k가 있다)
xs = rng.uniform(-1, 1, size=1000)
xs = xs[xs != 0]
check("비예 3.1.10: x ≠ 0이면 어떤 k에서 x ∉ (-1/k, 1/k)",
      all(abs(x) >= 1 / (int(np.ceil(1 / abs(x))) + 1) for x in xs))


# 예 3.1.12(d), 비예 3.1.13(a), 연습 3.1.2: 유한집합 위의 위상 판정
def is_topology(X, T):
    T = {frozenset(U) for U in T}
    X = frozenset(X)
    if frozenset() not in T or X not in T:
        return False
    for k in range(2, len(T) + 1):             # 모든 부분모임의 합집합과 교집합
        for sub in combinations(T, k):
            if frozenset().union(*sub) not in T:
                return False
            inter = X
            for U in sub:
                inter = inter & U
            if inter not in T:
                return False
    return True


X = "abc"
check("예 3.1.12(d): {∅,{a},{a,b},X}는 위상", is_topology(X, [set(), {"a"}, {"a", "b"}, set(X)]))
check("비예 3.1.13(a): {∅,{a},{b},X}는 위상이 아님", not is_topology(X, [set(), {"a"}, {"b"}, set(X)]))
check("연습 3.1.2(a): 위상", is_topology(X, [set(), {"a"}, {"b"}, {"a", "b"}, {"b", "c"}, set(X)]))
check("연습 3.1.2(b): 위상이 아님", not is_topology(X, [set(), {"a", "b"}, {"b", "c"}, set(X)]))
check("연습 3.1.2(c): 위상", is_topology(X, [set(), {"b"}, set(X)]))

# 예 3.1.18(a): 닫힌공의 여집합이 열림: d(y,p) > r, s = d(y,p) - r이면 B_s(y)는 K와 만나지 않는다
ok = True
for _ in range(300):
    p = rng.normal(size=2); r = rng.uniform(0.2, 1.5)
    y = p + rng.normal(size=2) * 3
    if d2(y, p) <= r:
        continue
    s = d2(y, p) - r
    z = y + rng.normal(size=(300, 2)); z = z[d2(z, y) < s]
    ok &= bool(np.all(d2(z, p) > r))
check("예 3.1.18(a): B_s(y) ∩ {d(x,p) <= r} = ∅", ok)

# 예 3.1.18(d): |x| = 1, y = (1 - eps/2)x이면 |y| < 1, |x - y| = eps/2
ok = True
for _ in range(200):
    x = rng.normal(size=4); x /= np.linalg.norm(x)
    e = rng.uniform(1e-4, 0.999)
    y = (1 - e / 2) * x
    ok &= bool(np.linalg.norm(y) < 1 and abs(np.linalg.norm(x - y) - e / 2) < 1e-12)
check("예 3.1.18(d): 경계점의 모든 공이 B_1(0)과 만난다", ok)

# 예 3.1.26: d_inf <= d <= d_1 <= n d_inf (표본 + n=2 기호 확인)
ok = True
for n in range(1, 8):
    A = rng.normal(size=(5000, n)) * rng.uniform(0.001, 10, size=(5000, 1))
    Z = np.zeros_like(A)
    ok &= bool(np.all(dinf(A, Z) <= d2(A, Z) + 1e-12))
    ok &= bool(np.all(d2(A, Z) <= d1(A, Z) + 1e-12))
    ok &= bool(np.all(d1(A, Z) <= n * dinf(A, Z) + 1e-12))
check("예 3.1.26: (3.1.3) d_inf <= d <= d_1 <= n d_inf (n = 1..7, 표본)", ok)
a1, a2 = sp.symbols("a1 a2", real=True)
sym_equal("예 3.1.26: d_1^2 - d^2 = 2|a1||a2| (n = 2)",
          (sp.Abs(a1) + sp.Abs(a2)) ** 2 - (a1 ** 2 + a2 ** 2), 2 * sp.Abs(a1) * sp.Abs(a2))
check("예 3.1.26 본문: (0,0)-(1,1) 거리 = sqrt2, 1, 2",
      np.isclose(d2(np.array([0, 0.]), np.array([1, 1.])), np.sqrt(2))
      and dinf(np.array([0, 0.]), np.array([1, 1.])) == 1 and d1(np.array([0, 0.]), np.array([1, 1.])) == 2)

# 연습 3.1.3: B^rho_1(0) = (-1, 1), B^rho_1(2) = (7^{1/3}, 9^{1/3})
xs = np.linspace(-3, 3, 60001)
rho3 = lambda x, y: np.abs(x ** 3 - y ** 3)
inball0 = xs[rho3(xs, 0) < 1]
inball2 = xs[rho3(xs, 2) < 1]
close("연습 3.1.3: B_1(0)의 끝점", [inball0.min(), inball0.max()], [-1, 1], tol=2e-4)
close("연습 3.1.3: B_1(2)의 끝점", [inball2.min(), inball2.max()], [7 ** (1 / 3), 9 ** (1 / 3)], tol=2e-4)

# 연습 3.1.7: |x - y|^2 = 4 sin^2(θ/2), (2/π)θ <= 2 sin(θ/2) <= θ
th = sp.symbols("theta", real=True)
sym_equal("연습 3.1.7: 2 - 2cosθ = 4 sin^2(θ/2)", 2 - 2 * sp.cos(th), 4 * sp.sin(th / 2) ** 2, {th: (0, sp.pi)})
T = np.linspace(0, np.pi, 100001)
check("연습 3.1.7: (2/π)θ <= 2sin(θ/2) <= θ on [0, π]",
      bool(np.all(2 / np.pi * T <= 2 * np.sin(T / 2) + 1e-15) and np.all(2 * np.sin(T / 2) <= T + 1e-15)))
ok = True
for _ in range(500):
    x = rng.normal(size=3); x /= np.linalg.norm(x)
    y = rng.normal(size=3); y /= np.linalg.norm(y)
    t = np.arccos(np.clip(x @ y, -1, 1))
    ok &= bool(abs(np.linalg.norm(x - y) - 2 * np.sin(t / 2)) < 1e-12)
check("연습 3.1.7: 구면 위 무작위 점에서 |x-y| = 2 sin(θ/2)", ok)

# 명제 3.1.27: 폐포(A를 포함하는 닫힌집합들의 교집합) = A ∪ ∂A.
# {a,b,c} 위의 모든 위상(부분집합 모임 256개 가운데 위상인 것)과 모든 A에 대해 전수 확인한다.
from itertools import chain
Xf = frozenset(X)
subsets = [frozenset(c) for c in chain.from_iterable(combinations(X, k) for k in range(len(X) + 1))]
tops = []
for mask in range(1 << len(subsets)):
    T = [subsets[i] for i in range(len(subsets)) if mask >> i & 1]
    if is_topology(X, T):
        tops.append(T)
ok = len(tops) == 29                              # 세 점 집합 위의 위상은 29개
for T in tops:
    closed = [Xf - U for U in T]
    for A in subsets:
        clos = Xf
        for C in closed:
            if A <= C:
                clos = clos & C
        bd = {p for p in X if all((U & A) and (U - A) for U in T if p in U)}
        ok &= clos == (A | bd)
check("명제 3.1.27: {a,b,c}의 위상 29개, 모든 A에서 closure(A) = A ∪ ∂A", ok)

summary()
