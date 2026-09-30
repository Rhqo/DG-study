"""3.4절 몫공간: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

원환면의 매개화는 tools/dgsym.py의 EXAMPLES["torus"]에서 가져온다 (§3.6).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/verify/v-3-4-quotient-spaces.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES

rng = np.random.default_rng(34)

# 비예 3.4.3: |x - y| < 1은 추이적이지 않다
check("비예 3.4.3: 0~0.6, 0.6~1.2, 0 !~ 1.2", abs(0 - 0.6) < 1 and abs(0.6 - 1.2) < 1 and not abs(0 - 1.2) < 1)

# 예 3.4.9: (3.4.3) |f(t) - f(t0)|^2 = 4 sin^2(π(t - t0))
t, t0 = sp.symbols("t t0", real=True)
f = lambda x: sp.Matrix([sp.cos(2 * sp.pi * x), sp.sin(2 * sp.pi * x)])
d = f(t) - f(t0)
sym_equal("예 3.4.9: (3.4.3)", (d.T * d)[0], 4 * sp.sin(sp.pi * (t - t0)) ** 2)
fn = lambda x: np.stack([np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)], -1)
# f(t) = f(t')이면 t - t' ∈ Z (표본)
T1 = rng.uniform(-5, 5, 20000); T2 = rng.uniform(-5, 5, 20000)
same = np.linalg.norm(fn(T1) - fn(T2), axis=1) < 1e-9
check("예 3.4.9: f(t) = f(t')이면 t - t' ∈ Z (표본; 서로 다른 무작위 점은 거의 항상 다름)", not np.any(same & (np.abs((T1 - T2) - np.round(T1 - T2)) > 1e-6)))
check("예 3.4.9: t' = t + k이면 f(t) = f(t')", bool(np.allclose(fn(T1), fn(T1 + rng.integers(-3, 4, T1.size)))))
# 열린사상 논증: |w - f(t0)| < 2 sin(πδ), w = f(t), t ∈ (t0 - 1/2, t0 + 1/2]이면 |t - t0| < δ
ok = True
for _ in range(2000):
    a0 = rng.uniform(-3, 3); de = rng.uniform(1e-3, 0.499)
    tt = a0 + rng.uniform(-0.5, 0.5)
    if np.linalg.norm(fn(tt) - fn(a0)) < 2 * np.sin(np.pi * de):
        ok &= abs(tt - a0) < de
check("예 3.4.9: 열린사상 논증 (표본)", bool(ok))

# 예 3.4.10: g(s,t) = g(s',t') ⇔ (3.4.1), Φ∘g = x(2πs, 2πt)
ex = EXAMPLES["torus"]
u, v = ex["coords"]; R, r = ex["params"]; X = ex["expr"]
s_, t_ = sp.symbols("s t", real=True)
Phi_g = sp.Matrix([(R + r * sp.cos(2 * sp.pi * s_)) * sp.cos(2 * sp.pi * t_),
                   (R + r * sp.cos(2 * sp.pi * s_)) * sp.sin(2 * sp.pi * t_),
                   r * sp.sin(2 * sp.pi * s_)])
sym_equal("예 3.4.10: Φ(g(s,t)) = x(2πs, 2πt)", Phi_g, X.subs({u: 2 * sp.pi * s_, v: 2 * sp.pi * t_}),
          {s_: (0, 1), t_: (0, 1), R: (1.1, 2), r: (0.2, 0.9)})
G = lambda S, T: np.concatenate([fn(S), fn(T)], -1)
grid = np.linspace(0, 1, 41)
Ss, Tt = [a.ravel() for a in np.meshgrid(grid, grid)]
Pts = G(Ss, Tt)
Dm = np.linalg.norm(Pts[:, None] - Pts[None], axis=2)
eqv = (np.abs((Ss[:, None] - Ss[None]) - np.round(Ss[:, None] - Ss[None])) < 1e-12) & \
      (np.abs((Tt[:, None] - Tt[None]) - np.round(Tt[:, None] - Tt[None])) < 1e-12)
check("예 3.4.10: g(p) = g(q) ⇔ p ~ q (격자 41×41)", bool(np.array_equal(Dm < 1e-9, eqv)))
# 연습 3.4.2: 동치류의 크기
cls = lambda p: {(a, b) for a in (0.0, 0.25, 0.5, 1 / 3, 1.0) for b in (0.0, 0.25, 0.5, 1 / 3, 1.0)
                 if abs((p[0] - a) - round(p[0] - a)) < 1e-12 and abs((p[1] - b) - round(p[1] - b)) < 1e-12}
check("연습 3.4.2: 동치류 크기 1, 2, 2, 4",
      [len(cls(p)) for p in ((0.5, 0.5), (0.0, 1 / 3), (0.25, 1.0), (1.0, 0.0))] == [1, 2, 2, 4])

# 명제 3.4.14: ρ(λx) = ±ρ(x)
ok = True
for _ in range(500):
    x = rng.normal(size=4); lam = rng.normal() * 3
    rho = lambda y: y / np.linalg.norm(y)
    ok &= bool(np.allclose(rho(lam * x), np.sign(lam) * rho(x)))
check("명제 3.4.14: ρ(λx) = sgn(λ) ρ(x)", ok)

# 명제 3.4.15: φ_i 잘 정의됨, φ_i∘ψ_i = id, ψ_i∘φ_i = id (n = 2, 기호)
n = 2
xs = sp.symbols("x1:4", real=True, nonzero=True)
us = sp.symbols("u1:3", real=True)
lam = sp.symbols("lambda", real=True, nonzero=True)


def phi(i, x):          # i = 1..n+1
    return sp.Matrix([x[k] / x[i - 1] for k in range(n + 1) if k != i - 1])


def psi(i, uu):
    return sp.Matrix(list(uu[:i - 1]) + [1] + list(uu[i - 1:]))


for i in range(1, n + 2):
    sym_equal(f"명제 3.4.15: φ_{i}(λx) = φ_{i}(x)", phi(i, [lam * c for c in xs]), phi(i, xs))
    sym_equal(f"명제 3.4.15: φ_{i}(ψ_{i}(u)) = u", phi(i, list(psi(i, us))), sp.Matrix(us))
    # ψ_i(φ_i[x]) = [x]: 대표 ψ_i(φ_i(x))에 x^i를 곱하면 x
    sym_equal(f"명제 3.4.15: x^i ψ_{i}(φ_{i}(x)) = x", xs[i - 1] * psi(i, list(phi(i, xs))), sp.Matrix(xs))

# 예 3.4.16: RP^1에서 φ_2 = 1/φ_1, [1:2] → 2, 1/2
x1, x2 = sp.symbols("x1 x2", real=True, nonzero=True)
check("예 3.4.16: φ_2 = 1/φ_1 on U_1 ∩ U_2", sp.simplify((x1 / x2) - 1 / (x2 / x1)) == 0)
check("예 3.4.16: [1:2]의 φ_1 = 2, φ_2 = 1/2", (2 / 1, 1 / 2) == (2.0, 0.5))
# 그림 3.4.2: tan 20°, tan 55°, tan(-35°)
close("그림 3.4.2: φ_1 = tan θ = 0.36, 1.43, -0.70", np.tan(np.deg2rad([20, 55, -35])), [0.36, 1.43, -0.70], tol=5e-3)

# 연습 3.4.3
P = [1, 2, -2]
phiN = lambda i, x: [x[k] / x[i - 1] for k in range(3) if k != i - 1]
close("연습 3.4.3(a): φ_1, φ_2, φ_3", phiN(1, P) + phiN(2, P) + phiN(3, P), [2, -2, 0.5, -1, -0.5, -1])
close("연습 3.4.3(c): [2:4:-4]도 같은 좌표", phiN(1, [2, 4, -4]) + phiN(2, [2, 4, -4]), [2, -2, 0.5, -1])
u1, u2 = sp.symbols("u1 u2", real=True, nonzero=True)
sym_equal("연습 3.4.3(b): φ_2(φ_1^{-1}(u)) = (1/u1, u2/u1)", phi(2, [1, u1, u2]), sp.Matrix([1 / u1, u2 / u1]))

# 연습 3.4.6: Q(ρ cosθ, ρ sinθ) = (cos 2θ, sin 2θ), Q(λx) = Q(x)
rh, th = sp.symbols("rho theta", positive=True)
Q = lambda a, b: sp.Matrix([a ** 2 - b ** 2, 2 * a * b]) / (a ** 2 + b ** 2)
sym_equal("연습 3.4.6: Q(ρ(cosθ, sinθ)) = (cos2θ, sin2θ)", Q(rh * sp.cos(th), rh * sp.sin(th)),
          sp.Matrix([sp.cos(2 * th), sp.sin(2 * th)]))
sym_equal("연습 3.4.6: Q(λx) = Q(x)", Q(lam * x1, lam * x2), Q(x1, x2))

summary()
