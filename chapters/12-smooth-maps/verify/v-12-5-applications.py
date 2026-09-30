"""12.5절 단위분할의 응용: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

원환면 매개화는 dgsym.EXAMPLES["torus"], 입체사영은 dgsym.stereographic(1)에서 가져온다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/verify/v-12-5-applications.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES, stereographic


def f0(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = np.exp(-1.0 / t[m])
    return out


def h(t):
    p, q = f0(2 - np.asarray(t, float)), f0(np.asarray(t, float) - 1)
    return p / (p + q)


def k(s):
    return 1 - h(1 + np.asarray(s, float))


rng = np.random.default_rng(125)

# ---------------------------------------------------------------- 본문: ℝ의 우릉 예 ψ(t) = k(2t + 1)
t = np.linspace(-3, 3, 60001)
psi = k(2 * t + 1)
check("정리 12.5.1 뒤의 예: t ≥ 0이면 ψ = 1", np.all(psi[t >= 0] == 1))
check("정리 12.5.1 뒤의 예: t ≤ -1/2이면 ψ = 0 (supp ψ = [-1/2, ∞) ⊆ (-1, ∞))", np.all(psi[t <= -0.5] == 0) and np.all(t[psi != 0] > -0.5))

# ---------------------------------------------------------------- 예 12.5.6: 원환면 위의 cos u
ex = EXAMPLES["torus"]
R, r = ex["params"]
u, v = ex["coords"]
X = ex["expr"]
dom = {u: (-3.1, 3.1), v: (-3.1, 3.1), R: (1.2, 2.0), r: (0.2, 0.9)}
rho = sp.sqrt(X[0] ** 2 + X[1] ** 2)
sym_equal("예 12.5.6: 원환면 위에서 ρ = R + r cos u", rho, R + r * sp.cos(u), dom)
sym_equal("예 12.5.6: f∘x = (ρ - R)/r = cos u", (rho - R) / r, sp.cos(u), dom)
for Rv, rv in ((2.0, 0.8), (1.5, 0.3), (3.0, 2.5)):
    a, b = (Rv - rv) / 3, 2 * (Rv - rv) / 3
    check(f"예 12.5.6: 0 < a < b < R - r (R = {Rv}, r = {rv})", 0 < a < b < Rv - rv)
    uu, vv = rng.uniform(-np.pi, np.pi, (2, 2000))
    P = np.array([(Rv + rv * np.cos(uu)) * np.cos(vv), (Rv + rv * np.cos(uu)) * np.sin(vv), rv * np.sin(uu)])
    rr = np.hypot(P[0], P[1])
    F = k((rr - a) / (b - a)) * (rr - Rv) / rv
    close(f"예 12.5.6: 원환면 위에서 F = cos u (R = {Rv}, r = {rv})", F, np.cos(uu), 1e-12)
    Q = rng.uniform(-1, 1, (3, 2000)) * np.array([[a], [a], [5]])
    Qr = np.hypot(Q[0], Q[1])
    FQ = k((Qr - a) / (b - a)) * (Qr - Rv) / rv
    check(f"예 12.5.6: ρ ≤ a (z축 근처)에서 F = 0 (R = {Rv}, r = {rv})", np.all(FQ[Qr <= a] == 0))

# ---------------------------------------------------------------- 예 12.5.8(d), 그림 12.5.2: ℝ의 소진 함수
b1 = lambda s: h(0.5 + 2 * np.abs(np.asarray(s, float)))
ks = np.arange(-9, 10)
tt = np.linspace(-6.5, 6.5, 26001)
g = np.array([b1(tt - q) for q in ks])
ps = g / g.sum(axis=0)
fx = ((np.abs(ks)[:, None] + 1) * ps).sum(axis=0)
check("예 12.5.8(d): f ≥ 1", np.all(fx >= 1 - 1e-12))
for N in range(0, 5):
    check(f"예 12.5.8(d): |t| ≥ {N + 1}이면 f(t) ≥ {N + 2}", np.all(fx[np.abs(tt) >= N + 1] >= N + 2 - 1e-12))
sub = fx <= 3
check("그림 12.5.2: f^{-1}((-∞, 3]) = [-9/4, 9/4] (수치, 경계 여유 0.05)",
      np.all(sub[np.abs(tt) <= 2.25]) and not np.any(sub[(np.abs(tt) > 2.30) & (np.abs(tt) < 6)]))

# ---------------------------------------------------------------- 명제 12.5.9의 부등식 (12.5.2)
ok = True
for _ in range(2000):
    w = rng.dirichlet(np.ones(12))              # ψ_j(x) (j = N+1, …, N+12), 합 1
    N = int(rng.integers(0, 20)); js = np.arange(N + 1, N + 13)
    ok &= (js * w).sum() >= N + 1 - 1e-12
check("명제 12.5.9 (12.5.2): 무작위 2000개에서 Σ_{j>N} jψ_j ≥ N + 1", ok)

# ---------------------------------------------------------------- 연습 12.5.1
ts = np.linspace(-3, 3, 60001)
psi1 = h(2 * np.abs(ts) - 1)
check("연습 12.5.1: |t| ≤ 1이면 ψ = 1", np.all(psi1[np.abs(ts) <= 1] == 1))
check("연습 12.5.1: supp ψ = [-3/2, 3/2] ⊆ (-2, 2)", np.all(np.abs(ts[psi1 != 0]) < 1.5))
check("연습 12.5.1: h(1 + (|t| - 1)/(1/2)) = h(2|t| - 1)", np.allclose(h(1 + (np.abs(ts) - 1) / 0.5), psi1))

# ---------------------------------------------------------------- 연습 12.5.3
tq = sp.symbols("t", positive=True)
sym_equal("연습 12.5.3(a): 1/t + 1/(1-t) = 1/(t(1-t))", 1 / tq + 1 / (1 - tq), 1 / (tq * (1 - tq)), {tq: (0.05, 0.95)})
t01 = np.linspace(1e-4, 1 - 1e-4, 100001)
f01 = 1 / t01 + 1 / (1 - t01)
check("연습 12.5.3(a): f ≥ 4, 최솟값은 t = 1/2", f01.min() >= 4 - 1e-9 and abs(t01[np.argmin(f01)] - 0.5) < 1e-4)
for c in (4.5, 10.0, 100.0):
    m = f01 <= c
    check(f"연습 12.5.3(a): {{f ≤ {c}}} ⊆ [1/c, 1 - 1/c]", np.all((t01[m] >= 1 / c) & (t01[m] <= 1 - 1 / c)))
rr = np.exp(rng.uniform(-5, 5, 5000))
gg = rr ** 2 + rr ** -2
for c in (2.5, 10.0, 50.0):
    m = gg <= c
    check(f"연습 12.5.3(b): {{g ≤ {c}}} ⊆ {{1/√c ≤ |x| ≤ √c}}", np.all((rr[m] >= 1 / np.sqrt(c)) & (rr[m] <= np.sqrt(c))))

# ---------------------------------------------------------------- 연습 12.5.4
j = np.arange(1, 50)
fj = 1 / (1 - (1 - 1 / j) ** 2)
check("연습 12.5.4: f(p_j) → ∞ (단조 증가, f(p_49) > 24)", np.all(np.diff(fj) > 0) and fj[-1] > 24)

# ---------------------------------------------------------------- 연습 12.5.5: S¹의 국소적 확장
st = stereographic(1)
x1, x2 = st["coords"]
U1 = st["chart_coords"][0]
sig_bar = x1 / (1 - x2)
sigt_bar = x1 / (1 + x2)
sym_equal("연습 12.5.5: σ̄(σ^{-1}(u)) = u", sig_bar.subs({x1: st["sigma_inv"][0], x2: st["sigma_inv"][1]}, simultaneous=True), U1)
sym_equal("연습 12.5.5: σ̄~(σ̃^{-1}(u)) = u", sigt_bar.subs({x1: st["sigma_south_inv"][0], x2: st["sigma_south_inv"][1]}, simultaneous=True), U1)
# 예: f(x) = (x¹)³ + x² 에 대해 f1 = (f∘σ^{-1})∘σ̄ 가 S¹∖{N}에서 f와 같음 (수치)
fS = lambda p, q: p ** 3 + q
th = rng.uniform(-np.pi, np.pi, 500)
th = th[np.abs(np.sin(th) - 1) > 1e-3]
P1, P2 = np.cos(th), np.sin(th)
uu = P1 / (1 - P2)
back = (2 * uu / (uu ** 2 + 1), (uu ** 2 - 1) / (uu ** 2 + 1))
close("연습 12.5.5: f_1 = f∘σ^{-1}∘σ̄ = f on S¹∖{N} (수치)", fS(*back), fS(P1, P2), 1e-10)

# ---------------------------------------------------------------- 연습 12.5.6: 매끄러운 근사 (M = ℝ, f(t) = |t|, δ(t) = 0.1 + 0.05t²)
f_ = np.abs
delta = lambda s: 0.1 + 0.05 * s ** 2
# 덮개 U_p를 구간 중심 p = 0.05ℤ로 잡고, 반지름을 U_p 안에 들어가게 고른 범프들로 단위분할을 만든다
centers = np.arange(-4, 4.0001, 0.05)
grid = np.linspace(-3.5, 3.5, 7001)
bumps = []
for p in centers:
    rad = 0.06                                          # |f(x) - f(p)| ≤ |x - p| < 0.06 < δ(x)이므로 [p-0.06, p+0.06] ⊆ U_p
    bumps.append(h(1 + (np.abs(grid - p) - rad / 2) / (rad / 2)))
bumps = np.array(bumps)
Gs = bumps.sum(axis=0)
check("연습 12.5.6: 범프들의 합 > 0 (|t| ≤ 3.5)", np.all(Gs > 0))
gapprox = (f_(centers)[:, None] * bumps).sum(axis=0) / Gs
check("연습 12.5.6: |g - f| < δ (|t| ≤ 3.5)", np.all(np.abs(gapprox - f_(grid)) < delta(grid)))
check("연습 12.5.6: 각 범프의 받침 ⊆ U_p (|x - p| ≤ 0.06 ⇒ |f(x) - f(p)| < δ(x))",
      all(np.all(np.abs(f_(grid[(bb > 0)]) - f_(p)) < delta(grid[bb > 0])) for p, bb in zip(centers, bumps)))

summary()
