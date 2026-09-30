"""12.2절 미분동형사상: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

원환면 매개화는 dgsym.EXAMPLES["torus"], 입체사영·ℝℙⁿ 차트는 dgsym.stereographic, dgsym.rpn_charts에서 가져온다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/verify/v-12-2-diffeomorphisms.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import EXAMPLES, rpn_charts, stereographic

rng = np.random.default_rng(122)

# ---------------------------------------------------------------- 예 12.2.5 (12.2.2): x³은 (ℝ,𝒜₃) → (ℝ,𝒜₁) 미분동형사상
u = sp.symbols("u", positive=True)       # u > 0에서 기호 확인, 음수는 홀함수성으로
F = lambda x: x ** 3
psi = lambda x: x ** 3
psi_inv = lambda t: t ** sp.Rational(1, 3)
sym_equal("예 12.2.5 (12.2.2): id∘F∘ψ^{-1}(u) = u (u > 0)", F(psi_inv(u)), u)
sym_equal("예 12.2.5 (12.2.2): ψ∘F^{-1}(y) = y (y > 0)", psi(psi_inv(u)), u)
ts = np.linspace(-2, 2, 41)
close("예 12.2.5 (12.2.2): id∘F∘ψ^{-1} = id (수치, 음수 포함)", np.cbrt(ts) ** 3, ts, 1e-12)
# 비예 12.2.6(a): id∘id∘ψ^{-1}(u) = u^{1/3}의 차분몫 발산
h = sp.symbols("h", positive=True)
check("비예 12.2.6(a): u^{1/3}의 0에서 차분몫 → ∞", sp.limit(h ** sp.Rational(1, 3) / h, h, 0, "+") == sp.oo)

# ---------------------------------------------------------------- 연습 12.2.2: (ℝ,𝓑_a) ≅ (ℝ,𝓑_b)
for a_, b_ in ((sp.Rational(1), sp.Rational(3)), (sp.Rational(5, 2), sp.Rational(1, 3)), (sp.Rational(2), sp.Rational(7))):
    x = sp.symbols("x", positive=True)
    psi_a = lambda t, a=a_: t * sp.Abs(t) ** (a - 1)
    psi_a_inv = lambda t, a=a_: t * sp.Abs(t) ** (1 / a - 1)
    psi_b = lambda t, b=b_: t * sp.Abs(t) ** (b - 1)
    Fab = lambda t, a=a_, b=b_: t * sp.Abs(t) ** (a / b - 1)
    sym_equal(f"연습 12.2.2: ψ_b∘F∘ψ_a^{{-1}}(u) = u (a = {a_}, b = {b_}, u > 0)", sp.simplify(psi_b(Fab(psi_a_inv(x)))), x)
    for t0 in (-1.7, -0.3, 0.6, 2.2):
        val = float(psi_b(Fab(psi_a_inv(sp.Float(t0)))))
        close(f"연습 12.2.2: ψ_b∘F∘ψ_a^{{-1}}({t0}) = {t0} (a = {a_}, b = {b_})", val, t0, 1e-10)

# ---------------------------------------------------------------- 명제 12.2.3(a) (12.2.1): 끌어온 차트의 좌표변환 (예: 𝒜₃ = F^*𝒜₁)
# 차트 θ ∈ 𝒜₁ 을 θ(y) = 2y + 1 로 두면 끌어온 차트 θ∘F(x) = 2x³ + 1, ψ와의 좌표변환은 u ↦ 2u + 1 (매끄러움)
xx = sp.symbols("x", real=True)
theta = lambda y: 2 * y + 1
pulled = theta(F(xx))
sym_equal("명제 12.2.3(a): (θ∘F)∘ψ^{-1}(u) = 2u + 1", pulled.subs(xx, psi_inv(u)), 2 * u + 1)

# ---------------------------------------------------------------- 예 12.2.8: T² ≅ T_{R,r}
ex = EXAMPLES["torus"]
R, r = ex["params"]
U_, V_ = ex["coords"]
X = ex["expr"]
a = (sp.cos(U_), sp.sin(U_))
b = (sp.cos(V_), sp.sin(V_))
Phi = sp.Matrix([(R + r * a[0]) * b[0], (R + r * a[0]) * b[1], r * a[1]])
sym_equal("예 12.2.8: Φ∘(α×α)^{-1}(u, v) = x(u, v) (기준 매개화)", Phi, X)
rho = sp.sqrt(X[0] ** 2 + X[1] ** 2)
dom = {U_: (-3.1, 3.1), V_: (-3.1, 3.1), R: (1.2, 2.0), r: (0.2, 0.9)}
sym_equal("예 12.2.8: x(u,v)에서 ρ = R + r cos u", rho, R + r * sp.cos(U_), dom)
Psi = sp.Matrix([(rho - R) / r, X[2] / r, X[0] / rho, X[1] / rho])
sym_equal("예 12.2.8: Ψ(Φ(a,b)) = (a,b)", Psi, sp.Matrix([a[0], a[1], b[0], b[1]]), dom)
# 수치: Φ∘Ψ = id on T_{R,r} (무작위 점)
Rv, rv = 2.0, 0.8
for _ in range(20):
    u0, v0 = rng.uniform(-np.pi, np.pi, 2)
    p = np.array([(Rv + rv * np.cos(u0)) * np.cos(v0), (Rv + rv * np.cos(u0)) * np.sin(v0), rv * np.sin(u0)])
    rr = np.hypot(p[0], p[1])
    aa, bb = np.array([(rr - Rv) / rv, p[2] / rv]), np.array([p[0] / rr, p[1] / rr])
    check("예 12.2.8: Ψ(p) ∈ S¹ × S¹", abs(aa @ aa - 1) < 1e-12 and abs(bb @ bb - 1) < 1e-12)
    back = np.array([(Rv + rv * aa[0]) * bb[0], (Rv + rv * aa[0]) * bb[1], rv * aa[1]])
    close("예 12.2.8: Φ(Ψ(p)) = p", back, p, 1e-12)
# 좌표표현 x^{-1}∘Φ∘(α×α)^{-1} = id: 각 차트 α (11.3.10)로 되읽기
alpha = lambda q1, q2: 2 * np.arctan(q2 / (1 + q1))
for _ in range(10):
    u0, v0 = rng.uniform(-3.0, 3.0, 2)
    aa, bb = np.array([np.cos(u0), np.sin(u0)]), np.array([np.cos(v0), np.sin(v0)])
    close("예 12.2.8: (α×α)(Ψ(x(u,v))) = (u,v) (u, v ∈ (-π, π))", [alpha(*aa), alpha(*bb)], [u0, v0], 1e-12)

# ---------------------------------------------------------------- 예 12.2.11(a) (12.2.4): π_S의 국소 역사상
for n in (1, 2, 3):
    rp = rpn_charts(n)
    Uc = rp["chart_coords"]
    W = sp.symbols(f"w1:{n + 1}", real=True)
    for i in range(n + 1):
        j = rp["psi"](i)                                   # j_i(u)
        nj = sp.sqrt(sum(c ** 2 for c in j))
        y = sp.Matrix(Uc) / sp.sqrt(1 + sum(c ** 2 for c in Uc))   # G(u) = u/√(1+|u|²) ∈ B^n
        s_ = sp.sqrt(1 - sum(c ** 2 for c in y))
        hemi = sp.Matrix(list(y[:i]) + [s_] + list(y[i:]))          # (φ_i^+)^{-1}(G(u))
        sym_equal(f"예 12.2.11(a) (12.2.4): (φ_{i + 1}^+)^{{-1}}(u/√(1+|u|²)) = j_{i + 1}(u)/|j_{i + 1}(u)| (n = {n})",
                  hemi, j / nj, {c: (-1.5, 1.5) for c in Uc})
        sym_equal(f"예 12.2.11(a): |j_{i + 1}(u)|² = 1 + |u|² (n = {n})", nj ** 2, 1 + sum(c ** 2 for c in Uc))
        # π_S(j/|j|) = [j] = φ_i^{-1}(u): h_i(j/|j|) = u
        hi = rp["phi"](i).subs(dict(zip(rp["coords"], j / nj)), simultaneous=True)
        sym_equal(f"예 12.2.11(a): φ_{i + 1}(π_S(j_{i + 1}(u)/|j_{i + 1}(u)|)) = u (n = {n})", hi, sp.Matrix(Uc))

# 예 12.2.11(a): f∘π_S = (x¹)² (단위벡터에서)
x = sp.symbols("x1:4", real=True)
check("예 12.2.11(a): |x| = 1이면 (x¹)²/|x|² = (x¹)²",
      sp.simplify((x[0] ** 2 / sum(c ** 2 for c in x)).subs(x[2], sp.sqrt(1 - x[0] ** 2 - x[1] ** 2)) - x[0] ** 2) == 0)

# 예 12.2.11(b): e(t) = -e(t - π), e|_I = α^{-1}∘τ
t = sp.symbols("t", real=True)
sym_equal("예 12.2.11(b): e(t) = -e(t - π)", sp.Matrix([sp.cos(t), sp.sin(t)]), -sp.Matrix([sp.cos(t - sp.pi), sp.sin(t - sp.pi)]))
for k in (-2, 0, 3):
    tt = np.linspace(2 * k * np.pi - 3.0, 2 * k * np.pi + 3.0, 25)
    close(f"예 12.2.11(b): α(e(t)) = t - 2kπ on (2kπ-π, 2kπ+π) (k = {k})", alpha(np.cos(tt), np.sin(tt)), tt - 2 * k * np.pi, 1e-9)

# ---------------------------------------------------------------- 연습 12.2.1: G∘σ(x) = x'/√(2(1 - x^{n+1}))
for n in (1, 2, 3):
    st = stereographic(n)
    Uc = st["chart_coords"]
    xs = st["sigma_inv"]                                      # x = σ^{-1}(u) ∈ S^n
    sig = sp.Matrix(Uc)
    G = sig / sp.sqrt(1 + sum(c ** 2 for c in sig))
    target = sp.Matrix(xs[:n]) / sp.sqrt(2 * (1 - xs[n]))
    sym_equal(f"연습 12.2.1: G(σ(x)) = x'/√(2(1-x^{{n+1}})) (n = {n})", G, target, {c: (-1.5, 1.5) for c in Uc})

# ---------------------------------------------------------------- 연습 12.2.5: ℝℙ¹ ≅ S¹
x1, x2, lam = sp.symbols("x1 x2 lambda", real=True)
Gx = sp.Matrix([x1 ** 2 - x2 ** 2, 2 * x1 * x2]) / (x1 ** 2 + x2 ** 2)
sym_equal("연습 12.2.5: G[λx] = G[x]", Gx.subs({x1: lam * x1, x2: lam * x2}, simultaneous=True), Gx)
sym_equal("연습 12.2.5: |G[x]|² = 1", (Gx.T * Gx)[0], 1)
th = sp.symbols("theta", real=True)
sym_equal("연습 12.2.5: Q(e(t)) = e(2t)", Gx.subs({x1: sp.cos(th), x2: sp.sin(th)}), sp.Matrix([sp.cos(2 * th), sp.sin(2 * th)]))
# 전단사: 무작위 w ∈ S¹의 원상 [cos(θ/2) : sin(θ/2)]
for _ in range(10):
    th0 = rng.uniform(-np.pi, np.pi)
    pre = np.array([np.cos(th0 / 2), np.sin(th0 / 2)])
    close("연습 12.2.5: G[cos(θ/2) : sin(θ/2)] = (cos θ, sin θ)",
          [pre[0] ** 2 - pre[1] ** 2, 2 * pre[0] * pre[1]], [np.cos(th0), np.sin(th0)], 1e-12)

# ---------------------------------------------------------------- 연습 12.2.6: S²∖{N,S} ≅ S¹ × ℝ
tt_ = sp.symbols("t", real=True)
c, s = sp.symbols("c s", real=True)
Gm = sp.Matrix([sp.sech(tt_) * c, sp.sech(tt_) * s, sp.tanh(tt_)])
sym_equal("연습 12.2.6: |G(a,t)|² = 1 (a = (c, s), c² + s² = 1)", ((Gm.T * Gm)[0]).subs(s ** 2, 1 - c ** 2), 1)
sv = sp.symbols("s", real=True)
sym_equal("연습 12.2.6: sech(artanh s) = √(1 - s²)", sp.sech(sp.atanh(sv)), sp.sqrt(1 - sv ** 2), {sv: (-0.95, 0.95)})
sym_equal("연습 12.2.6: artanh(tanh t) = t", sp.atanh(sp.tanh(tt_)), tt_, {tt_: (-3, 3)})
for _ in range(10):
    p = rng.normal(size=3); p /= np.linalg.norm(p)
    xp = p[:2]; aa = xp / np.linalg.norm(xp); t0 = np.arctanh(p[2])
    close("연습 12.2.6: G(F(x)) = x", [aa[0] / np.cosh(t0), aa[1] / np.cosh(t0), np.tanh(t0)], p, 1e-12)

summary()
