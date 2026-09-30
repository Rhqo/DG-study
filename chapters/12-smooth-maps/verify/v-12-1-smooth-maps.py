"""12.1절 매끄러운 함수와 매끄러운 사상: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

입체사영과 ℝℙⁿ 차트는 dgsym.stereographic(n), dgsym.rpn_charts(n)에서 가져온다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/verify/v-12-1-smooth-maps.py``
"""

import itertools

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary
from dgsym import rpn_charts, stereographic

rng = np.random.default_rng(121)

# ---------------------------------------------------------------- 예 12.1.3(c): 높이함수의 좌표표현 (12.1.3)
for n in (1, 2, 3):
    st = stereographic(n)
    U = st["chart_coords"]
    r2 = sum(u ** 2 for u in U)
    hN = st["sigma_inv"][n]
    hS = st["sigma_south_inv"][n]
    sym_equal(f"예 12.1.3(c) (12.1.3): h∘σ^{{-1}} = (|u|²-1)/(|u|²+1) (n = {n})", hN, (r2 - 1) / (r2 + 1))
    sym_equal(f"예 12.1.3(c) (12.1.3): h∘σ̃^{{-1}} = (1-|v|²)/(1+|v|²) (n = {n})", hS, (1 - r2) / (1 + r2))
    # (12.1.2)의 예: h∘σ̃^{-1}(v) = (h∘σ^{-1})(σ∘σ̃^{-1}(v)),  σ∘σ̃^{-1}(v) = v/|v|²
    sub = {U[i]: U[i] / r2 for i in range(n)}
    sym_equal(f"예 12.1.3(c): h∘σ̃^{{-1}} = (h∘σ^{{-1}})∘(σ∘σ̃^{{-1}}) (n = {n})", hN.subs(sub, simultaneous=True), hS)
    # 분모가 양수: |u|²+1 > 0 은 자명. σ^{-1}의 값이 단위구면 위
    sym_equal(f"예 12.1.3(c): |σ^{{-1}}(u)|² = 1 (n = {n})", sum(c ** 2 for c in st["sigma_inv"]), 1)
    sym_equal(f"예 12.1.3(c): |σ̃^{{-1}}(u)|² = 1 (n = {n})", sum(c ** 2 for c in st["sigma_south_inv"]), 1)

# 위도원의 반지름 (그림 12.1.2)
for c in np.linspace(-0.9, 0.9, 7):
    rN, rS = np.sqrt((1 + c) / (1 - c)), np.sqrt((1 - c) / (1 + c))
    close(f"예 12.1.3(c): 위도원 h = {c:+.2f}의 σ-반지름", (rN ** 2 - 1) / (rN ** 2 + 1), c, 1e-12)
    close(f"예 12.1.3(c): 위도원 h = {c:+.2f}의 σ̃-반지름, 두 반지름은 역수", [(1 - rS ** 2) / (1 + rS ** 2), rN * rS], [c, 1.0], 1e-12)

# ---------------------------------------------------------------- 예 12.1.3(d), 연습 12.1.3: ℝℙⁿ 위의 함수
for n in (1, 2, 3):
    rp = rpn_charts(n)
    X, U = rp["coords"], rp["chart_coords"]
    lam = sp.symbols("lambda", nonzero=True, real=True)
    fx = X[0] ** 2 / sum(x ** 2 for x in X)
    sym_equal(f"예 12.1.3(d): f[λx] = f[x] (n = {n})", fx.subs({x: lam * x for x in X}, simultaneous=True), fx)
    r2 = sum(u ** 2 for u in U)
    for i in range(n + 1):
        rep = rp["psi"](i)
        fi = fx.subs(dict(zip(X, rep)), simultaneous=True)
        sym_equal(f"예 12.1.3(d): f∘φ_{i + 1}^{{-1}}(u) = (j_{i + 1}(u)^1)²/(1+|u|²) (n = {n})", fi, rep[0] ** 2 / (1 + r2))
        # 좌표변환과의 일관성: f∘φ_j^{-1} = f∘φ_i^{-1} ∘ (φ_i∘φ_j^{-1})
        for j in range(n + 1):
            if i == j:
                continue
            tij = rp["transition"](j, i)  # φ_i∘φ_j^{-1}
            fj = fx.subs(dict(zip(X, rp["psi"](j))), simultaneous=True)
            lhs = fi.subs(dict(zip(U, tij)), simultaneous=True)
            # 수치 대입: 정의역 φ_j(U_i∩U_j)에서 (j_j(u)의 i번째 성분 ≠ 0)
            ok = True
            for _ in range(5):
                pt = {u: float(rng.uniform(0.3, 1.7)) for u in U}
                ok &= abs(float(lhs.subs(pt)) - float(fj.subs(pt))) < 1e-10
            check(f"예 12.1.3(d): f∘φ_{j + 1}^{{-1}} = (f∘φ_{i + 1}^{{-1}})∘(φ_{i + 1}∘φ_{j + 1}^{{-1}}) (n = {n})", ok)
u, v = sp.symbols("u v", real=True)
rp1 = rpn_charts(1)
f1 = rp1["coords"][0] ** 2 / sum(x ** 2 for x in rp1["coords"])
uu = rp1["chart_coords"][0]
sym_equal("예 12.1.3(d): n = 1, f∘φ_1^{-1}(u) = 1/(1+u²)", f1.subs(dict(zip(rp1["coords"], rp1["psi"](0)))), 1 / (1 + uu ** 2))
sym_equal("예 12.1.3(d): n = 1, f∘φ_2^{-1}(v) = v²/(1+v²)", f1.subs(dict(zip(rp1["coords"], rp1["psi"](1)))), uu ** 2 / (1 + uu ** 2))
x1, x2 = sp.symbols("x1 x2", real=True)
g = x1 * x2 / (x1 ** 2 + x2 ** 2)
sym_equal("연습 12.1.3: f∘φ_1^{-1}(u) = u/(1+u²)", g.subs({x1: 1, x2: u}), u / (1 + u ** 2))
sym_equal("연습 12.1.3: f∘φ_2^{-1}(v) = v/(v²+1)", g.subs({x1: v, x2: 1}), v / (v ** 2 + 1))
sym_equal("연습 12.1.3: v = 1/u이면 두 식이 같다", (v / (v ** 2 + 1)).subs(v, 1 / u), u / (1 + u ** 2))

# ---------------------------------------------------------------- 비예 12.1.4
h = sp.symbols("h", positive=True)
check("비예 12.1.4(a): u^{1/3}의 0에서의 오른쪽 차분몫 h^{-2/3} → ∞", sp.limit(h ** sp.Rational(1, 3) / h, h, 0, "+") == sp.oo)
w = sp.symbols("w", real=True)
check("비예 12.1.4(a): g∘ψ^{-1}(u) = (u^{1/3})³ = u (u > 0)", sp.simplify((h ** sp.Rational(1, 3)) ** 3 - h) == 0)
phi1p_inv = sp.Matrix([sp.sqrt(1 - w ** 2), w])
check("비예 12.1.4(b): (φ_1^+)^{-1}(w) = (√(1-w²), w)는 S¹ 위", sp.simplify(sum(c ** 2 for c in phi1p_inv) - 1) == 0)
check("비예 12.1.4(b): |w|의 좌·우 미분계수가 -1, +1",
      sp.limit(sp.Abs(h) / h, h, 0, "+") == 1 and sp.limit(sp.Abs(-h) / (-h), h, 0, "+") == -1)

# ---------------------------------------------------------------- 명제 12.1.8(b): σ̄∘σ^{-1} = id, σ̄~∘σ̃^{-1} = id
for n in (1, 2, 3):
    st = stereographic(n)
    X, U = st["coords"], st["chart_coords"]
    sig_bar = sp.Matrix([X[i] / (1 - X[n]) for i in range(n)])
    sigt_bar = sp.Matrix([X[i] / (1 + X[n]) for i in range(n)])
    sym_equal(f"명제 12.1.8(b): σ̄(σ^{{-1}}(u)) = u (n = {n})", sig_bar.subs(dict(zip(X, st["sigma_inv"])), simultaneous=True), sp.Matrix(U))
    sym_equal(f"명제 12.1.8(b): σ̄~(σ̃^{{-1}}(u)) = u (n = {n})", sigt_bar.subs(dict(zip(X, st["sigma_south_inv"])), simultaneous=True), sp.Matrix(U))
    sym_equal(f"명제 12.1.8(b): σ = σ̄ on S^n (식 비교, n = {n})", st["sigma"], sig_bar)

# ---------------------------------------------------------------- 예 12.1.9: π, π_S의 좌표표현 (12.1.7)
for n in (1, 2, 3):
    rp = rpn_charts(n)
    X = rp["coords"]
    W = sp.symbols(f"w1:{n + 1}", real=True)
    s = sp.sqrt(1 - sum(c ** 2 for c in W))
    lam = sp.symbols("lambda", positive=True)
    for i in range(n + 1):
        hi = rp["phi"](i)                                       # h_i
        sym_equal(f"예 12.1.9(a): h_{i + 1}(λx) = h_{i + 1}(x) (n = {n})", hi.subs({x: lam * x for x in X}, simultaneous=True), hi)
        for sign in (+1, -1):
            hemi = sp.Matrix(list(W[:i]) + [sign * s] + list(W[i:]))   # (φ_i^±)^{-1}(w)
            rep = hi.subs(dict(zip(X, hemi)), simultaneous=True)
            sym_equal(f"예 12.1.9(b) (12.1.7): φ_{i + 1}∘π_S∘(φ_{i + 1}^{'+' if sign > 0 else '-'})^{{-1}}(w) = ±w/√(1-|w|²) (n = {n})",
                      rep, sign * sp.Matrix(W) / s, {c: (-0.5, 0.5) for c in W})
    # 예 2.3.2(c)의 F(w) = w/√(1-|w|²)와 G(y) = y/√(1+|y|²)가 서로 역
    Y = sp.symbols(f"y1:{n + 1}", real=True)
    Fw = sp.Matrix(W) / s
    Gy = sp.Matrix(Y) / sp.sqrt(1 + sum(c ** 2 for c in Y))
    sym_equal(f"예 12.1.9(b): (12.1.7)은 B^n → R^n의 미분동형사상 (G∘F = id, n = {n})",
              Gy.subs(dict(zip(Y, Fw)), simultaneous=True), sp.Matrix(W), {c: (-0.5, 0.5) for c in W})

# ---------------------------------------------------------------- 예 12.1.10: GL(n)의 연산, (12.1.8)
for n in (2, 3):
    A = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{i + 1}{j + 1}", real=True))
    B = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"b{i + 1}{j + 1}", real=True))
    sym_equal(f"예 12.1.10(b): det(AB) = det A det B (n = {n})", (A * B).det(), A.det() * B.det())
    Ainv = A.inv()
    cramer = sp.zeros(n, n)
    for i, j in itertools.product(range(n), range(n)):
        Aij = A.copy()
        Aij[:, i] = sp.eye(n)[:, j]
        cramer[i, j] = Aij.det() / A.det()
    sym_equal(f"예 12.1.10(c) (12.1.8): (A^{{-1}})^i_j = det A_{{i←e_j}}/det A (n = {n})", cramer, Ainv)
    check(f"예 12.1.10(a): det은 성분의 다항식 (n = {n})", A.det().is_polynomial(*list(A)))
    check(f"예 12.1.10(b): (AB)^i_j는 성분의 다항식 (n = {n})", all(e.is_polynomial(*(list(A) + list(B))) for e in A * B))
a, b, c, d = sp.symbols("a b c d", real=True)
M2 = sp.Matrix([[a, b], [c, d]])
sym_equal("예 12.1.10(c): n = 2 역행렬 공식", M2.inv(), sp.Matrix([[d, -b], [-c, a]]) / (a * d - b * c))

# ---------------------------------------------------------------- 연습 12.1.1: S² 위의 x¹x²
st = stereographic(2)
U = st["chart_coords"]
r2 = sum(u_ ** 2 for u_ in U)
sym_equal("연습 12.1.1: f∘σ^{-1}(u) = 4u¹u²/(|u|²+1)²", st["sigma_inv"][0] * st["sigma_inv"][1], 4 * U[0] * U[1] / (r2 + 1) ** 2)
sym_equal("연습 12.1.1: f∘σ̃^{-1}(v) = 4v¹v²/(|v|²+1)²", st["sigma_south_inv"][0] * st["sigma_south_inv"][1], 4 * U[0] * U[1] / (r2 + 1) ** 2)

# ---------------------------------------------------------------- 연습 12.1.4: S¹의 제곱사상, α∘F∘α^{-1}(t) = 2t
t = sp.symbols("t", real=True)
xx, yy = sp.cos(t), sp.sin(t)
Fx, Fy = xx ** 2 - yy ** 2, 2 * xx * yy
sym_equal("연습 12.1.4: |F(x,y)|² = (x²+y²)²", (x1 ** 2 - x2 ** 2) ** 2 + (2 * x1 * x2) ** 2, (x1 ** 2 + x2 ** 2) ** 2)
sym_equal("연습 12.1.4: F(cos t, sin t) = (cos 2t, sin 2t)", sp.Matrix([Fx, Fy]), sp.Matrix([sp.cos(2 * t), sp.sin(2 * t)]))
alpha = lambda p, q: 2 * np.arctan(q / (1 + p))   # (11.3.10)
ts = np.linspace(-1.5, 1.5, 13)
close("연습 12.1.4: α(F(α^{-1}(t))) = 2t (|t| < π/2, 수치)", alpha(np.cos(2 * ts), np.sin(2 * ts)), 2 * ts, 1e-12)

# ---------------------------------------------------------------- 연습 12.1.5: [x] ↦ [Ax]의 좌표표현 (n = 2 한 예)
rp = rpn_charts(2)
X, Uc = rp["coords"], rp["chart_coords"]
Amat = sp.Matrix([[2, 1, 0], [0, 1, 1], [1, 0, 1]])
check("연습 12.1.5: 예의 행렬은 가역", Amat.det() != 0)
Aj = Amat * rp["psi"](0)                               # A j_1(u)
rep = rp["phi"](2).subs(dict(zip(X, Aj)), simultaneous=True)   # φ_3∘Ã∘φ_1^{-1}
pt = {Uc[0]: 0.4, Uc[1]: -0.7}
xv = np.array([float(e) for e in rp["psi"](0).subs(pt)])
Ax = np.array(Amat, dtype=float) @ xv
close("연습 12.1.5: φ_3∘Ã∘φ_1^{-1}(u) = h_3(A j_1(u)) (수치)", [float(e) for e in rep.subs(pt)], [Ax[0] / Ax[2], Ax[1] / Ax[2]], 1e-12)
lam = sp.symbols("lambda", positive=True)
sym_equal("연습 12.1.5: h_3(A(λx)) = h_3(Ax)", rp["phi"](2).subs(dict(zip(X, Amat * (lam * sp.Matrix(X)))), simultaneous=True),
          rp["phi"](2).subs(dict(zip(X, Amat * sp.Matrix(X))), simultaneous=True))

summary()
