"""8.1절 가우스 사상과 형태작용소: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/verify/v-8-1-gauss-map.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES


def dN_matrix(X, a, b, pos=(), N=None):
    """d𝐍_p의 기저 (x_a, x_b)에 대한 행렬 (열 j = d𝐍(x_j)의 성분). 𝐍 = N_x (또는 주어진 N)."""
    X = sp.Matrix(X)
    if N is None:
        N = dgsym.unit_normal(X, a, b, pos)
    B = sp.Matrix.hstack(X.diff(a), X.diff(b))
    Gm = B.T * B
    return dgsym.simp(Gm.inv() * B.T * sp.Matrix.hstack(N.diff(a), N.diff(b)), pos), N


# 예 8.1.2: 기준 예제의 가우스 사상 ---------------------------------------------------------------
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
sym_equal("예 8.1.2(b): 구면 N = x/r (바깥쪽)", dgsym.unit_normal(sph["expr"], th, ph, sph["positive"]), sph["expr"] / r, sph["domain"])
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
sym_equal("예 8.1.2(c): 원기둥 N = (cos u, sin u, 0), 셋째 성분 0 (가우스 상 ⊆ 적도)",
          dgsym.unit_normal(cyl["expr"], cu, cv), cyl["expected"]["normal"], cyl["domain"])
sad = EX["saddle"]
u, v = sad["coords"]
Xs = sad["expr"]
Ns = dgsym.unit_normal(Xs, u, v)
D = sp.sqrt(1 + 4 * u ** 2 + 4 * v ** 2)
sym_equal("(8.1.1): 안장면 N = (−2u, 2v, 1)/√(1 + 4u² + 4v²)", Ns, sp.Matrix([-2 * u, 2 * v, 1]) / D, sad["domain"])
A_, B_ = sp.symbols("a b", real=True)
Cc = sp.sqrt(1 - A_ ** 2 - B_ ** 2)
sym_equal("예 8.1.2(d): N(x(−a/(2c), b/(2c))) = (a, b, c) (위쪽 반구 위로 전사)",
          Ns.subs({u: -A_ / (2 * Cc), v: B_ / (2 * Cc)}), sp.Matrix([A_, B_, Cc]), {A_: (-0.5, 0.5), B_: (-0.5, 0.5)})

# 명제 8.1.4: N_u ⊥ N, (8.1.2) ---------------------------------------------------------------------
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
Xt = tor["expr"]
Nt = dgsym.unit_normal(Xt, tu, tv, tor["positive"])
check("명제 8.1.4: 원환면 ⟨N_u, N⟩ = ⟨N_v, N⟩ = 0", sp.simplify(Nt.diff(tu).dot(Nt)) == 0 and sp.simplify(Nt.diff(tv).dot(Nt)) == 0)
check("명제 8.1.4: 안장면 ⟨N_u, N⟩ = 0", sp.simplify(Ns.diff(u).dot(Ns)) == 0)

# 예 8.1.5: d𝐍_p의 계산 ---------------------------------------------------------------------------
Nsp = sph["expr"] / r
sym_equal("예 8.1.5(b): 구면 N_θ = x_θ/r, N_φ = x_φ/r",
          sp.Matrix.hstack(Nsp.diff(th), Nsp.diff(ph)), sp.Matrix.hstack(sph["expr"].diff(th), sph["expr"].diff(ph)) / r, sph["domain"])
Mc, _ = dN_matrix(cyl["expr"], cu, cv)
sym_equal("예 8.1.5(c): 원기둥 dN(x_u) = x_u/r, dN(x_v) = 0", Mc, sp.diag(1 / cyl["params"][0], 0), cyl["domain"])
sym_equal("예 8.1.5(d): 안장면 N_u(0,0) = (−2,0,0)", Ns.diff(u).subs({u: 0, v: 0}), sp.Matrix([-2, 0, 0]))
sym_equal("예 8.1.5(d): 안장면 N_v(0,0) = (0,2,0)", Ns.diff(v).subs({u: 0, v: 0}), sp.Matrix([0, 2, 0]))
beta = sp.symbols("beta", real=True)
w = sp.Matrix([sp.cos(beta), sp.sin(beta), 0])
dNw = sp.cos(beta) * Ns.diff(u).subs({u: 0, v: 0}) + sp.sin(beta) * Ns.diff(v).subs({u: 0, v: 0})
sym_equal("예 8.1.5(d): dN_0(w) = (−2cos β, 2 sin β, 0)", dNw, sp.Matrix([-2 * sp.cos(beta), 2 * sp.sin(beta), 0]))
sym_equal("예 8.1.5(e): 원환면 N_u = −x_u/r", Nt.diff(tu), -Xt.diff(tu) / rr, tor["domain"])
sym_equal("예 8.1.5(e): 원환면 N_v = −cos u/(R + r cos u) x_v", Nt.diff(tv), -sp.cos(tu) / (R + rr * sp.cos(tu)) * Xt.diff(tv), tor["domain"])

# 명제 8.1.7, 예 8.1.8, 비예 8.1.9, 주의 상자 ---------------------------------------------------------
J = sp.Matrix([[0, -1], [1, 0]])
e1, e2 = sp.Matrix([1, 0]), sp.Matrix([0, 1])
check("비예 8.1.9(a): ⟨Je₁, e₂⟩ = 1, ⟨e₁, Je₂⟩ = −1", (J * e1).dot(e2) == 1 and e1.dot(J * e2) == -1)
P = sp.Rational(1, 2) * sp.Matrix([[1, 1], [1, 1]])
check("예 8.1.8(b): 예 1.2.7의 P는 대칭", P == P.T)
Bb = sp.Matrix([[1, 1], [0, 1]])        # 열 = b₁ = e₁, b₂ = e₁ + e₂
Adiag = sp.diag(1, 2)
Mat = Bb.inv() * Adiag * Bb
check("주의: diag(1,2)의 기저 (e₁, e₁+e₂) 행렬 = [[1,−1],[0,2]]", Mat == sp.Matrix([[1, -1], [0, 2]]))
Gram = Bb.T * Bb
check("주의: 그람 행렬 [[1,1],[1,2]], 곱 [[1,1],[1,3]] (대칭)", Gram == sp.Matrix([[1, 1], [1, 2]]) and Gram * Mat == sp.Matrix([[1, 1], [1, 3]]))
# (8.1.3) 일반 확인: ⟨A b_j, b_k⟩ = Σ_i g_ki A^i_j
b11, b12, b21, b22 = sp.symbols("b11 b12 b21 b22", real=True)
Bg = sp.Matrix([[b11, b12], [b21, b22]])
Ag = sp.Matrix(2, 2, sp.symbols("a11 a12 a21 a22", real=True))
Gg = Bg.T * Bg
lhs = sp.Matrix(2, 2, lambda k, j: (Bg * Ag[:, j]).dot(Bg[:, k]))
check("(8.1.3): ⟨A b_j, b_k⟩ = Σ_i g_ki A^i_j (일반)", sp.expand(lhs - Gg * Ag) == sp.zeros(2, 2))

# 정리 8.1.10: ⟨N_u, x_v⟩ = −⟨N, x_uv⟩ = ⟨N_v, x_u⟩ ------------------------------------------------------
f1, f2, f3 = (sp.Function(n)(u, v) for n in ("f1", "f2", "f3"))
Xg = sp.Matrix([f1, f2, f3])
n = Xg.diff(u).cross(Xg.diff(v))
# N = n/|n|이고 ⟨n, x_v⟩ = 0이므로 ⟨N_u, x_v⟩ = ⟨n_u, x_v⟩/|n|. 분자끼리 비교한다.
check("(8.1.4) 분자: ⟨n_u, x_v⟩ = ⟨n_v, x_u⟩ (일반 매개화, n = x_u × x_v)",
      sp.expand(n.diff(u).dot(Xg.diff(v)) - n.diff(v).dot(Xg.diff(u))) == 0)
check("(8.1.4) 분자: ⟨n_u, x_v⟩ = −⟨n, x_uv⟩ (일반 매개화)",
      sp.expand(n.diff(u).dot(Xg.diff(v)) + n.dot(Xg.diff(u).diff(v))) == 0)
check("(8.1.4) 보조: ⟨N_u, x_v⟩ = ⟨n_u, x_v⟩/|n| 은 ⟨n, x_v⟩ = 0에서", sp.expand(n.dot(Xg.diff(v))) == 0)
for key in ("sphere", "torus", "saddle", "cylinder"):
    e = EX[key]
    a, b = e["coords"]
    X = e["expr"]
    N = dgsym.unit_normal(X, a, b, e["positive"])
    sym_equal(f"정리 8.1.10: {e['name']} ⟨N_u, x_v⟩ = ⟨N_v, x_u⟩", N.diff(a).dot(X.diff(b)), N.diff(b).dot(X.diff(a)), e["domain"])
    sym_equal(f"(8.1.4): {e['name']} ⟨N_u, x_v⟩ = −⟨N, x_uv⟩", N.diff(a).dot(X.diff(b)), -N.dot(X.diff(a).diff(b)), e["domain"])

# 예 8.1.11: 안장면 x(1/2, 1/2) ----------------------------------------------------------------------
half = {u: sp.Rational(1, 2), v: sp.Rational(1, 2)}
Ms, _ = dN_matrix(Xs, u, v)
M11 = sp.simplify(Ms.subs(half))
sym_equal("예 8.1.11: dN 행렬 = (2/(3√3)) [[−2, 1], [−1, 2]]", M11, 2 / (3 * sp.sqrt(3)) * sp.Matrix([[-2, 1], [-1, 2]]))
E_, F_, G_ = (c.subs(half) for c in dgsym.first_ff(Xs, u, v))
check("예 8.1.11: E = 2, F = −1, G = 2", (E_, F_, G_) == (2, -1, 2))
sym_equal("예 8.1.11: N_u = (−4, −2, −2)/(3√3)", Ns.diff(u).subs(half), sp.Matrix([-4, -2, -2]) / (3 * sp.sqrt(3)))
sym_equal("예 8.1.11: N_v = (2, 4, −2)/(3√3)", Ns.diff(v).subs(half), sp.Matrix([2, 4, -2]) / (3 * sp.sqrt(3)))
Gh = sp.Matrix([[E_, F_], [F_, G_]])
GM = sp.simplify(Gh * M11)
check("예 8.1.11: 행렬은 비대칭, 그람 행렬을 곱하면 (2/(3√3)) diag(−3, 3) (대칭)",
      M11 != M11.T and sp.simplify(GM - 2 / (3 * sp.sqrt(3)) * sp.diag(-3, 3)) == sp.zeros(2, 2))

# 예 8.1.13(e): 원환면의 W -------------------------------------------------------------------------------
Mt, _ = dN_matrix(Xt, tu, tv, tor["positive"])
sym_equal("(8.1.6): 원환면 W = diag(1/r, cos u/(R + r cos u)) (N_x 안쪽)", -Mt, sp.diag(1 / rr, sp.cos(tu) / (R + rr * sp.cos(tu))), tor["domain"])
sym_equal("(8.1.6)과 dgsym.shape_operator 일치", -Mt, dgsym.shape_operator(Xt, tu, tv, tor["positive"]), tor["domain"])

# 연습 8.1.1: 원뿔 ---------------------------------------------------------------------------------------
rho, phi, t, psi = sp.symbols("rho phi t psi", real=True)
Xc = sp.Matrix([u, v, sp.sqrt(u ** 2 + v ** 2)])
Nc = dgsym.unit_normal(Xc, u, v)
pol = {u: rho * sp.cos(phi), v: rho * sp.sin(phi)}
Ncp = sp.simplify(Nc.subs(pol).subs(sp.sqrt(rho ** 2), rho))
Ncp = sp.simplify(sp.refine(Ncp, sp.Q.positive(rho)))
sym_equal("연습 8.1.1: 원뿔 N = (−cos φ, −sin φ, 1)/√2", Ncp, sp.Matrix([-sp.cos(phi), -sp.sin(phi), 1]) / sp.sqrt(2), {rho: (0.2, 2), phi: (0, 6)})
er = sp.Matrix([sp.cos(phi), sp.sin(phi), 1]) / sp.sqrt(2)
eh = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
Nfun = sp.Matrix([-sp.cos(psi), -sp.sin(psi), 1]) / sp.sqrt(2)    # 수평 원 ψ ↦ x(ρ cos ψ, ρ sin ψ) 위의 N
sym_equal("연습 8.1.1: dN(ρ e_h) = −e_h/√2 ⇒ W(e_h) = e_h/(√2 ρ)", Nfun.diff(psi).subs(psi, phi), -eh / sp.sqrt(2))
gen = t * sp.Matrix([sp.cos(phi), sp.sin(phi), 1])
sym_equal("연습 8.1.1: 모선의 속도 = √2 e_r, 그 위에서 N 상수", gen.diff(t), sp.sqrt(2) * er)
Mcone, _ = dN_matrix(Xc, u, v)
Wc = -Mcone.subs(pol)
Bc = sp.Matrix.hstack(Xc.diff(u), Xc.diff(v)).subs(pol)
coef_h = Bc.solve_least_squares(eh)
coef_r = Bc.solve_least_squares(er)
sym_equal("연습 8.1.1: 좌표로 확인 W(e_h) = e_h/(√2 ρ)", Bc * Wc * coef_h, eh / (sp.sqrt(2) * rho), {rho: (0.2, 2), phi: (0, 6)})
sym_equal("연습 8.1.1: 좌표로 확인 W(e_r) = 0", Bc * Wc * coef_r, sp.zeros(3, 1), {rho: (0.2, 2), phi: (0, 6)})

# 연습 8.1.3 ---------------------------------------------------------------------------------------------
g = sp.diag(2, 1)
a, bq, cq, d = sp.symbols("a b c d", real=True)
Aq = sp.Matrix([[a, bq], [cq, d]])
GA = g * Aq
check("연습 8.1.3(b): g에 대해 자기수반 ⟺ A²₁ = 2A¹₂", sp.solve(GA[0, 1] - GA[1, 0], cq) == [2 * bq])
Aswap = sp.Matrix([[0, 1], [1, 0]])
check("연습 8.1.3(a): g(Ae₁, e₂) = 1, g(e₁, Ae₂) = 2", ((Aswap * e1).T * g * e2)[0] == 1 and (e1.T * g * (Aswap * e2))[0] == 2)

# 연습 8.1.4 ---------------------------------------------------------------------------------------------
sym_equal("연습 8.1.4(a): |dN_0(w)|² = 4", dNw.dot(dNw), 4)
sym_equal("연습 8.1.4(b): det(w, dN w) = 2 sin 2β", w[0] * dNw[1] - w[1] * dNw[0], 2 * sp.sin(2 * beta))
sym_equal("연습 8.1.4(b): ⟨w, dN w⟩ = −2 cos 2β", w.dot(dNw), -2 * sp.cos(2 * beta))

# 연습 8.1.5: 나선면 x(u, v) = (v cos u, v sin u, u) -----------------------------------------------------------
Xh = sp.Matrix([v * sp.cos(u), v * sp.sin(u), u])
Mh, Nh = dN_matrix(Xh, u, v)
ell = sp.sqrt(1 + v ** 2)
check("연습 8.1.5(a): x(u, v)는 나선면 위 (x sin z − y cos z = 0)", sp.simplify(Xh[0] * sp.sin(Xh[2]) - Xh[1] * sp.cos(Xh[2])) == 0)
sym_equal("연습 8.1.5(a): N = (−sin u, cos u, −v)/√(1+v²)", Nh, sp.Matrix([-sp.sin(u), sp.cos(u), -v]) / ell)
sym_equal("연습 8.1.5(b): dN 행렬 [[0, −ℓ⁻³], [−ℓ⁻¹, 0]]", Mh, sp.Matrix([[0, -ell ** -3], [-1 / ell, 0]]))
Eh, Fh, Gh_ = dgsym.first_ff(Xh, u, v)
check("연습 8.1.5: E = 1 + v², F = 0, G = 1", (sp.simplify(Eh - 1 - v ** 2), Fh, Gh_) == (0, 0, 1))
check("연습 8.1.5(b): 대칭 ⟺ v = 0", sp.solve(sp.Eq(Mh[0, 1], Mh[1, 0]), v) == [0])
Gmh = sp.Matrix([[Eh, Fh], [Fh, Gh_]]) * Mh
check("연습 8.1.5(c): 그람 행렬을 곱하면 대칭 (자기수반)", sp.simplify(Gmh - Gmh.T) == sp.zeros(2, 2))
Y = sp.Matrix([sp.sinh(u) * sp.cos(v), sp.sinh(u) * sp.sin(v), v])
My, _ = dN_matrix(Y, u, v)
check("연습 8.1.5: 연습 7.1.6의 등온 매개화 y에서는 행렬이 대칭", sp.simplify(My - My.T) == sp.zeros(2, 2))

# 연습 8.1.6: 강체운동과 닮음 (수치: 원환면 R = 2, r = 0.8) -----------------------------------------------------
rng = np.random.default_rng(7)
Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
if np.linalg.det(Q) < 0:
    Q[:, 0] *= -1
cval, bval = 1.7, np.array([0.3, -1.0, 2.0])
vals = {R: 2.0, rr: 0.8}
Xt_n = sp.lambdify((tu, tv), list(Xt.subs(vals)), "numpy")
Nt_n = sp.lambdify((tu, tv), list(Nt.subs(vals)), "numpy")
W_n = sp.lambdify((tu, tv), dgsym.shape_operator(Xt, tu, tv, tor["positive"]).subs(vals), "numpy")
ok = True
for (a0, b0) in [(0.4, 1.1), (2.5, 4.0), (-1.0, 0.3)]:
    h = 1e-6
    def Xb(a1, b1):
        return cval * Q @ np.array(Xt_n(a1, b1), float) + bval
    def Nb(a1, b1):
        return Q @ np.array(Nt_n(a1, b1), float)
    Xbu = (Xb(a0 + h, b0) - Xb(a0 - h, b0)) / (2 * h)
    Xbv = (Xb(a0, b0 + h) - Xb(a0, b0 - h)) / (2 * h)
    Nbu = (Nb(a0 + h, b0) - Nb(a0 - h, b0)) / (2 * h)
    Nbv = (Nb(a0, b0 + h) - Nb(a0, b0 - h)) / (2 * h)
    Bm = np.stack([Xbu, Xbv], axis=1)
    Wbar = -np.linalg.lstsq(Bm, np.stack([Nbu, Nbv], axis=1), rcond=None)[0]
    # 기저 (x̄_u, x̄_v) = cQ(x_u, x_v)이므로 (1/c) Q W Q⁻¹의 이 기저 행렬은 (1/c)·[W]
    ok &= np.allclose(Wbar, np.array(W_n(a0, b0), float) / cval, atol=1e-6)
    ok &= abs(Nb(a0, b0) @ Xbu) < 1e-9 and abs(Nb(a0, b0) @ Xbv) < 1e-9
check("연습 8.1.6: W̄ = (1/c) A W A⁻¹ (원환면, 무작위 회전, 수치 미분)", ok)

summary()
