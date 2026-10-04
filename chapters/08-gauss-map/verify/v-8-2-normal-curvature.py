"""8.2절 제2기본형식과 법곡률: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/verify/v-8-2-normal-curvature.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES


def II_matrix(X, a, b, pos=()):
    """II(w₁, w₂) = ⟨W w₁, w₂⟩의 기저 (x_a, x_b)에 대한 행렬 = I·[W] (명제 8.1.7(b), (8.1.3))."""
    E, F, G = dgsym.first_ff(X, a, b, pos)
    W = dgsym.shape_operator(X, a, b, pos)
    return dgsym.simp(sp.Matrix([[E, F], [F, G]]) * W, pos)


# 예 8.2.2 ------------------------------------------------------------------------------------------
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
E, F, G = dgsym.first_ff(sph["expr"], th, ph, sph["positive"])
sym_equal("예 8.2.2(b): 구면 II = −(1/r) I (바깥쪽)", II_matrix(sph["expr"], th, ph, sph["positive"]), -sp.Matrix([[E, F], [F, G]]) / r, sph["domain"])
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(rc,) = cyl["params"]
sym_equal("예 8.2.2(c): 원기둥 II(a x_u + b x_v) = −r a²", II_matrix(cyl["expr"], cu, cv), sp.diag(-rc, 0), cyl["domain"])
sad = EX["saddle"]
u, v = sad["coords"]
sym_equal("예 8.2.2(d): 안장면 원점 II = diag(2, −2)", II_matrix(sad["expr"], u, v).subs({u: 0, v: 0}), sp.diag(2, -2))
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
sym_equal("(8.2.2): 원환면 II = diag(r, cos u (R + r cos u)) (N_x 안쪽)", II_matrix(tor["expr"], tu, tv, tor["positive"]),
          sp.diag(rr, sp.cos(tu) * (R + rr * sp.cos(tu))), tor["domain"])

# 명제 8.2.3: ⟨γ'', N⟩ = II(γ') (호의 길이가 아닌 곡선) -----------------------------------------------------
t = sp.symbols("t", real=True)
Xt = tor["expr"]
Nt = dgsym.unit_normal(Xt, tu, tv, tor["positive"])
IIt = II_matrix(Xt, tu, tv, tor["positive"])
curve = {tu: t, tv: t ** 2}
gam = Xt.subs(curve)
coef = sp.Matrix([1, 2 * t])                                  # γ' = u' x_u + v' x_v
lhs = gam.diff(t, 2).dot(Nt.subs(curve))
rhs = (coef.T * IIt.subs(curve) * coef)[0]
sym_equal("명제 8.2.3: 원환면 위 γ(t) = x(t, t²)에서 ⟨γ'', N⟩ = II(γ')", lhs, rhs, {t: (-2, 2), R: (1.5, 2), rr: (0.3, 0.9)})
Xs = sad["expr"]
Ns = dgsym.unit_normal(Xs, u, v)
IIs = II_matrix(Xs, u, v)
curve = {u: sp.cos(t), v: sp.sin(3 * t) / 2}
gam = Xs.subs(curve)
coef = sp.Matrix([-sp.sin(t), sp.Rational(3, 2) * sp.cos(3 * t)])
sym_equal("명제 8.2.3: 안장면 위 곡선에서 ⟨γ'', N⟩ = II(γ')", gam.diff(t, 2).dot(Ns.subs(curve)), (coef.T * IIs.subs(curve) * coef)[0], {t: (0, 6)})

# 예 8.2.6: 구면 위의 위도원과 대원 ----------------------------------------------------------------------
s, th0 = sp.symbols("s theta0", positive=True)
Xsph = sph["expr"]
lat = Xsph.subs({th: th0, ph: s / (r * sp.sin(th0))})          # 단위속력 위도원
sym_equal("예 8.2.6: 위도원 단위속력", lat.diff(s).dot(lat.diff(s)), 1, {th0: (0.2, 3.0), r: (0.5, 2)})
sym_equal("예 8.2.6: 위도원 ⟨γ'', N⟩ = −1/r", lat.diff(s, 2).dot(lat / r), -1 / r, {th0: (0.2, 3.0), r: (0.5, 2)})
sym_equal("예 8.2.6: 위도원 κ = 1/(r sin θ₀)", sp.sqrt(lat.diff(s, 2).dot(lat.diff(s, 2))), 1 / (r * sp.sin(th0)), {th0: (0.2, 3.0), r: (0.5, 2)})
p0 = Xsph.subs({th: th0, ph: 0})
e2 = sp.Matrix([0, 1, 0])
gc = sp.cos(s / r) * p0 + r * sp.sin(s / r) * e2
sym_equal("예 8.2.6: 대원은 구면 위, 단위속력, p에서 속도 e₂", sp.Matrix([gc.dot(gc), gc.diff(s).dot(gc.diff(s))]), sp.Matrix([r ** 2, 1]), {th0: (0.2, 3.0), r: (0.5, 2), s: (0.1, 3)})
sym_equal("예 8.2.6: 대원 γ'' = −N/r", gc.diff(s, 2), -gc / r ** 2, {th0: (0.2, 3.0), r: (0.5, 2), s: (0.1, 3)})
sym_equal("예 8.2.6: 위도원의 속도(φ = 0) = e₂", lat.diff(s).subs(s, 0), e2, {th0: (0.2, 3.0), r: (0.5, 2)})

# 예 8.2.7: 원기둥 위의 나선 --------------------------------------------------------------------------------
hel = EX["helix"]
(ht,) = hel["coords"]
a, b = hel["params"]
g = hel["expr"]
Ncyl = sp.Matrix([sp.cos(ht), sp.sin(ht), 0])                 # 반지름 a 원기둥의 바깥쪽 N
kn = g.diff(ht, 2).dot(Ncyl) / g.diff(ht).dot(g.diff(ht))
sym_equal("예 8.2.7: 나선의 법곡률 = −a/(a² + b²)", kn, -a / (a ** 2 + b ** 2), hel["domain"])
kap, _ = dgsym.curvature_torsion(g, ht)
sym_equal("예 8.2.7: κ cos ϑ = −κ (n = −N)", kn, -kap, hel["domain"])
IIc = II_matrix(cyl["expr"], cu, cv).subs(rc, a)
tc = sp.Matrix([1, b]) / sp.sqrt(a ** 2 + b ** 2)
sym_equal("예 8.2.7: II(t) = −a/(a² + b²) (예 8.2.2(c) 사용)", (tc.T * IIc * tc)[0], -a / (a ** 2 + b ** 2), hel["domain"])

# 보조정리 8.2.8, 명제 8.2.10(a): 접평면 위의 그래프 (수치, 원환면 R = 2, r = 0.8, p = x(0.7, 0.4)) ----------------
vals = {R: 2.0, rr: 0.8}
Xn = sp.lambdify((tu, tv), list(Xt.subs(vals)), "numpy")
Jn = sp.lambdify((tu, tv), Xt.jacobian([tu, tv]).subs(vals), "numpy")
u0, v0 = 0.7, 0.4
p = np.array(Xn(u0, v0), float)
Jq = np.array(Jn(u0, v0), float)
e1n = Jq[:, 0] / np.linalg.norm(Jq[:, 0])
e2n = Jq[:, 1] / np.linalg.norm(Jq[:, 1])                   # F = 0이라 이미 직교
e3n = np.cross(e1n, e2n)
Nn = np.array(sp.lambdify((tu, tv), list(Nt.subs(vals)), "numpy")(u0, v0), float)
check("보조정리 8.2.8: e₃ = e₁ × e₂ = N_x(p) (원환면)", np.allclose(e3n, Nn))


def f_height(xi, eta):
    """P∘x(q) = (ξ, η)를 뉴턴법으로 풀고 f = ⟨x(q) − p, e₃⟩."""
    q = np.array([u0, v0])
    for _ in range(50):
        y = np.array(Xn(*q), float) - p
        res = np.array([y @ e1n - xi, y @ e2n - eta])
        J = np.array(Jn(*q), float)
        D = np.stack([e1n @ J, e2n @ J])
        q = q - np.linalg.solve(D, res)
        if np.max(np.abs(res)) < 1e-14:
            break
    return (np.array(Xn(*q), float) - p) @ e3n


h = 1e-3
f0 = f_height(0, 0)
fx = (f_height(h, 0) - f_height(-h, 0)) / (2 * h)
fy = (f_height(0, h) - f_height(0, -h)) / (2 * h)
fxx = (f_height(h, 0) - 2 * f0 + f_height(-h, 0)) / h ** 2
fyy = (f_height(0, h) - 2 * f0 + f_height(0, -h)) / h ** 2
fxy = (f_height(h, h) - f_height(h, -h) - f_height(-h, h) + f_height(-h, -h)) / (4 * h ** 2)
close("보조정리 8.2.8: f(0) = 0, f_ξ(0) = f_η(0) = 0", [f0, fx, fy], [0, 0, 0], tol=1e-6)
Wn = np.array(dgsym.shape_operator(Xt, tu, tv, tor["positive"]).subs(vals).subs({tu: u0, tv: v0}), float)
IIe = lambda A, B: (Wn @ A) @ (np.array([[Jq[:, 0] @ Jq[:, 0], 0], [0, Jq[:, 1] @ Jq[:, 1]]]) @ B)
# e₁, e₂의 (x_u, x_v) 성분
c1 = np.array([1 / np.linalg.norm(Jq[:, 0]), 0.0])
c2 = np.array([0.0, 1 / np.linalg.norm(Jq[:, 1])])
close("명제 8.2.10(a): f_ξξ(0) = II(e₁), f_ηη(0) = II(e₂), f_ξη(0) = II(e₁, e₂)",
      [fxx, fyy, fxy], [IIe(c1, c1), IIe(c2, c2), IIe(c1, c2)], tol=1e-5)

# 예 8.2.11: 안장면의 법단면 ---------------------------------------------------------------------------------
beta, xi, eta = sp.symbols("beta xi eta", real=True)
x_ = xi * sp.cos(beta) - eta * sp.sin(beta)
y_ = xi * sp.sin(beta) + eta * sp.cos(beta)
fs = x_ ** 2 - y_ ** 2
sym_equal("예 8.2.11: f(ξ, η) = (ξ² − η²) cos 2β − 2ξη sin 2β", fs, (xi ** 2 - eta ** 2) * sp.cos(2 * beta) - 2 * xi * eta * sp.sin(2 * beta))
sym_equal("예 8.2.11: f(ξ, 0) = ξ² cos 2β", fs.subs(eta, 0), xi ** 2 * sp.cos(2 * beta))
check("예 8.2.11: f(0) = f_ξ(0) = f_η(0) = 0",
      all(sp.simplify(e.subs({xi: 0, eta: 0})) == 0 for e in (fs, fs.diff(xi), fs.diff(eta))))
wv = sp.Matrix([sp.cos(beta), sp.sin(beta)])
sym_equal("예 8.2.11: κ_n(w) = f_ξξ(0) = 2 cos 2β = II(w)", fs.diff(xi, 2).subs({xi: 0, eta: 0}), (wv.T * sp.diag(2, -2) * wv)[0])
gsec = sp.Matrix([xi * sp.cos(beta), xi * sp.sin(beta), xi ** 2 * sp.cos(2 * beta)])
sym_equal("명제 8.2.10(b): 법단면의 원점 곡률 = |2 cos 2β|", (gsec.diff(xi).cross(gsec.diff(xi, 2))).norm().subs(xi, 0), sp.Abs(2 * sp.cos(2 * beta)), {beta: (0, 3)})
check("예 8.2.11: 직선 x = ±y, z = 0은 안장면 위 (z = x² − y²에 대입)", all(sp.expand(Xs[2].subs({u: xi, v: sg * xi})) == 0 for sg in (1, -1)))
# 명제 8.2.10(b): 호의 길이 재매개화의 β''(0) = κ_n N (수치 대입, β = 0.3)
bb = 0.3
# 원점에서 |γ'| = 1이고 γ'' ⊥ γ'이므로 호의 길이 재매개화의 β''(0) = γ''(0)이다 (본문의 ξ''(0) = 0).
g1 = np.array(sp.lambdify(xi, list(gsec.diff(xi).subs(beta, bb)), "numpy")(0.0), float)
g2 = np.array(sp.lambdify(xi, list(gsec.diff(xi, 2).subs(beta, bb)), "numpy")(0.0), float)
close("명제 8.2.10(b): γ_w'(0) = w (단위), γ_w''(0) = 2cos 2β e₃ ⊥ w", np.concatenate([g1, g2]),
      np.array([np.cos(bb), np.sin(bb), 0, 0, 0, 2 * np.cos(2 * bb)]), tol=1e-12)

# 연습 8.2.1 -------------------------------------------------------------------------------------------------
Ec, Gc = rr ** 2, (R + rr * sp.cos(tu)) ** 2
IIdiag = sp.diag(rr, sp.cos(tu) * (R + rr * sp.cos(tu)))
e1c = sp.Matrix([1 / rr, 0])
e2c = sp.Matrix([0, 1 / (R + rr * sp.cos(tu))])
q = lambda c: (c.T * IIdiag * c)[0]
sym_equal("연습 8.2.1: κ_n(e₁) = 1/r", q(e1c), 1 / rr, tor["domain"])
sym_equal("연습 8.2.1: κ_n(e₂) = cos u/(R + r cos u)", q(e2c), sp.cos(tu) / (R + rr * sp.cos(tu)), tor["domain"])
sym_equal("연습 8.2.1: κ_n((e₁+e₂)/√2) = (1/r + cos u/(R + r cos u))/2", q((e1c + e2c) / sp.sqrt(2)),
          (1 / rr + sp.cos(tu) / (R + rr * sp.cos(tu))) / 2, tor["domain"])
bq = sp.symbols("b_q", real=True)
kn_top = q(sp.cos(bq) * e1c + sp.sin(bq) * e2c).subs(tu, sp.pi / 2)
sym_equal("연습 8.2.1: u = π/2에서 κ_n = cos²β / r", kn_top, sp.cos(bq) ** 2 / rr, {rr: (0.3, 0.9), R: (1.2, 2)})

# 연습 8.2.5: 원기둥의 법단면은 타원 --------------------------------------------------------------------------
ss, tt_ = sp.symbols("s t_", real=True)
pt = sp.Matrix([rc + tt_, ss * sp.cos(beta), ss * sp.sin(beta)])
check("연습 8.2.5: (r + t)² + s² cos²β = r² ⟺ (t + r)²/r² + s²/(r/|cos β|)² = 1",
      sp.simplify((pt[0] ** 2 + pt[1] ** 2 - rc ** 2) / rc ** 2 - ((tt_ + rc) ** 2 / rc ** 2 + ss ** 2 * sp.cos(beta) ** 2 / rc ** 2 - 1)) == 0)
A_, B_, tau = sp.symbols("A B tau", positive=True)
ell = sp.Matrix([A_ * sp.cos(tau), B_ * sp.sin(tau)])
ks = (ell[0].diff(tau) * ell[1].diff(tau, 2) - ell[0].diff(tau, 2) * ell[1].diff(tau)) / (ell[0].diff(tau) ** 2 + ell[1].diff(tau) ** 2) ** sp.Rational(3, 2)
sym_equal("연습 8.2.5: 타원의 B축 끝점 곡률 = B/A²", ks.subs(tau, sp.pi / 2), B_ / A_ ** 2)
wc = sp.Matrix([sp.cos(beta) / rc, sp.sin(beta)])
sym_equal("연습 8.2.5: κ_n(w) = −cos²β/r, |κ_n| = B/A² (A = r/|cos β|, B = r)", (wc.T * sp.diag(-rc, 0) * wc)[0],
          -(rc / (rc / sp.cos(beta)) ** 2), {beta: (0.1, 1.4), rc: (0.5, 2)})

# 연습 8.2.6: 곡률중심은 원 위에 있다 (수치) ------------------------------------------------------------------
rng = np.random.default_rng(3)
kn0 = -0.7
Nv, tv_ = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
m = np.cross(Nv, tv_)
pp = np.array([0.2, -0.1, 0.5])
ok = True
for _ in range(20):
    vt = rng.uniform(0, np.pi)
    if abs(np.cos(vt)) < 0.05:
        continue
    eps = rng.choice([-1, 1])
    nvec = np.cos(vt) * Nv + eps * np.sin(vt) * m
    kap_ = kn0 / np.cos(vt)
    if kap_ <= 0:
        continue                                                   # 곡률은 양수여야 함: cos ϑ와 κ_n의 부호가 같을 때
    c = pp + nvec / kap_
    ok &= abs(np.linalg.norm(c - (pp + Nv / (2 * kn0))) - 1 / (2 * abs(kn0))) < 1e-12
    ok &= abs((c - pp) @ tv_) < 1e-12
check("연습 8.2.6: 곡률중심은 중심 p + N/(2κ_n), 반지름 1/(2|κ_n|)인 원 위 (t에 수직)", ok)

summary()
