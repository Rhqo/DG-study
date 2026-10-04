"""8.5절 국소좌표 공식: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

기준 예제의 L, M, N, K, H는 dgsym(second_ff, K_H, shape_operator)과 EXAMPLES로 계산해 §7 값과 대조한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/verify/v-8-5-local-coordinates.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
u, v = sp.symbols("u v", real=True)

# 명제 8.5.2: L = −⟨N_u, x_u⟩ 등 (일반 매개화, N = N_x) --------------------------------------------------------------
f1, f2, f3 = (sp.Function(n)(u, v) for n in ("f1", "f2", "f3"))
Xg = sp.Matrix([f1, f2, f3])
n = Xg.diff(u).cross(Xg.diff(v))           # N = n/|n|; ⟨n, x_u⟩ = ⟨n, x_v⟩ = 0이므로 분자만 비교
check("명제 8.5.2: ⟨n_u, x_u⟩ = −⟨n, x_uu⟩ (L의 두 표현, 분자)", sp.expand(n.diff(u).dot(Xg.diff(u)) + n.dot(Xg.diff(u, 2))) == 0)
check("명제 8.5.2: ⟨n_v, x_v⟩ = −⟨n, x_vv⟩ (N의 두 표현, 분자)", sp.expand(n.diff(v).dot(Xg.diff(v)) + n.dot(Xg.diff(v, 2))) == 0)

# 명제 8.5.4, 정리 8.5.5: 일반 계수에서 --------------------------------------------------------------------------------
E, F, G, L, M, N = sp.symbols("E F G L M N", real=True)
I = sp.Matrix([[E, F], [F, G]])
II = sp.Matrix([[L, M], [M, N]])
Wm = I.inv() * II
D = E * G - F ** 2
check("(8.5.5): I⁻¹ II = (1/(EG−F²)) [[GL−FM, GM−FN], [EM−FL, EN−FM]]",
      sp.simplify(Wm - sp.Matrix([[G * L - F * M, G * M - F * N], [E * M - F * L, E * N - F * M]]) / D) == sp.zeros(2, 2))
check("(8.5.7): det(I⁻¹II) = (LN − M²)/(EG − F²)", sp.simplify(Wm.det() - (L * N - M ** 2) / D) == 0)
check("(8.5.7): ½ tr(I⁻¹II) = (EN − 2FM + GL)/(2(EG − F²))", sp.simplify(Wm.trace() / 2 - (E * N - 2 * F * M + G * L) / (2 * D)) == 0)
k = sp.symbols("kappa")
check("(8.5.8): det(II − κ I) = (EG−F²)κ² − (EN − 2FM + GL)κ + (LN − M²)",
      sp.expand((II - k * I).det() - (D * k ** 2 - (E * N - 2 * F * M + G * L) * k + (L * N - M ** 2))) == 0)
check("명제 8.5.4 뒤: I·[W] = II 는 대칭 (자기수반)", sp.simplify(I * Wm - (I * Wm).T) == sp.zeros(2, 2))

# 명제 8.5.3: 계수의 변환 (구면, 변수 교환 h(s,t) = (t,s)) -----------------------------------------------------------
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
Xs = sph["expr"]
Ls, Ms, Ns = dgsym.second_ff(Xs, th, ph, sph["positive"])
Y = Xs.subs({th: v, ph: u}, simultaneous=True)            # y(u, v) = x(v, u): u = φ, v = θ
posY = (sp.sin(v),)
Ly, My, Ny = dgsym.second_ff(Y, u, v, posY)
Dh = sp.Matrix([[0, 1], [1, 0]])
rhs = -Dh.T * sp.Matrix([[Ls, Ms], [Ms, Ns]]).subs({th: v, ph: u}, simultaneous=True) * Dh
sym_equal("(8.5.4): 변수를 바꾼 구면 매개화의 계수 = sgn(det Dh) Dhᵀ[II]Dh", sp.Matrix([[Ly, My], [My, Ny]]), rhs, {u: (0.1, 6), v: (0.1, 3), r: (0.5, 2)})

# 예 8.5.7–8.5.9: 기준 예제의 L, M, N, K, H (§7) ------------------------------------------------------------------------
for key in ("sphere", "cylinder"):
    e = EX[key]
    a, b = e["coords"]
    LMN = dgsym.second_ff(e["expr"], a, b, e["positive"])
    for nm, val in zip("LMN", LMN):
        sym_equal(f"예 8.5.{7 if key == 'sphere' else 8}: {e['name']} {nm} (§7, {e['normal']})", val, e["expected"][nm], e["domain"])
    K, H = dgsym.K_H(e["expr"], a, b, e["positive"])
    sym_equal(f"예 8.5.{7 if key == 'sphere' else 8}: {e['name']} K, H (§7)", sp.Matrix([K, H]), sp.Matrix([e["expected"]["K"], e["expected"]["H"]]), e["domain"])
e = EX["sphere"]
sym_equal("예 8.5.7: x_θθ = −x", Xs.diff(th, 2), -Xs, e["domain"])
sym_equal("예 8.5.7: 주곡률 L/E = N/G = −1/r", sp.Matrix([Ls / r ** 2, Ns / (r ** 2 * sp.sin(th) ** 2)]), sp.Matrix([-1 / r, -1 / r]), e["domain"])
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
Lt, Mt, Nt = dgsym.second_ff(tor["expr"], tu, tv, tor["positive"])
sym_equal("(8.5.9): 원환면 L = r, M = 0, N = cos u (R + r cos u) (N_x 안쪽)", sp.Matrix([Lt, Mt, Nt]),
          sp.Matrix([rr, 0, sp.cos(tu) * (R + rr * sp.cos(tu))]), tor["domain"])
Kt, Ht = dgsym.K_H(tor["expr"], tu, tv, tor["positive"])
sym_equal("예 8.5.9: 원환면 K (§7)", Kt, tor["expected"]["K"], tor["domain"])
sym_equal("예 8.5.9: 원환면 H = (R + 2r cos u)/(2r(R + r cos u))", Ht, (R + 2 * rr * sp.cos(tu)) / (2 * rr * (R + rr * sp.cos(tu))), tor["domain"])
sym_equal("예 8.5.9: 바깥쪽이면 L = −r, N = −cos u(R + r cos u)", sp.Matrix([-Lt, -Nt]), sp.Matrix([-rr, -sp.cos(tu) * (R + rr * sp.cos(tu))]), tor["domain"])
sym_equal("예 8.5.9: shape_operator = I⁻¹II = diag(L/E, N/G)", dgsym.shape_operator(tor["expr"], tu, tv, tor["positive"]),
          sp.diag(Lt / rr ** 2, Nt / (R + rr * sp.cos(tu)) ** 2), tor["domain"])

# 예 8.5.10: 회전면 --------------------------------------------------------------------------------------------------------
rev = EX["revolution"]
ru, rv = rev["coords"]
rho, zf = rev["functions"]
Lr, Mr, Nr = dgsym.second_ff(rev["expr"], ru, rv)
rp, rpp = sp.diff(rho(ru), ru), sp.diff(rho(ru), ru, 2)
zp, zpp = sp.diff(zf(ru), ru), sp.diff(zf(ru), ru, 2)
sa = sp.sqrt(rp ** 2 + zp ** 2)
sym_equal("예 8.5.10: 회전면 L = (ρ'z'' − ρ''z')/|α'|, M = 0, N = ρz'/|α'|", sp.Matrix([Lr, Mr, Nr]), sp.Matrix([(rp * zpp - rpp * zp) / sa, 0, rho(ru) * zp / sa]))

# 명제 8.5.11: 그래프 곡면 ---------------------------------------------------------------------------------------------------
gr = EX["graph"]
gu, gv = gr["coords"]
(ff,) = gr["functions"]
fx = ff(gu, gv)
Lg, Mg, Ng = dgsym.second_ff(gr["expr"], gu, gv)
ell = sp.sqrt(1 + fx.diff(gu) ** 2 + fx.diff(gv) ** 2)
sym_equal("(8.5.10): 그래프 L, M, N = f_uu/ℓ, f_uv/ℓ, f_vv/ℓ", sp.Matrix([Lg, Mg, Ng]), sp.Matrix([fx.diff(gu, 2), fx.diff(gu, gv), fx.diff(gv, 2)]) / ell)
Kg, Hg = dgsym.K_H(gr["expr"], gu, gv)
sym_equal("(8.5.10): 그래프 K (§7)", Kg, gr["expected"]["K"])
fu, fv, fuu, fuv, fvv = fx.diff(gu), fx.diff(gv), fx.diff(gu, 2), fx.diff(gu, gv), fx.diff(gv, 2)
sym_equal("(8.5.10): 그래프 H", Hg, ((1 + fv ** 2) * fuu - 2 * fu * fv * fuv + (1 + fu ** 2) * fvv) / (2 * (1 + fu ** 2 + fv ** 2) ** sp.Rational(3, 2)))
xs, ys = sp.symbols("x y", real=True)
rq = sp.symbols("r", positive=True)
fn = sp.sqrt(rq ** 2 - xs ** 2 - ys ** 2)
check("명제 8.5.11 뒤: 예 2.2.18의 북극 헤세 = −(1/r) I",
      sp.simplify(sp.hessian(fn, (xs, ys)).subs({xs: 0, ys: 0}) + sp.eye(2) / rq) == sp.zeros(2, 2))

# 예 8.5.12: 안장면 ---------------------------------------------------------------------------------------------------------
sad = EX["saddle"]
Ks, Hs = dgsym.K_H(sad["expr"], u, v)
sym_equal("예 8.5.12: 안장면 K = −4/(1 + 4u² + 4v²)²", Ks, -4 / (1 + 4 * u ** 2 + 4 * v ** 2) ** 2)
sym_equal("예 8.5.12: 안장면 H = 4(v² − u²)/(1 + 4u² + 4v²)^{3/2}", Hs, 4 * (v ** 2 - u ** 2) / (1 + 4 * u ** 2 + 4 * v ** 2) ** sp.Rational(3, 2))
sym_equal("예 8.5.12: 원점 K = −4, H = 0 (§7)", sp.Matrix([Ks, Hs]).subs(sad["point"]), sp.Matrix([sad["expected"]["K_at_origin"], sad["expected"]["H_at_origin"]]))
half = {u: sp.Rational(1, 2), v: sp.Rational(1, 2)}
sym_equal("예 8.5.12: x(1/2, 1/2)에서 K = −4/9, H = 0", sp.Matrix([Ks, Hs]).subs(half), sp.Matrix([-sp.Rational(4, 9), 0]))

# 명제 8.5.13: 테일러 전개와 부호 (수치: 원환면의 두 점) ------------------------------------------------------------------------
vals = {R: 2.0, rr: 0.8}
Xn = sp.lambdify((tu, tv), list(tor["expr"].subs(vals)), "numpy")
Nn = sp.lambdify((tu, tv), list(dgsym.unit_normal(tor["expr"], tu, tv, tor["positive"]).subs(vals)), "numpy")
ok = True
for u0, sign_expect in ((0.0, "one"), (np.pi, "both"), (0.9, "one"), (2.5, "both")):
    p = np.array(Xn(u0, 0.0), float)
    nv = np.array(Nn(u0, 0.0), float)
    g = np.linspace(-0.2, 0.2, 41)
    U, V = np.meshgrid(u0 + g, g, indexing="ij")
    Q = np.stack(Xn(U, V), axis=-1)
    hgt = (Q - p) @ nv
    mask = np.hypot(U - u0, V) > 1e-9
    if sign_expect == "one":
        ok &= np.all(hgt[mask] > 0) or np.all(hgt[mask] < 0)
    else:
        ok &= hgt.min() < 0 < hgt.max()
check("명제 8.5.13: 원환면 타원점(u = 0, 0.9)은 한쪽, 쌍곡점(u = π, 2.5)은 양쪽", ok)
# (8.5.11): 높이 f의 2차 근사 오차 = O(|h|³) (원환면 u = 0, 보조정리 8.2.8의 그래프)
xi, eta = sp.symbols("xi eta", real=True)
# 바깥쪽 적도 p = (R + r, 0, 0): e₁ = x_u/r = (0, 0, 1), e₂ = x_v/(R + r) = (0, 1, 0), e₁ × e₂ = (−1, 0, 0) = N_x(p)
# 곡면 위의 점 p + ξe₁ + ηe₂ + f·N = (R + r − f, η, ξ):  (√((R + r − f)² + η²) − R)² + ξ² = r²
fexact = (R + rr) - sp.sqrt((R + sp.sqrt(rr ** 2 - xi ** 2)) ** 2 - eta ** 2)
sym_equal("(8.5.11): 원환면 u = 0에서 f의 헤세 = diag(κ₁, κ₂) = diag(1/r, 1/(R + r))",
          sp.hessian(fexact, (xi, eta)).subs({xi: 0, eta: 0}), sp.diag(1 / rr, 1 / (R + rr)), {R: (1.5, 2), rr: (0.3, 0.8)})
fnum = sp.lambdify((xi, eta), fexact.subs(vals), "numpy")
errs = []
for hval in (0.08, 0.04, 0.02):
    val = fnum(hval / np.sqrt(2), hval / np.sqrt(2))
    quad = 0.5 * ((hval / np.sqrt(2)) ** 2 / 0.8 + (hval / np.sqrt(2)) ** 2 / 2.8)
    errs.append(abs(val - quad))
check("(8.5.11): 오차 |f − ½(κ₁ξ² + κ₂η²)|가 적어도 |h|³ 비율로 줄어든다 (여기서는 대칭이라 |h|⁴)", errs[0] / errs[1] > 6 and errs[1] / errs[2] > 6)

# 비예 8.5.14 --------------------------------------------------------------------------------------------------------------
tt = sp.symbols("t", real=True)
for expr, hess, name in ((v ** 2, sp.diag(0, 2), "z = v²"), (u ** 3 + v ** 2, sp.diag(0, 2), "z = u³ + v²"),
                         (u ** 4 + v ** 4, sp.zeros(2, 2), "z = u⁴ + v⁴"), (u ** 3 - 3 * u * v ** 2, sp.zeros(2, 2), "원숭이 안장")):
    W0 = dgsym.shape_operator(sp.Matrix([u, v, expr]), u, v).subs({u: 0, v: 0})
    check(f"비예 8.5.14: {name}의 원점 W = 헤세 = {list(hess)}", W0 == hess)
check("비예 8.5.14(b): z(t, 0) = t³ (양쪽)", (u ** 3 + v ** 2).subs({u: tt, v: 0}) == tt ** 3)

# 연습 8.5.1: 원숭이 안장 -------------------------------------------------------------------------------------------------
Km, _ = dgsym.K_H(sp.Matrix([u, v, u ** 3 - 3 * u * v ** 2]), u, v)
sym_equal("연습 8.5.1: K = −36(u² + v²)/(1 + 9(u² + v²)²)²", Km, -36 * (u ** 2 + v ** 2) / (1 + 9 * (u ** 2 + v ** 2) ** 2) ** 2)

# 연습 8.5.2: 구면의 변수 교환 ------------------------------------------------------------------------------------------------
sym_equal("연습 8.5.2: N_y = −x/r", dgsym.unit_normal(Y, u, v, posY), -Xs.subs({th: v, ph: u}, simultaneous=True) / r, {u: (0.1, 6), v: (0.1, 3), r: (0.5, 2)})
sym_equal("연습 8.5.2: L̄ = r sin²θ, M̄ = 0, N̄ = r", sp.Matrix([Ly, My, Ny]), sp.Matrix([r * sp.sin(v) ** 2, 0, r]), {u: (0.1, 6), v: (0.1, 3), r: (0.5, 2)})
Ky, Hy = dgsym.K_H(Y, u, v, posY)
sym_equal("연습 8.5.2: K = 1/r², H = +1/r", sp.Matrix([Ky, Hy]), sp.Matrix([1 / r ** 2, 1 / r]), {u: (0.1, 6), v: (0.1, 3), r: (0.5, 2)})

# 연습 8.5.3: 기둥면 z = f(u) --------------------------------------------------------------------------------------------------
g1 = sp.Function("g")(u)
Kc, Hc = dgsym.K_H(sp.Matrix([u, v, g1]), u, v)
sym_equal("연습 8.5.3: K = 0, H = f''/(2(1 + f'²)^{3/2})", sp.Matrix([Kc, Hc]), sp.Matrix([0, g1.diff(u, 2) / (2 * (1 + g1.diff(u) ** 2) ** sp.Rational(3, 2))]))

# 연습 8.5.4: 셰르크 곡면 ------------------------------------------------------------------------------------------------------
Ksh, Hsh = dgsym.K_H(sp.Matrix([u, v, sp.log(sp.cos(v)) - sp.log(sp.cos(u))]), u, v)
sym_equal("연습 8.5.4: 셰르크 곡면 H = 0", Hsh, 0, {u: (-1.4, 1.4), v: (-1.4, 1.4)})
check("연습 8.5.4: K < 0 (표본점)", all(float(Ksh.subs({u: a_, v: b_})) < 0 for a_, b_ in ((0.1, 0.2), (-1.0, 0.7), (1.2, -1.3))))

# 연습 8.5.5: 안장면의 전곡률 --------------------------------------------------------------------------------------------------
rho_, P = sp.symbols("rho P", positive=True)
Eg, Fg, Gg = dgsym.first_ff(sad["expr"], u, v)
integrand = sp.simplify(sp.Abs(Ks) * sp.sqrt(Eg * Gg - Fg ** 2))
sym_equal("연습 8.5.5: |K|√(EG − F²) = 4/(1 + 4u² + 4v²)^{3/2}", integrand, 4 / (1 + 4 * u ** 2 + 4 * v ** 2) ** sp.Rational(3, 2))
Ipart = 2 * sp.pi * sp.integrate(4 * rho_ / (1 + 4 * rho_ ** 2) ** sp.Rational(3, 2), (rho_, 0, P))
sym_equal("연습 8.5.5: 원판 적분 = 2π(1 − 1/√(1 + 4P²))", Ipart, 2 * sp.pi * (1 - 1 / sp.sqrt(1 + 4 * P ** 2)))
check("연습 8.5.5: P → ∞ 극한 = 2π", sp.limit(Ipart, P, sp.oo) == 2 * sp.pi)
# 극관의 넓이: 가우스 상은 셋째 성분 > 1/√(1 + 4P²)인 극관, 넓이 2π(1 − c₀)
c0 = 1 / sp.sqrt(1 + 4 * P ** 2)
sym_equal("연습 8.5.5: 원판의 가우스 상(극관)의 넓이 = 적분값", 2 * sp.pi * (1 - c0), Ipart)

# 연습 8.5.6: 정칙값의 역상 ---------------------------------------------------------------------------------------------------
x, y, z = sp.symbols("x y z", real=True)
ft = (sp.sqrt(x ** 2 + y ** 2) - R) ** 2 + z ** 2 - rr ** 2
Hf = sp.hessian(ft, (x, y, z)).subs({x: R + rr, y: 0, z: 0})
gf = sp.Matrix([ft]).jacobian([x, y, z]).subs({x: R + rr, y: 0, z: 0})
sym_equal("연습 8.5.6: 원환면 (R + r, 0, 0)에서 grad f = (2r, 0, 0)", gf, sp.Matrix([[2 * rr, 0, 0]]))
sym_equal("연습 8.5.6: II(e₂) = −1/(R + r), II(e₃) = −1/r (바깥쪽)", sp.Matrix([-Hf[1, 1] / (2 * rr), -Hf[2, 2] / (2 * rr)]),
          sp.Matrix([-1 / (R + rr), -1 / rr]), {R: (1.5, 2), rr: (0.3, 0.9)})
fs = x ** 2 + y ** 2 + z ** 2 - rq ** 2
check("연습 8.5.6: 구면 헤세 = 2I, |grad f| = 2r → II = −|w|²/r", sp.hessian(fs, (x, y, z)) == 2 * sp.eye(3))
# 일반 확인 (수치): 타원면 f = x²/4 + y² + z²/0.25 − 1, 무작위 점·접벡터에서 공식 = dgsym의 II (N = grad f/|grad f|)
aa, bb, cc = 2.0, 1.0, 0.5
Xe = sp.Matrix([aa * sp.sin(u) * sp.cos(v), bb * sp.sin(u) * sp.sin(v), cc * sp.cos(u)])
Ne = dgsym.unit_normal(Xe, u, v)
fE = x ** 2 / aa ** 2 + y ** 2 / bb ** 2 + z ** 2 / cc ** 2 - 1
ok = True
for (u0, v0) in ((0.7, 0.4), (1.9, 2.2), (2.5, 5.0)):
    pt = {x: Xe[0].subs({u: u0, v: v0}), y: Xe[1].subs({u: u0, v: v0}), z: Xe[2].subs({u: u0, v: v0})}
    grad = np.array(sp.Matrix([fE]).jacobian([x, y, z]).subs(pt), float).ravel()
    Hm = np.array(sp.hessian(fE, (x, y, z)).subs(pt), float)
    nvec = np.array(Ne.subs({u: u0, v: v0}), float).ravel()
    sgn = np.sign(grad @ nvec)                         # N_x와 grad f 방향의 관계
    xu = np.array(Xe.diff(u).subs({u: u0, v: v0}), float).ravel()
    II_formula = -(xu @ Hm @ xu) / np.linalg.norm(grad) * sgn
    II_direct = float((Xe.diff(u, 2).subs({u: u0, v: v0}).T * Ne.subs({u: u0, v: v0}))[0])
    ok &= abs(II_formula - II_direct) < 1e-10
check("연습 8.5.6: 타원면 표본점에서 −Σ∂ᵢ∂ⱼf wⁱwʲ/|grad f| = ⟨x_uu, N⟩ (N 방향 보정)", ok)

summary()
