"""8.4절 가우스 곡률과 평균곡률: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/verify/v-8-4-gaussian-mean-curvature.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES

# 예 8.4.3: 기준 예제의 K, H (dgsym.K_H, §7 값과 대조) -------------------------------------------------------
for key in ("sphere", "cylinder", "torus"):
    e = EX[key]
    a, b = e["coords"]
    K, H = dgsym.K_H(e["expr"], a, b, e["positive"])
    sym_equal(f"예 8.4.3: {e['name']} K (§7)", K, e["expected"]["K"], e["domain"])
    if "H" in e["expected"]:
        sym_equal(f"예 8.4.3: {e['name']} H (§7, {e['normal']})", H, e["expected"]["H"], e["domain"])
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
Kt, Ht = dgsym.K_H(tor["expr"], tu, tv, tor["positive"])
sym_equal("(8.4.2): 원환면 H = (R + 2r cos u)/(2r(R + r cos u)) (N_x 안쪽)", Ht, (R + 2 * rr * sp.cos(tu)) / (2 * rr * (R + rr * sp.cos(tu))), tor["domain"])
Kout, Hout = dgsym.K_H(tor["expr"].subs({tu: tv, tv: tu}, simultaneous=True), tu, tv, (R + rr * sp.cos(tv),))
sym_equal("예 8.4.3(e): 바깥쪽 법벡터(변수 교환)에서 K 같고 H 부호 반대",
          sp.Matrix([Kout, Hout]), sp.Matrix([Kt, -Ht]).subs({tu: tv, tv: tu}, simultaneous=True), {tu: (0, 6), tv: (0, 6), R: (1.2, 2), rr: (0.2, 0.9)})
sad = EX["saddle"]
u, v = sad["coords"]
Ks, Hs = dgsym.K_H(sad["expr"], u, v)
sym_equal("예 8.4.3(d): 안장면 원점 K = −4 (§7)", Ks.subs(sad["point"]), sad["expected"]["K_at_origin"])
sym_equal("예 8.4.3(d): 안장면 원점 H = 0 (§7)", Hs.subs(sad["point"]), sad["expected"]["H_at_origin"])

# 예 8.4.3(f): 회전면 -------------------------------------------------------------------------------------------
rev = EX["revolution"]
ru, rv = rev["coords"]
rho, zf = rev["functions"]
Kr, Hr = dgsym.K_H(rev["expr"], ru, rv)
rp, rpp = sp.diff(rho(ru), ru), sp.diff(rho(ru), ru, 2)
zp, zpp = sp.diff(zf(ru), ru), sp.diff(zf(ru), ru, 2)
sym_equal("예 8.4.3(f): 회전면 K = (ρ'z'' − ρ''z')z'/(ρ|α'|⁴)", Kr, (rp * zpp - rpp * zp) * zp / (rho(ru) * (rp ** 2 + zp ** 2) ** 2))
s = sp.symbols("s", real=True)
phi = sp.Function("phi")(s)                      # 단위속력: ρ' = cos φ, z' = sin φ
rhop, zp_ = sp.cos(phi), sp.sin(phi)
rhopp, zpp_ = sp.diff(rhop, s), sp.diff(zp_, s)
check("(8.4.3): 단위속력이면 (ρ'z'' − ρ''z')z' = −ρ''", sp.simplify((rhop * zpp_ - rhopp * zp_) * zp_ + rhopp) == 0)
r = sp.symbols("r", positive=True)
rhos = r * sp.sin(s / r)
sym_equal("예 8.4.3(f): 구면 ρ(s) = r sin(s/r)에서 −ρ''/ρ = 1/r²", -sp.diff(rhos, s, 2) / rhos, 1 / r ** 2, {s: (0.2, 2.5), r: (1, 2)})
uq = sp.symbols("u", real=True)
Xcat = sp.Matrix([sp.cosh(uq) * sp.cos(v), sp.cosh(uq) * sp.sin(v), uq])
Kc, Hc = dgsym.K_H(Xcat, uq, v)
sym_equal("예 8.4.3(g): 현수면 K = −1/cosh⁴u, H = 0", sp.Matrix([Kc, Hc]), sp.Matrix([-1 / sp.cosh(uq) ** 4, 0]), {uq: (-2, 2)})

# 명제 8.4.2(c) --------------------------------------------------------------------------------------------------
k1, k2 = 1 / rr, sp.cos(tu) / (R + rr * sp.cos(tu))
sym_equal("명제 8.4.2(c): 원환면 H ± √(H² − K) = κ₁, κ₂", sp.Matrix([Ht + sp.sqrt(Ht ** 2 - Kt), Ht - sp.sqrt(Ht ** 2 - Kt)]), sp.Matrix([k1, k2]), tor["domain"])
sym_equal("명제 8.4.2(c): H² − K = ((κ₁ − κ₂)/2)²", Ht ** 2 - Kt, ((k1 - k2) / 2) ** 2, tor["domain"])

# 예 8.4.5: 원환면의 부호 분포 ----------------------------------------------------------------------------------------
vals = {R: 2, rr: sp.Rational(4, 5)}
check("예 8.4.5: u = 0, π/4에서 K > 0; u = 3π/4, π에서 K < 0; u = ±π/2에서 K = 0",
      all(Kt.subs(vals).subs(tu, x) > 0 for x in (0, sp.pi / 4)) and all(Kt.subs(vals).subs(tu, x) < 0 for x in (3 * sp.pi / 4, sp.pi))
      and all(sp.simplify(Kt.subs(tu, x)) == 0 for x in (sp.pi / 2, -sp.pi / 2)))
Wt = dgsym.shape_operator(tor["expr"], tu, tv, tor["positive"])
check("예 8.4.5: u = π/2에서 W = diag(1/r, 0) ≠ 0 (포물점)", sp.simplify(Wt.subs(tu, sp.pi / 2) - sp.diag(1 / rr, 0)) == sp.zeros(2, 2))

# 비예 8.4.6: 원숭이 안장 ------------------------------------------------------------------------------------------
xi, eta, t = sp.symbols("xi eta t", real=True)
fm = xi ** 3 - 3 * xi * eta ** 2
check("비예 8.4.6: 원숭이 안장의 원점 헤세 = 0, ∇f = 0", sp.hessian(fm, (xi, eta)).subs({xi: 0, eta: 0}) == sp.zeros(2, 2)
      and all(sp.diff(fm, x).subs({xi: 0, eta: 0}) == 0 for x in (xi, eta)))
Xm = sp.Matrix([u, v, u ** 3 - 3 * u * v ** 2])
check("비예 8.4.6: dgsym으로도 W_0 = 0", dgsym.shape_operator(Xm, u, v).subs({u: 0, v: 0}) == sp.zeros(2, 2))
check("비예 8.4.6: f(t, 0) = t³ (양쪽 부호)", fm.subs({xi: t, eta: 0}) == t ** 3)

# 명제 8.4.8, 연습 8.4.4, 8.4.6: 점근방향 ---------------------------------------------------------------------------------
beta = sp.symbols("beta", real=True)
Ws0 = sp.diag(2, -2)
wv = sp.Matrix([sp.cos(beta), sp.sin(beta)])
check("명제 8.4.8: 안장면 원점 점근방향 β = ±π/4", set(sp.solve(sp.Eq((wv.T * Ws0 * wv)[0], 0), beta)) >= {sp.pi / 4, -sp.pi / 4} or
      all(sp.simplify((wv.T * Ws0 * wv)[0].subs(beta, x)) == 0 for x in (sp.pi / 4, -sp.pi / 4)))
k1n, k2n = 1 / 0.8, -1 / 1.2
b0 = np.arctan(np.sqrt(-k1n / k2n))
close("연습 8.4.4(c): 원환면 u = π의 점근방향 ≈ ±50.8°", np.degrees(b0), 50.768, tol=1e-3)
omega = min(2 * b0, np.pi - 2 * b0)
close("연습 8.4.6: 두 점근방향 사이 각 ω, tan²(ω/2) = −κ₂/κ₁ 또는 역수", min(abs(np.tan(omega / 2) ** 2 - (-k2n / k1n)), abs(np.tan(omega / 2) ** 2 - (-k1n / k2n))), 0.0, tol=1e-12)
close("연습 8.4.6: 원환면 안쪽 적도에서 2β₀ ≈ 101.5°, ω ≈ 78.5°", [np.degrees(2 * b0), np.degrees(omega)], [101.537, 78.463], tol=1e-3)
k1s, k2s = sp.symbols("k1 k2", real=True)
check("연습 8.4.6: 수직(ω = π/2) ⟺ −κ₂/κ₁ = 1 ⟺ H = 0", sp.solve(sp.Eq(-k2s / k1s, 1), k2s) == [-k1s])

# 정리 8.4.9: 타원면 (연습 8.4.7) --------------------------------------------------------------------------------------
a, b, c = sp.symbols("a b c", positive=True)
Xe = sp.Matrix([a * sp.sqrt(1 - u ** 2 / b ** 2 - v ** 2 / c ** 2), u, v])     # (a, 0, 0) 근처, e₁ = (0,1,0), e₂ = (0,0,1)
Ke, He = dgsym.K_H(Xe, u, v)
Ne = dgsym.unit_normal(Xe, u, v)
sym_equal("연습 8.4.7: 이 매개화의 N(0) = (1, 0, 0) (바깥쪽)", Ne.subs({u: 0, v: 0}), sp.Matrix([1, 0, 0]), {a: (2, 3), b: (1, 1.9), c: (0.3, 0.9)})
sym_equal("연습 8.4.7: 타원면 (a, 0, 0)에서 K = a²/(b²c²)", Ke.subs({u: 0, v: 0}), a ** 2 / (b ** 2 * c ** 2), {a: (2, 3), b: (1, 1.9), c: (0.3, 0.9)})
We = dgsym.shape_operator(Xe, u, v).subs({u: 0, v: 0})
sym_equal("연습 8.4.7: W = diag(−a/b², −a/c²)", We, sp.diag(-a / b ** 2, -a / c ** 2), {a: (2, 3), b: (1, 1.9), c: (0.3, 0.9)})
fe = a * sp.sqrt(1 - xi ** 2 / b ** 2 - eta ** 2 / c ** 2) - a
sym_equal("연습 8.4.7: 높이 f의 헤세(원점) = diag(−a/b², −a/c²)", sp.hessian(fe, (xi, eta)).subs({xi: 0, eta: 0}), sp.diag(-a / b ** 2, -a / c ** 2))
rng = np.random.default_rng(5)
av, bv, cv = 3.0, 2.0, 1.0
th = rng.uniform(0, np.pi, 20000)
ph = rng.uniform(0, 2 * np.pi, 20000)
pts = np.stack([av * np.sin(th) * np.cos(ph), bv * np.sin(th) * np.sin(ph), cv * np.cos(th)], axis=1)
check("정리 8.4.9: 타원면(3,2,1)의 표본점은 모두 |q| ≤ 3, K(3,0,0) = 9/4 ≥ 1/9",
      np.max(np.linalg.norm(pts, axis=1)) <= 3.0 + 1e-12 and av ** 2 / (bv ** 2 * cv ** 2) >= 1 / av ** 2)

# 명제 8.4.11 --------------------------------------------------------------------------------------------------
for key in ("torus", "saddle", "sphere"):
    e = EX[key]
    p1, p2 = e["coords"]
    X = e["expr"]
    N = dgsym.unit_normal(X, p1, p2, e["positive"])
    K, _ = dgsym.K_H(X, p1, p2, e["positive"])
    sym_equal(f"(8.4.4): {e['name']} N_u × N_v = K x_u × x_v", N.diff(p1).cross(N.diff(p2)), K * X.diff(p1).cross(X.diff(p2)), e["domain"])
A2 = sp.Matrix(2, 2, sp.symbols("a11 a12 a21 a22", real=True))
w1, w2 = sp.Matrix(sp.symbols("p1:4", real=True)), sp.Matrix(sp.symbols("q1:4", real=True))
Aw1 = A2[0, 0] * w1 + A2[1, 0] * w2
Aw2 = A2[0, 1] * w1 + A2[1, 1] * w2
check("명제 8.4.11(a) 보조: Aw₁ × Aw₂ = det A (w₁ × w₂)", sp.expand(Aw1.cross(Aw2) - A2.det() * w1.cross(w2)) == sp.zeros(3, 1))
# 넓이 비율의 극한 (수치): 원환면 p = x(0.4, −π/2), p = x(2.6, 5π/6)
valsn = {R: 2.0, rr: 0.8}
Xn = tor["expr"].subs(valsn)
Nn = dgsym.unit_normal(tor["expr"], tu, tv, tor["positive"]).subs(valsn)
cx = sp.lambdify((tu, tv), list(Xn.diff(tu).cross(Xn.diff(tv))), "numpy")
cn = sp.lambdify((tu, tv), list(Nn.diff(tu).cross(Nn.diff(tv))), "numpy")
Kn = sp.lambdify((tu, tv), Kt.subs(valsn), "numpy")
ok = True
for (u0, v0) in ((0.4, -np.pi / 2), (2.6, 5 * np.pi / 6)):
    errs = []
    for eps in (0.2, 0.1, 0.05):
        g = np.linspace(-eps, eps, 81)
        gm = 0.5 * (g[1:] + g[:-1])
        U, V = np.meshgrid(u0 + gm, v0 + gm, indexing="ij")
        Ax = np.sum(np.linalg.norm(np.stack(cx(U, V), axis=-1), axis=-1))
        An = np.sum(np.linalg.norm(np.stack(cn(U, V), axis=-1), axis=-1))
        errs.append(abs(An / Ax - abs(Kn(u0, v0))))
    ok &= errs[0] > errs[1] > errs[2] and errs[2] < 1e-3
check("(8.4.5): 넓이 비율 → |K(p)| (ε = 0.2, 0.1, 0.05에서 오차 감소)", ok)
close("그림 8.4.2: K(p₁) ≈ 0.42, K(p₂) ≈ −0.82", [Kn(0.4, 0.0), Kn(2.6, 0.0)], [0.4206, -0.8154], tol=1e-3)

# 연습 8.4.1: 나선면 -----------------------------------------------------------------------------------------------
Xh = sp.Matrix([v * sp.cos(u), v * sp.sin(u), u])
Kh, Hh = dgsym.K_H(Xh, u, v)
sym_equal("연습 8.4.1: 나선면 K = −1/(1 + v²)², H = 0", sp.Matrix([Kh, Hh]), sp.Matrix([-1 / (1 + v ** 2) ** 2, 0]))

# 연습 8.4.3: 원환면을 호의 길이로 --------------------------------------------------------------------------------------
rho_s = R + rr * sp.cos(s / rr)
z_s = rr * sp.sin(s / rr)
check("연습 8.4.3: 생성곡선 단위속력", sp.simplify(sp.diff(rho_s, s) ** 2 + sp.diff(z_s, s) ** 2) == 1)
sym_equal("연습 8.4.3: −ρ''/ρ = K (u = s/r)", -sp.diff(rho_s, s, 2) / rho_s, Kt.subs(tu, s / rr), {s: (0, 5), R: (1.2, 2), rr: (0.2, 0.9)})

# 연습 8.4.5: 원환면 두 절반의 곡률 적분 ------------------------------------------------------------------------------------
integrand = sp.simplify(Kt * rr * (R + rr * sp.cos(tu)))
check("연습 8.4.5: K|x_u × x_v| = cos u", sp.simplify(integrand - sp.cos(tu)) == 0)
Iplus = 2 * sp.pi * sp.integrate(sp.cos(tu), (tu, -sp.pi / 2, sp.pi / 2))
Iminus = 2 * sp.pi * (sp.integrate(sp.cos(tu), (tu, -sp.pi, -sp.pi / 2)) + sp.integrate(sp.cos(tu), (tu, sp.pi / 2, sp.pi)))
check("연습 8.4.5: 바깥쪽 절반 4π, 안쪽 절반 −4π, 합 0", (Iplus, Iminus, Iplus + Iminus) == (4 * sp.pi, -4 * sp.pi, 0))
Nt = dgsym.unit_normal(tor["expr"], tu, tv, tor["positive"])
sym_equal("연습 8.4.5: −N_x = (cos u cos v, cos u sin v, sin u) (구면좌표)", -Nt, sp.Matrix([sp.cos(tu) * sp.cos(tv), sp.cos(tu) * sp.sin(tv), sp.sin(tu)]), tor["domain"])

summary()
