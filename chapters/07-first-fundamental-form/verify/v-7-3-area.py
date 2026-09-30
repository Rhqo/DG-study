"""7.3절 넓이: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/verify/v-7-3-area.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
first_ff = dgsym.first_ff
u, v, s, t = sp.symbols("u v s t", real=True)

# 보조정리 7.3.1: y = x∘h이면 y_s × y_t = det(Dh) (x_u × x_v)∘h — 일반 기호 함수로 ------------------
f1, f2, f3 = (sp.Function(n)(u, v) for n in ("f1", "f2", "f3"))
Xg = sp.Matrix([f1, f2, f3])
h1, h2 = sp.Function("h1")(s, t), sp.Function("h2")(s, t)
Yg = Xg.subs({u: h1, v: h2})
lhs = Yg.diff(s).cross(Yg.diff(t))
Dh = sp.Matrix([h1, h2]).jacobian([s, t])
rhs = Dh.det() * (Xg.diff(u).cross(Xg.diff(v))).subs({u: h1, v: h2})
check("보조정리 7.3.1: y_s × y_t = det(Dh)·(x_u × x_v)∘h (일반)", sp.simplify((lhs - rhs).doit()) == sp.zeros(3, 1))

# 정의 7.3.2: |x_u × x_v| = √(EG − F²) ----------------------------------------------------------
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
Xs = sph["expr"]
Es, Fs, Gs = first_ff(Xs, th, ph, sph["positive"])
ns = Xs.diff(th).cross(Xs.diff(ph))
sym_equal("(7.3.2): 구면에서 |x_θ × x_φ| = √(EG − F²) = r² sin θ", sp.sqrt(ns.dot(ns)), r ** 2 * sp.sin(th), sph["domain"])

# 비예 7.3.4: 두 번 세기 ----------------------------------------------------------------------
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(cr,) = cyl["params"]
Xc = cyl["expr"]
nc = Xc.diff(cu).cross(Xc.diff(cv))
check("비예 7.3.4: 원기둥 |x_u × x_v| = r", sp.simplify(sp.sqrt(nc.dot(nc)) - cr) == 0)
AR = sp.integrate(sp.integrate(cr, (cu, sp.pi / 2, 3 * sp.pi / 2)), (cv, 0, 1))
check("비예 7.3.4: A(x([π/2, 3π/2] × [0, 1])) = πr", sp.simplify(AR - sp.pi * cr) == 0)
Adouble = AR + sp.integrate(sp.integrate(cr, (cu, 5 * sp.pi / 2, 7 * sp.pi / 2)), (cv, 0, 1))
check("비예 7.3.4: 두 직사각형 위의 적분 = 2πr = 2A(R)", sp.simplify(Adouble - 2 * sp.pi * cr) == 0)
check("비예 7.3.4: z(u + 2π, v) = z(u, v) (단사가 아님)", sp.simplify(Xc.subs(cu, cu + 2 * sp.pi) - Xc) == sp.zeros(3, 1))

# 예 7.3.5: 구면 조각의 넓이 --------------------------------------------------------------------
t1, t2, p1, p2 = sp.symbols("theta1 theta2 phi1 phi2", positive=True)
Apiece = sp.integrate(sp.integrate(r ** 2 * sp.sin(th), (th, t1, t2)), (ph, p1, p2))
check("예 7.3.5: A = r²(φ₂ − φ₁)(cos θ₁ − cos θ₂)", sp.simplify(Apiece - r ** 2 * (p2 - p1) * (sp.cos(t1) - sp.cos(t2))) == 0)

# 명제 7.3.7: 내부 근사 (구면, Q_n = [1/n, π − 1/n] × [1/n, 2π − 1/n]) ----------------------------
n = sp.symbols("n", positive=True)
An = sp.integrate(sp.integrate(r ** 2 * sp.sin(th), (th, 1 / n, sp.pi - 1 / n)), (ph, 1 / n, 2 * sp.pi - 1 / n))
check("명제 7.3.7 (예): A(x(Q_n)) → 4πr²", sp.simplify(sp.limit(An, n, sp.oo) - 4 * sp.pi * r ** 2) == 0)
# 오차 상한 M·vol(Q \ Q_n): M = r²
volQ = sp.pi * 2 * sp.pi
volQn = (sp.pi - 2 / n) * (2 * sp.pi - 2 / n)
for nv in (3, 10, 100):
    err = float((4 * sp.pi * r ** 2 - An).subs({r: 1, n: nv}))
    bound = float((volQ - volQn).subs(n, nv))
    check(f"명제 7.3.7: n = {nv}에서 오차 {err:.4f} ≤ M·vol(Q∖Q_n) = {bound:.4f} (r = 1)", 0 <= err <= bound)

# 예 7.3.9: 구면, 원환면 전체의 넓이, 구면 띠 ------------------------------------------------------
Asph = sp.integrate(sp.integrate(r ** 2 * sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi))
check("예 7.3.9(a): A(S²(r)) = 4πr²", sp.simplify(Asph - 4 * sp.pi * r ** 2) == 0)
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
Et, Ft, Gt = first_ff(tor["expr"], tu, tv, tor["positive"])
dAt = sp.simplify(sp.sqrt(Et * Gt - Ft ** 2))
sym_equal("예 7.3.9(b): 원환면 √(EG − F²) = r(R + r cos u)", dAt, rr * (R + rr * sp.cos(tu)), tor["domain"])
Ator = sp.integrate(sp.integrate(rr * (R + rr * sp.cos(tu)), (tu, 0, 2 * sp.pi)), (tv, 0, 2 * sp.pi))
check("예 7.3.9(b): A(원환면) = 4π²Rr", sp.simplify(Ator - 4 * sp.pi ** 2 * R * rr) == 0)
Vtor = 2 * sp.pi ** 2 * R * rr ** 2         # 연습 2.6.6
check("예 7.3.9(b): d/dr(2π²Rr²) = 4π²Rr (연습 2.6.6과 비교)", sp.simplify(sp.diff(Vtor, rr) - Ator) == 0)
check("예 7.3.9(a): d/dr(4πr³/3) = 4πr²", sp.simplify(sp.diff(sp.Rational(4, 3) * sp.pi * r ** 3, r) - Asph) == 0)
Azone = sp.integrate(sp.integrate(r ** 2 * sp.sin(th), (th, t1, t2)), (ph, 0, 2 * sp.pi))
check("예 7.3.9(c): 띠의 넓이 = 2πr·h, h = r(cos θ₁ − cos θ₂)", sp.simplify(Azone - 2 * sp.pi * r * r * (sp.cos(t1) - sp.cos(t2))) == 0)
# 빠진 부분 C는 정칙곡선의 자취
gam = sp.Matrix([r * sp.sin(t), 0, r * sp.cos(t)])
check("예 7.3.9(a): C를 그리는 γ(t) = (r sin t, 0, r cos t)는 정칙 (|γ'| = r)", sp.simplify(gam.diff(t).dot(gam.diff(t)) - r ** 2) == 0)

# 명제 7.3.10: 등거리사상은 넓이를 보존 — 원기둥 조각과 평면 직사각형 ----------------------------
a, b, c, d = sp.symbols("a b c d", positive=True)
Acyl = sp.integrate(sp.integrate(cr, (cu, a, b)), (cv, c, d))
Aplane = sp.integrate(sp.integrate(1, (u, cr * a, cr * b)), (v, c, d))
check("예 (명제 7.3.10): 원기둥 조각 x([a, b] × [c, d])와 평면 [ra, rb] × [c, d]의 넓이가 같다", sp.simplify(Acyl - Aplane) == 0)

# 예 7.3.11: 아르키메데스의 투영 ----------------------------------------------------------------
Ys = Xc.subs({cu: ph, cv: r * sp.cos(th)})
Ey, Fy, Gy = first_ff(Ys, th, ph, sph["positive"])
check("예 7.3.11: ψ∘x = (r cos φ, r sin φ, r cos θ)의 계수 (r² sin²θ, 0, r²)",
      (sp.simplify(Ey - r ** 2 * sp.sin(th) ** 2), Fy, sp.simplify(Gy - r ** 2)) == (0, 0, 0))
sym_equal("예 7.3.11: √(ĒḠ − F̄²) = √(EG − F²) = r² sin θ (넓이 보존)", sp.sqrt(Ey * Gy - Fy ** 2), sp.sqrt(Es * Gs - Fs ** 2), sph["domain"])
check("예 7.3.11: Ē ≠ E (등거리가 아님)", sp.simplify(Ey - Es) != 0)
check("예 7.3.11: ψ∘x는 원기둥 매개화 x_cyl(φ, r cos θ)", sp.simplify(Ys - Xc.subs({cu: ph, cv: r * sp.cos(th)})) == sp.zeros(3, 1))
close("그림 7.3.1: 칸의 넓이 ≈ 0.213, 평행사변형 ≈ 0.177 (θ₀ = π/4, Δ = 0.5)",
      (round(0.5 * (np.cos(np.pi / 4) - np.cos(np.pi / 4 + 0.5)), 3), round(np.sin(np.pi / 4) * 0.25, 3)), (0.213, 0.177), tol=1e-12)
close("그림 7.3.2: 조각 넓이 ≈ 0.405 (r = 1, θ ∈ [0.5, 1], φ ∈ [0.2, 1.4])", round(1.2 * (np.cos(0.5) - np.cos(1.0)), 3), 0.405, tol=1e-12)

# 연습 7.3.1: 그래프 곡면, 포물면 ------------------------------------------------------------
fg = sp.Function("f")(u, v)
Eg, Fg, Gg = first_ff(sp.Matrix([u, v, fg]), u, v)
check("연습 7.3.1: 그래프 곡면 √(EG − F²) = √(1 + f_u² + f_v²)",
      sp.simplify(Eg * Gg - Fg ** 2 - (1 + sp.diff(fg, u) ** 2 + sp.diff(fg, v) ** 2)) == 0)
rho, al = sp.symbols("rho a", positive=True)
Apar = 2 * sp.pi * sp.integrate(sp.sqrt(1 + 4 * rho ** 2) * rho, (rho, 0, al))
check("연습 7.3.1: 포물면 조각의 넓이 = (π/6)((1 + 4a²)^{3/2} − 1)", sp.simplify(Apar - sp.pi / 6 * ((1 + 4 * al ** 2) ** sp.Rational(3, 2) - 1)) == 0)

# 연습 7.3.2: 원기둥 띠 ------------------------------------------------------------------------
hh = sp.symbols("h", positive=True)
check("연습 7.3.2: 원기둥 띠 {0 ≤ z ≤ h}의 넓이 = 2πrh", sp.integrate(sp.integrate(cr, (cu, 0, 2 * sp.pi)), (cv, 0, hh)) == 2 * sp.pi * cr * hh)
check("연습 7.3.2: 평면 극좌표 넓이요소 = ρ (예 2.6.10)", sp.simplify(sp.sqrt(1 * rho ** 2 - 0)) == rho)

# 연습 7.3.4: 회전면의 넓이 (파푸스) --------------------------------------------------------------
rev = EX["revolution"]
ru, rv = rev["coords"]
rf, zf = rev["functions"]
Er, Fr, Gr = first_ff(rev["expr"], ru, rv)
check("연습 7.3.4: 회전면 √(EG − F²) = ρ|α'|",
      sp.simplify((Er * Gr - Fr ** 2) - rf(ru) ** 2 * (sp.diff(rf(ru), ru) ** 2 + sp.diff(zf(ru), ru) ** 2)) == 0)
Acat = 2 * sp.pi * sp.integrate(sp.cosh(u) * sp.cosh(u), (u, -1, 1))
check("연습 7.3.4: 현수면 조각 |z| ≤ 1의 넓이 = π(2 + sinh 2)", sp.simplify(Acat - sp.pi * (2 + sp.sinh(2))) == 0)

# 연습 7.3.5: A(R_ε)/ε² → √(EG − F²)(q) ----------------------------------------------------------
eps = sp.symbols("epsilon", positive=True)
q0 = sp.pi / 3
Aeps = sp.integrate(sp.integrate(r ** 2 * sp.sin(th), (th, q0, q0 + eps)), (ph, 1, 1 + eps))
check("연습 7.3.5 (예): 구면에서 A(R_ε)/ε² → r² sin(π/3)", sp.simplify(sp.limit(Aeps / eps ** 2, eps, 0) - r ** 2 * sp.sin(q0)) == 0)

# 연습 7.3.6: 구면 극관의 넓이를 두 매개화로 --------------------------------------------------------
th0 = sp.symbols("theta0", positive=True)
Acap1 = sp.integrate(sp.integrate(r ** 2 * sp.sin(th), (th, 0, th0)), (ph, 0, 2 * sp.pi))
# 위쪽 반구 그래프 매개화: √(1 + f_u² + f_v²) = r/√(r² − ρ²), 극좌표로 (ρ ≤ r sin θ₀)
Acap2 = 2 * sp.pi * sp.integrate(r / sp.sqrt(r ** 2 - rho ** 2) * rho, (rho, 0, r * sp.sin(th0)))
check("연습 7.3.6: 극관 {θ ≤ θ₀}의 넓이 (기준 매개화) = 2πr²(1 − cos θ₀)", sp.simplify(Acap1 - 2 * sp.pi * r ** 2 * (1 - sp.cos(th0))) == 0)
sym_equal("연습 7.3.6: 그래프 매개화로 계산해도 같다 (0 < θ₀ < π/2)", Acap2, Acap1, {th0: (0.1, 1.5), r: (0.5, 2)})
fu, fv = sp.symbols("fu fv", real=True)
uu, vv = sp.symbols("uu vv", real=True)
fh = sp.sqrt(r ** 2 - uu ** 2 - vv ** 2)
sym_equal("연습 7.3.6: 반구 그래프에서 1 + f_u² + f_v² = r²/(r² − u² − v²)", 1 + sp.diff(fh, uu) ** 2 + sp.diff(fh, vv) ** 2,
          r ** 2 / (r ** 2 - uu ** 2 - vv ** 2), {uu: (0.1, 0.4), vv: (0.1, 0.4), r: (1, 2)})

summary()
