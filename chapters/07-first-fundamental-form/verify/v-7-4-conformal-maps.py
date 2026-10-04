"""7.4절 등각사상: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/verify/v-7-4-conformal-maps.py``
"""

import math

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
first_ff = dgsym.first_ff
u, v, t = sp.symbols("u v t", real=True)


def ratio_const(A, B, positive=()):
    """(E', F', G') = μ (E, F, G)인 μ를 돌려준다 (없으면 None)."""
    E, F, G = A
    E2, F2, G2 = B
    mu = sp.simplify(E2 / E)
    ok = sp.simplify(G2 - mu * G) == 0 and sp.simplify(F2 - mu * F) == 0
    return mu if ok else None


# 명제 7.4.4: 각을 보존하는 선형사상 ---------------------------------------------------------------
rng = np.random.default_rng(0)
# 닮음변환 (λ × 회전)은 각을 보존, 층밀림은 보존하지 않음
lam, al = 1.7, 0.6
T1 = lam * np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
T2 = np.array([[1.0, 0.5], [0.0, 1.0]])


def ang(a, b):
    return math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))


ok1 = ok2 = True
for _ in range(200):
    a, b = rng.normal(size=2), rng.normal(size=2)
    ok1 &= abs(ang(T1 @ a, T1 @ b) - ang(a, b)) < 1e-10
    ok2 &= abs(ang(T2 @ a, T2 @ b) - ang(a, b)) < 1e-10
check("명제 7.4.4: λ·회전은 모든 각을 보존한다 (수치)", ok1)
check("명제 7.4.4: 층밀림 [[1, 1/2], [0, 1]]은 각을 보존하지 않는다", not ok2)
e1, e2 = np.array([1.0, 0]), np.array([0, 1.0])
check("명제 7.4.4 증명: (e1 + e2) ⊥ (e1 − e2)이고, T가 각을 보존하면 |Te1| = |Te2|",
      abs((e1 + e2) @ (e1 - e2)) < 1e-15 and np.isclose(np.linalg.norm(T1 @ e1), np.linalg.norm(T1 @ e2)))

# 예 7.4.2: 닮음, 지수 사상 ---------------------------------------------------------------------
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
Is = first_ff(sph["expr"], th, ph, sph["positive"])
c = sp.symbols("c", positive=True)
check("예 7.4.2(b): 닮음 p ↦ cp의 계수 비 = c²", ratio_const(Is, first_ff(c * sph["expr"], th, ph, sph["positive"])) == c ** 2)
Yexp = sp.Matrix([sp.exp(u) * sp.cos(v), sp.exp(u) * sp.sin(v), 0])
check("예 7.4.2(c): 지수 사상 ψ∘ι의 계수 = e^{2u}·(1, 0, 1)", ratio_const((1, 0, 1), first_ff(Yexp, u, v)) == sp.exp(2 * u))

# 정리 7.4.6: 판정 — λ²∘x = μ ---------------------------------------------------------------------
# 예 7.4.8: 등온 매개화들
Ecat = first_ff(sp.Matrix([sp.cosh(u) * sp.cos(v), sp.cosh(u) * sp.sin(v), u]), u, v)
check("예 7.4.8(b): 현수면 매개화는 등온 (E = G = cosh²u, F = 0)", sp.simplify(Ecat[0] - Ecat[2]) == 0 and Ecat[1] == 0)
Epl = first_ff(sp.Matrix([u, v, 0]), u, v)
check("예 7.4.8(a): 평면 ι는 등온 (E = G = 1)", Epl == (1, 0, 1))
Epol = first_ff(sp.Matrix([sp.Symbol("rho", positive=True) * sp.cos(v), sp.Symbol("rho", positive=True) * sp.sin(v), 0]), sp.Symbol("rho", positive=True), v)
check("예 7.4.8(c): 극좌표 매개화는 등온이 아니다 (E = 1, G = ρ²)", sp.simplify(Epol[0] - Epol[2]) != 0)

# 예 7.4.9: 입체사영 ---------------------------------------------------------------------------
st = dgsym.stereographic(2)
X1, X2, X3 = st["coords"]
U1, U2 = st["chart_coords"]
inv = st["sigma_inv"]
Ei, Fi, Gi = first_ff(inv, U1, U2)
w = U1 ** 2 + U2 ** 2 + 1
check("예 7.4.9: σ⁻¹의 E = G = 4/(1 + u² + v²)², F = 0",
      sp.simplify(Ei - 4 / w ** 2) == 0 and Fi == 0 and sp.simplify(Gi - 4 / w ** 2) == 0)
check("예 7.4.9: σ⁻¹(u, v)는 단위구면 위", sp.simplify(inv.dot(inv) - 1) == 0)
check("예 7.4.9: σ∘σ⁻¹ = id", sp.simplify(st["sigma"].subs({X1: inv[0], X2: inv[1], X3: inv[2]}) - sp.Matrix([U1, U2])) == sp.zeros(2, 1))
zz = inv[2]
check("예 7.4.9: 1 − z = 2/(1 + u² + v²) (σ⁻¹(u, v)의 z)", sp.simplify(1 - zz - 2 / w) == 0)
check("예 7.4.9: σ의 배율 λ² = (1 + u² + v²)²/4 = 1/(1 − z)²", sp.simplify(w ** 2 / 4 - 1 / (1 - zz) ** 2) == 0)
check("예 7.4.9: 남극 (u, v) = (0, 0)에서 λ_{σ⁻¹} = 2", sp.sqrt(Ei.subs({U1: 0, U2: 0})) == 2)
# 6.1절 연습 6.1.5와의 일치: |σ⁻¹_u × σ⁻¹_v| = 4/w²
nn = inv.diff(U1).cross(inv.diff(U2))
check("예 7.4.9: |σ⁻¹_u × σ⁻¹_v| = 4/(1 + u² + v²)² (연습 6.1.5)", sp.simplify(nn.dot(nn) - 16 / w ** 4) == 0)

# 예 7.4.10: 메르카토르 ---------------------------------------------------------------------------
Ym = sp.Matrix([ph, sp.log(sp.cot(th / 2)), 0])
Im = first_ff(Ym, th, ph, sph["positive"])
mu = ratio_const(Is, Im, sph["positive"])
sym_equal("예 7.4.10: 메르카토르 ψ∘x의 계수 = (1/(r² sin²θ))·(E, F, G)", mu, 1 / (r ** 2 * sp.sin(th) ** 2), sph["domain"])
sym_equal("예 7.4.10: d/dθ log cot(θ/2) = −1/sin θ", sp.diff(sp.log(sp.cot(th / 2)), th), -1 / sp.sin(th), sph["domain"])
tt = sp.symbols("tt", real=True)
sym_equal("예 7.4.10: 역함수 θ = 2 arctan(e^{−t})", sp.log(sp.cot(sp.atan(sp.exp(-tt)))), tt, {tt: (-3, 3)})

# 비예 7.4.11: 아르키메데스의 투영은 등각이 아니다 ------------------------------------------------
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(cr,) = cyl["params"]
Ya = cyl["expr"].subs(cr, r).subs({cu: ph, cv: r * sp.cos(th)})
Ia = first_ff(Ya, th, ph, sph["positive"])
check("비예 7.4.11: 아르키메데스 투영의 계수는 (E, F, G)의 상수배가 아니다", ratio_const(Is, Ia, sph["positive"]) is None)
sym_equal("비예 7.4.11: Ē/E = sin²θ, Ḡ/G = 1/sin²θ", sp.Matrix([Ia[0] / Is[0], Ia[2] / Is[2]]), sp.Matrix([sp.sin(th) ** 2, 1 / sp.sin(th) ** 2]), sph["domain"])

# 연습 7.4.2: (s, φ) ↦ (e^s cos φ, e^s sin φ, 0) 은 등온 --------------------------------------------
check("연습 7.4.2: 로그 극좌표는 등온 매개화", ratio_const((1, 0, 1), first_ff(Yexp, u, v)) is not None)

# 연습 7.4.3: 넓이요소는 λ²배, 입체사영으로 본 남반구의 넓이 --------------------------------------
Eg3, Fg3, Gg3, mu3 = sp.symbols("E F G mu", positive=True)
check("연습 7.4.3: (Ē, F̄, Ḡ) = μ(E, F, G)이면 ĒḠ − F̄² = μ²(EG − F²)",
      sp.expand((mu3 * Eg3) * (mu3 * Gg3) - (mu3 * Fg3) ** 2 - mu3 ** 2 * (Eg3 * Gg3 - Fg3 ** 2)) == 0)
rho3 = sp.symbols("rho3", positive=True)
lam2 = (2 / (1 + rho3 ** 2)) ** 2  # σ⁻¹의 배율의 제곱, u² + v² = ρ²
A_south = sp.integrate(sp.integrate(lam2 * rho3, (rho3, 0, 1)), (tt, 0, 2 * sp.pi))
check("연습 7.4.3(b): ∬_{원판} λ² dA = 2π (닫힌 남반구의 넓이)", sp.simplify(A_south - 2 * sp.pi) == 0)
check("연습 7.4.3(b): 예 7.3.10(c)의 2πr²(cos θ₁ − cos θ₂) (r = 1, θ₁ = π/2, θ₂ = π)와 같다",
      sp.simplify(2 * sp.pi * (sp.cos(sp.pi / 2) - sp.cos(sp.pi)) - A_south) == 0)
# 같은 값을 σ⁻¹의 넓이요소 √(EG − F²)로 직접 (수치, 극좌표 격자)
inv_np = sp.lambdify((U1, U2), list(inv), "numpy")
rr3 = np.linspace(0.0, 1.0, 1201)
aa3 = np.linspace(0.0, 2 * np.pi, 1201)
RR3, AA3 = np.meshgrid(rr3, aa3, indexing="ij")
h3 = 1e-6
P0 = np.array(inv_np(RR3 * np.cos(AA3), RR3 * np.sin(AA3)), dtype=float)
Pu = (np.array(inv_np(RR3 * np.cos(AA3) + h3, RR3 * np.sin(AA3)), dtype=float) - np.array(inv_np(RR3 * np.cos(AA3) - h3, RR3 * np.sin(AA3)), dtype=float)) / (2 * h3)
Pv = (np.array(inv_np(RR3 * np.cos(AA3), RR3 * np.sin(AA3) + h3), dtype=float) - np.array(inv_np(RR3 * np.cos(AA3), RR3 * np.sin(AA3) - h3), dtype=float)) / (2 * h3)
dA3 = np.linalg.norm(np.cross(Pu, Pv, axis=0), axis=0) * RR3
close("연습 7.4.3(b): |σ⁻¹_u × σ⁻¹_v|를 원판에서 적분 = 2π (수치)", float(np.trapezoid(np.trapezoid(dA3, aa3, axis=1), rr3)), 2 * math.pi, tol=1e-5)
check("연습 7.4.3(b): 원판의 상은 z ≤ 0 (경계 원은 적도)", sp.simplify(inv[2].subs({U1: 1, U2: 0})) == 0 and float(inv[2].subs({U1: 0.3, U2: 0.4})) < 0)

# 연습 7.4.4: 배율의 합성 규칙 (수치 예: 닮음 두 개) --------------------------------------------------
check("연습 7.4.4: λ_{ψ∘φ} = (λ_ψ∘φ)λ_φ (닮음 c₁, c₂의 합성은 c₁c₂)", ratio_const(Is, first_ff(2 * 3 * sph["expr"], th, ph, sph["positive"])) == 36)

# 연습 7.4.5: 항정선의 길이 --------------------------------------------------------------------
be = sp.symbols("beta", positive=True)
# 메르카토르 직선 ds/dt = tan β, t = log cot(θ/2): dφ/dθ = tan β · dt/dθ = −tan β/sin θ
dphi = -sp.tan(be) / sp.sin(th)
speed2 = Is[0] * 1 + Is[2] * dphi ** 2
sym_equal("연습 7.4.5: |α'(θ)|² = r²/cos²ϑ₀ (상수)", speed2, r ** 2 / sp.cos(be) ** 2, {th: (0.2, 3.0), be: (0.2, 1.3), r: (0.5, 2)})
check("연습 7.4.5: 적도에서 북극까지 길이 = πr/(2 cos ϑ₀)", sp.simplify(sp.integrate(r / sp.cos(be), (th, 0, sp.pi / 2)) - sp.pi * r / (2 * sp.cos(be))) == 0)
# 각: 경선 방향 (1, 0)과 (1, dφ/dθ) 사이의 cos = √E/|α'| = cos β
sym_equal("연습 7.4.5: 경선과 이루는 각의 cos = cos ϑ₀", sp.sqrt(Is[0]) / sp.sqrt(speed2), sp.cos(be), {th: (0.2, 3.0), be: (0.2, 1.3), r: (0.5, 2)})

# 연습 7.4.6: 입체사영은 원을 원으로 --------------------------------------------------------------
a, b, cc, d = sp.symbols("a b c d", real=True)
plane_eq = a * inv[0] + b * inv[1] + cc * inv[2] - d
circ = sp.simplify(plane_eq * w)
check("연습 7.4.6: (ax + by + cz − d)(1 + u² + v²) = (c − d)(u² + v²) + 2au + 2bv − (c + d)",
      sp.expand(circ - ((cc - d) * (U1 ** 2 + U2 ** 2) + 2 * a * U1 + 2 * b * U2 - (cc + d))) == 0)

# 연습 7.4.7: 코시-리만 --------------------------------------------------------------------------
f = sp.Function("f")(u, v)
g = sp.Function("g")(u, v)
Yfg = sp.Matrix([f, g, 0])
Efg, Ffg, Gfg = first_ff(Yfg, u, v)
fu, fv, gu, gv = (sp.diff(f, u), sp.diff(f, v), sp.diff(g, u), sp.diff(g, v))
cr_sub = {sp.diff(g, v): sp.diff(f, u), sp.diff(g, u): -sp.diff(f, v)}
check("연습 7.4.7: 코시-리만이면 E = G, F = 0", sp.simplify((Efg - Gfg).subs(cr_sub)) == 0 and sp.simplify(Ffg.subs(cr_sub)) == 0)
zc = sp.symbols("zc")
fz = (U1 + sp.I * U2) ** 2
check("연습 7.4.7 (예): z² = (u² − v², 2uv)는 코시-리만을 만족한다",
      sp.simplify(sp.diff(sp.re(sp.expand(fz)), U1) - sp.diff(sp.im(sp.expand(fz)), U2)) == 0)

summary()
