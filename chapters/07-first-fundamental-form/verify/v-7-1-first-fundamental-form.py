"""7.1절 제1기본형식: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/verify/v-7-1-first-fundamental-form.py``
"""

import math

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
first_ff = dgsym.first_ff


def gram(E, F, G):
    return sp.Matrix([[E, F], [F, G]])


# 정의 7.1.1 뒤: EG − F² = |x_u × x_v|² (정리 1.6.16(d)) — 일반 매개화로 --------------------------
u, v = sp.symbols("u v", real=True)
f1, f2, f3 = (sp.Function(n)(u, v) for n in ("f1", "f2", "f3"))
Xg = sp.Matrix([f1, f2, f3])
Eg, Fg, Gg = (Xg.diff(u).dot(Xg.diff(u)), Xg.diff(u).dot(Xg.diff(v)), Xg.diff(v).dot(Xg.diff(v)))
ng = Xg.diff(u).cross(Xg.diff(v))
check("(7.1.3): 일반 매개화에서 EG − F² = |x_u × x_v|²", sp.expand(Eg * Gg - Fg ** 2 - ng.dot(ng)) == 0)

# 예 7.1.2: 구면 -----------------------------------------------------------------------------
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
Xs = sph["expr"]
Es, Fs, Gs = first_ff(Xs, th, ph, sph["positive"])
for nm, val in zip("EFG", (Es, Fs, Gs)):
    sym_equal(f"예 7.1.2: 구면 {nm}", val, sph["expected"][nm], sph["domain"])
ns = Xs.diff(th).cross(Xs.diff(ph))
sym_equal("예 7.1.2: EG − F² = |x_θ × x_φ|² = r⁴ sin²θ", Es * Gs - Fs ** 2, r ** 4 * sp.sin(th) ** 2, sph["domain"])
sym_equal("예 7.1.2: |x_θ × x_φ|² = r⁴ sin²θ (식 (6.1.5))", ns.dot(ns), r ** 4 * sp.sin(th) ** 2, sph["domain"])
a, b = sp.symbols("a b", real=True)
w = a * Xs.diff(th) + b * Xs.diff(ph)
sym_equal("예 7.1.2: I(a x_θ + b x_φ) = r²a² + r² sin²θ b²", w.dot(w), r ** 2 * a ** 2 + r ** 2 * sp.sin(th) ** 2 * b ** 2,
          sph["domain"])

# 비예 7.1.3: (u³, v, 0) ---------------------------------------------------------------------
Xbad = sp.Matrix([u ** 3, v, 0])
Eb, Fb, Gb = first_ff(Xbad, u, v)
check("비예 7.1.3: E = 9u⁴, F = 0, G = 1", (sp.simplify(Eb - 9 * u ** 4), Fb, Gb) == (0, 0, 1))
check("비예 7.1.3: u = 0에서 EG − F² = 0", sp.simplify((Eb * Gb - Fb ** 2).subs(u, 0)) == 0)
check("비예 7.1.3: u = 0에서 좌표 방향 (1, 0)의 길이의 제곱 E·1² = 0", Eb.subs(u, 0) == 0)

# 예 7.1.4: 원기둥 ---------------------------------------------------------------------------
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(cr,) = cyl["params"]
Ec, Fc, Gc = first_ff(cyl["expr"], cu, cv)
for nm, val in zip("EFG", (Ec, Fc, Gc)):
    sym_equal(f"예 7.1.4: 원기둥 {nm}", val, cyl["expected"][nm], cyl["domain"])

# 예 7.1.5: 원환면 ---------------------------------------------------------------------------
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
Xt = tor["expr"]
Et, Ft, Gt = first_ff(Xt, tu, tv, tor["positive"])
for nm, val in zip("EFG", (Et, Ft, Gt)):
    sym_equal(f"예 7.1.5: 원환면 {nm}", val, tor["expected"][nm], tor["domain"])
nt = Xt.diff(tu).cross(Xt.diff(tv))
sym_equal("예 7.1.5: EG − F² = r²(R + r cos u)² = |x_u × x_v|² (식 (6.5.4))", nt.dot(nt), Et * Gt - Ft ** 2, tor["domain"])

# 예 7.1.6: 회전면, 구면·원기둥·원환면·현수면이 특수한 경우 -------------------------------------
rev = EX["revolution"]
ru, rv = rev["coords"]
rho, zf = rev["functions"]
Er, Fr, Gr = first_ff(rev["expr"], ru, rv)
for nm, val in zip("EFG", (Er, Fr, Gr)):
    sym_equal(f"예 7.1.6: 회전면 {nm}", val, rev["expected"][nm])
nr = rev["expr"].diff(ru).cross(rev["expr"].diff(rv))
check("예 7.1.6: 회전면 EG − F² = |x_u × x_v|² = ρ²(ρ'² + z'²) (식 (6.5.6))",
      sp.simplify(nr.dot(nr) - rho(ru) ** 2 * (sp.diff(rho(ru), ru) ** 2 + sp.diff(zf(ru), ru) ** 2)) == 0)


def rev_EG(rho_e, z_e, var):
    return sp.simplify(sp.diff(rho_e, var) ** 2 + sp.diff(z_e, var) ** 2), sp.simplify(rho_e ** 2)


Esp, Gsp = rev_EG(r * sp.sin(th), r * sp.cos(th), th)
check("예 7.1.6(a): 구면 ρ = r sin θ, z = r cos θ ⇒ E = r², G = r² sin²θ", (Esp, sp.simplify(Gsp - r ** 2 * sp.sin(th) ** 2)) == (r ** 2, 0))
Ecy, Gcy = rev_EG(cr + 0 * cu, cu, cu)
check("예 7.1.6(b): 원기둥을 회전면으로 (ρ, z) = (r, u) ⇒ E = 1, G = r² (변수 이름이 바뀐 것)", (Ecy, Gcy) == (1, cr ** 2))
Eto, Gto = rev_EG(R + rr * sp.cos(tu), rr * sp.sin(tu), tu)
check("예 7.1.6(c): 원환면 ρ = R + r cos u, z = r sin u ⇒ E = r², G = (R + r cos u)²",
      (sp.simplify(Eto - rr ** 2), sp.simplify(Gto - (R + rr * sp.cos(tu)) ** 2)) == (0, 0))
Eca, Gca = rev_EG(sp.cosh(u), u, u)
check("예 7.1.6(d): 현수면 ρ = cosh u, z = u ⇒ E = G = cosh²u", sp.simplify(Eca - sp.cosh(u) ** 2) == 0 and sp.simplify(Gca - sp.cosh(u) ** 2) == 0)
Xcat = sp.Matrix([sp.cosh(u) * sp.cos(v), sp.cosh(u) * sp.sin(v), u])
check("예 7.1.6(d): 현수면 매개화로 직접 계산해도 (cosh²u, 0, cosh²u)",
      all(sp.simplify(x - y) == 0 for x, y in zip(first_ff(Xcat, u, v), (sp.cosh(u) ** 2, 0, sp.cosh(u) ** 2))))

# 명제 7.1.7: 계수의 변환 Ī = Dhᵀ I Dh ---------------------------------------------------------
s, t = sp.symbols("s t", real=True)
h = sp.Matrix([s, s + t])            # 연습 7.1.3의 매개변수 변환
Dh = h.jacobian([s, t])
Ys = Xs.subs({th: h[0], ph: h[1]})
Ey, Fy, Gy = first_ff(Ys, s, t, (sp.sin(s),))
Imat = gram(Es, Fs, Gs).subs({th: h[0], ph: h[1]})
sym_equal("명제 7.1.7 (구면, h(s, t) = (s, s + t)): (Ē F̄; F̄ Ḡ) = Dhᵀ (E F; F G) Dh",
          gram(Ey, Fy, Gy), Dh.T * Imat * Dh, {s: (0.2, 2.9), t: (-1, 1), r: (0.5, 2)})
# 일반 매개화와 일반 매개변수 변환(기호 함수)으로도
h1, h2 = sp.Function("h1")(s, t), sp.Function("h2")(s, t)
Xgh = Xg.subs({u: h1, v: h2})
lhs = gram(*[Xgh.diff(p).dot(Xgh.diff(q)) for p, q in ((s, s), (s, t), (t, t))])
Dhg = sp.Matrix([h1, h2]).jacobian([s, t])
rhs = Dhg.T * gram(Eg, Fg, Gg).subs({u: h1, v: h2}) * Dhg
check("명제 7.1.7 (일반): y = x∘h이면 Ī = Dhᵀ I(h) Dh (기호 함수로)",
      sp.simplify(sp.expand((lhs - rhs).doit())) == sp.zeros(2, 2))

# 예 7.1.8: 평면의 두 매개화 --------------------------------------------------------------------
rho_, phi_ = sp.symbols("rho phi", positive=True)
Xpl = sp.Matrix([u, v, 0])
check("예 7.1.8: 평면 ι(u, v) = (u, v, 0)의 E, F, G = 1, 0, 1", first_ff(Xpl, u, v) == (1, 0, 1))
Xpol = sp.Matrix([rho_ * sp.cos(phi_), rho_ * sp.sin(phi_), 0])
check("예 7.1.8: 극좌표 매개화의 E, F, G = 1, 0, ρ²", tuple(sp.simplify(c) for c in first_ff(Xpol, rho_, phi_)) == (1, 0, rho_ ** 2))
hp = sp.Matrix([rho_ * sp.cos(phi_), rho_ * sp.sin(phi_)])
Dhp = hp.jacobian([rho_, phi_])
check("예 7.1.8: Dhᵀ I Dh = diag(1, ρ²) (명제 7.1.7)", sp.simplify(Dhp.T * Dhp - sp.diag(1, rho_ ** 2)) == sp.zeros(2, 2))

# 명제 7.1.9 / 예 7.1.10: 곡선의 길이 ------------------------------------------------------------
th0 = sp.symbols("theta0", positive=True)
speed_lat = sp.sqrt(Gs.subs(th, th0) * 1)
sym_equal("예 7.1.10(a): 위도원 θ = θ₀의 속력 = r sin θ₀", speed_lat, r * sp.sin(th0), {th0: (0.1, 3.0), r: (0.5, 2)})
check("예 7.1.10(a): 위도원 길이 2πr sin θ₀", sp.simplify(sp.integrate(r * sp.sin(th0), (t, 0, 2 * sp.pi)) - 2 * sp.pi * r * sp.sin(th0)) == 0)
check("예 7.1.10(a): 경선 (θ = t, 0 < t < π)의 길이 = πr", sp.integrate(sp.sqrt(Es), (t, 0, sp.pi)) == sp.pi * r)
bb = sp.symbols("b", real=True, nonzero=True)
# 원기둥 위 곡선 β(t) = (t, bt): α = x∘β는 나선 (r cos t, r sin t, bt)
alpha_cyl = cyl["expr"].subs({cu: t, cv: bb * t})
hel = EX["helix"]["expr"].subs({sp.Symbol("a", positive=True): cr, sp.Symbol("b", real=True, nonzero=True): bb, EX["helix"]["coords"][0]: t})
check("예 7.1.10(b): x(t, bt)는 나선 (a = r)", sp.simplify(alpha_cyl - hel) == sp.zeros(3, 1))
sp2 = Ec * 1 ** 2 + 2 * Fc * 1 * bb + Gc * bb ** 2
sym_equal("예 7.1.10(b): E u'² + 2F u'v' + G v'² = r² + b²", sp2, cr ** 2 + bb ** 2)
sym_equal("예 7.1.10(b): 직접 계산한 |α'|² = r² + b²", alpha_cyl.diff(t).dot(alpha_cyl.diff(t)), cr ** 2 + bb ** 2)
Lhel = sp.integrate(sp.sqrt(cr ** 2 + bb ** 2), (t, 0, 2 * sp.pi))
sym_equal("예 7.1.10(b): 한 바퀴 길이 2π√(r² + b²) (식 (4.2.6))", Lhel, 2 * sp.pi * sp.sqrt(cr ** 2 + bb ** 2))
# 원환면 위 곡선 x(t, t)의 길이는 수치로 (본문에서는 적분식만)
RV, rV = 2.0, 0.8
Lnum = np.trapezoid(np.sqrt(rV ** 2 + (RV + rV * np.cos(np.linspace(0, 2 * np.pi, 200001))) ** 2), np.linspace(0, 2 * np.pi, 200001))
alpha_t = sp.lambdify(t, list(Xt.subs({tu: t, tv: t, R: RV, rr: rV}).diff(t)), "numpy")
ts = np.linspace(0, 2 * np.pi, 200001)
Ldir = np.trapezoid(np.linalg.norm(np.array(alpha_t(ts), float), axis=0), ts)
close("예 7.1.12: 원환면 곡선 x(t, t)의 길이, (7.1.5)와 직접 계산이 같다 (R = 2, r = 0.8)", Lnum, Ldir, tol=1e-8)
close("예 7.1.12: 그 길이 ≈ 13.60", round(Lnum, 2), 13.60, tol=1e-12)

# 정의 7.1.11, 식 (7.1.6)–(7.1.7): 각 -----------------------------------------------------------
a1, b1, a2, b2 = sp.symbols("a1 b1 a2 b2", real=True)
w1 = a1 * Xg.diff(u) + b1 * Xg.diff(v)
w2 = a2 * Xg.diff(u) + b2 * Xg.diff(v)
check("(7.1.6): <w1, w2> = E a1a2 + F(a1b2 + a2b1) + G b1b2",
      sp.expand(w1.dot(w2) - (Eg * a1 * a2 + Fg * (a1 * b2 + a2 * b1) + Gg * b1 * b2)) == 0)
# 예 7.1.12: 원환면 대각선과 u-좌표곡선의 각
cos_formula = rr / sp.sqrt(rr ** 2 + (R + rr * sp.cos(tu)) ** 2)
cos_direct = (Xt.diff(tu).dot(Xt.diff(tu) + Xt.diff(tv))) / sp.sqrt(Et) / sp.sqrt(Xt.diff(tu).dot(Xt.diff(tu)) + 2 * Xt.diff(tu).dot(Xt.diff(tv)) + Xt.diff(tv).dot(Xt.diff(tv)))
sym_equal("예 7.1.12: cos ϑ = r/√(r² + (R + r cos u)²)", cos_direct, cos_formula, tor["domain"])
for tval, deg in ((sp.pi / 4, 72.68), (5 * sp.pi / 6, 58.53), (0, 74.05), (sp.pi, 56.31)):
    ang = math.degrees(math.acos(float(cos_formula.subs({R: 2, rr: sp.Rational(4, 5), tu: tval}))))
    close(f"예 7.1.12: u = {tval}에서 ϑ ≈ {deg}° (R = 2, r = 0.8)", round(ang, 2), deg, tol=1e-9)
check("예 7.1.12: U에서의 각은 π/4 (cos = 1/√2)", sp.simplify(sp.Matrix([1, 0]).dot(sp.Matrix([1, 1])) / sp.sqrt(2) - 1 / sp.sqrt(2)) == 0)
# 좌표곡선 사이의 각 cos = F/√(EG) (식 (7.1.7))
sym_equal("(7.1.7): cos(좌표곡선의 각) = F/√(EG) (원환면에서 확인: F = 0이므로 직교)", Xt.diff(tu).dot(Xt.diff(tv)) / sp.sqrt(Et * Gt), 0, tor["domain"])
sym_equal("(7.1.7): 연습 7.1.3의 매개화에서 <y_s, y_t>/(|y_s||y_t|) = F/√(EG)", Ys.diff(s).dot(Ys.diff(t)) / sp.sqrt(Ys.diff(s).dot(Ys.diff(s))) / sp.sqrt(Ys.diff(t).dot(Ys.diff(t))), Fy / sp.sqrt(Ey * Gy), {s: (0.1, 3.0), t: (-1, 1), r: (0.5, 2)})

# 연습 7.1.1: 그래프 곡면 ------------------------------------------------------------------------
fg = sp.Function("f")(u, v)
Egr, Fgr, Ggr = first_ff(sp.Matrix([u, v, fg]), u, v)
fu, fv = sp.diff(fg, u), sp.diff(fg, v)
check("연습 7.1.1: 그래프 곡면 E = 1 + f_u², F = f_u f_v, G = 1 + f_v²",
      (sp.simplify(Egr - 1 - fu ** 2), sp.simplify(Fgr - fu * fv), sp.simplify(Ggr - 1 - fv ** 2)) == (0, 0, 0))
check("연습 7.1.1: EG − F² = 1 + f_u² + f_v²", sp.simplify(Egr * Ggr - Fgr ** 2 - (1 + fu ** 2 + fv ** 2)) == 0)
sad = EX["saddle"]["expr"]
Esd = [c.subs({u: 1, v: 1}) for c in first_ff(sad, u, v)]
check("연습 7.1.1: 안장면의 (1, 1)에서 E, F, G = 5, −4, 5", Esd == [5, -4, 5])
check("연습 7.1.1: 안장면의 (1, 1)에서 좌표곡선의 cos = −4/5", sp.Rational(-4, 5) == Esd[1] / sp.sqrt(Esd[0] * Esd[2]))

# 연습 7.1.2: 평면 극좌표의 곡선 ρ = e^{−t}, φ = t ---------------------------------------------
sp_sq = (sp.diff(sp.exp(-t), t)) ** 2 + sp.exp(-2 * t) * 1
check("연습 7.1.2: 속력의 제곱 = 2e^{−2t}", sp.simplify(sp_sq - 2 * sp.exp(-2 * t)) == 0)
Lsp = sp.integrate(sp.sqrt(2) * sp.exp(-t), (t, 0, 2 * sp.pi))
check("연습 7.1.2: 길이 = √2(1 − e^{−2π})", sp.simplify(Lsp - sp.sqrt(2) * (1 - sp.exp(-2 * sp.pi))) == 0)

# 연습 7.1.3: 구면의 다른 매개화 y(s, t) = x(s, s + t) ----------------------------------------
check("연습 7.1.3: E = r²(1 + sin²s), F = G = r² sin²s",
      (sp.simplify(Ey - r ** 2 * (1 + sp.sin(s) ** 2)), sp.simplify(Fy - r ** 2 * sp.sin(s) ** 2), sp.simplify(Gy - r ** 2 * sp.sin(s) ** 2)) == (0, 0, 0))
sym_equal("연습 7.1.3: 좌표곡선의 cos = sin s/√(1 + sin²s)", Fy / sp.sqrt(Ey * Gy), sp.sin(s) / sp.sqrt(1 + sp.sin(s) ** 2),
          {s: (0.1, 3.0), r: (0.5, 2)})

# 연습 7.1.5: 경선이 가장 짧다 (수치로 무작위 곡선 확인) ------------------------------------------
rng = np.random.default_rng(7)
ok = True
for _ in range(200):
    t1, t2 = sorted(rng.uniform(0.2, 2.9, 2))
    ts = np.linspace(0, 1, 2001)
    thc = t1 + (t2 - t1) * ts + 0.3 * np.sin(np.pi * ts) * rng.normal()
    phc = 1.0 + 0.8 * np.sin(np.pi * ts * rng.integers(1, 4)) * rng.normal()
    thc = np.clip(thc, 0.05, np.pi - 0.05)
    L = np.trapezoid(np.sqrt(np.gradient(thc, ts) ** 2 + np.sin(thc) ** 2 * np.gradient(phc, ts) ** 2), ts)
    ok &= L >= abs(thc[-1] - thc[0]) - 1e-6
check("연습 7.1.5: 무작위 곡선 200개의 길이 ≥ r|θ₂ − θ₁| (r = 1)", ok)

# 연습 7.1.6: 나선면 ---------------------------------------------------------------------------
Xhel = sp.Matrix([sp.sinh(u) * sp.cos(v), sp.sinh(u) * sp.sin(v), v])
Eh, Fh, Gh = first_ff(Xhel, u, v)
check("연습 7.1.6: 나선면 (sinh u cos v, sinh u sin v, v)의 E, F, G = cosh²u, 0, cosh²u",
      (sp.simplify(Eh - sp.cosh(u) ** 2), Fh, sp.simplify(Gh - sp.cosh(u) ** 2)) == (0, 0, 0))

# 연습 7.1.7: 대각선 방향이 직교 ⇔ E = G ----------------------------------------------------------
check("연습 7.1.7: <x_u + x_v, x_u − x_v> = E − G (일반)",
      sp.expand((Xg.diff(u) + Xg.diff(v)).dot(Xg.diff(u) - Xg.diff(v)) - (Eg - Gg)) == 0)
Rn, rn = sp.symbols("Rn rn", positive=True)
for RV_, rV_, expect in ((sp.Rational(3, 2), 1, 2), (2, 1, 1), (3, 1, 0)):
    sols = sp.solveset(sp.Eq(RV_ + rV_ * sp.cos(tu), rV_), tu, sp.Interval.Ropen(0, 2 * sp.pi))
    check(f"연습 7.1.7: R = {RV_}, r = {rV_}에서 E = G인 수평 원의 개수 = {expect}", len(list(sols)) == expect)
sol = list(sp.solveset(sp.Eq(sp.Rational(3, 2) + sp.cos(tu), 1), tu, sp.Interval.Ropen(0, 2 * sp.pi)))
check("연습 7.1.7: R = 3/2, r = 1에서 두 원은 π/2 < u < 3π/2 (안쪽 절반)", all(sp.pi / 2 < sv < 3 * sp.pi / 2 for sv in sol))

summary()
