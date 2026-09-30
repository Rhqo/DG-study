"""7.2절 등거리사상: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/verify/v-7-2-isometries.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
first_ff = dgsym.first_ff
u, v = sp.symbols("u v", real=True)
s, t = sp.symbols("s t", real=True)


def same_fff(X, Y, a, b, positive=(), domain=None):
    """두 사상 X, Y (같은 변수 a, b)의 E, F, G가 같은가."""
    A = first_ff(X, a, b, positive)
    B = first_ff(Y, a, b, positive)
    return all(sp.simplify(x - y) == 0 for x, y in zip(A, B))


# 예 7.2.2: 직교변환과 평행이동의 제한 ------------------------------------------------------------
al = sp.symbols("alpha", real=True)
Rz = sp.Matrix([[sp.cos(al), -sp.sin(al), 0], [sp.sin(al), sp.cos(al), 0], [0, 0, 1]])
check("예 7.2.2: z축 회전은 직교행렬 (AᵀA = I)", sp.simplify(Rz.T * Rz) == sp.eye(3))
Sig = sp.diag(1, 1, -1)
check("예 7.2.2: xy 평면에 대한 반사는 직교행렬", Sig.T * Sig == sp.eye(3))
w1 = sp.Matrix(sp.symbols("a1:4", real=True))
w2 = sp.Matrix(sp.symbols("b1:4", real=True))
check("예 7.2.2: 직교행렬은 내적을 보존한다 <Aw1, Aw2> = <w1, w2>", sp.simplify((Rz * w1).dot(Rz * w2) - w1.dot(w2)) == 0)
tor = EX["torus"]
tu, tv = tor["coords"]
Xt = tor["expr"]
check("예 7.2.2: 원환면에서 R_z,α ∘ x(u, v) = x(u, v + α)", sp.simplify(Rz * Xt - Xt.subs(tv, tv + al)) == sp.zeros(3, 1))
check("예 7.2.2: x와 R∘x의 E, F, G가 같다 (원환면)", same_fff(Xt, Rz * Xt, tu, tv, tor["positive"]))
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(cr,) = cyl["params"]
c0 = sp.symbols("c", real=True)
Xc = cyl["expr"]
check("예 7.2.2: 원기둥을 축 방향으로 c만큼 옮기면 x(u, v + c)", sp.simplify(Xc + sp.Matrix([0, 0, c0]) - Xc.subs(cv, cv + c0)) == sp.zeros(3, 1))

# 비예 7.2.3: 닮음 p ↦ λp -----------------------------------------------------------------------
lam = sp.symbols("lambda", positive=True)
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
E1 = first_ff(sph["expr"], th, ph, sph["positive"])
E2 = first_ff(lam * sph["expr"], th, ph, sph["positive"])
check("비예 7.2.3(a): λx의 계수는 λ²(E, F, G)", all(sp.simplify(b - lam ** 2 * a) == 0 for a, b in zip(E1, E2)))
A = sp.diag(2, 1, 1)
wv = sp.Matrix([1, 0, 0])
check("비예 7.2.3(b): 늘이기 diag(2, 1, 1)은 e₁의 길이를 2배로", (A * wv).norm() == 2)

# 정리 7.2.7(a): y = φ∘x이면 y_u = dφ(x_u) — 원기둥 감기에서 ---------------------------------
x, y = sp.symbols("x y", real=True)
Phi = sp.Matrix([cr * sp.cos(x / cr), cr * sp.sin(x / cr), y])            # 연습 6.5.4의 φ (z는 무시)
Xpl = sp.Matrix([u, v, 0])
Yv = Phi.subs({x: u, y: v})
DPhi = sp.Matrix([[sp.diff(Phi[i], x), sp.diff(Phi[i], y), 0] for i in range(3)])
check("정리 7.2.7(a): (φ∘ι)_u = Dφ(ι_u), (φ∘ι)_v = Dφ(ι_v)",
      sp.simplify(Yv.diff(u) - DPhi.subs({x: u, y: v}) * Xpl.diff(u)) == sp.zeros(3, 1)
      and sp.simplify(Yv.diff(v) - DPhi.subs({x: u, y: v}) * Xpl.diff(v)) == sp.zeros(3, 1))

# 예 7.2.9: 평면을 원기둥에 감기 ---------------------------------------------------------------
Ey = tuple(sp.simplify(c) for c in first_ff(Yv, u, v))
check("예 7.2.9: y = φ∘ι = (r cos(u/r), r sin(u/r), v)의 E, F, G = 1, 0, 1", Ey == (1, 0, 1))
check("예 7.2.9: ι의 E, F, G = 1, 0, 1 (예 7.1.8)", first_ff(Xpl, u, v) == (1, 0, 1))
check("예 7.2.9: φ(x + 2πr, y) = φ(x, y) (단사가 아님)", sp.simplify(Phi.subs(x, x + 2 * sp.pi * cr) - Phi) == sp.zeros(3, 1))
check("예 7.2.9: y(u, v) = x_cyl(u/r, v) (원기둥 기준 매개화와의 관계)", sp.simplify(Yv - Xc.subs({cu: u / cr, cv: v})) == sp.zeros(3, 1))
check("예 7.2.9: 원기둥 매개화 x와 평면 매개화 (ru, v, 0)의 E, F, G가 같다 (예 7.1.4)",
      same_fff(Xc, sp.Matrix([cr * cu, cv, 0]), cu, cv))
t = sp.symbols("t", real=True)
bb = sp.symbols("b", real=True, nonzero=True)
sig = sp.Matrix([cr * t, bb * t])
hel = Phi.subs({x: sig[0], y: sig[1]})
sym_equal("예 7.2.9: φ∘σ는 나선 (r cos t, r sin t, bt)", hel, sp.Matrix([cr * sp.cos(t), cr * sp.sin(t), bb * t]))
sym_equal("예 7.2.9: |σ'|² = |(φ∘σ)'|² = r² + b²", hel.diff(t).dot(hel.diff(t)), sig.diff(t).dot(sig.diff(t)))
check("예 7.2.9: 길이 2π√(r² + b²)", sp.simplify(sp.integrate(sp.sqrt(sig.diff(t).dot(sig.diff(t))), (t, 0, 2 * sp.pi)) - 2 * sp.pi * sp.sqrt(cr ** 2 + bb ** 2)) == 0)
# 주의 상자: ℝ³ 거리는 보존되지 않는다
p1, p2 = Phi.subs({x: sp.pi * cr / 2, y: 0}), Phi.subs({x: 3 * sp.pi * cr / 2, y: 0})
check("주의 (7.2절): φ(πr/2, 0) = (0, r, 0), φ(3πr/2, 0) = (0, −r, 0)",
      sp.simplify(p1 - sp.Matrix([0, cr, 0])) == sp.zeros(3, 1) and sp.simplify(p2 - sp.Matrix([0, -cr, 0])) == sp.zeros(3, 1))
check("주의 (7.2절): ℝ³ 거리 2r < 평면 거리 πr", bool(sp.simplify(2 * cr - sp.pi * cr) < 0))

# 비예 7.2.10: 지수 사상 (e^x cos y, e^x sin y) ------------------------------------------------
Ye = sp.Matrix([sp.exp(u) * sp.cos(v), sp.exp(u) * sp.sin(v), 0])
Ee = tuple(sp.simplify(c) for c in first_ff(Ye, u, v))
check("비예 7.2.10: 계수 e^{2u}(1, 0, 1)", Ee == (sp.exp(2 * u), 0, sp.exp(2 * u)))
check("비예 7.2.10: u ≠ 0이면 E ≠ 1 (국소 등거리가 아님)", Ee[0].subs(u, 1) != 1)

# 연습 7.2.1: 나선면의 나사 운동 ----------------------------------------------------------------
xs, ys, zs = sp.symbols("xs ys zs", real=True)
fH = xs * sp.sin(zs) - ys * sp.cos(zs)
Phi_a = sp.Matrix([xs * sp.cos(al) - ys * sp.sin(al), xs * sp.sin(al) + ys * sp.cos(al), zs + al])
check("연습 7.2.1: f(Φ_α(p)) = f(p) (f = x sin z − y cos z)", sp.simplify(fH.subs({xs: Phi_a[0], ys: Phi_a[1], zs: Phi_a[2]}, simultaneous=True) - fH) == 0)
Xhel0 = sp.Matrix([sp.sinh(u) * sp.cos(v), sp.sinh(u) * sp.sin(v), v])
check("연습 7.2.1: Φ_α(y(u, v)) = y(u, v + α)",
      sp.simplify(Phi_a.subs({xs: Xhel0[0], ys: Xhel0[1], zs: Xhel0[2]}, simultaneous=True) - Xhel0.subs(v, v + al)) == sp.zeros(3, 1))
check("연습 7.2.1: y의 계수는 v에 무관", all(sp.diff(c, v) == 0 for c in (sp.simplify(c) for c in first_ff(Xhel0, u, v))))
# (회전면의 회전: 예 7.2.2와 같은 논법)
rev = EX["revolution"]
ru, rv = rev["coords"]
Xr = rev["expr"]
check("예 7.2.2 (회전면): R_z,α ∘ x(u, v) = x(u, v + α)", sp.simplify(Rz * Xr - Xr.subs(rv, rv + al)) == sp.zeros(3, 1))

# 연습 7.2.2: 반지름이 다른 두 원기둥 -------------------------------------------------------------
r1, r2 = sp.symbols("r1 r2", positive=True)
X1 = sp.Matrix([r1 * sp.cos(u / r1), r1 * sp.sin(u / r1), v])
X2 = sp.Matrix([r2 * sp.cos(u / r2), r2 * sp.sin(u / r2), v])
check("연습 7.2.2: 두 원기둥의 매개화 (r_i cos(u/r_i), r_i sin(u/r_i), v)의 E, F, G = 1, 0, 1",
      tuple(sp.simplify(c) for c in first_ff(X1, u, v)) == (1, 0, 1) and tuple(sp.simplify(c) for c in first_ff(X2, u, v)) == (1, 0, 1))

# 연습 7.2.5: 평면곡선 위의 기둥 --------------------------------------------------------------
c1, c2 = sp.Function("c1")(s), sp.Function("c2")(s)
Xgc = sp.Matrix([c1, c2, v])
Egc, Fgc, Ggc = first_ff(Xgc, s, v)
check("연습 7.2.5: 기둥 (c(s), v)의 계수 = (|c'|², 0, 1)",
      (sp.simplify(Egc - (sp.diff(c1, s) ** 2 + sp.diff(c2, s) ** 2)), Fgc, Ggc) == (0, 0, 1))
cc = sp.Matrix([cr * sp.cos(s / cr), cr * sp.sin(s / cr)])
check("연습 7.2.5: 원 c(s) = (r cos(s/r), r sin(s/r))은 호의 길이 매개화", sp.simplify(cc.diff(s).dot(cc.diff(s))) == 1)

# 연습 7.2.6: 현수면과 나선면 ---------------------------------------------------------------------
Xcat = sp.Matrix([sp.cosh(u) * sp.cos(v), sp.cosh(u) * sp.sin(v), u])
Xhel = Xhel0
check("연습 7.2.6(a): 현수면과 나선면 매개화의 E, F, G가 같다 (cosh²u, 0, cosh²u)", same_fff(Xcat, Xhel, u, v))
check("연습 7.2.6(b): 허리 원 x(0, v) = (cos v, sin v, 0) ↦ y(0, v) = (0, 0, v) (축)",
      Xcat.subs(u, 0) == sp.Matrix([sp.cos(v), sp.sin(v), 0]) and Xhel.subs(u, 0) == sp.Matrix([0, 0, v]))
check("연습 7.2.6(b): 허리 원 속력 |x_v(0, v)| = 1 = 축 선분의 속력", sp.simplify(Xcat.diff(v).subs(u, 0).norm() - 1) == 0 and Xhel.diff(v).subs(u, 0).norm() == 1)
b0, v0 = sp.symbols("b0 v0", positive=True)
Lcat = sp.integrate(sp.sqrt(sp.simplify(Xcat.diff(u).dot(Xcat.diff(u)))), (u, 0, b0))
check("연습 7.2.6(b): 현수선 조각의 길이 = sinh b", sp.simplify(Lcat - sp.sinh(b0)) == 0)
chord = (Xhel.subs({u: b0, v: v0}) - Xhel.subs({u: 0, v: v0}))
check("연습 7.2.6(b): 나선면 모선 조각의 길이 = sinh b", sp.simplify(sp.sqrt(chord.dot(chord)) - sp.sinh(b0)) == 0)
Zmap = sp.Matrix([xs / sp.cosh(zs), ys / sp.cosh(zs), zs])
img = Zmap.subs({xs: Xcat[0], ys: Xcat[1], zs: Xcat[2]}, simultaneous=True)
check("연습 7.2.6(c): (x/cosh z, y/cosh z, z)는 현수면을 원기둥 C₁로 보낸다", sp.simplify(img[0] ** 2 + img[1] ** 2 - 1) == 0)
fc = sp.lambdify(t, list(Xcat.subs({u: sp.sin(t), v: 2 * t}).diff(t)), "numpy")
fh = sp.lambdify(t, list(Xhel.subs({u: sp.sin(t), v: 2 * t}).diff(t)), "numpy")
ts = np.linspace(0, 3, 20001)


def speed(f):
    return np.linalg.norm(np.stack([np.broadcast_to(c, ts.shape) for c in f(ts)]), axis=0)


close("연습 7.2.6(a): 곡선 (sin t, 2t)의 상의 길이가 두 곡면에서 같다 (수치)", np.trapezoid(speed(fc), ts), np.trapezoid(speed(fh), ts), tol=1e-9)

# 연습 7.2.7: 층밀림 -------------------------------------------------------------------------------
be = sp.symbols("beta", positive=True)
Ysh = sp.Matrix([u + v * sp.cos(be), v * sp.sin(be), 0])
Esh = tuple(sp.simplify(c) for c in first_ff(Ysh, u, v))
check("연습 7.2.7: 층밀림의 계수 = (1, cos β, 1)", Esh == (1, sp.cos(be), 1))
check("연습 7.2.7: 대각선 길이의 제곱 = 2 + 2cos β ≠ 2", sp.simplify(Esh[0] + 2 * Esh[1] + Esh[2] - (2 + 2 * sp.cos(be))) == 0)

summary()
