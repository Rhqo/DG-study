"""6.5절 정칙값의 역상: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/verify/v-6-5-regular-values.py``
"""

import math
import random

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
rng = random.Random(3)
x, y, z = sp.symbols("x y z", real=True)
P = sp.Matrix([x, y, z])


def grad(f):
    return sp.Matrix([f]).jacobian(P).T


def Arg(a, b):
    return math.pi - 2 * math.atan(b / (1 - a))


# 예 6.5.2 ---------------------------------------------------------------------------------
check("예 6.5.2(a): grad(x²+y²+z²) = 2(x,y,z)", grad(x ** 2 + y ** 2 + z ** 2) == 2 * P)
check("예 6.5.2(b): grad(x²+y²−z²) = (2x, 2y, −2z)", grad(x ** 2 + y ** 2 - z ** 2) == sp.Matrix([2 * x, 2 * y, -2 * z]))
check("예 6.5.2(c): grad(x²+y²) = (2x, 2y, 0)", grad(x ** 2 + y ** 2) == sp.Matrix([2 * x, 2 * y, 0]))
check("예 6.5.2(d): grad(z²) = (0, 0, 2z)", grad(z ** 2) == sp.Matrix([0, 0, 2 * z]))
sols = sp.solve(list(grad(x ** 2 + y ** 2 - z ** 2)), [x, y, z], dict=True)
check("예 6.5.2(b): 임계점은 원점뿐", sols == [{x: 0, y: 0, z: 0}])

# 정리 6.5.3 증명의 예: 구면에서 z에 대해 푼 음함수는 √(r² − u² − v²) -------------------------
r = sp.symbols("r", positive=True)
u, v = sp.symbols("u v", real=True)
gimp = sp.sqrt(r ** 2 - u ** 2 - v ** 2)
check("정리 6.5.3 (예): F(u, v, g(u, v)) = r² (구면, 위쪽)", sp.simplify(u ** 2 + v ** 2 + gimp ** 2 - r ** 2) == 0)

# 예 6.5.5 ---------------------------------------------------------------------------------
a, b, c = sp.symbols("a b c", positive=True)
fe = x ** 2 / a ** 2 + y ** 2 / b ** 2 + z ** 2 / c ** 2
check("예 6.5.5(b): 타원면 기울기의 영점은 원점뿐", sp.solve(list(grad(fe)), [x, y, z], dict=True) == [{x: 0, y: 0, z: 0}])
check("예 6.5.5(a): 구면에서 grad f(p) = 2p", grad(x ** 2 + y ** 2 + z ** 2) == 2 * P)
pts = np.random.default_rng(1).normal(size=(1000, 2))
zz = np.sqrt(1 + (pts ** 2).sum(1))
check("예 6.5.5(c): 두 겹 쌍곡면 위에서 |z| ≥ 1", bool(np.all(zz >= 1)))

# 예 6.5.7: 원기둥 -------------------------------------------------------------------------
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(cr,) = cyl["params"]
Xc = cyl["expr"]
check("예 6.5.7: 기준 매개화의 상은 x² + y² = r²", sp.simplify(Xc[0] ** 2 + Xc[1] ** 2 - cr ** 2) == 0)
ncyl = Xc.diff(cu).cross(Xc.diff(cv))
check("예 6.5.7: x_u × x_v = (r cos u, r sin u, 0) (바깥쪽)", sp.simplify(ncyl - sp.Matrix([cr * sp.cos(cu), cr * sp.sin(cu), 0])) == sp.zeros(3, 1))
gc = grad(x ** 2 + y ** 2).subs({x: Xc[0], y: Xc[1], z: Xc[2]})
check("예 6.5.7: 기울기는 x_u × x_v와 평행 (외적 = 0)", sp.simplify(gc.cross(ncyl)) == sp.zeros(3, 1))

# 예 6.5.8: 원환면 -------------------------------------------------------------------------
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
Xt = tor["expr"]
rho = sp.sqrt(x ** 2 + y ** 2)
ft = (rho - R) ** 2 + z ** 2
sub = {x: Xt[0], y: Xt[1], z: Xt[2]}
sym_equal("예 6.5.8: f(x(u, v)) = r²", ft.subs(sub), rr ** 2, tor["domain"])
gt = grad(ft)
sym_equal("예 6.5.8 (6.5.3): grad f = (2(ρ−R)x/ρ, 2(ρ−R)y/ρ, 2z)", gt,
          sp.Matrix([2 * (rho - R) * x / rho, 2 * (rho - R) * y / rho, 2 * z]), {x: (0.3, 2), y: (0.3, 2), z: (-1, 1), R: (1, 2)})
nt = Xt.diff(tu).cross(Xt.diff(tv))
dir_ = sp.Matrix([sp.cos(tu) * sp.cos(tv), sp.cos(tu) * sp.sin(tv), sp.sin(tu)])
sym_equal("예 6.5.8 (6.5.4): x_u × x_v = −r(R + r cos u)(cos u cos v, cos u sin v, sin u)", nt,
          -rr * (R + rr * sp.cos(tu)) * dir_, tor["domain"])
sym_equal("예 6.5.8: 단위 x_u × x_v는 EXAMPLES의 법선(안쪽)과 같다", nt / (rr * (R + rr * sp.cos(tu))), tor["expected"]["normal"], tor["domain"])
sym_equal("예 6.5.8: grad f(x(u, v)) = 2r(cos u cos v, cos u sin v, sin u) (바깥쪽)", gt.subs(sub), 2 * rr * dir_, tor["domain"])
ok = True
for _ in range(300):
    Rv, rv = rng.uniform(1.1, 2.0), rng.uniform(0.2, 0.9)
    U0, V0 = rng.uniform(0.01, 2 * math.pi - 0.01), rng.uniform(0.01, 2 * math.pi - 0.01)
    X0 = np.array([float(cc) for cc in Xt.subs({R: Rv, rr: rv, tu: U0, tv: V0})])
    rh = math.hypot(X0[0], X0[1])
    ok &= abs(Arg((rh - Rv) / rv, X0[2] / rv) - U0) < 1e-8 and abs(Arg(X0[0] / rh, X0[1] / rh) - V0) < 1e-8
check("예 6.5.8: (u, v) = (Arg((ρ−R)/r, z/r), Arg(x/ρ, y/ρ)) (단사성)", ok)
# 세 매개화가 덮는다: 각 점 (u, v)마다 u, v 모두 a_k와 합동이 아닌 k가 있다
A = [0.0, math.pi, math.pi / 2]
grid = [(i * math.pi / 4, j * math.pi / 4) for i in range(8) for j in range(8)] + [(rng.uniform(0, 7), rng.uniform(0, 7)) for _ in range(500)]
def cong(s, t):
    d = (s - t) % (2 * math.pi)
    return min(d, 2 * math.pi - d) < 1e-9
check("예 6.5.8: (0,0), (π,π), (π/2,π/2)의 세 매개화가 원환면을 덮는다",
      all(any(not cong(uu, ak) and not cong(vv, ak) for ak in A) for uu, vv in grid))

# 명제 6.5.10, 예 6.5.11: 회전면 ------------------------------------------------------------
rev = EX["revolution"]
ru, rv_ = rev["coords"]
rho_f, z_f = rev["functions"]
Xr = rev["expr"]
nr = Xr.diff(ru).cross(Xr.diff(rv_))
p_ = rho_f(ru)
zp = z_f(ru)
check("명제 6.5.10 (6.5.6): x_u × x_v = (−ρz' cos v, −ρz' sin v, ρρ')",
      sp.simplify(nr - sp.Matrix([-p_ * zp.diff(ru) * sp.cos(rv_), -p_ * zp.diff(ru) * sp.sin(rv_), p_ * p_.diff(ru)])) == sp.zeros(3, 1))
check("명제 6.5.10 (6.5.6): |x_u × x_v|² = ρ²(ρ'² + z'²)",
      sp.simplify(nr.dot(nr) - p_ ** 2 * (p_.diff(ru) ** 2 + zp.diff(ru) ** 2)) == 0)
check("본문: ⟨x_u, x_v⟩ = 0 (회전면)", sp.simplify(Xr.diff(ru).dot(Xr.diff(rv_))) == 0)
sph = EX["sphere"]
th, ph = sph["coords"]
(rs,) = sph["params"]
Xs_rev = Xr.subs({rho_f(ru): rs * sp.sin(ru), z_f(ru): rs * sp.cos(ru)}).subs({ru: th, rv_: ph})
check("예 6.5.11(a): α(θ) = (r sinθ, r cosθ)의 회전면 = 구면의 기준 매개화", sp.simplify(Xs_rev - sph["expr"]) == sp.zeros(3, 1))
Xt_rev = Xr.subs({rho_f(ru): R + rr * sp.cos(ru), z_f(ru): rr * sp.sin(ru)}).subs({ru: tu, rv_: tv})
check("예 6.5.11(c): α(u) = (R + r cos u, r sin u)의 회전면 = 원환면의 기준 매개화", sp.simplify(Xt_rev - Xt) == sp.zeros(3, 1))
Xcat = Xr.subs({rho_f(ru): sp.cosh(ru), z_f(ru): ru}).doit()
ncat = Xcat.diff(ru).cross(Xcat.diff(rv_))
check("예 6.5.11(d): 현수면 |x_u × x_v| = cosh² u", sp.simplify(ncat.dot(ncat) - sp.cosh(ru) ** 4) == 0)
check("예 6.5.11(d): 현수면은 x² + y² = cosh² z 위", sp.simplify(Xcat[0] ** 2 + Xcat[1] ** 2 - sp.cosh(Xcat[2]) ** 2) == 0)

# 연습 6.5.1 ------------------------------------------------------------------------------
r2 = x ** 2 + y ** 2 + z ** 2
f1 = (r2 - 1) ** 2
check("연습 6.5.1: grad f = 4(|p|²−1)(x, y, z)", sp.simplify(grad(f1) - 4 * (r2 - 1) * P) == sp.zeros(3, 1))
check("연습 6.5.1: 임계값은 0(단위구면)과 f(0) = 1", f1.subs({x: 1, y: 0, z: 0}) == 0 and f1.subs({x: 0, y: 0, z: 0}) == 1)
cpos = sp.symbols("c", positive=True)
wr = sp.symbols("w", real=True)  # w = |p|²
check("연습 6.5.1: f = c (c > 0) ⟺ |p|² = 1 ± √c",
      set(sp.solve(sp.Eq((wr - 1) ** 2, cpos), wr)) == {1 - sp.sqrt(cpos), 1 + sp.sqrt(cpos)})
check("연습 6.5.1: c = 1이면 |p|² ∈ {0, 2} (원점과 반지름 √2인 구면)", set(sp.solve(sp.Eq((wr - 1) ** 2, 1), wr)) == {0, 2})

# 연습 6.5.2 ------------------------------------------------------------------------------
x0, y0, z0 = sp.symbols("x0 y0 z0", real=True)
g2 = grad(x ** 2 + y ** 2 - z ** 2).subs({x: x0, y: y0, z: z0})
plane = sp.expand(g2.dot(sp.Matrix([x - x0, y - y0, z - z0])) / 2)
check("연습 6.5.2: 아핀 접평면 ⟨grad f(p), X − p⟩/2 = x0x + y0y − z0z − (x0² + y0² − z0²)",
      sp.expand(plane - (x0 * x + y0 * y - z0 * z - (x0 ** 2 + y0 ** 2 - z0 ** 2))) == 0)
check("연습 6.5.2: ⟨grad f(p), e3⟩ = −2z0", g2[2] == -2 * z0)

# 연습 6.5.4 ------------------------------------------------------------------------------
phat = sp.Matrix([u / cr, v])
check("연습 6.5.4: det Dφ̂ = 1/r", sp.simplify(phat.jacobian([u, v]).det() - 1 / cr) == 0)
phi = sp.Matrix([cr * sp.cos(u / cr), cr * sp.sin(u / cr), v])
check("연습 6.5.4: φ = x∘φ̂ (원기둥의 기준 매개화)", sp.simplify(phi - Xc.subs({cu: u / cr, cv: v})) == sp.zeros(3, 1))
check("연습 6.5.4: φ(x + 2πr, y) = φ(x, y)", sp.simplify(phi.subs(u, u + 2 * sp.pi * cr) - phi) == sp.zeros(3, 1))

# 연습 6.5.5 ------------------------------------------------------------------------------
rho3 = sp.sqrt(x ** 2 + y ** 2)
Rsym, Zsym = sp.symbols("rho zeta", positive=True)
gpoly = Rsym ** 3 * Zsym + Rsym * Zsym ** 2 - 2 * Rsym + sp.sin(Zsym)   # 임의의 매끄러운 g의 표본
fpoly = gpoly.subs({Rsym: rho3, Zsym: z})
gnorm = (sp.diff(gpoly, Rsym) ** 2 + sp.diff(gpoly, Zsym) ** 2).subs({Rsym: rho3, Zsym: z})
sym_equal("연습 6.5.5: |grad f|² = (g_ρ² + g_z²)(ρ, z) (표본 g)", grad(fpoly).dot(grad(fpoly)), gnorm,
          {x: (0.3, 2), y: (0.3, 2), z: (-1, 1)})
gcon = (rho3 - R) ** 2 + z ** 2 - rr ** 2
gradg = sp.Matrix([2 * (Rsym - R), 2 * Zsym])
lhs2 = grad(gcon).dot(grad(gcon))
sym_equal("연습 6.5.5: 원환면의 g에서 |grad f|² = |grad g|²(ρ, z)", lhs2,
          gradg.dot(gradg).subs({Rsym: rho3, Zsym: z}), {x: (0.3, 2), y: (0.3, 2), z: (-1, 1), R: (1, 2), rr: (0.2, 0.9)})

# 연습 6.5.6 ------------------------------------------------------------------------------
aa, bb, cc2 = sp.symbols("a b c", real=True, nonzero=True)
f61 = x ** 2 + y ** 2 + z ** 2 - aa * x
f62 = x ** 2 + y ** 2 + z ** 2 - bb * y
ip = grad(f61).dot(grad(f62))
check("연습 6.5.6: ⟨grad f1, grad f2⟩ = 4|p|² − 2ax − 2by", sp.expand(ip - (4 * (x ** 2 + y ** 2 + z ** 2) - 2 * aa * x - 2 * bb * y)) == 0)
check("연습 6.5.6: grad f1 = 0인 점 (a/2, 0, 0)에서 f1 = −a²/4 ≠ 0", sp.simplify(f61.subs({x: aa / 2, y: 0, z: 0}) + aa ** 2 / 4) == 0)

m1 = sp.Matrix([aa / 2, 0, 0]); m2 = sp.Matrix([0, bb / 2, 0])
check("연습 6.5.6: f1 = |p − m1|² − a²/4", sp.expand(f61 - ((P - m1).dot(P - m1) - aa ** 2 / 4)) == 0)
check("연습 6.5.6: grad f1 = 2(p − m1)", sp.simplify(grad(f61) - 2 * (P - m1)) == sp.zeros(3, 1))
check("연습 6.5.6: |m1 − m2|² = r1² + r2² (a²/4 + b²/4)", sp.expand((m1 - m2).dot(m1 - m2) - (aa ** 2 / 4 + bb ** 2 / 4)) == 0)
# 수치: 두 구면의 교점 한 곳에서 기울기가 수직
av, bv = 1.3, -0.7
# 교원 위의 점: ax = by = |p|² 를 만족하는 점을 매개변수로 찾는다 (x = t, y = a t / b, |p|² = a t)
t_ = 0.2
zz = (av * t_ - t_ ** 2 - (av * t_ / bv) ** 2)
if zz > 0:
    pt = np.array([t_, av * t_ / bv, math.sqrt(zz)])
    g1n = np.array([2 * pt[0] - av, 2 * pt[1], 2 * pt[2]]); g2n = np.array([2 * pt[0], 2 * pt[1] - bv, 2 * pt[2]])
    close("연습 6.5.6: 교점에서 ⟨grad f1, grad f2⟩ = 0 (수치)", float(g1n @ g2n), 0.0, 1e-12)
summary()
