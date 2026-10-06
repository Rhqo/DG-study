"""0.5절 Intrinsic Geometry: Geodesics and Gauss–Bonnet — 본문, In computer vision 상자, Quick check의 계산 검증.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-5-intrinsic.py``
"""

import numpy as np
import sympy as sp

import dgnum
import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
sph, cyl, tor, hyp = EX["sphere"], EX["cylinder"], EX["torus"], EX["hyperbolic_plane"]


def K_orthogonal(E, G, a, b):
    """(0.5.1): F = 0일 때 K = −1/(2√(EG)) [ ∂_a(G_a/√(EG)) + ∂_b(E_b/√(EG)) ]."""
    W = sp.sqrt(E * G)
    return -(sp.diff(sp.diff(G, a) / W, a) + sp.diff(sp.diff(E, b) / W, b)) / (2 * W)


# Theorem 0.5.1 / (0.5.1) -----------------------------------------------------------------------
th, ph = sph["coords"]
(r,) = sph["params"]
E, F, G = dgsym.first_ff(sph["expr"], th, ph, sph["positive"])
Kf = sp.simplify(K_orthogonal(E, G, th, ph).subs(sp.sqrt(r ** 4 * sp.sin(th) ** 2), r ** 2 * sp.sin(th)))
sym_equal("(0.5.1): sphere, E, F, G만으로 K = 1/r²", Kf, sph["expected"]["K"], sph["domain"])
u, v = tor["coords"]
R, rt = tor["params"]
Et, Ft, Gt = dgsym.first_ff(tor["expr"], u, v, tor["positive"])
Kt = K_orthogonal(Et, Gt, u, v)
sym_equal("(0.5.1): torus, E, F, G만으로 K = cos u/(r(R + r cos u))", Kt, tor["expected"]["K"], tor["domain"])
Kt2, _ = dgsym.K_H(tor["expr"], u, v, tor["positive"])
sym_equal("Theorem 0.5.1: torus에서 (0.4.2)(N을 쓴 공식)와 (0.5.1)이 같다", Kt, Kt2, tor["domain"])
cu, cv = cyl["coords"]
Ec, Fc, Gc = dgsym.first_ff(cyl["expr"], cu, cv)
check("(0.5.1): cylinder K = 0 = 평면의 K (E, G가 상수)", sp.simplify(K_orthogonal(Ec, Gc, cu, cv)) == 0)
x, y = hyp["coords"]
Kh = sp.simplify(K_orthogonal(1 / y ** 2, 1 / y ** 2, x, y))
check("Example 0.5.2: E = G = 1/y², F = 0이면 K = −1 (ℝ³ 없이 계량만)", Kh == hyp["expected"]["K"])
Kh2 = dgsym.sectional(hyp["expr"], [x, y], sp.Matrix([1, 0]), sp.Matrix([0, 1]))
check("Example 0.5.2: Riemann curvature tensor로 잰 K도 −1 (dgsym.sectional)", sp.simplify(Kh2 + 1) == 0)

# Corollary 0.5.3과 map projection ----------------------------------------------------------------
mer = sp.Matrix([ph, sp.log(sp.cot(th / 2))])
lam = sp.Matrix([ph, sp.cos(th)])
for name, psi in (("Mercator", mer), ("Lambert", lam)):
    J = psi.jacobian([th, ph])
    g_pull = sp.simplify(J.T * J)            # 평면의 metric을 (θ, φ)로 당긴 것
    globals()["g_" + name] = g_pull
gS = sp.diag(1, sp.sin(th) ** 2)              # r = 1
ratio = sp.simplify(g_Mercator * gS.inv())
check("Example (Mercator): 평면 metric = (1/sin²θ) × sphere metric (conformal, 배율 1/sin θ)",
      sp.simplify(ratio - sp.eye(2) / sp.sin(th) ** 2) == sp.zeros(2, 2))
check("Example (Lambert): det 비 = 1 (넓이 보존)", sp.simplify(g_Lambert.det() / gS.det()) == 1)
check("Example (Lambert): conformal이 아니다 (θ 방향 sin²θ, φ 방향 1/sin²θ)",
      sp.simplify(g_Lambert - sp.diag(sp.sin(th) ** 2, 1)) == sp.zeros(2, 2))
lam_ = sp.symbols("lambda", positive=True)
check("Exercise 0.5.1: conformal(λ²배)이면서 넓이 보존(λ² = 1)이면 λ = 1", sp.solve(sp.Eq(lam_ ** 2, 1), lam_) == [1])

# Definition 0.5.4 / Example 0.5.5: geodesic ------------------------------------------------------
t0 = sp.symbols("theta_0", positive=True)
k_lat = 1 / (r * sp.sin(t0))                  # 반지름 r sin θ0인 원
kn = -1 / r                                   # sphere의 normal curvature (바깥쪽 N)
kg2 = sp.simplify(k_lat ** 2 - kn ** 2)
sym_equal("Example 0.5.5: circle of latitude의 κ_g² = cot²θ₀/r²", kg2, sp.cot(t0) ** 2 / r ** 2, {t0: (0, sp.pi), r: (0.5, 2)})
check("Example 0.5.5: 적도(θ₀ = π/2)에서만 κ_g = 0", sp.simplify(kg2.subs(t0, sp.pi / 2)) == 0)
pp = np.array([np.sin(np.pi / 4) * np.cos(-np.pi / 3), np.sin(np.pi / 4) * np.sin(-np.pi / 3), np.cos(np.pi / 4)])
qq = np.array([np.sin(np.pi / 4) * np.cos(np.pi / 3), np.sin(np.pi / 4) * np.sin(np.pi / 3), np.cos(np.pi / 4)])
close("Example 0.5.5: ⟨p, q⟩ = 1/4", pp @ qq, 0.25, 1e-12)
close("Example 0.5.5: circle of latitude의 호 = (√2/2)(2π/3) ≈ 1.481", np.sin(np.pi / 4) * 2 * np.pi / 3, 1.4810, 5e-5)
close("Example 0.5.5: great circle의 호 = arccos(1/4) ≈ 1.318", np.arccos(0.25), 1.3181, 5e-5)
a, b, t = sp.symbols("a b t", positive=True)
hel = sp.Matrix([a * sp.cos(t), a * sp.sin(t), b * t])
acc = hel.diff(t, 2)
Ncyl = sp.Matrix([sp.cos(t), sp.sin(t), 0])
check("Example 0.5.5: cylinder 위의 helix는 가속도가 N에 평행 (geodesic)", sp.simplify(acc.cross(Ncyl)) == sp.zeros(3, 1))
# 수치: 적도에서 비스듬히 출발한 geodesic은 원점을 지나는 평면 위에 머문다
g = dgsym.induced_metric(sph["expr"], (th, ph), sph["positive"])
Gam = dgnum.numeric_christoffel(g, (th, ph), {r: 1}, sph["positive"])
ts = np.linspace(0, 3.0, 3001)
xs, vs = dgnum.integrate_geodesic(Gam, np.array([np.pi / 2, 0.0]), np.array([-0.6, 0.8]), ts)
pts = np.stack([np.sin(xs[:, 0]) * np.cos(xs[:, 1]), np.sin(xs[:, 0]) * np.sin(xs[:, 1]), np.cos(xs[:, 0])], axis=1)
nplane = np.cross(pts[0], pts[100])
check("Example 0.5.5: sphere의 geodesic(수치)은 great circle의 평면 위에 있다",
      np.max(np.abs(pts @ (nplane / np.linalg.norm(nplane)))) < 1e-8)

# Example 0.5.7: parallel transport around a circle of latitude -----------------------------------
for th0 in (np.pi / 3, np.pi / 4):
    tt = np.linspace(0, 2 * np.pi, 4001)
    V = dgnum.parallel_transport(Gam, lambda s: np.array([th0, s]), lambda s: np.array([0.0, 1.0]),
                                 np.array([1.0, 0.0]), tt)
    Xt = np.array([np.cos(th0), 0.0, -np.sin(th0)])                     # x_θ at φ = 0
    Xp = np.array([0.0, np.sin(th0), 0.0])                              # x_φ at φ = 0
    v0 = V[0, 0] * Xt + V[0, 1] * Xp
    v1 = V[-1, 0] * Xt + V[-1, 1] * Xp
    N0 = np.array([np.sin(th0), 0.0, np.cos(th0)])
    ang = np.arctan2(np.cross(v0, v1) @ N0, v0 @ v1)
    target = 2 * np.pi * (1 - np.cos(th0))
    close(f"Example 0.5.7: θ₀ = {th0:.4f}에서 반시계 회전각 = 2π(1 − cos θ₀) (mod 2π)",
          np.mod(ang, 2 * np.pi), np.mod(target, 2 * np.pi), 1e-6)
    close(f"Example 0.5.7 / §7: θ₀ = {th0:.4f}에서 시계 방향으로 재면 2π cos θ₀ (mod 2π)",
          np.mod(-ang, 2 * np.pi), np.mod(2 * np.pi * np.cos(th0), 2 * np.pi), 1e-6)
cap = sp.integrate(sp.sin(th), (th, 0, t0)) * 2 * sp.pi
sym_equal("Example 0.5.7: 극관의 넓이 × K = 2π(1 − cos θ₀) (r = 1)", cap, 2 * sp.pi * (1 - sp.cos(t0)))
close("Exercise 0.5.3: θ₀ = π/4에서 2π(1 − √2/2) ≈ 1.840 rad ≈ 105.4°", np.degrees(2 * np.pi * (1 - np.cos(np.pi / 4))),
      105.44, 5e-3)

# Theorem 0.5.8 / Example 0.5.9: Gauss–Bonnet ------------------------------------------------------
sym_equal("Example 0.5.9: sphere ∫∫K dA = 4π = 2π·2",
          sp.integrate(sph["expected"]["K"] * r ** 2 * sp.sin(th), (th, 0, sp.pi), (ph, 0, 2 * sp.pi)), 4 * sp.pi)
dA_t = rt * (R + rt * sp.cos(u))
check("Example 0.5.9: torus ∫∫K dA = 0 = 2π·0",
      sp.simplify(sp.integrate(sp.simplify(tor["expected"]["K"] * dA_t), (u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi))) == 0)
pos = sp.integrate(sp.simplify(tor["expected"]["K"] * dA_t), (u, -sp.pi / 2, sp.pi / 2), (v, 0, 2 * sp.pi))
check("Example 0.5.9: torus의 바깥쪽 절반 ∫∫K dA = 4π, 안쪽 절반 −4π", sp.simplify(pos - 4 * sp.pi) == 0)
check("Example 0.5.9: octant triangle의 내각의 합 3π/2 = π + (1)(4π/8)", sp.simplify(3 * sp.pi / 2 - (sp.pi + 4 * sp.pi / 8)) == 0)
check("Exercise 0.5.2: 세 각이 2π/3인 geodesic triangle의 넓이 = 3·2π/3 − π = π (r = 1)",
      sp.simplify(3 * 2 * sp.pi / 3 - sp.pi - sp.pi) == 0)
check("Exercise 0.5.2: S²(r)에서는 넓이 = π/K = πr²", sp.simplify((3 * 2 * sp.pi / 3 - sp.pi) / sph["expected"]["K"] - sp.pi * r ** 2) == 0)


# (0.5.4): discrete Gauss–Bonnet ------------------------------------------------------------------
def defects(V, Fc):
    s = np.zeros(len(V))
    for f in Fc:
        for k in range(len(f)):
            i, j, l = f[k], f[(k + 1) % len(f)], f[(k - 1) % len(f)]
            a_, b_ = V[j] - V[i], V[l] - V[i]
            s[i] += np.arccos(np.clip(a_ @ b_ / np.linalg.norm(a_) / np.linalg.norm(b_), -1, 1))
    return 2 * np.pi - s


Vc = np.array([(x_, y_, z_) for x_ in (0, 1) for y_ in (0, 1) for z_ in (0, 1)], float)
Fcube = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
dc = defects(Vc, Fcube)
check("Exercise 0.5.4: cube의 각 꼭짓점 angle defect = π/2", np.allclose(dc, np.pi / 2))
close("Exercise 0.5.4: cube Σδ = 4π = 2π(8 − 12 + 6)", dc.sum(), 2 * np.pi * (8 - 12 + 6), 1e-12)
nU, nV = 12, 24
Vt, Ft = [], []
for i in range(nU):
    for j in range(nV):
        uu, vv = 2 * np.pi * i / nU, 2 * np.pi * j / nV
        Vt.append(((2 + 0.8 * np.cos(uu)) * np.cos(vv), (2 + 0.8 * np.cos(uu)) * np.sin(vv), 0.8 * np.sin(uu)))
for i in range(nU):
    for j in range(nV):
        a0, a1 = i * nV + j, i * nV + (j + 1) % nV
        b0, b1 = ((i + 1) % nU) * nV + j, ((i + 1) % nU) * nV + (j + 1) % nV
        Ft += [(a0, a1, b1), (a0, b1, b0)]
Vt = np.array(Vt)
dt_ = defects(Vt, Ft)
n0, n2 = len(Vt), len(Ft)
n1 = 3 * n2 // 2
check("(0.5.4): triangulated torus의 χ = n0 − n1 + n2 = 0", n0 - n1 + n2 == 0)
close("(0.5.4): triangulated torus Σδ = 0", dt_.sum(), 0.0, 1e-9)
check("(0.5.4): torus mesh의 바깥쪽 정점은 δ > 0, 안쪽 정점은 δ < 0", dt_[0] > 0 and dt_[(nU // 2) * nV] < 0)

# CV 상자: heat method의 바탕 (Varadhan): ℝⁿ의 heat kernel에서 −4t log k_t → d² ------------------------
tt_, d, n = sp.symbols("t d n", positive=True)
kt = (4 * sp.pi * tt_) ** (-n / 2) * sp.exp(-d ** 2 / (4 * tt_))
check("CV box (heat method): lim_{t→0} −4t log k_t(x, y) = d(x, y)² (ℝⁿ)",
      sp.limit(-4 * tt_ * sp.log(kt), tt_, 0) == d ** 2)

summary()
