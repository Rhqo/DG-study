"""0.4절 How Surfaces Bend: Curvature — 본문, In computer vision 상자, Quick check의 계산 검증 (GUIDELINES.md §13).

규약(§6): N = x_u × x_v/|x_u × x_v|, W_p = −dN_p, L = ⟨x_uu, N⟩ …, K = det W, H = ½ tr W, Δ = div grad.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-4-curvature.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
sph, cyl, tor, sad = EX["sphere"], EX["cylinder"], EX["torus"], EX["saddle"]

# Example 0.4.2: shape operator ------------------------------------------------------------------
th, ph = sph["coords"]
(r,) = sph["params"]
Ws = dgsym.shape_operator(sph["expr"], th, ph, sph["positive"])
sym_equal("Example 0.4.2: sphere(바깥쪽 N)의 W = −(1/r) id", Ws, -sp.eye(2) / r, sph["domain"])
cu, cv = cyl["coords"]
(rc,) = cyl["params"]
Wc = dgsym.shape_operator(cyl["expr"], cu, cv)
sym_equal("Example 0.4.2: cylinder(바깥쪽 N)의 W = diag(−1/r, 0) (basis x_u, x_v)", Wc, sp.diag(-1 / rc, 0), cyl["domain"])
pu, pv = sp.symbols("u v", real=True)
Wp = dgsym.shape_operator(sp.Matrix([pu, pv, 0]), pu, pv)
check("Example 0.4.2: 평면의 W = 0", Wp == sp.zeros(2, 2))

# Example 0.4.4: saddle surface ----------------------------------------------------------------
u, v = sad["coords"]
W0 = dgsym.shape_operator(sad["expr"], u, v).subs(sad["point"])
check("Example 0.4.4: saddle 원점의 W = diag(2, −2) (= f의 Hessian)", W0 == sp.diag(2, -2))
t = sp.symbols("theta", real=True)
w = sp.Matrix([sp.cos(t), sp.sin(t)])
kn = sp.simplify((w.T * W0 * w)[0])
sym_equal("(0.4.1): Euler's formula κ_n = κ₁cos²θ + κ₂sin²θ (saddle)", kn, 2 * sp.cos(t) ** 2 - 2 * sp.sin(t) ** 2)
check("Example 0.4.4: θ = π/4에서 κ_n = 0", sp.simplify(kn.subs(t, sp.pi / 4)) == 0)
s = sp.symbols("s", real=True)
sec = sp.Matrix([s, 0, s ** 2])
k_sec = dgsym.signed_curvature(sp.Matrix([s, s ** 2, 0]), s)
check("Example 0.4.4: normal section z = x²의 원점 curvature = 2 (N 쪽으로)", k_sec.subs(s, 0) == 2)

# Example 0.4.6: L, M, N, K, H -------------------------------------------------------------------
Ls, Ms, Ns = dgsym.second_ff(sph["expr"], th, ph, sph["positive"])
for nm, val in zip("LMN", (Ls, Ms, Ns)):
    sym_equal(f"Example 0.4.6: sphere {nm}", val, sph["expected"][nm], sph["domain"])
Ks, Hs = dgsym.K_H(sph["expr"], th, ph, sph["positive"])
sym_equal("Example 0.4.6: sphere K = 1/r²", Ks, sph["expected"]["K"], sph["domain"])
sym_equal("Example 0.4.6: sphere H = −1/r", Hs, sph["expected"]["H"], sph["domain"])
Kc, Hc = dgsym.K_H(cyl["expr"], cu, cv)
sym_equal("Example 0.4.6: cylinder K = 0", Kc, 0, cyl["domain"])
sym_equal("Example 0.4.6: cylinder H = −1/(2r)", Hc, cyl["expected"]["H"], cyl["domain"])
Ks0, Hs0 = dgsym.K_H(sad["expr"], u, v)
check("Example 0.4.6: saddle 원점 K = −4, H = 0", Ks0.subs(sad["point"]) == -4 and Hs0.subs(sad["point"]) == 0)
# CV box (minimal surface): saddle의 H = 0인 점은 두 직선 x = ±y 위에만 있다 (원점만이 아니다)
Hs_num = sp.numer(sp.together(sp.simplify(Hs0)))
check("CV box: saddle의 H 분자 ∝ (u − v)(u + v), 곧 H = 0 ⟺ u = ±v",
      sp.simplify(sp.factor(Hs_num) / ((u - v) * (u + v))).free_symbols == set())
check("CV box: saddle은 (1, 1)에서 H = 0, (1, 0)에서 H ≠ 0 (minimal surface가 아니다)",
      sp.simplify(Hs0.subs({u: 1, v: 1})) == 0 and sp.simplify(Hs0.subs({u: 1, v: 0})) != 0)
tu, tv = tor["coords"]
R, rt = tor["params"]
Kt, Ht = dgsym.K_H(tor["expr"], tu, tv, tor["positive"])
sym_equal("Example 0.4.6: torus K = cos u/(r(R + r cos u))", Kt, tor["expected"]["K"], tor["domain"])
sym_equal("Example 0.4.6: torus H (안쪽 N) = (R + 2r cos u)/(2r(R + r cos u))", Ht,
          (R + 2 * rt * sp.cos(tu)) / (2 * rt * (R + rt * sp.cos(tu))), tor["domain"])
# (0.4.2)의 행렬: W = g⁻¹ h
E, F, G = dgsym.first_ff(tor["expr"], tu, tv, tor["positive"])
L, M, N = dgsym.second_ff(tor["expr"], tu, tv, tor["positive"])
Wt = sp.Matrix([[E, F], [F, G]]).inv() * sp.Matrix([[L, M], [M, N]])
sym_equal("(0.4.2): det(g⁻¹h) = (LN − M²)/(EG − F²) = K", sp.simplify(Wt.det()), Kt, tor["domain"])
sym_equal("(0.4.2): ½ tr(g⁻¹h) = H", sp.simplify(Wt.trace() / 2), Ht, tor["domain"])

# Proposition (부호): N을 뒤집으면 H → −H, K 그대로. cylinder를 (v, u) 순서로 쓰면 N이 안쪽이 된다 ----------
Xin = cyl["expr"].subs({cu: pv, cv: pu}, simultaneous=True)
Nin = dgsym.unit_normal(Xin, pu, pv)
sym_equal("Exercise 0.4.1: (v, u) 순서의 cylinder N = −(cos, sin, 0) (안쪽)", Nin,
          -sp.Matrix([sp.cos(pv), sp.sin(pv), 0]), {pu: (-2, 2), pv: (0, 2 * sp.pi), rc: (0.5, 2)})
Kin, Hin = dgsym.K_H(Xin, pu, pv)
sym_equal("Exercise 0.4.1: 안쪽 N에서 K = 0", Kin, 0, {rc: (0.5, 2)})
sym_equal("Exercise 0.4.1: 안쪽 N에서 H = 1/(2r)", Hin, 1 / (2 * rc), {rc: (0.5, 2)})
eig_in = sorted(dgsym.shape_operator(Xin, pu, pv).eigenvals().keys(), key=lambda e: float(e.subs(rc, 1)))
check("Exercise 0.4.1: 안쪽 N에서 κ₁ = 1/r, κ₂ = 0", sp.simplify(eig_in[1] - 1 / rc) == 0 and eig_in[0] == 0)

# Exercise 0.4.2: graph z = (a x² + b y²)/2 ------------------------------------------------------
a, b = sp.symbols("a b", real=True)
Xg = sp.Matrix([u, v, (a * u ** 2 + b * v ** 2) / 2])
Wg = dgsym.shape_operator(Xg, u, v).subs({u: 0, v: 0})
check("Exercise 0.4.2: 원점의 W = diag(a, b)", sp.simplify(Wg - sp.diag(a, b)) == sp.zeros(2, 2))
Kg, Hg = dgsym.K_H(Xg, u, v)
check("Exercise 0.4.2: K = ab, H = (a + b)/2", sp.simplify(Kg.subs({u: 0, v: 0}) - a * b) == 0
      and sp.simplify(Hg.subs({u: 0, v: 0}) - (a + b) / 2) == 0)

# Exercise 0.4.3: torus의 K 최대·최소 ---------------------------------------------------------------
Kn = sp.lambdify(tu, Kt.subs({R: 2, rt: sp.Rational(4, 5)}), "numpy")
close("Exercise 0.4.3: K(0) = 1/(r(R + r)) ≈ 0.446", Kn(0.0), 1 / (0.8 * 2.8), 1e-12)
close("Exercise 0.4.3: K(π) = −1/(r(R − r)) ≈ −1.042", Kn(np.pi), -1 / (0.8 * 1.2), 1e-12)
check("Exercise 0.4.3: 반올림 값 0.446, −1.042", abs(Kn(0.0) - 0.446) < 5e-4 and abs(Kn(np.pi) + 1.042) < 5e-4)
uu = np.linspace(-np.pi, np.pi, 2001)
check("Exercise 0.4.3: K는 u = 0에서 최대, u = π에서 최소", np.argmax(Kn(uu)) == 1000 and np.argmin(Kn(uu)) in (0, 2000))

# (0.4.3): Laplace–Beltrami Δ_S x = 2H N (Δ = div grad) -------------------------------------------
def laplace_beltrami(X, q1, q2, f, positive=()):
    E, F, G = dgsym.first_ff(X, q1, q2, positive)
    g = sp.Matrix([[E, F], [F, G]])
    gi = g.inv()
    sq = dgsym.simp(sp.sqrt(g.det()), positive)
    qs = (q1, q2)
    out = 0
    for i in range(2):
        out += sp.diff(sq * sum(gi[i, j] * sp.diff(f, qs[j]) for j in range(2)), qs[i])
    return dgsym.simp(out / sq, positive)


for nm, ex, (q1, q2) in (("sphere", sph, (th, ph)), ("cylinder", cyl, (cu, cv)), ("torus", tor, (tu, tv))):
    X = ex["expr"]
    pos = ex["positive"]
    lap = sp.Matrix([laplace_beltrami(X, q1, q2, X[k], pos) for k in range(3)])
    Kx, Hx = dgsym.K_H(X, q1, q2, pos)
    Nx = dgsym.unit_normal(X, q1, q2, pos)
    sym_equal(f"(0.4.3): {nm}에서 Δ_S x = 2H N", lap, 2 * Hx * Nx, ex["domain"])
f_test = sp.cos(th)
sym_equal("(0.4.3): sphere에서 Δ_S cos θ = −(2/r²) cos θ (Δ는 음의 정부호 쪽, div grad 규약)",
          laplace_beltrami(sph["expr"], th, ph, f_test, sph["positive"]), -2 * sp.cos(th) / r ** 2, sph["domain"])


# CV 상자: icosphere에서 cotan Laplacian과 angle defect ---------------------------------------------
def icosphere(level):
    t_ = (1 + 5 ** 0.5) / 2
    V = [(-1, t_, 0), (1, t_, 0), (-1, -t_, 0), (1, -t_, 0), (0, -1, t_), (0, 1, t_), (0, -1, -t_), (0, 1, -t_),
         (t_, 0, -1), (t_, 0, 1), (-t_, 0, -1), (-t_, 0, 1)]
    V = [np.array(p, float) / np.linalg.norm(p) for p in V]
    Fc = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
          (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
          (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        mid = {}
        nf = []

        def m(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid:
                p = V[i] + V[j]
                V.append(p / np.linalg.norm(p))
                mid[key] = len(V) - 1
            return mid[key]

        for (i, j, k) in Fc:
            a_, b_, c_ = m(i, j), m(j, k), m(k, i)
            nf += [(i, a_, c_), (j, b_, a_), (k, c_, b_), (a_, b_, c_)]
        Fc = nf
    return np.array(V), np.array(Fc)


def cotan_and_defect(V, Fc):
    n = len(V)
    Lx = np.zeros((n, 3))
    area = np.zeros(n)
    angsum = np.zeros(n)
    for tri in Fc:
        P = V[tri]
        for k in range(3):
            i, j, l = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            e1, e2 = V[j] - V[i], V[l] - V[i]
            angsum[i] += np.arccos(np.clip(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2), -1, 1))
            # 꼭짓점 i의 맞은편 변 (j, l)에 cot(∠i)를 준다
            cot = (e1 @ e2) / np.linalg.norm(np.cross(e1, e2))
            Lx[j] += 0.5 * cot * (V[l] - V[j])
            Lx[l] += 0.5 * cot * (V[j] - V[l])
            # Voronoi 넓이 [Meyer03]: 변 (j, l)의 기여 (1/8) cot(∠i) |x_j − x_l|² (acute triangle)
            d2 = np.sum((V[j] - V[l]) ** 2)
            area[j] += cot * d2 / 8
            area[l] += cot * d2 / 8
        e = [P[1] - P[0], P[2] - P[1], P[0] - P[2]]
        assert all(-e[k] @ e[(k + 1) % 3] > 0 for k in range(3))        # acute triangle
    return Lx / area[:, None], 2 * np.pi - angsum, area


Vs, Fs = icosphere(4)
Lx, defect, Ai = cotan_and_defect(Vs, Fs)
Nout = Vs                                          # unit sphere, 바깥쪽 N
mc = np.einsum("ij,ij->i", Lx, Nout)               # ⟨Δx, N⟩ ≈ 2H = −2
close("CV box: icosphere(level 4)에서 ⟨Δx, N⟩ 평균 ≈ 2H = −2", mc.mean(), -2.0, 1e-2)
check("CV box: 모든 정점에서 |⟨Δx, N⟩ + 2| < 0.01 (Voronoi 넓이)", np.max(np.abs(mc + 2)) < 0.01, f"max {np.max(np.abs(mc + 2)):.4f}")
tang = Lx - mc[:, None] * Nout
check("CV box: cotan Laplacian of x는 거의 normal 방향 (tangential 성분 < 1%)",
      np.max(np.linalg.norm(tang, axis=1)) < 0.02, f"max {np.max(np.linalg.norm(tang, axis=1)):.4f}")
Kest = defect / Ai
close("CV box: angle defect / area ≈ K = 1 (평균)", Kest.mean(), 1.0, 1e-2)
check("CV box: 모든 정점에서 |angle defect/area − 1| < 0.02", np.max(np.abs(Kest - 1)) < 0.02, f"max {np.max(np.abs(Kest - 1)):.4f}")
close("CV box / Section 0.5: Σ angle defect = 4π = 2πχ(S²)", defect.sum(), 4 * np.pi, 1e-9)

# CV 상자: mean curvature flow ∂x/∂t = 2HN에서 sphere ------------------------------------------------
tt, r0 = sp.symbols("t r_0", positive=True)
rf = sp.Function("rho")
sol = sp.dsolve(sp.Eq(rf(tt).diff(tt), 2 * (-1 / rf(tt))), rf(tt), ics={rf(0): r0})
sols = sol if isinstance(sol, list) else [sol]
ok = any(sp.simplify(sl.rhs ** 2 - (r0 ** 2 - 4 * tt)) == 0 and sl.rhs.subs(tt, 0) == r0 for sl in sols)
check("Exercise 0.4.4: dr/dt = 2H = −2/r의 해는 r(t)² = r₀² − 4t", ok)
check("Exercise 0.4.4: r₀ = 1이면 t = 1/4에 한 점으로 줄어든다", sp.solve(sp.Eq(1 - 4 * tt, 0), tt) == [sp.Rational(1, 4)])
A_t = 4 * sp.pi * (r0 ** 2 - 4 * tt)
dA = sp.diff(A_t, tt)
first_var = -4 * (1 / r0 ** 2) * 4 * sp.pi * r0 ** 2        # −∫ 4H² dA at t = 0
check("CV box: sphere에서 dA/dt = −∫4H² dA = −16π", sp.simplify(dA - first_var) == 0 and sp.simplify(dA + 16 * sp.pi) == 0)

# intuition: |K| = Gauss map 넓이의 비 (sphere) -----------------------------------------------------
check("Intuition: sphere의 Gauss image 넓이 4π / 넓이 4πr² = 1/r² = K",
      sp.simplify(4 * sp.pi / (4 * sp.pi * r ** 2) - sph["expected"]["K"]) == 0)

summary()
