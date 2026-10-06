"""0.3절 Surfaces and Tangent Planes — 본문과 Quick check의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-3-surfaces.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES

# Example 0.3.2: implicit equation ---------------------------------------------------------------
sph, cyl, tor = EX["sphere"], EX["cylinder"], EX["torus"]
th, ph = sph["coords"]
(r,) = sph["params"]
Xs = sph["expr"]
sym_equal("Example 0.3.2: sphere는 x²+y²+z² = r²", Xs.dot(Xs), r ** 2, sph["domain"])
u, v = tor["coords"]
R, rt = tor["params"]
Xt = tor["expr"]
rho = sp.sqrt(Xt[0] ** 2 + Xt[1] ** 2)
sym_equal("Example 0.3.2: torus는 (√(x²+y²) − R)² + z² = r²", sp.simplify((rho - R) ** 2 + Xt[2] ** 2), rt ** 2,
          tor["domain"])
cu, cv = cyl["coords"]
(rc,) = cyl["params"]
Xc = cyl["expr"]
sym_equal("Example 0.3.2: cylinder는 x²+y² = r²", Xc[0] ** 2 + Xc[1] ** 2, rc ** 2, cyl["domain"])

# Example 0.3.4: unit normal vector의 방향 --------------------------------------------------------
Ns = dgsym.unit_normal(Xs, th, ph, sph["positive"])
sym_equal("Example 0.3.4: sphere N = x/r (바깥쪽)", Ns, Xs / r, sph["domain"])
sym_equal("Example 0.3.4: sphere x_θ × x_φ = r sin θ · x", Xs.diff(th).cross(Xs.diff(ph)), r * sp.sin(th) * Xs, sph["domain"])
Nc = dgsym.unit_normal(Xc, cu, cv)
sym_equal("Example 0.3.4: cylinder N = (cos u, sin u, 0) (바깥쪽)", Nc, cyl["expected"]["normal"], cyl["domain"])
Nt = dgsym.unit_normal(Xt, u, v, tor["positive"])
sym_equal("Example 0.3.4: torus N = −(cos u cos v, cos u sin v, sin u) (안쪽)", Nt, tor["expected"]["normal"], tor["domain"])
center = sp.Matrix([R * sp.cos(v), R * sp.sin(v), 0])           # 관의 중심원 위의 점
check("Example 0.3.4: torus의 N은 중심원 쪽(안쪽)을 향한다",
      sp.simplify(Nt.dot(center - Xt) - rt) == 0)
# implicit surface의 normal은 gradient에 평행
x, y, z = sp.symbols("x y z", real=True)
f = (sp.sqrt(x ** 2 + y ** 2) - R) ** 2 + z ** 2 - rt ** 2
gradf = sp.Matrix([f.diff(w) for w in (x, y, z)]).subs({x: Xt[0], y: Xt[1], z: Xt[2]})
sym_equal("Example 0.3.4: torus의 ∇f ∥ N (∇f × N = 0)", sp.simplify(gradf.cross(Nt)), sp.zeros(3, 1), tor["domain"])

# Example 0.3.6: E, F, G와 넓이 ------------------------------------------------------------------
for name, X, (a1, a2), ex in (("sphere", Xs, (th, ph), sph), ("cylinder", Xc, (cu, cv), cyl), ("torus", Xt, (u, v), tor)):
    EFG = dgsym.first_ff(X, a1, a2, ex["positive"])
    for k, val in zip("EFG", EFG):
        sym_equal(f"Example 0.3.6: {name} {k}", val, ex["expected"][k], ex["domain"])
Es, Fs, Gs = dgsym.first_ff(Xs, th, ph, sph["positive"])
dA_s = r ** 2 * sp.sin(th)
sym_equal("Example 0.3.6: sphere √(EG−F²) = r² sin θ", sp.sqrt(Es * Gs - Fs ** 2), dA_s, sph["domain"])
sym_equal("Example 0.3.6: sphere 넓이 4πr²", sp.integrate(dA_s, (th, 0, sp.pi), (ph, 0, 2 * sp.pi)), 4 * sp.pi * r ** 2)
Et, Ft, Gt = dgsym.first_ff(Xt, u, v, tor["positive"])
dA_t = rt * (R + rt * sp.cos(u))
sym_equal("Exercise 0.3.2: torus √(EG−F²) = r(R + r cos u)", sp.sqrt(Et * Gt - Ft ** 2), dA_t, tor["domain"])
At = sp.integrate(dA_t, (u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi))
sym_equal("Exercise 0.3.2: torus 넓이 4π²Rr", At, 4 * sp.pi ** 2 * R * rt)
close("Exercise 0.3.2: R = 2, r = 0.8에서 넓이 ≈ 63.17", float(At.subs({R: 2, rt: sp.Rational(4, 5)})), 63.165, 1e-3)

# (0.3.2): sphere 위의 위도원 길이 2πr sin θ0 (E, F, G로) -------------------------------------------
t = sp.symbols("t", real=True)
th0 = sp.symbols("theta_0", positive=True)
speed2 = (Es * 0 + Gs * 1).subs(th, th0)          # u' = 0, v' = 1
sym_equal("(0.3.2): 위도원 θ = θ0의 길이 = 2π r sin θ0", sp.integrate(sp.sqrt(speed2), (t, 0, 2 * sp.pi)).subs(
    sp.sqrt(sp.sin(th0) ** 2), sp.sin(th0)), 2 * sp.pi * r * sp.sin(th0), {th0: (0, sp.pi), r: (0.5, 2)})

# Example 0.3.7 / Exercise 0.3.1: cylinder를 arc length로 다시 쓰면 E = G = 1, F = 0 -----------------
Y = sp.Matrix([rc * sp.cos(cu / rc), rc * sp.sin(cu / rc), cv])
EFGy = dgsym.first_ff(Y, cu, cv)
check("Exercise 0.3.1: (r cos(u/r), r sin(u/r), v)의 E, F, G = 1, 0, 1", tuple(sp.simplify(e) for e in EFGy) == (1, 0, 1))
check("Exercise 0.3.1: 평면 (u, v, 0)의 E, F, G = 1, 0, 1", dgsym.first_ff(sp.Matrix([cu, cv, 0]), cu, cv) == (1, 0, 1))

# Figure 0.3.3 / Exercise 0.3.4: ellipse의 반축 ---------------------------------------------------
eps = sp.symbols("epsilon", positive=True)
check("Figure 0.3.3: sphere(r = 1)에서 θ 방향 반축 ε, φ 방향 반축 ε/sin θ",
      sp.simplify(Es.subs(r, 1) * eps ** 2 - eps ** 2) == 0 and sp.simplify(Gs.subs(r, 1) * (eps / sp.sin(th)) ** 2 - eps ** 2) == 0)
close("Exercise 0.3.4: θ = 30°에서 φ 한 눈금의 길이 / θ 한 눈금의 길이 = sin 30° = 0.5",
      float(sp.sqrt(Gs / Es).subs({th: sp.pi / 6, r: 1})), 0.5, 1e-12)
Gt_num = sp.lambdify(u, Gt.subs({R: 2, rt: sp.Rational(4, 5)}), "numpy")
close("Figure 0.3.3: torus u = 0에서 v 방향 반축 ε/2.8", 1 / np.sqrt(Gt_num(0.0)), 1 / 2.8, 1e-12)
close("Figure 0.3.3: torus u = π에서 v 방향 반축 ε/1.2", 1 / np.sqrt(Gt_num(np.pi)), 1 / 1.2, 1e-12)
close("Figure 0.3.1: x_v의 길이 / x_u의 길이 = (R + r cos(π/3))/r = 3", np.sqrt(Gt_num(np.pi / 3)) / 0.8, 3.0, 1e-12)

# CV 상자: PCA의 최소 eigenvector가 최소제곱 평면의 normal ------------------------------------------
rng = np.random.default_rng(3)
pts = rng.normal(size=(200, 3)) * np.array([1.0, 0.6, 0.02])
Rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
pts = pts @ Rot.T + np.array([0.3, -1.0, 2.0])
c = pts.mean(axis=0)
Cov = (pts - c).T @ (pts - c) / len(pts)
lam, vec = np.linalg.eigh(Cov)
nrm = vec[:, 0]
best = min(np.sum(((pts - c) @ (w / np.linalg.norm(w))) ** 2) for w in rng.normal(size=(2000, 3)))
check("CV box: 최소 eigenvector 방향으로의 분산 Σ⟨q − c, n⟩² 이 어떤 방향보다도 작다",
      np.sum(((pts - c) @ nrm) ** 2) <= best + 1e-12)
close("CV box: 최소 eigenvector ≈ 참 normal (부호 무시)", abs(nrm @ Rot[:, 2]), 1.0, 1e-3)
check("CV box: −n도 같은 분산 (부호를 정할 수 없다)",
      np.isclose(np.sum(((pts - c) @ nrm) ** 2), np.sum(((pts - c) @ (-nrm)) ** 2)))

summary()
