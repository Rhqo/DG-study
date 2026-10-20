"""Verification for Section 41.3 (A Panorama of Discrete Curvatures).

Run from the project root: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/verify/v-41-3-panorama.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "41-3-panorama.html"
rng = np.random.default_rng(41303)


def ang(A, B):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(A, B), axis=-1), np.sum(A * B, -1)))


# ---------------------------------------------------------------- Implementation 41.3 and Exercise 41.3.6
check("Implementation 41.3: at most 30 lines", dgcode.code_lines(PAGE, "code-41-3-1") <= 30)
ns, out = ddg41.run_printed(PAGE, "code-41-3-1")
check("Implementation 41.3: printed output matches the page", out.strip() == ddg41.printed_block(PAGE, "out-41-3-1").strip(), out)
V, F = dm.torus_mesh(2, 0.8, 12, 24)
close("Implementation 41.3: sphere_inscribed_normals = lib", ns["sphere_inscribed_normals"](V, F), ddg41.sphere_inscribed_normals(V, F), 1e-14)
ns2, out2 = ddg41.run_printed(PAGE, "code-ex-41-3-6", dict(ns))
check("Exercise 41.3.6: printed output matches the page", out2.strip() == ddg41.printed_block(PAGE, "out-ex-41-3-6").strip(), out2)
x = 0.3
f1 = lambda t: 0.6 * np.tanh(4 * t)  # noqa: E731
d1 = (f1(x + 1e-5) - f1(x - 1e-5)) / 2e-5
d2 = (f1(x + 1e-4) - 2 * f1(x) + f1(x - 1e-4)) / 1e-8
close("Exercise 41.3.6: H_exact = f'' / (2 (1 + f'^2)^(3/2)) for the graph z = 0.6 tanh 4x", ns2["H_exact"](x), d2 / (2 * (1 + d1 ** 2) ** 1.5), 1e-5)
xs = np.linspace(-0.5, 0.5, 20001)
Hs = ns2["H_exact"](xs)
close("Exercise 41.3.6 solution: max |H| = 1.31 at x = +-0.33", [np.abs(Hs).max(), abs(xs[np.argmax(np.abs(Hs))])], [1.31, 0.33], 6e-3)
Vs, Fs, inner6 = ns2["step_mesh"](13)
Lx6, A6 = dm.cotan_apply(Vs, Fs, Vs), dm.mass_voronoi(Vs, Fs)
i6 = np.argmin(np.linalg.norm(Vs - np.array([-0.274, -0.147, -0.48]), axis=1))
fp6 = 2.4 / np.cosh(4 * Vs[i6, 0]) ** 2
N6 = np.array([-fp6, 0, 1]) / np.hypot(fp6, 1)                      # exact upward normal of the graph
close("Exercise 41.3.6 solution: |(L x)_i| / (2 A_i) = 2.96, its exact-normal part 0.13",
      [np.linalg.norm(Lx6[i6]) / (2 * A6[i6]), Lx6[i6] @ N6 / (2 * A6[i6])], [2.96, 0.13], 6e-3)

# ---------------------------------------------------------------- Proposition 41.3.1
for name, (V, F) in [("torus_mesh(2, 0.8, 12, 24)", dm.torus_mesh(2, 0.8, 12, 24)), ("jittered sphere", ddg41.jittered_sphere(2, 0.15, 7))]:
    Fr = F[:, ::-1]
    A = dm.mass_voronoi(V, F)
    Lx, Lr = dm.cotan_laplacian(V, F) @ V, dm.cotan_laplacian(V, Fr) @ V
    KN, KNr = ddg41.gauss_curvature_normal(V, F), ddg41.gauss_curvature_normal(V, Fr)
    ok = np.allclose(Lx, Lr, atol=1e-13) and np.allclose(KN, -KNr, atol=1e-13)
    for k in ("uniform", "area", "angle"):
        N, Nr = dm.vertex_normals(V, F, kind=k), dm.vertex_normals(V, Fr, kind=k)
        ok &= np.allclose(np.sum(Lx * N, 1), -np.sum(Lr * Nr, 1), atol=1e-13) and np.allclose(np.sum(KN * N, 1), np.sum(KNr * Nr, 1), atol=1e-13)
    ok &= np.allclose(dm.angle_defect(V, F), dm.angle_defect(V, Fr), atol=1e-13)
    ok &= np.allclose(dm.mean_curvature_dihedral(V, F, mass=A), -dm.mean_curvature_dihedral(V, Fr, mass=A), atol=1e-13)
    check(f"Proposition 41.3.1(a): orientation flips H^w and H_ij, keeps K^w and delta ({name})", ok)
    Hs = {k: np.sum(Lx * dm.vertex_normals(V, F, kind=k), 1) / (2 * A) for k in ("uniform", "area", "angle")}
    bound = np.linalg.norm(Lx, axis=1) / (2 * A)
    check(f"Proposition 41.3.1(b): |H^w| <= |(L x)_i| / (2 A_i) ({name})", all(np.all(np.abs(h) <= bound + 1e-14) for h in Hs.values()))
    Nu, Na = dm.vertex_normals(V, F, kind="uniform"), dm.vertex_normals(V, F, kind="area")
    check(f"Proposition 41.3.1(c): |H^U - H^A| <= |(L x)| |N^U - N^A| / (2 A) ({name})",
          np.all(np.abs(Hs["uniform"] - Hs["area"]) <= bound * np.linalg.norm(Nu - Na, axis=1) + 1e-14))

# ---------------------------------------------------------------- Example 41.3.2
V, F = ddg41.cube_mesh(1, center=(0.5, 0.5, 0.5))
i = int(np.argmin(np.linalg.norm(V - np.array([1.0, 0, 0]), axis=1)))
Lx, A = dm.cotan_laplacian(V, F) @ V, dm.mass_voronoi(V, F)
close("Example 41.3.2(a): (L x)_i = -(1, -1, -1), A_i = 0.75, four triangles", np.r_[Lx[i], A[i], np.sum(np.any(F == i, axis=1))], [-1, 1, 1, 0.75, 4], 1e-13)
NU, NT, NS = dm.vertex_normals(V, F, kind="uniform")[i], dm.vertex_normals(V, F, kind="angle")[i], ddg41.sphere_inscribed_normals(V, F)[i]
close("Example 41.3.2(a): N^U = N^A = (2, -1, -1)/sqrt 6, N^theta = N^S = (1, -1, -1)/sqrt 3",
      np.r_[NU, dm.vertex_normals(V, F, kind="area")[i], NT, NS], np.r_[[2, -1, -1] / np.sqrt(6), [2, -1, -1] / np.sqrt(6),
                                                                       [1, -1, -1] / np.sqrt(3), [1, -1, -1] / np.sqrt(3)], 1e-13)
HU, HT = Lx[i] @ NU / (2 * A[i]), Lx[i] @ NT / (2 * A[i])
close("Example 41.3.2(a): H^U = -1.0887, H^theta = -1.1547, ratio 0.9428 = cos 19.47", [HU, HT, HU / HT, np.cos(np.radians(19.47))],
      [-1.0887, -1.1547, 0.9428, 0.9428], 6e-5)
V, F = dm.torus_mesh(2, 0.8, 24, 48)
Lx, A = dm.cotan_laplacian(V, F) @ V, dm.mass_voronoi(V, F)
u, Ke, He, e1, e2 = ddg41.torus_exact(V)
arr = np.array([np.sum(Lx * dm.vertex_normals(V, F, kind=k), 1) / (2 * A) for k in ("uniform", "area", "angle")]
               + [np.sum(Lx * ddg41.sphere_inscribed_normals(V, F), 1) / (2 * A), -np.linalg.norm(Lx, axis=1) / (2 * A)])
check(f"Example 41.3.2(b): torus: the five values agree within 2.4e-5 (max {(arr.max(0) - arr.min(0)).max():.2e})", (arr.max(0) - arr.min(0)).max() < 2.5e-5)
check("Example 41.3.2(b): torus: all within 0.0029 of the exact H", np.abs(arr - He).max() < 0.0029 + 1e-4)

# ---------------------------------------------------------------- Definition 41.3.3, Proposition 41.3.4, Example 41.3.5
V, F = ddg41.saddle_grid(17, 0.5)
o = int(np.argmin(np.linalg.norm(V, axis=1)))
Ks, Hs_ = dm.angle_defect(V, F)[o] / dm.mass_voronoi(V, F)[o], dd.mean_curvature(V, F)[o]
k1, k2, cl = ddg41.principal_curvatures(Hs_, Ks)
close("Definition 41.3.3 (saddle origin): H = 0, K = -3.9536, kappa = +-1.988", [Hs_, Ks, k1, k2], [0, -3.9536, 1.988, -1.988], 6e-4)
worst = 0.0
for t in range(2000):
    a, b = rng.normal(size=2) * 3
    worst = max(worst, abs(np.sqrt(max(a, 0)) - np.sqrt(max(b, 0))) - np.sqrt(abs(a - b)))
check("Proposition 41.3.4: |sqrt(a_+) - sqrt(b_+)| <= sqrt|a - b| (2000 random pairs)", worst <= 1e-15)
for t in range(2000):
    H, K, H2, K2 = rng.normal(size=4)
    for sgn in (0, 1):
        lhs = abs(ddg41.principal_curvatures(H2, K2)[sgn] - ddg41.principal_curvatures(H, K)[sgn])
        if lhs > abs(H2 - H) + np.sqrt(abs((H2 * H2 - K2) - (H * H - K))) + 1e-14:
            check("Proposition 41.3.4(b) on random data", False)
check("Proposition 41.3.4(b): the bound holds on 2000 random pairs", True)
k1, k2, _ = ddg41.principal_curvatures(-1.0, 1 - 1e-6)
close("Exercise 41.3.3(a): K error 1e-6 at an umbilic gives kappa = -1 +- 1e-3", [k1, k2], [-1 + 1e-3, -1 - 1e-3], 1e-12)
V, F = dm.torus_mesh(2, 0.8, 24, 48)
A = dm.mass_voronoi(V, F)
k1, k2, cl = ddg41.principal_curvatures(dd.mean_curvature(V, F), dm.angle_defect(V, F) / A)
u, Ke, He, e1, e2 = ddg41.torus_exact(V)
close("Example 41.3.5(a): exact kappa_1 in [-0.357, 0.833], kappa_2 = -1.25", [e1.min(), e1.max(), e2.min(), e2.max()], [-0.357, 0.833, -1.25, -1.25], 5e-4)
check("Example 41.3.5(a): no umbilic point: min (H^2 - K) = 0.199", abs((He ** 2 - Ke).min() - 0.199) < 5e-4)
close("Example 41.3.5(a): kappa errors 0.0059, 0.0051, no clamp", [np.abs(k1 - e1).max(), np.abs(k2 - e2).max(), cl.sum()], [0.0059, 0.0051, 0], 6e-5)
for lev, want_split, want_err in zip((1, 2, 3, 4), (0.565, 0.283, 0.148, 0.075), (0.080, 0.020, 0.0055, 0.0014)):
    V, F = dm.icosphere(lev)
    Ks = dm.angle_defect(V, F) / dm.mass_voronoi(V, F)
    Hs_ = dd.mean_curvature(V, F)
    disc = Hs_ ** 2 - Ks
    check(f"Example 41.3.5(b): icosphere({lev}): H^2 < K at every vertex", np.all(disc < 0) and np.all(Ks > 1))
    close(f"Example 41.3.5(b): icosphere({lev}): split {want_split}, max|K - 1| {want_err}", [2 * np.sqrt(np.abs(disc).max()), np.abs(Ks - 1).max()],
          [want_split, want_err], max(5e-4, 0.01 * want_err) if lev > 1 else 6e-4)

# ---------------------------------------------------------------- Theorem 41.3.7 and Example 41.3.8
for t in range(20):
    c, rad = rng.normal(size=3), rng.uniform(0.5, 3)
    m = int(rng.integers(3, 9))
    xi = c + rad * np.array([0, 0, 1.0])
    phis = np.sort(rng.uniform(0, 2 * np.pi, m))
    th = rng.uniform(0.1, 0.9, m)
    link = c + rad * np.c_[np.sin(th) * np.cos(phis), np.sin(th) * np.sin(phis), np.cos(th)]
    e = link - xi
    S = sum(np.cross(e[k], e[(k + 1) % m]) / ((e[k] @ e[k]) * (e[(k + 1) % m] @ e[(k + 1) % m])) for k in range(m))
    if np.linalg.norm(np.cross(S, xi - c)) > 1e-10 * np.linalg.norm(S) * rad:
        check(f"Theorem 41.3.7: random star {t} on a sphere", False)
check("Theorem 41.3.7: S_i is parallel to x_i - c for 20 random stars on random spheres", True)
V, F = dm.icosphere(3)
res = [ang(dm.vertex_normals(V, F, kind=k), V).max() for k in ("uniform", "area", "angle")] + [ang(ddg41.sphere_inscribed_normals(V, F), V).max()]
close("Example 41.3.8(a): icosphere(3): 0.47, 0.68, 0.30 degrees", res[:3], [0.47, 0.68, 0.30], 6e-3)
check("Example 41.3.8(a): sphere-inscribed exact (< 1e-12 degrees)", res[3] < 1e-12)
V, F = ddg41.jittered_sphere(3, 0.15, 7)
check("Example 41.3.8(b): jittered sphere is still a discrete immersion with outward faces", (np.sum(dm.face_cross(V, F) * V[F].mean(1), 1) < 0).sum() == 0)
close("Example 41.3.8(b): 13% obtuse triangles", (dm.corner_angles(V, F).max(1) > np.pi / 2).mean(), 0.13, 5e-3)
res = [ang(dm.vertex_normals(V, F, kind=k), V).max() for k in ("uniform", "area", "angle")]
NSj = ddg41.sphere_inscribed_normals(V, F)
close("Example 41.3.8(b) / Figure 41.3.2(a): 3.27, 5.43, 2.19 degrees", res, [3.27, 5.43, 2.19], 6e-3)
check("Example 41.3.8(b): sphere-inscribed normal exact and outward (+ sign) at every vertex", ang(NSj, V).max() < 1e-11 and np.all(np.sum(NSj * V, 1) > 0))
Vb, Fb, Nb = ddg41.bumpy_revolution()
res = [ang(dm.vertex_normals(Vb, Fb, kind=k), Nb).max() for k in ("uniform", "area", "angle")] + [ang(ddg41.sphere_inscribed_normals(Vb, Fb), Nb).max()]
close("Example 41.3.8(c) / Figure 41.3.2(b): bumpy surface of revolution 2.18, 1.70, 2.48, 3.90 degrees", res, [2.18, 1.70, 2.48, 3.90], 6e-3)
for nu, nv in ((8, 16), (12, 24), (24, 48)):
    V, F = dm.torus_mesh(2, 0.8, nu, nv)
    u, *_ = ddg41.torus_exact(V)
    Nex = np.c_[np.cos(u)[:, None] * V[:, :2] / np.hypot(V[:, 0], V[:, 1])[:, None], np.sin(u)]
    check(f"Example 41.3.8(c): torus_mesh(2, 0.8, {nu}, {nv}): sphere-inscribed normal exact (observation)", ang(ddg41.sphere_inscribed_normals(V, F), Nex).max() < 1e-11)
V, F = dm.torus_mesh(2, 0.8, 12, 24)
i = 5 * 24 + 3
ring = [i] + dm.neighbors(F, len(V))[i]
P = V[ring]
Am = np.c_[2 * P, -np.ones(len(P))]
sol, *_ = np.linalg.lstsq(Am, np.sum(P * P, 1), rcond=None)
check("Exercise 41.3.5(d): the one-ring of a torus_mesh vertex is not on a sphere (fit residual > 1e-4)", np.abs(Am @ sol - np.sum(P * P, 1)).max() > 1e-4)
# Exercise 41.3.4
t, ph = [0.3, 0.5, 0.3, 0.7], [0, np.pi / 2, np.pi, 1.5 * np.pi]
V = np.array([[0, 0, 1.0]] + [[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)] for a, b in zip(t, ph)])
F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
close("Exercise 41.3.4: N^S = e_3", ddg41.sphere_inscribed_normals(V, F)[0], [0, 0, 1], 1e-15)
close("Exercise 41.3.4: N^A = (0, -0.0998, 0.9950), 5.73 degrees", np.r_[dm.vertex_normals(V, F, kind="area")[0], ang(dm.vertex_normals(V, F, kind="area")[0], np.array([0, 0, 1.0]))],
      [0, -0.0998, 0.9950, 5.73], 6e-5 * 100)
close("Exercise 41.3.4: uniform 2.87, angle 2.79 degrees", [ang(dm.vertex_normals(V, F, kind=k)[0], np.array([0, 0, 1.0])) for k in ("uniform", "angle")], [2.87, 2.79], 6e-3)
summary()
