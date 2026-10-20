"""Verification for Section 42.4 (Discretization via DEC).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/verify/v-42-4-dec.py``
"""

import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch42  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "42-4-dec.html"
rng = np.random.default_rng(4204)


def boundary_flux(V, F, u, i):
    """sum over the faces at i of the integrals of *dW0u along [m_ij, c_f] + [c_f, m_ik] (face plane, signed)."""
    C = dd.circumcenters(V, F)
    total = 0.0
    for f, tri in enumerate(F):
        tri = list(tri)
        if i not in tri:
            continue
        a = tri.index(i)
        j, k = tri[(a + 1) % 3], tri[(a + 2) % 3]
        P = V[[i, j, k]]
        g = dd.barycentric_gradients(P)
        grad = u[[i, j, k]] @ g
        n = np.cross(P[1] - P[0], P[2] - P[0])
        n /= np.linalg.norm(n)
        mij, mik = 0.5 * (V[i] + V[j]), 0.5 * (V[i] + V[k])
        for s0, s1 in ((mij, C[f]), (C[f], mik)):
            seg = s1 - s0
            total += grad @ np.cross(seg, n)          # (*du)(v) = -du(Jv) = du(v x n)... flux to the right of v
    return total


# ---------------------------------------------------------------- Proposition 42.4.1 on planar and curved meshes with obtuse triangles
for name, (V, F) in {"jittered sphere(2)": ch42.jittered_sphere(2, s=0.15, seed=42),
                     "torus_mesh(2, 0.8, 8, 16)": dm.torus_mesh(2.0, 0.8, 8, 16)}.items():
    L = dm.cotan_laplacian(V, F)
    u = rng.normal(size=len(V))
    fl = np.array([boundary_flux(V, F, u, i) for i in range(len(V))])
    close(f"Proposition 42.4.1 on {name}: flux of grad W0 u out of the circumcentric cell = (L u)_i", fl, L @ u, tol=1e-11)
check("Proposition 42.4.1: the jittered sphere(2) has obtuse triangles and negative weights",
      (dm.corner_angles(*ch42.jittered_sphere(2, s=0.15, seed=42)).max(1) > np.pi / 2).sum() == 52
      and (dd.hodge1(*ch42.jittered_sphere(2, s=0.15, seed=42)) < 0).sum() > 0)
# smooth version on a planar polygon: int over C_i of Delta u = boundary flux, for u = x^2 + 3y^2 (Delta u = 8)
V, F = ch42.disk_mesh(8)
inner = np.setdiff1d(np.arange(len(V)), dm.boundary_vertices(F))
i0 = inner[np.argmin(np.linalg.norm(V[inner, :2] - [0.13, 0.2], axis=1))]
A0 = dd.dual_areas(V, F, "circumcentric")[i0]
# for a quadratic u the gradient is linear: integrate exactly along each straight piece with the midpoint rule
C = dd.circumcenters(V, F)
tot = 0.0
for f, tri in enumerate(F):
    tri = list(tri)
    if i0 not in tri:
        continue
    a = tri.index(i0)
    j, k = tri[(a + 1) % 3], tri[(a + 2) % 3]
    for s0, s1 in ((0.5 * (V[i0] + V[j]), C[f]), (C[f], 0.5 * (V[i0] + V[k]))):
        mid = 0.5 * (s0 + s1)
        gr = np.array([2 * mid[0], 6 * mid[1], 0.0])
        tot += gr @ np.cross(s1 - s0, [0, 0, 1.0])
close("Proposition 42.4.1 (smooth): divergence theorem on a circumcentric cell, int Delta u = 8 |C_i|", tot, 8 * A0, tol=1e-12)

# ---------------------------------------------------------------- Figure 42.4.1
nb = np.array([[1.0, 0.0], [0.35, 0.3], [-0.55, 0.85], [-1.05, -0.1], [-0.4, -0.95], [0.55, -0.25]])
Vs = np.c_[np.vstack([[0.0, 0.0], nb]), np.zeros(7)]
Fs = np.array([[0, k + 1, (k + 1) % 6 + 1] for k in range(6)])
Ls = dm.cotan_laplacian(Vs, Fs)
us = Vs[:, 0] ** 2 + 0.6 * Vs[:, 1]
angs = np.degrees(dm.corner_angles(Vs, Fs))
close("Figure 42.4.1: obtuse angles 115 and 127 degrees, L_01 = -0.599", [angs[0, 2], angs[5, 1], Ls[0, 1]], [115, 127, -0.599],
      tol=0.6)
close("Figure 42.4.1: L_01 = -0.599 (3 digits)", Ls[0, 1], -0.599, tol=6e-4)
close("Figure 42.4.1: (L u)_0 = 1.055, signed |C_0| = 0.562, ratio 1.877",
      [(Ls @ us)[0], dd.dual_areas(Vs, Fs, "circumcentric")[0], (Ls @ us)[0] / dd.dual_areas(Vs, Fs, "circumcentric")[0]],
      [1.055, 0.562, 1.877], tol=6e-4)
close("Figure 42.4.1: segment fluxes sum to (L u)_0", boundary_flux(Vs, Fs, us, 0), (Ls @ us)[0], tol=1e-13)

# ---------------------------------------------------------------- Example 42.4.2
vals = []
for lv in (1, 2, 3, 4):
    V, F = dm.icosphere(lv)
    L = dm.cotan_laplacian(V, F)
    m = dd.dual_areas(V, F, "circumcentric")
    z = V[:, 2]
    vals.append((np.abs(L @ z).max(), m.max(), np.abs(L @ z / m + 2 * z).max()))
vals = np.array(vals)
close("Example 42.4.2: max |(L z)_i| = 0.573, 0.154, 0.044, 0.011", vals[:, 0], [0.573, 0.154, 0.044, 0.011], tol=6e-4)
close("Example 42.4.2: largest dual area 0.287, 0.081, 0.023, 0.006", vals[:, 1], [0.287, 0.081, 0.023, 0.006], tol=6e-4)
close("Example 42.4.2: pointwise error with circumcentric area 0.000, 0.028, 0.015, 0.008", vals[:, 2],
      [0.0, 0.028, 0.015, 0.008], tol=6e-4)

# ---------------------------------------------------------------- Definition 42.4.4 example / non-example, Proposition 42.4.5
Vo = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.5, 0.1, 0]])
Fo = np.array([[0, 1, 2]])
close("Definition 42.4.4: circumcentric pieces -0.1375, -0.1375, 0.325; mixed 0.0125, 0.0125, 0.025",
      [dd.dual_areas(Vo, Fo, "circumcentric"), dd.dual_areas(Vo, Fo, "mixed")],
      [[-0.1375, -0.1375, 0.325], [0.0125, 0.0125, 0.025]], tol=1e-12)
for name, (V, F) in {"jittered sphere(3)": ch42.jittered_sphere(3, s=0.15, seed=42), "disk mesh": ch42.disk_mesh(8),
                     "torus": dm.torus_mesh(2.0, 0.8, 12, 24)}.items():
    tot = dm.face_areas(V, F).sum()
    sums = [dd.dual_areas(V, F, k).sum() for k in ("barycentric", "circumcentric", "mixed")]
    close(f"Proposition 42.4.5(a) on {name}: the three dual areas sum to the total area", sums, [tot] * 3, tol=1e-11)
    check(f"Proposition 42.4.5(b) on {name}: barycentric and mixed areas are positive",
          dd.dual_areas(V, F, "barycentric").min() > 0 and dd.dual_areas(V, F, "mixed").min() > 0)
Vr, Fr = ch42.jittered_sphere(3, s=0.3, seed=7)
check("Proposition 42.4.5(c): circumcentric areas can be negative (jittered sphere, s = 0.3)",
      dd.dual_areas(Vr, Fr, "circumcentric").min() < 0)
Vq, Fq = dm.icosphere(3)
check("Definition 42.4.4 example: on the icosphere and on grid_mesh mixed = circumcentric",
      np.allclose(dd.dual_areas(Vq, Fq, "mixed"), dd.dual_areas(Vq, Fq, "circumcentric"))
      and np.allclose(*[dd.dual_areas(*dm.grid_mesh(np.linspace(0, 1, 5), np.linspace(0, 1, 5)), k) for k in ("mixed", "circumcentric")]))

# ---------------------------------------------------------------- Example 42.4.6
grid_err = {"barycentric": [], "circumcentric": []}
for n in (8, 16, 32):
    xs = np.linspace(0, 1, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    L = dm.cotan_laplacian(V, F)
    inn = (V[:, 0] > 1e-9) & (V[:, 0] < 1 - 1e-9) & (V[:, 1] > 1e-9) & (V[:, 1] < 1 - 1e-9)
    deg = np.array([len(nbr) for nbr in dm.neighbors(F, len(V))])
    ex = np.sin(np.pi * V[:, 0]) * np.sin(np.pi * V[:, 1])
    for k in ("barycentric", "circumcentric", "mixed"):
        m = ch42.mass(V, F, k)
        lap = (L @ V[:, 0] ** 2 / m)[inn]
        if k == "barycentric":
            check(f"Example 42.4.6 (n = {n}): barycentric areas 2/3 l^2 (valence 4) and 4/3 l^2 (valence 8), pointwise 3 and 1.5",
                  np.allclose(m[inn & (deg == 4)] * n * n, 2 / 3) and np.allclose(m[inn & (deg == 8)] * n * n, 4 / 3)
                  and np.allclose(np.sort(np.unique(np.round(lap, 10))), [1.5, 3.0]))
        else:
            check(f"Example 42.4.6 (n = {n}): {k} area l^2 and pointwise Delta(x^2) = 2 exactly",
                  np.allclose(m[inn] * n * n, 1.0) and np.allclose(lap, 2.0))
        if k in grid_err:
            uu = np.zeros(len(V))
            uu[inn] = np.linalg.solve(L[np.ix_(inn, inn)], m[inn] * (-2 * np.pi ** 2 * ex[inn]))
            grid_err[k].append(np.abs(uu - ex).max())
close("Example 42.4.6: barycentric Poisson errors 2.6e-2, 6.5e-3, 1.6e-3 (relative)",
      np.array(grid_err["barycentric"]) / [2.6e-2, 6.5e-3, 1.6e-3], np.ones(3), tol=0.02)
check("Example 42.4.6: about twice the circumcentric errors",
      np.all(np.abs(np.array(grid_err["barycentric"]) / np.array(grid_err["circumcentric"]) - 2) < 0.05))

# ---------------------------------------------------------------- Example 42.4.7 and Figure 42.4.2
pw = {k: [] for k in ("barycentric", "circumcentric", "mixed")}
so = {k: [] for k in pw}
nob = []
for lv in range(1, 5):
    V, F = ch42.jittered_sphere(lv, s=0.15, seed=42)
    nob.append(int((dm.corner_angles(V, F).max(1) > np.pi / 2).sum()))
    L = dm.cotan_laplacian(V, F)
    z = V[:, 2]
    for k in pw:
        m = ch42.mass(V, F, k)
        pw[k].append(np.abs(L @ z / m + 2 * z).max())
        b = m * (-2 * z)
        b -= m * (b.sum() / m.sum())
        uu = np.linalg.solve(-L + np.outer(m, m), -b)
        e = uu - (z - (m @ z) / m.sum())
        so[k].append(np.sqrt((e ** 2 * m).sum() / m.sum()))
check("Example 42.4.7: obtuse triangles 4, 52, 168, 690", nob == [4, 52, 168, 690])
close("Example 42.4.7: pointwise errors (barycentric)", pw["barycentric"], [0.287, 0.351, 0.559, 0.523], tol=6e-4)
close("Example 42.4.7: pointwise errors (mixed)", pw["mixed"], [0.159, 0.218, 0.451, 0.566], tol=6e-4)
close("Example 42.4.7: pointwise errors (circumcentric)", pw["circumcentric"], [0.159, 0.197, 0.098, 0.061], tol=6e-4)
close("Figure 42.4.2: level-4 solve errors 8.7e-5, 2.8e-5, 6.2e-5 (relative)",
      np.array([so["barycentric"][-1], so["circumcentric"][-1], so["mixed"][-1]]) / [8.7e-5, 2.8e-5, 6.2e-5], np.ones(3), tol=0.02)
check("Example 42.4.7: level-1 solve errors near 8e-3 (7.8e-3 to 9.4e-3)",
      all(7.7e-3 < so[k][0] < 9.4e-3 for k in so))
check("Example 42.4.7: solve errors fall faster than mean edge length squared",
      all(so[k][0] / so[k][-1] > (0.58 / 0.076) ** 2 for k in so))

# ---------------------------------------------------------------- Proposition 42.4.8
for name, (V, F) in {"icosphere(2)": dm.icosphere(2), "jittered sphere(2)": ch42.jittered_sphere(2, s=0.15, seed=42),
                     "disk mesh": ch42.disk_mesh(6)}.items():
    M1 = ch42.whitney1_mass(V, F)
    D0 = dd.d0(F, len(V))
    L = dm.cotan_laplacian(V, F)
    off = np.abs(M1 - np.diag(np.diag(M1))).max()
    check(f"Proposition 42.4.8 on {name}: d0^T M1 d0 = -L although M1 is not diagonal (max off-diagonal {off:.3f})",
          np.abs(D0.T @ M1 @ D0 + L).max() < 1e-12 and off > 1e-3)
V, F = dm.icosphere(2)
M1 = ch42.whitney1_mass(V, F)
close("Section 42.4 (why): icosphere(2) max off-diagonal of M1 = 0.091", np.abs(M1 - np.diag(np.diag(M1))).max(), 0.091, tol=6e-4)

# ---------------------------------------------------------------- Implementation 42.4
ns, same, printed = ch42.run_printed(PAGE, "code-42-4-1", "out-42-4-1")
check("Implementation 42.4: code-42-4-1 prints exactly out-42-4-1", same, printed)
check("Implementation 42.4: code-42-4-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-42-4-1") <= 30)
Vt, Ft = dm.torus_mesh(2.0, 0.8, 10, 20)
Lt, bt, ct, mt = ns["cotan_and_areas"](Vt, Ft)
check("Implementation 42.4: cotan_and_areas also matches on a torus with obtuse triangles",
      np.allclose(Lt, dm.cotan_laplacian(Vt, Ft)) and np.allclose(mt, dd.dual_areas(Vt, Ft, "mixed"))
      and np.allclose(ct, dd.dual_areas(Vt, Ft, "circumcentric")))

# ---------------------------------------------------------------- exercises
V1 = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
close("Exercise 42.4.1: right isosceles triangle: barycentric 1/6; circumcentric = mixed = 1/4, 1/8, 1/8",
      [dd.dual_areas(V1, np.array([[0, 1, 2]]), k) for k in ("barycentric", "circumcentric", "mixed")],
      [[1 / 6] * 3, [0.25, 0.125, 0.125], [0.25, 0.125, 0.125]], tol=1e-14)
V2 = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0.0, 0.3, 0]])
F2 = np.array([[0, 1, 2]])
close("Exercise 42.4.2: circumcentric -0.304, -0.304, 0.908; mixed 0.075, 0.075, 0.15; barycentric 0.1",
      [dd.dual_areas(V2, F2, "circumcentric"), dd.dual_areas(V2, F2, "mixed"), dd.dual_areas(V2, F2, "barycentric")],
      [[-0.304, -0.304, 0.908], [0.075, 0.075, 0.15], [0.1, 0.1, 0.1]], tol=6e-4)
close("Exercise 42.4.2: theta_2 = 146.6 degrees, cot = -1.5167", [np.degrees(dm.corner_angles(V2, F2)[0, 2]),
                                                                    1 / np.tan(dm.corner_angles(V2, F2)[0, 2])],
      [146.6, -1.5167], tol=6e-2)
ok = True
for _ in range(20):
    p, q = rng.uniform(0.2, 3.0, size=2)
    Vr3 = np.array([[0.0, 0, 0], [p, 0, 0], [0.0, q, 0]])          # right angle at vertex 0
    A = 0.5 * p * q
    ok &= np.allclose(dd.dual_areas(Vr3, np.array([[0, 1, 2]]), "circumcentric"), [A / 2, A / 4, A / 4])
check("Exercise 42.4.4: right triangles get A/2, A/4, A/4 from the Voronoi pieces (20 random leg lengths)", ok)
ns5, same5, printed5 = ch42.run_printed(PAGE, "code-ex-42-4-5", "out-ex-42-4-5")
check("Exercise 42.4.5: code-ex-42-4-5 prints exactly out-ex-42-4-5", same5, printed5)
check("Exercise 42.4.5: code block is at most 30 lines", dgcode.code_lines(PAGE, "code-ex-42-4-5") <= 30)
V, F = ch42.jittered_sphere(4, s=0.15, seed=42)
close("Exercise 42.4.5: smallest circumcentric area at level 4 is 0.0023", dd.dual_areas(V, F, "circumcentric").min(), 0.0023,
      tol=6e-5)


def solid_angle(a, b, c):
    """signed solid angle of the spherical triangle spanned by the directions a, b, c (Van Oosterom-Strackee)."""
    a, b, c = (x / np.linalg.norm(x) for x in (a, b, c))
    return 2 * np.arctan2(a @ np.cross(b, c), 1 + a @ b + b @ c + c @ a)


ok_proj, ok_normal, ratio_err = True, True, []
for level in (1, 2, 3, 4):
    V, F = ch42.jittered_sphere(level, s=0.15, seed=42)
    C = dd.circumcenters(V, F)
    Nf = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
    ok_normal &= np.allclose(C / np.linalg.norm(C, axis=1, keepdims=True), Nf, atol=1e-12)
    proj = np.zeros(len(V))
    for f, tri in enumerate(F):
        for a in range(3):
            i, j, k = tri[a], tri[(a + 1) % 3], tri[(a + 2) % 3]
            proj[i] += solid_angle(V[i], 0.5 * (V[i] + V[j]), C[f]) + solid_angle(V[i], C[f], 0.5 * (V[i] + V[k]))
    ok_proj &= np.abs(proj - dm.angle_defect(V, F)).max() < 1e-13
    ratio_err.append(np.abs(proj / dd.dual_areas(V, F, "circumcentric") - 1).max())
check("Exercise 42.4.5: on the jittered sphere the circumcenter of each face points along the outward face normal",
      ok_normal)
check("Exercise 42.4.5: the radial projection of C_i to the sphere has area exactly delta_i (levels 1-4, 1e-13)", ok_proj)
close("Exercise 42.4.5: so K_i - 1 is the area ratio of C_i and its projection (level 4: 0.003)", ratio_err[-1], 0.003,
      tol=5e-4)

summary()
