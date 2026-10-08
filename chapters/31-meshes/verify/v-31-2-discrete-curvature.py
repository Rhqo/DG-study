"""Verification for Section 31.2 (Discrete Curvature).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/verify/v-31-2-discrete-curvature.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

import dgsym  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


# ---------------------------------------------------------------- Example 31.2.2 / Figure 31.2.1(a)
lon = np.radians([0, 25, 50, 75, 160, 235, 300])
col = np.array([0.40, 0.30, 0.38, 0.30, 0.62, 0.50, 0.55])
P = np.stack([np.sin(col) * np.cos(lon), np.sin(col) * np.sin(lon), np.cos(col)], 1)
V = np.vstack([[0, 0, 1.0], P])
F = np.array([[0, 1 + k, 1 + (k + 1) % 7] for k in range(7)])
tilt = {k: np.degrees(np.arccos(dm.vertex_normals(V, F, k)[0][2])) for k in ("uniform", "area", "angle")}
close("Example 31.2.2: tilt of the uniform normal = 1.24 deg", round(tilt["uniform"], 2), 1.24, tol=1e-9)
close("Example 31.2.2: tilt of the area-weighted normal = 6.50 deg", round(tilt["area"], 2), 6.50, tol=1e-9)
close("Example 31.2.2: tilt of the angle-weighted normal = 4.22 deg", round(tilt["angle"], 2), 4.22, tol=1e-9)
ang0 = dm.corner_angles(V, F)[:, 0]
close("Example 31.2.2: angle defect at the pole = 10.58 deg", round(np.degrees(2 * np.pi - ang0.sum()), 2), 10.58,
      tol=1e-9)
# area-weighted normal = normalized sum of cross products (the PyTorch3D formula)
cr = dm.face_cross(V, F).sum(0)
close("Def 31.2.1: area weights = sum of face cross products (PyTorch3D Meshes._compute_vertex_normals)",
      cr / np.linalg.norm(cr), dm.vertex_normals(V, F, "area")[0], tol=1e-14)

# re-triangulations of the same piecewise-flat surface (Proposition 31.2.3)
f = 4
a, b = F[f, 1], F[f, 2]
# (i) split face f through the vertex 0: new vertex m on the opposite edge ab; the neighbour across ab is not in the star
m = (V[a] + V[b]) / 2
V2 = np.vstack([V, m])
F2 = np.vstack([np.delete(F, f, 0), [[0, a, 8], [0, 8, b]]])
# (ii) cut the corner: new vertices p on 0a and q on 0b; split every face that contains edge 0a or 0b
p, q = V[0] + 0.5 * (V[a] - V[0]), V[0] + 0.5 * (V[b] - V[0])
V3 = np.vstack([V, p, q])
ip, iq = 8, 9
F3 = []
for tri in F:
    t = list(tri)
    if set(t) == {0, a, b}:
        F3 += [[0, ip, iq], [ip, a, b], [ip, b, iq]]
    elif 0 in t and a in t:   # neighbour (0, a_prev, a)
        o = [x for x in t if x not in (0, a)][0]
        F3 += [[0, o, ip], [ip, o, a]]
    elif 0 in t and b in t:   # neighbour (0, b, b_next)
        o = [x for x in t if x not in (0, b)][0]
        F3 += [[0, iq, o], [iq, b, o]]
    else:
        F3.append(t)
F3 = np.array(F3)
for kind, ch1, ch2 in (("uniform", True, False), ("area", False, True), ("angle", False, False)):
    n0 = dm.vertex_normals(V, F, kind)[0]
    d1 = angle_between(n0, dm.vertex_normals(V2, F2, kind)[0])
    d2 = angle_between(n0, dm.vertex_normals(V3, F3, kind)[0])
    check(f"Prop 31.2.3: {kind} normal {'changes' if ch1 else 'unchanged'} when a face is split through the vertex"
          f" ({d1:.3f} deg)", (d1 > 0.1) == ch1 and (ch1 or d1 < 1e-9))
    check(f"Prop 31.2.3: {kind} normal {'changes' if ch2 else 'unchanged'} when the corner is cut ({d2:.3f} deg)",
          (d2 > 0.1) == ch2 and (ch2 or d2 < 1e-9))
close("Example 31.2.2: the uniform normal moves 2.61 deg under the split", round(angle_between(
    dm.vertex_normals(V, F, "uniform")[0], dm.vertex_normals(V2, F2, "uniform")[0]), 2), 2.61, tol=1e-9)
close("Example 31.2.2: the area-weighted normal moves 5.91 deg under the corner cut", round(angle_between(
    dm.vertex_normals(V, F, "area")[0], dm.vertex_normals(V3, F3, "area")[0]), 2), 5.91, tol=1e-9)
check("Prop 31.2.3: the corner cut keeps the same surface (all new triangles lie in old faces)",
      np.isclose(dm.face_areas(V3, F3).sum(), dm.face_areas(V, F).sum()))

# ---------------------------------------------------------------- Figure 31.2.1(b): jittered icospheres
rng = np.random.default_rng(1)
errs = {k: [] for k in ("uniform", "area", "angle")}
hs = []
for lv in range(1, 6):
    V0, F0 = dm.icosphere(lv)
    h = np.mean([np.linalg.norm(V0[a_] - V0[b_]) for a_, b_ in dm.edges(F0)])
    J = V0 + 0.1 * h * rng.normal(size=V0.shape)
    J /= np.linalg.norm(J, axis=1, keepdims=True)
    hs.append(h)
    for k in errs:
        N = dm.vertex_normals(J, F0, k)
        errs[k].append(np.degrees(np.arccos(np.clip((N * J).sum(1), -1, 1))).mean())
for k, e in errs.items():
    rate = np.log(e[-2] / e[-1]) / np.log(hs[-2] / hs[-1])
    check(f"Fig 31.2.1(b): {k} normal error decreases like h (rate {rate:.2f})", 0.8 < rate < 1.2)
check("Fig 31.2.1(b): at level 4 the errors are 0.26 (angle) < 0.41 (uniform) < 0.64 (area) degrees",
      [round(errs[k][3], 2) for k in ("angle", "uniform", "area")] == [0.26, 0.41, 0.64])


# Insight box: with folded (flipped) slivers the angle weights can fail badly, the area weights do not
rng2 = np.random.default_rng(1)
for lv in range(1, 5):
    V0, F0 = dm.icosphere(lv)
    h = np.mean([np.linalg.norm(V0[a_] - V0[b_]) for a_, b_ in dm.edges(F0)])
    Jb = V0 + 0.2 * h * rng2.normal(size=V0.shape)
    Jb /= np.linalg.norm(Jb, axis=1, keepdims=True)
flips = (np.einsum("ij,ij->i", dm.face_normals(Jb, F0), Jb[F0].mean(1)) < 0).sum()
mx = {k: np.degrees(np.arccos(np.clip((dm.vertex_normals(Jb, F0, k) * Jb).sum(1), -1, 1))).max()
      for k in ("uniform", "area", "angle")}
check(f"Insight box: jitter 0.2 h, level 4: {flips} folded triangles; max error angle {mx['angle']:.1f}, "
      f"uniform {mx['uniform']:.1f}, area {mx['area']:.1f} deg",
      flips == 10 and round(mx["angle"], 1) == 92.9 and round(mx["uniform"], 1) == 8.1 and round(mx["area"], 1) == 4.2)

# ---------------------------------------------------------------- Theorem 31.2.4: discrete Gauss-Bonnet
n0, n1, n2 = sp.symbols("n0 n1 n2")
sym_equal("Thm 31.2.4 (derivation): 2 pi n0 - pi n2 = 2 pi (n0 - n1 + n2) when 3 n2 = 2 n1",
          (2 * sp.pi * n0 - sp.pi * n2 - 2 * sp.pi * (n0 - n1 + n2)).subs(n1, sp.Rational(3, 2) * n2), 0)
for name, (V_, F_), chi in [("icosphere 3", dm.icosphere(3), 2), ("torus 12x24", dm.torus_mesh(2, 0.8, 12, 24), 0),
                            ("genus 2", dm.genus2_mesh(h=0.16), -2)]:
    close(f"Thm 31.2.4: sum of angle defects = 2 pi chi on {name}", dm.angle_defect(V_, F_).sum(), 2 * np.pi * chi,
          tol=1e-9)
# random positions, same combinatorics: still exact
V_, F_ = dm.icosphere(2)
V_ = V_ * (1 + 0.3 * np.random.default_rng(5).uniform(size=(len(V_), 1)))
close("Thm 31.2.4: exact for any vertex positions (randomly scaled icosphere)", dm.angle_defect(V_, F_).sum(),
      4 * np.pi, tol=1e-9)
# with boundary (Exercise 31.2.4): sum_int delta + sum_bnd (pi - angles) = 2 pi chi
Vi, Fi = dm.icosphere(1)
close("Exercise 31.2.4: icosphere minus a face, chi = 1", dm.angle_defect(Vi, Fi[1:]).sum(), 2 * np.pi, tol=1e-9)
Vm, Fm = dm.mobius_mesh(n=12)
close("Exercise 31.2.4: Moebius strip, chi = 0 (orientability not needed)", dm.angle_defect(Vm, Fm).sum(), 0.0,
      tol=1e-9)
tri_V = np.array([[0, 0, 0], [2.0, 0.3, 0.1], [0.4, 1.7, -0.2]])
close("Exercise 31.2.4: a single triangle: sum (pi - theta) = 2 pi", dm.angle_defect(tri_V, np.array([[0, 1, 2]])).sum(),
      2 * np.pi, tol=1e-12)

# Figure 31.2.2: unfolding
t = 2 * np.pi * np.arange(6) / 6
for name, z, val in (("cone", np.full(6, -0.55), 0.84), ("flat", np.zeros(6), 0.0),
                     ("saddle", 0.32 * (-1.0) ** np.arange(6), -0.93)):
    Pz = np.stack([np.cos(t), np.sin(t), z], 1)
    Vz = np.vstack([[0, 0, 0], Pz])
    Fz = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
    dz = 2 * np.pi - dm.corner_angles(Vz, Fz)[:, 0].sum()
    close(f"Fig 31.2.2: {name} star has delta = {val}", round(dz, 2) + 0.0, val, tol=1e-12)

# ---------------------------------------------------------------- Figure 31.2.3 / 31.2.4: torus (running example)
ex = dgsym.EXAMPLES["torus"]
u_, v_ = ex["coords"]
Rs, rs = ex["params"]
Ksym, Hsym = dgsym.K_H(ex["expr"], u_, v_, positive=ex["positive"])
sym_equal("Fig 31.2.3: K of the torus = cos u / (r (R + r cos u)) (dgsym, GUIDELINES 7)", Ksym, ex["expected"]["K"],
          {u_: (0, 2 * np.pi), Rs: (1.1, 2), rs: (0.2, 0.9)})
H_out = -(Rs + 2 * rs * sp.cos(u_)) / (2 * rs * (Rs + rs * sp.cos(u_)))
sym_equal("Fig 31.2.4: H of the torus with x_u x x_v (inward) is (R + 2 r cos u)/(2r(R + r cos u)); outward = minus",
          -Hsym, H_out, {u_: (0, 2 * np.pi), Rs: (1.1, 2), rs: (0.2, 0.9)})
R, r = 2.0, 0.8
for (nu, nv), tolK, tolH in (((12, 24), 0.03, 0.02), ((24, 48), 0.008, 0.005)):
    V_, F_ = dm.torus_mesh(R, r, nu, nv)
    u = np.repeat(2 * np.pi * np.arange(nu) / nu, nv)
    d = dm.angle_defect(V_, F_)
    A = dm.mass_barycentric(V_, F_)
    Kex = np.cos(u) / (r * (R + r * np.cos(u)))
    Hex = -(R + 2 * r * np.cos(u)) / (2 * r * (R + r * np.cos(u)))
    errK = np.abs(d / A - Kex).max()
    errH = np.abs(dm.mean_curvature_dihedral(V_, F_) - Hex).max()
    check(f"Fig 31.2.3: torus {nu}x{nv}: max |delta_i/A_i - K| = {errK:.4f} < {tolK}", errK < tolK)
    check(f"Fig 31.2.4: torus {nu}x{nv}: max |H_i - H| = {errH:.4f} < {tolH}", errH < tolH)
    check(f"Fig 31.2.3: torus {nu}x{nv}: |sum delta| < 1e-11", abs(d.sum()) < 1e-11)
    check(f"Fig 31.2.3: torus {nu}x{nv}: outer vertices have delta > 0, inner delta < 0",
          (d[np.cos(u) > 0.05] > 0).all() and (d[np.cos(u) < -0.05] < 0).all())

# ---------------------------------------------------------------- Proposition 31.2.6: Steiner formula
eps = sp.symbols("epsilon", positive=True)
k1, k2 = sp.symbols("kappa1 kappa2", real=True)
W = sp.diag(k1, k2)
sym_equal("Prop 31.2.6 (smooth): area factor of x + eps N is det(I - eps W) = 1 - 2 H eps + K eps^2",
          (sp.eye(2) - eps * W).det(), 1 - 2 * ((k1 + k2) / 2) * eps + k1 * k2 * eps ** 2)
rr = sp.symbols("r", positive=True)
sym_equal("Prop 31.2.6 (smooth): sphere 4 pi (r + eps)^2 = A - 2 eps int H + eps^2 int K with H = -1/r",
          4 * sp.pi * (rr + eps) ** 2, 4 * sp.pi * rr ** 2 - 2 * eps * (-1 / rr) * 4 * sp.pi * rr ** 2
          + eps ** 2 * 4 * sp.pi)


def spherical_polygon_area(N):
    """Area of the spherical polygon with unit vertices N (in order), by fan triangulation from N[0]."""
    tot = 0.0
    for k in range(1, len(N) - 1):
        a_, b_, c_ = N[0], N[k], N[k + 1]
        num = abs(np.dot(a_, np.cross(b_, c_)))
        den = 1 + np.dot(a_, b_) + np.dot(b_, c_) + np.dot(c_, a_)
        tot += 2 * np.arctan2(num, den)
    return tot


# convex polyhedra: area of the Gauss image of a vertex (normals of the faces around it) = angle defect
for name, (Vp, Fp) in (("icosahedron", dm.icosahedron()),
                       ("tetrahedron", (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float),
                                        np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])))):
    if dm.signed_volume(Vp, Fp) < 0:
        Fp = Fp[:, ::-1]
    fn = dm.face_normals(Vp, Fp)
    dlt = dm.angle_defect(Vp, Fp)
    ok = True
    for i in range(len(Vp)):
        # faces around i in cyclic order: follow the link
        faces = [f_ for f_ in range(len(Fp)) if i in Fp[f_]]
        order = [faces[0]]
        while len(order) < len(faces):
            last = list(Fp[order[-1]])
            k = last.index(i)
            nxt_v = last[(k + 2) % 3]  # previous vertex in this face -> shared with the next face
            cand = [f_ for f_ in faces if f_ not in order and nxt_v in Fp[f_]]
            order.append(cand[0])
        ok &= abs(spherical_polygon_area(fn[order]) - dlt[i]) < 1e-12
    check(f"Prop 31.2.6 (derivation): on the {name} the Gauss image of each vertex has area = angle defect", ok)
    be = dm.edge_bending_angles(Vp, Fp)
    S = sum(t_ * l_ for t_, l_ in be.values())
    for e_ in (0.1, 0.5):
        Acurv = dm.face_areas(Vp, Fp).sum() + e_ * S + e_ ** 2 * dlt.sum()
        # independent: area of the offset = derivative of the offset volume, which equals
        # Vol + A e + (S/2) e^2 + (sum delta / 3) e^3 for a convex polyhedron; check via finite difference
        vol = lambda x: abs(dm.signed_volume(Vp, Fp)) + dm.face_areas(Vp, Fp).sum() * x + S / 2 * x ** 2 + dlt.sum() / 3 * x ** 3  # noqa: E731
        close(f"Prop 31.2.6: {name}, eps = {e_}: A + eps S + eps^2 sum delta = d/deps Vol(eps)",
              Acurv, (vol(e_ + 1e-6) - vol(e_ - 1e-6)) / 2e-6, tol=1e-6)
# the cube of side 2: Steiner area agrees with the rounded cube of Section 31.1
Vc = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
Fc = np.array([t_ for a_, b_, c_, d_ in quads for t_ in ([a_, b_, c_], [a_, c_, d_])])
Fc, _, _ = dm.orient(Fc)
if dm.signed_volume(Vc, Fc) < 0:
    Fc = Fc[:, ::-1]
Sc = sum(t_ * l_ for t_, l_ in dm.edge_bending_angles(Vc, Fc).values())
close("Example 31.2.7: cube of side 2: sum theta_e l_e = 12 pi", Sc, 12 * np.pi, tol=1e-12)
close("Example 31.2.7: cube: int H = -(1/2) sum theta l = -6 pi (rounded cube, Section 31.1)", -Sc / 2, -6 * np.pi,
      tol=1e-12)
for e_ in (0.25, 0.7):
    close(f"Example 31.2.7: cube Steiner area at eps = {e_} = 24 + 12 pi eps + 4 pi eps^2",
          24 + e_ * Sc + e_ ** 2 * dm.angle_defect(Vc, Fc).sum(), 24 + 12 * np.pi * e_ + 4 * np.pi * e_ ** 2, tol=1e-12)
# icosphere: (1/2) sum theta l -> 4 pi = -int H dA on the unit sphere
vals = []
for lv in range(0, 6):
    Vs, Fs = dm.icosphere(lv)
    vals.append(sum(t_ * l_ for t_, l_ in dm.edge_bending_angles(Vs, Fs).values()) / 2)
check("Example 31.2.7: (1/2) sum theta l on icospheres increases to 4 pi: 11.51, 12.26, 12.49, 12.55, 12.56, 12.57",
      [round(x, 2) for x in vals] == [11.51, 12.26, 12.49, 12.55, 12.56, 12.57] and vals[-1] < 4 * np.pi)
Vs, Fs = dm.icosphere(4)
Hs = dm.mean_curvature_dihedral(Vs, Fs)
check("Example 31.2.7: icosphere level 4: H_i in [-1.146, -0.997], mean -1.001 (exact -1)",
      round(Hs.min(), 3) == -1.146 and round(Hs.max(), 3) == -0.998 and round(Hs.mean(), 3) == -1.001)

# Exercise 31.2.2: regular octahedron
Vo = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
Fo = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
if dm.signed_volume(Vo, Fo) < 0:
    Fo = Fo[:, ::-1]
close("Exercise 31.2.2: octahedron angle defect 2 pi / 3 at every vertex", dm.angle_defect(Vo, Fo), np.full(6, 2 * np.pi / 3),
      tol=1e-12)
th_o = [t_ for t_, _ in dm.edge_bending_angles(Vo, Fo).values()]
close("Exercise 31.2.2: octahedron bending angle = arccos(1/3) = 70.53 deg", np.array(th_o), np.full(12, np.arccos(1 / 3)),
      tol=1e-12)
close("Exercise 31.2.2: octahedron: -(1/2) sum theta l = -6 sqrt2 arccos(1/3)",
      -0.5 * sum(t_ * l_ for t_, l_ in dm.edge_bending_angles(Vo, Fo).values()), -6 * np.sqrt(2) * np.arccos(1 / 3),
      tol=1e-12)

# ---------------------------------------------------------------- Example 31.2.8: the 4-8 cylinder
def cyl48(m, nrows=12):
    hu = 2 * np.sin(np.pi / m)
    u = 2 * np.pi * np.arange(m) / m
    z = hu * np.arange(nrows)
    Vy = np.array([[np.cos(a_), np.sin(a_), zz] for a_ in u for zz in z])
    idx = lambda i, j: (i % m) * nrows + j  # noqa: E731
    Fy = []
    for i in range(m):
        for j in range(nrows - 1):
            a_, b_, c_, d_ = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            Fy += [[a_, b_, c_], [a_, c_, d_]] if (i + j) % 2 == 0 else [[a_, b_, d_], [b_, c_, d_]]
    Fy = np.array(Fy)
    fn = dm.face_normals(Vy, Fy)
    cen = Vy[Fy].mean(1)
    if ((fn[:, :2] * cen[:, :2]).sum(1) > 0).mean() < 0.5:
        Fy = Fy[:, ::-1]
    return Vy, Fy, nrows


for m in (16, 64):
    Vy, Fy, nr = cyl48(m)
    deg = np.array([len(x) for x in dm.neighbors(Fy)])
    inter = np.array([(k % nr) not in (0, nr - 1) for k in range(len(Vy))])
    LX = dm.cotan_laplacian(Vy, Fy) @ Vy
    Ab, Av = dm.mass_barycentric(Vy, Fy), dm.mass_voronoi(Vy, Fy)
    Hb = np.linalg.norm(LX, axis=1) / (2 * Ab)
    Hv = np.linalg.norm(LX, axis=1) / (2 * Av)
    Hd = np.abs(dm.mean_curvature_dihedral(Vy, Fy))
    close(f"Example 31.2.8: 4-8 cylinder m = {m}: |H| (cotan, barycentric) = 0.75 at valence 4",
          Hb[inter & (deg == 4)].mean(), 0.75, tol=1e-12)
    close(f"Example 31.2.8: 4-8 cylinder m = {m}: |H| (cotan, barycentric) = 0.375 at valence 8",
          Hb[inter & (deg == 8)].mean(), 0.375, tol=1e-12)
    close(f"Example 31.2.8: 4-8 cylinder m = {m}: |H| (cotan, Voronoi) = 0.5 at both", np.r_[Hv[inter]].std() + Hv[inter].mean(),
          0.5, tol=1e-12)
    check(f"Example 31.2.8: 4-8 cylinder m = {m}: dihedral formula with barycentric A_i -> 0.75 / 0.375",
          abs(Hd[inter & (deg == 4)].mean() - 0.75) < 0.006 and abs(Hd[inter & (deg == 8)].mean() - 0.375) < 0.003)
    hu = 2 * np.sin(np.pi / m)
    close(f"Exercise 31.2.5: m = {m}: barycentric areas 2h^2/3 (valence 4) and 4h^2/3 (valence 8)",
          np.array([Ab[inter & (deg == 4)].mean(), Ab[inter & (deg == 8)].mean()]),
          np.array([2 * hu ** 2 / 3, 4 * hu ** 2 / 3]), tol=1e-12)
    close(f"Exercise 31.2.5: m = {m}: Voronoi area h^2 at both kinds", Av[inter], np.full(inter.sum(), hu ** 2), tol=1e-12)

# Example 31.2.8: [Hildebrandt06, Example 1] divides by the consistent (finite element) mass matrix
# M_pq = int phi_p phi_q dA restricted to the interior vertices instead of a lumped vertex area. Then the mean
# curvature at valence-4 vertices tends to 3 times the exact value and at valence-8 vertices to 0 (review R1).
def consistent_mass(V_, F_):
    Mc = np.zeros((len(V_), len(V_)))
    ar = dm.face_areas(V_, F_)
    for f_, t_ in enumerate(F_):
        for a_ in range(3):
            for b_ in range(3):
                Mc[t_[a_], t_[b_]] += ar[f_] / (6 if a_ == b_ else 12)
    return Mc


for m in (16, 64):
    nr = m // 2 + 1
    Vy, Fy, _ = cyl48(m, nr)
    deg = np.array([len(x) for x in dm.neighbors(Fy)])
    inter = np.array([(k % nr) not in (0, nr - 1) for k in range(len(Vy))])
    z = Vy[:, 2]
    mid = inter & (np.abs(z - z[inter].mean()) < 0.25 * np.ptp(z))
    Mc = consistent_mass(Vy, Fy)
    Hc = np.zeros_like(Vy)
    Hc[inter] = np.linalg.solve(Mc[np.ix_(inter, inter)], (dm.cotan_laplacian(Vy, Fy) @ Vy)[inter]) / 2
    radial = -(Hc[:, :2] * Vy[:, :2]).sum(1)          # |H| along the inward direction; exact value 1/2
    r4, r8 = radial[mid & (deg == 4)].mean(), radial[mid & (deg == 8)].mean()
    check(f"Example 31.2.8 / [Hildebrandt06]: consistent mass, m = {m}: |H| = {r4:.3f} (valence 4) and {r8:.3f} "
          f"(valence 8) -> 3 x 0.5 and 0", abs(r4 - 1.5) < 0.04 * 16 / m and abs(r8) < 0.04 * 16 / m)

# ---------------------------------------------------------------- Figure 31.2.5: pointwise vs integrated
def jtorus(nu, nv, jit):
    V_, F_ = dm.torus_mesh(R, r, nu, nv)
    rg = np.random.default_rng(0)
    u = np.repeat(2 * np.pi * np.arange(nu) / nu, nv)
    v = np.tile(2 * np.pi * np.arange(nv) / nv, nu)
    u = u + jit * (2 * np.pi / nu) * rg.uniform(-1, 1, u.shape)
    v = v + jit * (2 * np.pi / nv) * rg.uniform(-1, 1, v.shape)
    V_ = np.stack([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)], 1)
    return V_, F_, u


uu, vv = sp.symbols("u v", real=True)
weakK = sp.integrate(sp.integrate(sp.cos(uu) / (r * (R + r * sp.cos(uu))) * sp.cos(uu) * r * (R + r * sp.cos(uu)),
                                  (uu, 0, 2 * sp.pi)), (vv, 0, 2 * sp.pi))
close("Fig 31.2.5: int K cos u dA = 2 pi^2 on T_{R,r}", float(weakK), 2 * np.pi ** 2, tol=1e-9)
weakH = -2 * np.pi * np.pi * r   # int H cos u dA with the outward normal = -2 pi^2 r
res = {}
for jit in (0.0, 0.2):
    rows = []
    for nu, nv in ((12, 24), (24, 48), (48, 96), (96, 192)):
        V_, F_, u = jtorus(nu, nv, jit)
        d = dm.angle_defect(V_, F_)
        A = dm.mass_barycentric(V_, F_)
        Hd = dm.mean_curvature_dihedral(V_, F_)
        Kex = np.cos(u) / (r * (R + r * np.cos(u)))
        Hex = -(R + 2 * r * np.cos(u)) / (2 * r * (R + r * np.cos(u)))
        rows.append((np.abs(d / A - Kex).max(), abs((d * np.cos(u)).sum() - 2 * np.pi ** 2),
                     np.abs(Hd - Hex).max(), abs((Hd * A * np.cos(u)).sum() - weakH)))
    res[jit] = np.array(rows)
jt, rg_ = res[0.2], res[0.0]
check("Fig 31.2.5: jittered torus: pointwise max error of K grows 0.20 -> 0.30 (does not converge)",
      [round(x, 2) for x in jt[:, 0]] == [0.20, 0.22, 0.26, 0.30])
check("Fig 31.2.5: jittered torus: pointwise max error of H does not decrease (0.18 -> 0.27)",
      jt[-1, 2] > jt[0, 2] and round(jt[0, 2], 2) == 0.18 and round(jt[-1, 2], 2) == 0.27)
for col_, name in ((1, "K"), (3, "H")):
    for lab, arr in (("regular", rg_), ("jittered", jt)):
        ratios = arr[:-1, col_] / arr[1:, col_]
        check(f"Fig 31.2.5: {lab}: integrated error of {name} drops by about 4 per refinement {np.round(ratios, 2)}",
              (ratios > 3.5).all() and (ratios < 4.3).all())
check("Fig 31.2.5: regular torus: pointwise error of K drops by about 4 per refinement",
      (rg_[:-1, 0] / rg_[1:, 0] > 3.5).all())

summary()
