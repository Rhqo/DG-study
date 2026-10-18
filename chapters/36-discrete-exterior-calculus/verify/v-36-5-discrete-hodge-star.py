"""Verification for Section 36.5 (Dual Forms and the Discrete Hodge Star).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/verify/v-36-5-discrete-hodge-star.py``
"""

import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch36  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "36-5-discrete-hodge-star.html"
rng = np.random.default_rng(3605)
J = np.array([[0.0, -1.0], [1.0, 0.0]])

# a jittered acute planar patch (as in Figure 36.5.3(a))
rng_fig = np.random.default_rng(365)
VP, FP = ch36.equilateral_patch(2)
VP = VP + 0.09 * rng_fig.uniform(-1, 1, size=VP.shape)
EP, idxP = dd.edge_list(FP)

# ---------------------------------------------------------------- Remark 36.5.1
D1P = dd.d1(FP)
ok = True
for f, tri in enumerate(FP):
    for c in range(3):
        a, b, k = int(tri[c]), int(tri[(c + 1) % 3]), int(tri[(c + 2) % 3])
        i, j = min(a, b), max(a, b)
        left = ch36.cross2(VP[j] - VP[i], VP[k] - VP[i]) > 0
        ok &= (D1P[f, idxP[(i, j)]] == 1) == left
check("Remark 36.5.1: o(f, e) = +1 exactly for the face on the left of i -> j (counterclockwise planar mesh)", ok)
polys = ch36.dual_edge_polylines(VP, FP)
ok = all((P[2] - P[0]) @ (J @ (VP[j] - VP[i])) > 0 for P, (i, j) in zip(polys, EP))
check("Remark 36.5.1: every dual edge c_g -> c_f points along J(x_j - x_i) on an acute mesh", ok)

# ---------------------------------------------------------------- Proposition 36.5.5
for name, (Vm, Fm) in (("icosphere(1)", dm.icosphere(1)), ("torus 8x12", dm.torus_mesh(2, 0.8, 8, 12))):
    Em, idxm = dd.edge_list(Fm)
    D0m, D1m = dd.d0(Fm, len(Vm)), dd.d1(Fm)
    lr = ch36.left_right_faces(Fm)
    dual_d0 = np.zeros((len(Em), len(Fm)))
    for e, (g, f) in enumerate(lr):
        dual_d0[e, f] += 1.0
        dual_d0[e, g] -= 1.0                                      # boundary of e* = [f*] - [g*]
    # orientation of e* in the counterclockwise boundary of i*, read geometrically (outward normals)
    cent = Vm[Fm].mean(1)
    nrm = dm.vertex_normals(Vm, Fm, kind="area")
    dual_d1 = np.zeros((len(Vm), len(Em)))
    for e, (i, j) in enumerate(Em):
        g, f = lr[e]
        direction = cent[f] - cent[g]
        for v in (i, j):
            mid = 0.5 * (cent[f] + cent[g])
            tang = np.cross(nrm[v], mid - Vm[v])                   # counterclockwise around v
            dual_d1[v, e] = np.sign(direction @ tang)
    check(f"Proposition 36.5.5: dual d0 (from Remark 36.5.1) = d1^T on {name}", np.array_equal(dual_d0, D1m.T))
    check(f"Proposition 36.5.5: dual d1 (geometric orientation of i*) = -d0^T on {name}",
          np.array_equal(dual_d1, -D0m.T))
    check(f"Proposition 36.5.5: lib dual_d0, dual_d1 agree and dual_d1 dual_d0 = 0 on {name}",
          np.array_equal(ch36.dual_d0(Fm), D1m.T) and np.array_equal(ch36.dual_d1(Fm, len(Vm)), -D0m.T)
          and not np.any(ch36.dual_d1(Fm, len(Vm)) @ ch36.dual_d0(Fm)))
# with boundary (paragraph after Proposition 36.5.5): (36.5.1) holds on the rows of interior edges and interior vertices
D0P = dd.d0(FP, len(VP))
lrP = ch36.left_right_faces(FP)
int_e = (lrP >= 0).all(1)
bnd_v = set(np.unique(dm.boundary_edges(FP)).tolist())
ok_d0 = all(D1P[f, e] == 1 and D1P[g, e] == -1 and np.count_nonzero(D1P[:, e]) == 2
            for e, (g, f) in enumerate(lrP) if int_e[e])
ok_d1 = True
for v in range(len(VP)):
    if v in bnd_v:
        continue
    fs = [f for f in range(len(FP)) if v in FP[f]]
    fs.sort(key=lambda f: np.arctan2(*(VP[FP[f]].mean(0) - VP[v])[::-1]))   # counterclockwise around v
    row = np.zeros(len(EP))
    for a_, b_ in zip(fs, fs[1:] + fs[:1]):
        (j,) = set(FP[a_].tolist()) & set(FP[b_].tolist()) - {v}
        e = idxP[(min(v, j), max(v, j))]
        row[e] += 1 if (lrP[e, 0], lrP[e, 1]) == (a_, b_) else -1        # e* runs right -> left
    ok_d1 &= np.array_equal(row, -D0P[:, v])
check("Proposition 36.5.5 (with boundary): d1^T on interior-edge rows, -d0^T on interior-vertex rows (planar patch)",
      ok_d0 and ok_d1 and int_e.sum() < len(EP) and len(bnd_v) > 0)

# ---------------------------------------------------------------- Example 36.5.7, Non-example 36.5.8
ell = 1.7
Ve, Fe = ch36.equilateral_triangle(ell)
he = dd.dual_edge_lengths(Ve, Fe)
close("Example 36.5.7(a): equilateral triangle, h = l / (2 sqrt 3) for every edge", he, np.full(3, ell / (2 * np.sqrt(3))),
      tol=1e-12)
Vr = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
Er, idxr = dd.edge_list(np.array([[0, 1, 2]]))
hr = dd.dual_edge_lengths(Vr, np.array([[0, 1, 2]]))
close("Example 36.5.7(b): right triangle, h = 0 on the hypotenuse [1, 2]; circumcenter = its midpoint",
      [hr[idxr[(1, 2)]], *(dd.circumcenters(Vr, np.array([[0, 1, 2]]))[0] - np.array([1.0, 0.5]))], [0, 0, 0], tol=1e-14)
Vo = np.array([[-1.0, 0.0], [1.0, 0.0], [0.3, 0.4]])
co = dd.circumcenters(Vo, np.array([[0, 1, 2]]))[0]
close("Example 36.5.7(c): obtuse example, c = (0, -0.9375), h = -0.9375",
      [*co, dd.dual_edge_lengths(Vo, np.array([[0, 1, 2]]))[0]], [0, -0.9375, -0.9375], tol=1e-12)
bfv = Vr.mean(0) - np.array([1.0, 0.0])
check("Non-example 36.5.8: barycenter - midpoint = (-1/3, 1/3), at 45 degrees to the base edge",
      np.allclose(bfv, [-1 / 3, 1 / 3]) and abs(np.degrees(np.arctan2(bfv[1], -bfv[0])) - 45) < 1e-12)

# ---------------------------------------------------------------- Lemma 36.5.9 (sympy + random triangles)
a, p1 = sp.symbols("a p1", real=True)
p2 = sp.symbols("p2", positive=True)
c = (p1 ** 2 + p2 ** 2 - a ** 2) / (2 * p2)
check("Lemma 36.5.9: c = (p1^2 + p2^2 - a^2) / (2 p2) is equidistant from (-a, 0), (a, 0), p (sympy)",
      sp.simplify((a ** 2 + c ** 2) - (p1 ** 2 + (p2 - c) ** 2)) == 0)
u = sp.Matrix([-a - p1, -p2])
w = sp.Matrix([a - p1, -p2])
cot = (u.dot(w)) / (u[0] * w[1] - u[1] * w[0])
check("Lemma 36.5.9: cot(theta) = <u, w> / (u1 w2 - u2 w1) = (p1^2 + p2^2 - a^2) / (2 a p2), so c = a cot(theta)",
      sp.simplify(u[0] * w[1] - u[1] * w[0] - 2 * a * p2) == 0 and sp.simplify(a * cot - c) == 0)
ok = True
for _ in range(200):
    T = rng.normal(size=(3, 2))
    if ch36.cross2(T[1] - T[0], T[2] - T[0]) < 0:
        T = T[[0, 2, 1]]
    C = dd.circumcenters(T, np.array([[0, 1, 2]]))[0]
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        m = 0.5 * (T[i] + T[j])
        e = T[j] - T[i]
        nrm_ = J @ e / np.linalg.norm(e)
        side = np.sign(nrm_ @ (T[k] - m))
        h = side * (nrm_ @ (C - m))
        u_, w_ = T[i] - T[k], T[j] - T[k]
        th = np.arccos(u_ @ w_ / np.linalg.norm(u_) / np.linalg.norm(w_))
        ok &= abs((C - m) @ e) < 1e-9 * (1 + np.abs(C).max()) and abs(h - 0.5 * np.linalg.norm(e) / np.tan(th)) < 1e-8 * (
            1 + abs(h))
check("Lemma 36.5.9: (c - m) perpendicular to e and h = |e|/2 cot(theta) on 200 random triangles (all three edges)", ok)

# ---------------------------------------------------------------- Theorem 36.5.10, Corollary 36.5.11
Vn, Fn = ch36.non_delaunay_pair()
for name, (Vm, Fm) in (("jittered patch", (VP, FP)), ("non-Delaunay pair", (Vn, Fn)), ("icosphere(2)", dm.icosphere(2)),
                       ("random grid", (ch36.grid(5)[0] + 0.06 * rng.normal(size=(36, 2)), ch36.grid(5)[1]))):
    Em, idxm = dd.edge_list(Fm)
    V3 = np.c_[Vm, np.zeros(len(Vm))] if Vm.shape[1] == 2 else Vm
    ang = dm.corner_angles(V3, Fm)
    cots = {}
    for f, tri in enumerate(Fm):
        for k in range(3):
            i, j = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            cots.setdefault((min(i, j), max(i, j)), []).append(1 / np.tan(ang[f, k]))
    target = np.array([0.5 * sum(cots[(int(i), int(j))]) for i, j in Em])
    L = dd.primal_edge_lengths(Vm, Fm)
    close(f"Theorem 36.5.10: |e*| / |e| = (cot alpha + cot beta) / 2 on {name}", dd.dual_edge_lengths(Vm, Fm) / L, target,
          tol=1e-10)
    s0 = np.zeros(len(Vm))
    for f, tri in enumerate(Fm):
        for k in range(3):
            i, j, kk = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            s0[i] += (np.sum((V3[j] - V3[i]) ** 2) / np.tan(ang[f, (k + 2) % 3])
                      + np.sum((V3[kk] - V3[i]) ** 2) / np.tan(ang[f, (k + 1) % 3])) / 8
    close(f"Corollary 36.5.11: (36.5.5) = dgdec circumcentric dual areas on {name}", s0,
          dd.dual_areas(Vm, Fm, "circumcentric"), tol=1e-10)
angn = dm.corner_angles(np.c_[Vn, np.zeros(4)], Fn)
check("Theorem 36.5.10: sign of star1 is the sign of sin(alpha + beta) (non-Delaunay pair: alpha + beta > pi)",
      np.sin(angn[0, 2] + angn[1, 2]) < 0 and dd.hodge1(Vn, Fn)[0] < 0)
# Corollary 36.5.11 as the area of the polygon of circumcenters and midpoints (acute patch, interior vertex 0)
he_ = dd.HalfedgeMesh(FP)
ring = he_.vertex_faces(0)
Cp = dd.circumcenters(VP, FP)
cell = []
for a_, b_ in zip(ring, ring[1:] + ring[:1]):
    shared = sorted(set(FP[a_]) & set(FP[b_]))
    cell += [Cp[a_], 0.5 * (VP[shared[0]] + VP[shared[1]])]
close("Corollary 36.5.11: |0*| = area of the polygon c_f1, m, c_f2, m, ... (Voronoi cell of vertex 0)",
      ch36.signed_polygon_area(np.array(cell)), dd.dual_areas(VP, FP, "circumcentric")[0], tol=1e-12)

# ---------------------------------------------------------------- Example 36.5.12 / Figure 36.5.3(b)
close("Example 36.5.12: cot alpha = -1.5167, cot beta = -1.875, alpha + beta = 298.5 deg, |e*| = -3.392, star1 = -1.696",
      [round(float(v), d) for v, d in zip([1 / np.tan(angn[0, 2]), 1 / np.tan(angn[1, 2]),
                                           np.degrees(angn[0, 2] + angn[1, 2]), dd.dual_edge_lengths(Vn, Fn)[0],
                                           dd.hodge1(Vn, Fn)[0]], [4, 3, 1, 3, 3])],
      [-1.5167, -1.875, 298.5, -3.392, -1.696], tol=1e-12)
Cn = dd.circumcenters(Vn, Fn)
check("Example 36.5.12: c_f below the edge, c_g above (both circumcenters cross over)", Cn[0][1] < 0 < Cn[1][1])

# ---------------------------------------------------------------- Example 36.5.15
Vq, Fq = ch36.equilateral_patch(3, h=0.7)
Eq, _ = dd.edge_list(Fq)
bd_e = np.array([len(fs) == 1 for fs in (dm.edge_faces(Fq)[(int(i), int(j))] for i, j in Eq)])
s1q = dd.hodge1(Vq, Fq)
close("Example 36.5.15(a): equilateral grid star1 = 1/sqrt3 (interior), 1/(2 sqrt3) (boundary)",
      np.concatenate([s1q[~bd_e], s1q[bd_e]]),
      np.concatenate([np.full((~bd_e).sum(), 1 / np.sqrt(3)), np.full(bd_e.sum(), 0.5 / np.sqrt(3))]), tol=1e-12)
close("Example 36.5.15(a): interior star0 = (sqrt3/2) l^2 (also barycentric), star2 = 1 / ((sqrt3/4) l^2)",
      [dd.dual_areas(Vq, Fq, "circumcentric")[0], dd.dual_areas(Vq, Fq, "barycentric")[0], dd.hodge2(Vq, Fq)[0]],
      [np.sqrt(3) / 2 * 0.49, np.sqrt(3) / 2 * 0.49, 1 / (np.sqrt(3) / 4 * 0.49)], tol=1e-12)
Vi, Fi = dm.icosphere(1)
degi = np.bincount(dm.edges(Fi).ravel())
s0i = dd.dual_areas(Vi, Fi, "circumcentric")
close("Example 36.5.15(b): icosphere(1) star0 = 0.2560 (valence 5), 0.2865 (valence 6), sum 11.6659",
      np.round([s0i[degi == 5].min(), s0i[degi == 5].max(), s0i[degi == 6].min(), s0i[degi == 6].max(), s0i.sum()], 4),
      [0.2560, 0.2560, 0.2865, 0.2865, 11.6659], tol=1e-12)
check("Example 36.5.15(b): icosphere(1) is acute; total area = sum of star0 < 4 pi",
      dm.corner_angles(Vi, Fi).max() < np.pi / 2 and abs(s0i.sum() - dm.face_areas(Vi, Fi).sum()) < 1e-12
      and s0i.sum() < 4 * np.pi)
V2, F2 = dm.icosphere(2)
close("Example 36.5.15(b): icosphere(2) star0 sums to 12.3298; all star1 > 0",
      [round(dd.dual_areas(V2, F2, "circumcentric").sum(), 4), float(dd.hodge1(V2, F2).min() > 0)], [12.3298, 1.0], 1e-12)
bary = dd.dual_areas(Vi, Fi, "barycentric")
close("Figure 36.5.4(b): barycentric star0 = 0.2322 (valence 5), 0.2960 (valence 6)",
      np.round([bary[degi == 5][0], bary[degi == 6][0]], 4), [0.2322, 0.2960], tol=1e-12)

# ---------------------------------------------------------------- Theorem 36.5.16 (exact on constant forms)
s0P, s1P, s2P = dd.dual_areas(VP, FP, "circumcentric"), dd.hodge1(VP, FP), dd.hodge2(VP, FP)
cst = 1.7
# k = 0: integral of cst dx^dy over the 2-chain i* = sum of oriented right triangles
cells = np.zeros(len(VP))
for f, tri in enumerate(FP):
    for k in range(3):
        i, j, kk = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        for tri_ in ((VP[i], 0.5 * (VP[i] + VP[j]), Cp[f]), (VP[i], Cp[f], 0.5 * (VP[i] + VP[kk]))):
            cells[i] += ch36.signed_polygon_area(np.array(tri_))
close("Theorem 36.5.16 (k = 0): integral over the chain i* of c dx^dy = star0 R0(c)", cst * cells, s0P * cst, 1e-12)
ab = rng.normal(size=2)
alpha = lambda p: ab  # noqa: E731
flux = np.array([sum((ab @ (-J @ (q - p))) for p, q in zip(P[:-1], P[1:])) for P in polys])
close("Theorem 36.5.16 (k = 1): integral of star alpha over e* = star1 R1(alpha) for a constant alpha", flux,
      s1P * dd.derham1(alpha, VP, EP), tol=1e-12)
close("Theorem 36.5.16 (k = 2): star2 R2(c dx^dy) = c at every dual vertex", s2P * dd.derham2(lambda p: cst, VP, FP),
      np.full(len(FP), cst), tol=1e-12)
V51 = np.array([[0.0, 0.0], [2.0, 0.0], [1.05, 2.0], [0.9, -1.9]])
F51 = np.array([[0, 1, 2], [1, 0, 3]])
e51 = dd.edge_list(F51)[1][(0, 1)]
P51 = ch36.dual_edge_polylines(V51, F51)[e51]
X51 = np.array([0.8, 0.5])
flux51 = sum(X51 @ (-J @ (q - p)) for p, q in zip(P51[:-1], P51[1:]))
close("Figure 36.5.1: circulation 1.6, |e*| 1.4401, star1 0.7200, flux 1.1521 (rounded)",
      np.round([X51 @ np.array([2.0, 0.0]), dd.dual_edge_lengths(V51, F51)[e51], dd.hodge1(V51, F51)[e51], flux51], 4),
      [1.6, 1.4401, 0.72, 1.1521], tol=1e-12)

# ---------------------------------------------------------------- Remark 36.5.17
# (a) the barycentric dual edge measures flux of dy across the base of (0,0),(2,0),(0,1): 1/3 instead of 0
mb = np.array([1.0, 0.0])
bb = Vr.mean(0)
close("Remark 36.5.17(a) / Exercise 36.5.4: integral of star(dy) over [m_e, b_f] = 1/3, over [m_e, c_f] = 0",
      [np.array([0.0, 1.0]) @ (-J @ (bb - mb)), np.array([0.0, 1.0]) @ (-J @ (np.array([1.0, 0.5]) - mb))],
      [1 / 3, 0.0], tol=1e-14)
# (b) Galerkin star: M_ee' = int <phi_e, phi_e'> reproduces the L2 norm of constant 1-forms
B, wq = dd.triangle_quadrature(4)
M = np.zeros((len(EP), len(EP)))
for f, tri in enumerate(FP):
    A = 0.5 * ch36.cross2(VP[tri[1]] - VP[tri[0]], VP[tri[2]] - VP[tri[0]])
    loc = [idxP[(min(tri[a_], tri[b_]), max(tri[a_], tri[b_]))] for a_, b_ in ((0, 1), (1, 2), (2, 0))]
    for q, b_ in zip(wq, B):
        vals = np.zeros((len(EP), 2))
        for e in loc:
            unit = np.zeros(len(EP))
            unit[e] = 1.0
            vals[e] = dd.whitney_interpolate(VP, FP, unit, f, b_, idxP)
        M += A * q * vals @ vals.T
r1 = dd.derham1(alpha, VP, EP)
area = dm.face_areas(np.c_[VP, np.zeros(len(VP))], FP).sum()
close("Remark 36.5.17(b): Galerkin star reproduces ||alpha||^2 = |alpha|^2 Area for constant alpha", r1 @ M @ r1,
      (ab @ ab) * area, tol=1e-10)
check("Remark 36.5.17(b): Galerkin matrix is symmetric positive definite and not diagonal",
      np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > 0 and np.abs(M - np.diag(np.diag(M))).max() > 1e-3)
close("Remark 36.5.17: the diagonal circumcentric star also reproduces ||alpha||^2 for constant alpha", r1 @ (s1P * r1),
      (ab @ ab) * area, tol=1e-10)

# ---------------------------------------------------------------- Non-example 36.5.18 (the unsigned ratio)
en = dd.edge_list(Fn)[1][(0, 1)]
Pn = ch36.dual_edge_polylines(Vn, Fn)[en]  # c_g -> m_e -> c_f (right face g, left face f)
dx_ = lambda p: np.array([1.0, 0.0])  # noqa: E731
flux_n = sum(dx_(None) @ (-J @ (q - p)) for p, q in zip(Pn[:-1], Pn[1:]))  # integral of star(dx) = dy over e*
r1n = dd.derham1(dx_, Vn, dd.edge_list(Fn)[0])[en]
s1n = dd.hodge1(Vn, Fn)[en]
close("Non-example 36.5.18: e* runs from c_g = (1, 1.875) to c_f = (1, -1.5167)",
      np.round([Pn[0][0], Pn[0][1], Pn[-1][0], Pn[-1][1]], 4), [1.0, 1.875, 1.0, -1.5167], tol=1e-12)
close("Non-example 36.5.18: R1(dx)_e = 2, integral of star(dx) over e* = -3.392 = star1 R1(dx)",
      [r1n, round(flux_n, 3), round(s1n * r1n, 3)], [2.0, -3.392, -3.392], tol=1e-12)
check("Non-example 36.5.18: the unsigned ratio |star1| R1(dx) = +3.392 has the wrong sign",
      abs(abs(s1n) * r1n - 3.392) < 1e-3 and abs(s1n) * r1n * flux_n < 0)

# ---------------------------------------------------------------- Implementation 36.5
ns, same, printed = ch36.run_printed(PAGE, "code-36-5-1", "out-36-5-1")
check("Implementation 36.5: code-36-5-1 prints exactly out-36-5-1", same, printed)
check("Implementation 36.5: code-36-5-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-36-5-1") <= 30)

# ---------------------------------------------------------------- Exercises
close("Exercise 36.5.1: circumcentric and barycentric star0 = (sqrt3/2) l^2 at an interior vertex (l = 0.7)",
      [dd.dual_areas(Vq, Fq, "circumcentric")[0], dd.dual_areas(Vq, Fq, "barycentric")[0]],
      [np.sqrt(3) / 2 * 0.49] * 2, tol=1e-12)
Vg, Fg = ch36.grid(4)
Eg, _ = dd.edge_list(Fg)
s1g = dd.hodge1(Vg, Fg)
diag_e = np.array([(Vg[i] != Vg[j]).all() for i, j in Eg])
interior = np.array([len(dm.edge_faces(Fg)[(int(i), int(j))]) == 2 for i, j in Eg])
check("Exercise 36.5.2: grid diagonals star1 = 0, interior axis edges star1 = 1; star1 not invertible",
      np.abs(s1g[diag_e]).max() < 1e-14 and np.allclose(s1g[interior & ~diag_e], 1.0))
try:
    dd.hodge_inverse(s1g)
    check("Exercise 36.5.2: hodge_inverse(star1) raises on the grid", False)
except ZeroDivisionError:
    check("Exercise 36.5.2: hodge_inverse(star1) raises on the grid", True)
ns5, same5, printed5 = ch36.run_printed(PAGE, "code-ex-36-5-5", "out-ex-36-5-5")
check("Exercise 36.5.5: code-ex-36-5-5 prints exactly out-ex-36-5-5", same5, printed5)

summary()
