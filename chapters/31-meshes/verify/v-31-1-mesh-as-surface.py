"""Verification for Section 31.1 (What Makes a Mesh a Surface).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/verify/v-31-1-mesh-as-surface.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

from dgcheck import check, close, summary, sym_equal  # noqa: E402

# ---------------------------------------------------------------- Example 31.1.2 / Figure 31.1.3: counts
expected_ico = {0: (12, 30, 20), 1: (42, 120, 80), 2: (162, 480, 320), 3: (642, 1920, 1280), 4: (2562, 7680, 5120)}
for lv, cnt in expected_ico.items():
    V, F = dm.icosphere(lv)
    check(f"Fig 31.1.3(a): icosphere level {lv} counts {cnt}", dm.counts(F) == cnt)
    check(f"Fig 31.1.3(a): icosphere level {lv} chi = 2", dm.euler_characteristic(F) == 2)
    check(f"Def 31.1.3: icosphere level {lv} is a closed manifold mesh",
          dm.is_manifold(F) and not dm.boundary_edges(F))
    check(f"Def 31.1.5: icosphere level {lv} consistently oriented, outward", dm.oriented_consistently(F)
          and dm.signed_volume(V, F) > 0)

for nu, nv in ((8, 16), (16, 32), (5, 7)):
    V, F = dm.torus_mesh(2.0, 0.8, nu, nv)
    n0, n1, n2 = dm.counts(F)
    check(f"Fig 31.1.3(b): torus {nu}x{nv} counts ({nu*nv}, {3*nu*nv}, {2*nu*nv}) and chi = 0",
          (n0, n1, n2) == (nu * nv, 3 * nu * nv, 2 * nu * nv) and n0 - n1 + n2 == 0)
    check(f"Def 31.1.5: torus {nu}x{nv} manifold, oriented outward",
          dm.is_manifold(F) and dm.oriented_consistently(F) and dm.signed_volume(V, F) > 0)

for h in (0.16, 0.1):
    V, F = dm.genus2_mesh(h=h)
    check(f"Fig 31.1.3(c): genus-2 mesh (h = {h}) chi = -2, closed manifold, one component",
          dm.euler_characteristic(F) == -2 and dm.is_manifold(F) and not dm.boundary_edges(F)
          and len(dm.components(F)) == 1)
V, F = dm.genus2_mesh(h=0.16)
check("Fig 31.1.3(c): counts 4032 - 12102 + 8068", dm.counts(F) == (4032, 12102, 8068))

# cube (Exercise in Tour 0.5 and Example here): 8 vertices, each square cut into two triangles
Vc = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
Fc = np.array([t for a, b, c, d in quads for t in ([a, b, c], [a, c, d])])
Fc, ok, _ = dm.orient(Fc)
check("Example 31.1.2: triangulated cube 8 - 18 + 12 = 2", ok and dm.counts(Fc) == (8, 18, 12)
      and dm.euler_characteristic(Fc) == 2)

# ---------------------------------------------------------------- Figure 31.1.1: manifold vs non-manifold
Fa = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
check("Fig 31.1.1(a): closed fan -> link is a cycle", dm.link_type(dm.vertex_link(Fa, 0)) == "cycle")
Fb = np.array([[0, 1 + k, 2 + k] for k in range(3)])
check("Fig 31.1.1(b): open fan -> link is a path", dm.link_type(dm.vertex_link(Fb, 0)) == "path")
Fe = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
check("Fig 31.1.1(c): an edge in three faces is non-manifold", dm.manifold_report(Fe)["nonmanifold_edges"] == [(0, 1)])
Fd = np.array([[0, 1 + k, 1 + (k + 1) % 5] for k in range(5)] + [[0, 6 + (k + 1) % 5, 6 + k] for k in range(5)])
rep = dm.manifold_report(Fd)
check("Fig 31.1.1(d): two fans at one vertex: non-manifold vertex, every edge has <= 2 faces",
      rep["nonmanifold_vertices"] == [0] and rep["nonmanifold_edges"] == [])

# Exercise 31.1.2: two tetrahedra sharing a vertex / an edge
tet = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
two_v = np.vstack([tet, tet + 3])          # share vertex 3
r1 = dm.manifold_report(two_v)
check("Exercise 31.1.2(a): tetrahedra sharing one vertex -> non-manifold vertex only",
      r1["nonmanifold_vertices"] == [3] and r1["nonmanifold_edges"] == [])
two_e = np.vstack([tet, np.where(tet >= 2, tet + 2, tet)])  # second copy keeps vertices 0, 1 (edge 01)
r2 = dm.manifold_report(two_e)
check("Exercise 31.1.2(b): tetrahedra sharing one edge -> that edge lies in four faces",
      r2["nonmanifold_edges"] == [(0, 1)] and len(dm.edge_faces(two_e)[(0, 1)]) == 4)

# ---------------------------------------------------------------- Example 31.1.6: Moebius strip
for n in (6, 12, 24):
    V, F = dm.mobius_mesh(n=n)
    n0, n1, n2 = dm.counts(F)
    _, ok, conflict = dm.orient(F)
    bnd = dm.boundary_edges(F)
    # the boundary is a single closed polygon of 2n edges
    adj = {}
    for a, b in bnd:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = bnd[0][0]
    prev, cur, length = None, start, 0
    while True:
        nxt = [x for x in adj[cur] if x != prev][0]
        prev, cur, length = cur, nxt, length + 1
        if cur == start:
            break
    check(f"Example 31.1.6: Moebius mesh n = {n}: counts ({2*n}, {4*n}, {2*n}), chi = 0",
          (n0, n1, n2) == (2 * n, 4 * n, 2 * n) and n0 - n1 + n2 == 0)
    check(f"Example 31.1.6: Moebius mesh n = {n} is a manifold mesh but cannot be oriented",
          dm.is_manifold(F) and not ok and conflict is not None)
    check(f"Example 31.1.6: Moebius mesh n = {n} has one boundary loop of {2*n} edges",
          len(bnd) == 2 * n and length == 2 * n)

# ---------------------------------------------------------------- Proposition 31.1.7: counting identities
n0, n1, n2, chi = sp.symbols("n0 n1 n2 chi")
sol = sp.solve([sp.Eq(3 * n2, 2 * n1), sp.Eq(chi, n0 - n1 + n2)], [n1, n2], dict=True)[0]
sym_equal("Prop 31.1.7: n1 = 3(n0 - chi)", sol[n1], 3 * (n0 - chi))
sym_equal("Prop 31.1.7: n2 = 2(n0 - chi)", sol[n2], 2 * (n0 - chi))
sym_equal("Prop 31.1.7: mean valence 2 n1 / n0 = 6 - 6 chi / n0", 2 * sol[n1] / n0, 6 - 6 * chi / n0)
for (V, F, chi_v, name) in [(*dm.icosphere(3), 2, "icosphere 3"), (*dm.torus_mesh(2, 0.8, 10, 14), 0, "torus"),
                            (*dm.genus2_mesh(h=0.16), -2, "genus 2")]:
    deg = np.array([len(x) for x in dm.neighbors(F)])
    check(f"Exercise 31.1.4: sum (6 - d_i) = 6 chi on {name}", int((6 - deg).sum()) == 6 * chi_v)
V, F = dm.icosphere(2)
deg = np.array([len(x) for x in dm.neighbors(F)])
check("Exercise 31.1.4: icosphere has exactly 12 vertices of degree 5, the rest degree 6",
      (deg == 5).sum() == 12 and set(deg) == {5, 6})

# Exercise 31.1.1: genus 3, n0 = 1000
chi3 = 2 - 2 * 3
check("Exercise 31.1.1: n1 = 3012, n2 = 2008", (3 * (1000 - chi3), 2 * (1000 - chi3)) == (3012, 2008))
# Exercise 31.1.5: one closed component with chi = -6 has genus 4
check("Exercise 31.1.5: chi = -6 -> g = 4 (3 handles more than a mug)", (2 - (-6)) // 2 == 4)

# Exercise 31.1.3: removing faces (boundary loops)
V, F = dm.icosphere(1)
F1 = F[1:]
check("Exercise 31.1.3: icosphere minus one face: 42 - 120 + 79 = 1 = 2 - b", dm.euler_characteristic(F1) == 1
      and len(dm.boundary_edges(F1)) == 3)
xs = np.linspace(0, 1, 9)
Vg, Fg = dm.grid_mesh(np.linspace(0, 2 * np.pi, 13)[:-1], xs, pattern="same")
# wrap the first coordinate around a cylinder (glue column 0 to column 12)
ny = len(xs)
Fcyl = []
for i in range(12):
    for j in range(ny - 1):
        a, b = i * ny + j, ((i + 1) % 12) * ny + j
        Fcyl += [[a, b, b + 1], [a, b + 1, a + 1]]
Fcyl = np.array(Fcyl)
check("Exercise 31.1.3: open cylinder (annulus) chi = 0 with two boundary loops",
      dm.euler_characteristic(Fcyl) == 0 and len(dm.boundary_edges(Fcyl)) == 24)

# ---------------------------------------------------------------- Figure 31.1.4: concentration of curvature
for rho in (0.3, 0.1, 0.03):
    close(f"Fig 31.1.4(a): rounded corner, rho = {rho}: integral of kappa = (1/rho)(pi rho / 2) = pi/2",
          (1 / rho) * (np.pi * rho / 2), np.pi / 2, tol=1e-14)
rho = sp.symbols("rho", positive=True)
th, ph = sp.symbols("theta phi", positive=True)
corner_area = sp.integrate(sp.integrate(rho ** 2 * sp.sin(th), (th, 0, sp.pi / 2)), (ph, 0, sp.pi / 2))
sym_equal("Fig 31.1.4(b): eighth sphere area = pi rho^2 / 2", corner_area, sp.pi * rho ** 2 / 2)
sym_equal("Fig 31.1.4(b): integral of K over a corner = pi/2", corner_area / rho ** 2, sp.pi / 2)
ell = sp.symbols("ell", positive=True)
edge_int_H = -1 / (2 * rho) * (sp.pi * rho / 2) * ell
sym_equal("Fig 31.1.4(b): integral of H over an edge piece = -(pi/4) ell (outward N)", edge_int_H, -sp.pi * ell / 4)
A_round = 24 + 12 * 2 * (sp.pi / 2 * rho) + 8 * corner_area
sym_equal("Text: rounded cube area 24 + 12 pi rho + 4 pi rho^2", A_round, 24 + 12 * sp.pi * rho + 4 * sp.pi * rho ** 2)
sym_equal("Text: total integral of K over the rounded cube = 8 (pi/2) = 2 pi chi", 8 * corner_area / rho ** 2,
          2 * sp.pi * 2)
sym_equal("Text: edge terms: sum theta_e l_e = 12 (pi/2) 2 = 12 pi, integral of H = -6 pi = -(1/2) sum",
          12 * edge_int_H.subs(ell, 2), -sp.Rational(1, 2) * 12 * (sp.pi / 2) * 2)
# the triangulated cube's angle defects: pi/2 at each of the 8 corners
check("Text: triangulated cube: angle defect pi/2 at every vertex",
      np.allclose(dm.angle_defect(Vc, Fc), np.pi / 2))
be = dm.edge_bending_angles(Vc, Fc)
check("Text: triangulated cube: 12 edges with bending pi/2, 6 diagonals with bending 0",
      sum(np.isclose(t, np.pi / 2) for t, _ in be.values()) == 12 and sum(np.isclose(t, 0) for t, _ in be.values()) == 6)

# ---------------------------------------------------------------- marching tetrahedra on a sphere through grid points
xs = np.linspace(-1.3, 1.3, 27)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
V, F = dm.marching_tetrahedra(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 1, xs, xs, xs)
check("Insight box: level-set mesh of a sphere through grid points is a closed manifold with chi = 2",
      dm.euler_characteristic(F) == 2 and dm.is_manifold(F) and not dm.boundary_edges(F))
check("Insight box: ...but it contains numerically degenerate triangles (area < 1e-12)",
      (dm.face_areas(V, F) < 1e-12).sum() > 0)

summary()
