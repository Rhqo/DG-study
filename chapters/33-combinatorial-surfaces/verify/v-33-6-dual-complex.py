"""Verification for Section 33.6 (The Dual Complex and Poincare Duality).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/verify/v-33-6-dual-complex.py``
"""

import contextlib
import io
import itertools
import pathlib
import sys
from collections import Counter

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch33  # noqa: E402

from dgcheck import check, summary  # noqa: E402

PAGE = HERE / "33-6-dual-complex.html"


def run_printed(block, out_block, namespace=None):
    """Run <pre id=block> with dgcode.run_block; return (namespace, stdout == the text of <pre id=out_block>)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(PAGE, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(PAGE, out_block))


ns, same = run_printed("code-33-6-1", "out-33-6-1")
check("Implementation 33.6: code-33-6-1 prints exactly out-33-6-1", same)
check("Implementation 33.6: code-33-6-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-33-6-1") <= 30)
dual = ns["dual"]

closed = {"tetrahedron": ch33.tetrahedron()[1], "octahedron": ch33.octahedron()[1], "icosahedron": dm.icosphere(0)[1],
          "icosphere(1)": dm.icosphere(1)[1], "icosphere(2)": dm.icosphere(2)[1], "cube": ch33.cube_triangulated()[1],
          "torus 4x6": dm.torus_mesh(2, 0.8, 4, 6)[1], "torus 8x16": dm.torus_mesh(2, 0.8, 8, 16)[1]}

# ---------------------------------------------------------------- Definition 33.6.1 / Proposition 33.6.5 / 33.6.6
ok_def = ok5a = ok5b = ok5c = ok6 = True
for name, F in closed.items():
    dedges, dfaces, B0, B1 = dual(F)
    n0, n1, n2 = dm.counts(F)
    K = dd.complex_from_faces(F)
    deg = np.bincount(dm.edges(F).ravel())
    # Definition 33.6.1: the corners of i* are the triangles around i in the cyclic order of the link
    for i, poly in enumerate(dfaces):
        ok_def &= sorted(poly) == sorted(f for f in range(len(F)) if i in F[f])
        for f, g in zip(poly, poly[1:] + poly[:1]):
            shared = set(F[f]) & set(F[g])
            ok_def &= len(shared) == 2 and i in shared
    ok_def &= all(f != g and set(F[f]) >= set(e) and set(F[g]) >= set(e)
                  for e, (f, g) in zip(dd.edge_list(F)[0].tolist(), dedges))      # e* joins the two triangles at e
    ok5a &= (len(F), len(dedges), len(dfaces)) == (n2, n1, n0) and n2 - n1 + n0 == dm.euler_characteristic(F)
    ok5b &= (B0.sum(0) == 3).all() and all(len(p) == deg[i] for i, p in enumerate(dfaces)) and (B1.sum(0) == 2).all()
    he = dd.HalfedgeMesh(F)
    nn = he.next[he.twin]
    faces_new = dd.permutation_cycles(nn)
    verts_new = dd.permutation_cycles(nn[he.twin])
    ok5c &= np.array_equal(nn[he.twin], he.next)                                   # (rho eta) eta = rho
    ok5c &= sorted(tuple(sorted({int(he.vertex[h]) for h in c})) for c in faces_new) == [(i,) for i in range(n0)]
    ok5c &= sorted(sorted(int(he.face[h]) for h in c) for c in verts_new) == sorted([[f] * 3 for f in range(n2)])
    ok5c &= len(faces_new) - n1 + len(verts_new) == n2 - n1 + n0
    # the side h of i* joins face(h)* (new tail) and face(eta h)* (new head = tail of the new next)
    ok5c &= all(he.face[nn[h]] == he.face[he.twin[h]] for h in range(he.nh))
    ok6 &= np.array_equal(B0, np.abs(dd.d1(F)).T) and np.array_equal(B1, np.abs(dd.d0(F)).T)
check("Definition 33.6.1: dual face corners = triangles around i, consecutive ones share an edge at i (8 meshes)", ok_def)
check("Proposition 33.6.5(a): n_k* = n_{2-k}, chi* = chi (8 meshes)", ok5a)
check("Proposition 33.6.5(b): trivalent, |i*| = deg i, each dual edge in two dual faces (8 meshes)", ok5b)
check("Proposition 33.6.5(c): (H, eta, rho eta) has faces = primal vertices, vertices = triangles (8 meshes)", ok5c)
check("Proposition 33.6.6: A0* = A1^T and A1* = A0^T from the polygon structure (8 meshes)", ok6)

# ---------------------------------------------------------------- Example 33.6.3
exp = {"tetrahedron": ((4, 6, 4), {3: 4}), "octahedron": ((8, 12, 6), {4: 6}), "icosahedron": ((20, 30, 12), {5: 12}),
       "torus 8x16": ((256, 384, 128), {6: 128})}
for name, (cnt, shapes) in exp.items():
    dedges, dfaces, B0, B1 = dual(closed[name])
    check(f"Example 33.6.3: dual of the {name}: counts {cnt}, dual faces {shapes}",
          (B0.shape[1], len(dedges), len(dfaces)) == cnt and Counter(len(p) for p in dfaces) == shapes)
# the tetrahedron is self-dual: its dual faces are triangles meeting like the faces of a tetrahedron
dedges, dfaces, B0, B1 = dual(closed["tetrahedron"])
check("Example 33.6.3(a): the tetrahedron is self-dual (dual faces as vertex sets form all 3-subsets of 4)",
      sorted(tuple(sorted(p)) for p in dfaces) == sorted(itertools.combinations(range(4), 3)))
Vo, Fo = ch33.octahedron()
P = Vo[Fo].mean(1)
dE = np.array(dual(Fo)[0])
Ld = np.linalg.norm(P[dE[:, 0]] - P[dE[:, 1]], axis=1)
check("Figure 33.6.2: octahedron centroids are (+-1,+-1,+-1)/3, the 12 dual edges have equal length",
      sorted(map(tuple, np.round(3 * P).astype(int).tolist())) == sorted(itertools.product((-1, 1), repeat=3))
      and len(Ld) == 12 and np.allclose(Ld, 2 / 3))

# ---------------------------------------------------------------- Non-example 33.6.4
bow = np.array([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 5)])
Kb = dd.complex_from_faces(bow)
L = [s for s in dd.link(Kb, 0) if len(s) == 2]
upper = [e for e in L if set(e) <= {1, 2, 3, 4}]
lower = [e for e in L if set(e) <= {5, 6, 7, 8}]
check("Non-example 33.6.4(b): the triangles around a bowtie vertex form two cycles (link = two 4-cycles)",
      dd.vertex_link_type(Kb, 0) == "other" and len(upper) == len(lower) == 4
      and dm.link_type(upper) == dm.link_type(lower) == "cycle")

# ---------------------------------------------------------------- Exercises
dedges, dfaces, B0, B1 = dual(closed["icosphere(1)"])
check("Exercise 33.6.1: dual of icosphere(1): (80, 120, 42), 12 pentagons and 30 hexagons, 240 sides",
      (B0.shape[1], len(dedges), len(dfaces)) == (80, 120, 42)
      and Counter(len(p) for p in dfaces) == {5: 12, 6: 30} and sum(len(p) for p in dfaces) == 240)
dedges, dfaces, B0, B1 = dual(closed["cube"])
check("Exercise 33.6.2: dual of the triangulated cube: 2 hexagons and 6 quadrilaterals, (12, 18, 8), chi 2",
      Counter(len(p) for p in dfaces) == {6: 2, 4: 6} and (B0.shape[1], len(dedges), len(dfaces)) == (12, 18, 8))
ok = True
for lv in range(4):
    F = dm.icosphere(lv)[1]
    dedges, dfaces, B0, B1 = dual(F)
    ok &= sum(6 - len(p) for p in dfaces) == 6 * dm.euler_characteristic(F) == 12
    ok &= Counter(len(p) for p in dfaces)[5] == 12
for name in ("torus 4x6", "torus 8x16", "cube", "octahedron"):
    F = closed[name]
    dedges, dfaces, B0, B1 = dual(F)
    ok &= sum(6 - len(p) for p in dfaces) == 6 * dm.euler_characteristic(F)
check("Exercise 33.6.3: sum (6 - sides) = 6 chi on duals; icosphere duals have exactly 12 pentagons (levels 0-3)", ok)
Fg = dm.grid_mesh(np.arange(4.0), np.arange(4.0), pattern="same")[1]
Kg = dd.complex_from_faces(Fg)
interior = [i for i in range(16) if dd.vertex_link_type(Kg, i) == "cycle"]
check("Exercise 33.6.4: grid 4x4: 4 interior vertices with hexagonal dual faces, 12 cut dual edges",
      len(interior) == 4 and all(np.bincount(dm.edges(Fg).ravel())[i] == 6 for i in interior)
      and len(dm.boundary_edges(Fg)) == 12)
ns5, same5 = run_printed("code-ex-33-6-5", "out-ex-33-6-5", namespace=dict(ns))
check("Exercise 33.6.5 (coding): code-ex-33-6-5 runs after code-33-6-1 and prints exactly out-ex-33-6-5", same5)

# ---------------------------------------------------------------- remark after Proposition 33.6.5: orientation of i*
ok = True
for name in ("octahedron", "icosphere(1)"):
    V = ch33.octahedron()[0] if name == "octahedron" else dm.icosphere(1)[0]
    F = closed[name]
    he = dd.HalfedgeMesh(F)
    nn = he.next[he.twin]
    C = V[F].mean(1)                                    # centroids of the triangles = drawn dual vertices
    for c in dd.permutation_cycles(nn):                 # new faces i*, corners face(h) in the order of rho o eta
        i = int(he.vertex[c[0]])
        P = C[[int(he.face[h]) for h in c]] - V[i]
        turn = sum(np.dot(np.cross(P[r], P[(r + 1) % len(P)]), V[i]) for r in range(len(P)))
        ok &= turn < 0                                  # clockwise seen from the outward normal V[i]
check("Proposition 33.6.5(c) remark: the dual faces of (H, eta, rho eta) run clockwise around i (outward normal)", ok)

summary()
