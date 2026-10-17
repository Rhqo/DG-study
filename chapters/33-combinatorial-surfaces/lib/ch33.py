"""Shared helpers for Chapter 33 (Combinatorial Surfaces): small example complexes.

The figure and verify scripts of this chapter import it with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import ch33

Contents
--------
- Example meshes (V, F) with faces oriented outward on closed meshes (GUIDELINES 2.5):
  tetrahedron, octahedron, cube_triangulated, cylinder_band (the untwisted twin of dgmesh.mobius_mesh),
  klein_bottle (an abstract complex: combinatorics only, F only).
- clip, poly_area, ccw, meet_is_common_face: intersection of two triangles in the plane (condition (2) of
  Definition 33.1.15 for a pair of triangles).

The code blocks of the pages (<pre id="code-33-M-K">, <pre id="code-ex-33-M-K"> and their outputs <pre id="out-...">,
GUIDELINES 2.5) are run by the verify scripts with tools/dgcode.py.
Everything else (complexes, orientation, incidence matrices, halfedge) comes from tools/dgdec.py and tools/dgmesh.py.
"""

import dgmesh as dm
import numpy as np


# ---------------------------------------------------------------------------
# example meshes
# ---------------------------------------------------------------------------


def _orient_outward(V, F):
    """Flip each face of a convex polyhedron so that its normal points away from the origin."""
    F = np.array(F, int)
    for f, (i, j, k) in enumerate(F):
        n = np.cross(V[j] - V[i], V[k] - V[i])
        if n @ (V[i] + V[j] + V[k]) < 0:
            F[f] = [i, k, j]
    return F


def tetrahedron():
    """Regular tetrahedron inscribed in the unit sphere. F lists the four triangles of the boundary of the
    abstract 3-simplex {0, 1, 2, 3}, each oriented outward."""
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0)
    return V, _orient_outward(V, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])


def octahedron():
    """Regular octahedron with vertices 0: +e1, 1: -e1, 2: +e2, 3: -e2, 4: +e3, 5: -e3; faces oriented outward."""
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]
    return V, _orient_outward(V, F)


def cube_triangulated():
    """Boundary of the cube [-1/2, 1/2]^3, each square cut by one diagonal: n0 = 8, n1 = 18, n2 = 12.
    Vertex 4x + 2y + z sits at (x, y, z) - 1/2 for x, y, z in {0, 1} (the cube of Chapter 31's verify)."""
    V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) - 0.5
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return V, _orient_outward(V, F)


def cylinder_band(n=12, w=0.35, R=1.0):
    """The untwisted band: same vertex numbering and face order as dgmesh.mobius_mesh(n, w, R); only the last two
    faces (the seam) differ. Orientable, chi = 0, two boundary loops."""
    V = []
    for side in (-1, 1):
        for k in range(n):
            t = 2 * np.pi * k / n
            V.append([R * np.cos(t), R * np.sin(t), side * w])
    F = []
    for k in range(n):
        a, b, c, d = k, n + k, (k + 1) % n, n + (k + 1) % n
        F += [[a, c, b], [b, c, d]]
    return np.array(V), np.array(F, int)


def klein_bottle(m=5, n=6):
    """An abstract triangulated Klein bottle (F only): the m x n grid of squares with (i, j + n) ~ (-i, j) and
    (i + m, j) ~ (i, j), each square cut by the same diagonal. Vertex index (i mod m) * n + (j mod n)."""
    def idx(i, j):
        if j >= n:
            i, j = -i, j - n
        return (i % m) * n + j

    F = []
    for i in range(m):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F += [[a, b, c], [a, c, d]]
    return np.array(F, int)


def counts(F):
    """(n0, n1, n2) of a face list (dgmesh.counts)."""
    return dm.counts(np.asarray(F, int))


# ---------------------------------------------------------------------------
# triangles in the plane (Figure 33.1.2, Exercise 33.1.4)
# ---------------------------------------------------------------------------


def clip(subject, clipper):
    """Sutherland-Hodgman: intersection polygon of a convex polygon with a counterclockwise convex clipper."""
    out = [np.asarray(p, float) for p in subject]
    clipper = [np.asarray(p, float) for p in clipper]
    for a, b in zip(clipper, clipper[1:] + clipper[:1]):
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])  # noqa: E731
        for p, q in zip(inp, inp[1:] + inp[:1]):
            sp, sq = side(p), side(q)
            if sp >= -1e-12:
                out.append(p)
            if (sp >= -1e-12) != (sq >= -1e-12):
                t = sp / (sp - sq)
                out.append(p + t * (q - p))
    return out


def poly_area(P):
    P = np.asarray(P)
    if len(P) < 3:
        return 0.0
    return 0.5 * abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - P[:, 1] * np.roll(P[:, 0], -1)))


def ccw(T):
    T = np.asarray(T, float)
    if (T[1, 0] - T[0, 0]) * (T[2, 1] - T[0, 1]) - (T[1, 1] - T[0, 1]) * (T[2, 0] - T[0, 0]) < 0:
        T = T[[0, 2, 1]]
    return [p for p in T]


def unique_pts(P, tol=1e-9):
    out = []
    for p in P:
        if all(np.linalg.norm(p - q) > tol for q in out):
            out.append(p)
    return out


def meet_is_common_face(T1, T2, tol=1e-9):
    """True if the intersection of two triangles (vertex arrays) is empty, a shared vertex, or a shared edge."""
    I = unique_pts(clip(ccw(T1), ccw(T2)))
    if poly_area(I) > tol:
        return False
    shared = [p for p in np.asarray(T1, float) if any(np.linalg.norm(p - q) < tol for q in np.asarray(T2, float))]
    if not I:
        return True
    # every point of the intersection must be a shared vertex, or lie on the segment between two shared vertices
    if len(I) == 1:
        return any(np.linalg.norm(I[0] - s) < tol for s in shared)
    if len(shared) < 2:
        return False
    a, b = shared[0], shared[1]
    return all(abs((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) < tol for p in I) and len(I) == 2 \
        and all(any(np.linalg.norm(p - s) < tol for s in shared) for p in I)
