"""Chapter 38 helpers (Surfaces: Smooth and Discrete; Track 2, GUIDELINES 2.5).

Import from a figure or verify script of this chapter with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import ddg38

The canonical implementations are tools/dgsym.py (smooth conventions, EXAMPLES), tools/dgmesh.py (meshes, normals,
areas, cotan matrix) and tools/dgdec.py (d0, Hodge stars, dual areas). This module adds what Chapter 38 needs:

- smooth parametrizations as numeric functions (sphere, torus (2, 0.8), saddle, cylinder), the naive "eversion"
  family x_lambda, the figure-eight cylinder, the double cone, stereographic and Mercator coordinates of the sphere;
- small simplicial surfaces: octahedron, triangulated cube with a chosen diagonal pattern, the hinge (two triangles
  with bending angle psi), a folded planar star, the pentagram star (branch point), a saddle grid, the dimpled
  icosahedron;
- vertex stars: projected star test (Corollary 38.4.12), planar overlap test by polygon clipping;
- vertex normals of Section 38.5 (uniform / area / angle / vector-area with a chosen dual cell);
- Hodge stars and the cotan matrix from edge lengths only (Section 38.6);
- shape from face normals (Theorem 38.6.8) by a null space;
- similarity / rigid alignment errors.

Conventions: V is (n0, 3), F is (n2, 3); a face (i, j, k) is positively oriented and its normal is
(x_j - x_i) x (x_k - x_i) / |...|; closed meshes face outward (dgmesh). The torus is T_{2, 0.8} with x_u x x_v inward
(GUIDELINES 7), the sphere uses (theta, phi) with x_theta x x_phi outward.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

import dgmesh as dm  # noqa: E402

R_TORUS, r_TORUS = 2.0, 0.8

# ---------------------------------------------------------------------------
# smooth parametrizations (numeric, vectorized: x(u, v) returns an array of shape (3,) + u.shape)
# ---------------------------------------------------------------------------


def sphere(th, ph, r=1.0):
    return np.array([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th) + 0 * ph])


def torus(u, v, R=R_TORUS, r=r_TORUS):
    return np.array([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u) + 0 * v])


def saddle(u, v):
    return np.array([u + 0 * v, v + 0 * u, u ** 2 - v ** 2])


def cylinder(u, v, r=1.0):
    return np.array([r * np.cos(u) + 0 * v, r * np.sin(u) + 0 * v, v + 0 * u])


def eversion(th, ph, lam):
    """naive family x_lambda: squash the unit sphere in z by the factor 1 - 2 lambda (lambda = 1: reflected)."""
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), (1 - 2 * lam) * np.cos(th) + 0 * ph])


def figure_eight_cylinder(u, v):
    """(sin u, sin u cos u, v): the figure-eight curve of Chapter 37 times a line."""
    return np.array([np.sin(u) + 0 * v, np.sin(u) * np.cos(u) + 0 * v, v + 0 * u])


def double_cone(rho, t):
    return np.array([rho * np.cos(t), rho * np.sin(t), rho + 0 * t])


def stereographic_inv(u, v, r=1.0):
    """inverse of the stereographic projection from the north pole (Example 7.4.9), scaled by r."""
    w = 1 + u ** 2 + v ** 2
    return r * np.array([2 * u / w, 2 * v / w, (u ** 2 + v ** 2 - 1) / w])


def mercator_inv(ph, t, r=1.0):
    """inverse of the Mercator projection (7.4.4): theta = 2 arctan(e^{-t}), so sin theta = sech t."""
    return r * np.array([np.cos(ph) / np.cosh(t), np.sin(ph) / np.cosh(t), np.tanh(t) + 0 * ph])


def partials(x, u, v, h=1e-6, **kw):
    """(x_u, x_v) by central differences (shape (3,) + u.shape each)."""
    xu = (x(u + h, v, **kw) - x(u - h, v, **kw)) / (2 * h)
    xv = (x(u, v + h, **kw) - x(u, v - h, **kw)) / (2 * h)
    return xu, xv


def gram(xu, xv):
    """E, F, G from partial derivatives."""
    return (np.sum(xu * xu, 0), np.sum(xu * xv, 0), np.sum(xv * xv, 0))


# ---------------------------------------------------------------------------
# small simplicial surfaces
# ---------------------------------------------------------------------------


def orient_outward(V, F):
    """Flip faces of a star-shaped (w.r.t. the origin) closed mesh so that normals point away from the origin."""
    F = np.array(F, int).copy()
    for f, (i, j, k) in enumerate(F):
        n = np.cross(V[j] - V[i], V[k] - V[i])
        if n @ (V[i] + V[j] + V[k]) < 0:
            F[f] = [i, k, j]
    return F


def octahedron():
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    return V, orient_outward(V, F)


CUBE_V = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def cube(pattern):
    """Cube [-1, 1]^3 split into 12 triangles. ``pattern`` is a dict {face_key: bool}; face_key is one of
    'x+', 'x-', 'y+', 'y-', 'z+', 'z-'. True means the diagonal of that square passes through the vertex of the
    square with the largest coordinate sum (for 'x+' that is (1, 1, 1)); False uses the other diagonal."""
    idx = {tuple(v): n for n, v in enumerate(CUBE_V)}
    F = []
    for axis in range(3):
        for s in (1.0, -1.0):
            key = "xyz"[axis] + ("+" if s > 0 else "-")
            a, b = [c for c in range(3) if c != axis]
            sq = []
            for (pa, pb) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[a], p[b] = s, pa, pb
                sq.append(idx[tuple(p)])
            # sq is a cycle around the square; sq[2] has the largest (a, b) coordinate sum
            if pattern.get(key, False):
                F += [[sq[0], sq[1], sq[2]], [sq[0], sq[2], sq[3]]]
            else:
                F += [[sq[1], sq[2], sq[3]], [sq[1], sq[3], sq[0]]]
    return CUBE_V.copy(), orient_outward(CUBE_V, F)


def hinge(psi, a=(0.5, 0.8), b=(0.3, 0.9)):
    """Two triangles sharing the edge x0 x1 = [0, 1] x {0} x {0}. Triangle (0, 1, 2) lies in the plane z = 0 with
    x2 = (a0, a1, 0); triangle (1, 0, 3) has x3 = (b0, -b1 cos psi, b1 sin psi). psi = 0 is the flat open book,
    psi = pi folds the second triangle onto the side of the first. The angle between the face normals is psi."""
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [a[0], a[1], 0.0],
                  [b[0], -b[1] * np.cos(psi), b[1] * np.sin(psi)]])
    F = np.array([[0, 1, 2], [1, 0, 3]])
    return V, F


def planar_star(angles, radii=None):
    """Interior vertex 0 at the origin, link vertices 1..m at the given angles (radians) and radii, faces
    (0, k, k+1) in the given order (closing up). Returns (V, F) in the plane z = 0."""
    m = len(angles)
    radii = np.ones(m) if radii is None else np.asarray(radii, float)
    V = np.zeros((m + 1, 3))
    V[1:, 0] = radii * np.cos(angles)
    V[1:, 1] = radii * np.sin(angles)
    F = np.array([[0, 1 + k, 1 + (k + 1) % m] for k in range(m)])
    return V, F


def folded_star():
    """Hexagonal star with link vertex 2 moved from 60 deg to 150 deg (radius 0.8): face (0, 2, 3) turns clockwise."""
    ang = np.deg2rad([0.0, 150.0, 120.0, 180.0, 240.0, 300.0])
    rad = np.array([1.0, 0.8, 1.0, 1.0, 1.0, 1.0])
    return planar_star(ang, rad)


def pentagram_star():
    """Valence-5 star whose link is the pentagram: link vertex k at angle 90 + 144 k degrees. Every face is a
    nondegenerate counterclockwise triangle with angle 144 degrees at the center; the angle sum is 4 pi."""
    ang = np.deg2rad(90.0 + 144.0 * np.arange(5))
    return planar_star(ang)


def saddle_grid(n=9, a=1.0):
    """grid_mesh over [-a, a]^2 (alternate diagonals) lifted to z = x^2 - y^2 (GUIDELINES 2.5 running example)."""
    s = np.linspace(-a, a, n)
    V, F = dm.grid_mesh(s, s)
    V[:, 2] = V[:, 0] ** 2 - V[:, 1] ** 2
    return V, F


def dimpled_icosahedron():
    """Icosahedron (dgmesh) with vertex 0 reflected through the plane of its link: same edge lengths, nonconvex."""
    V, F = dm.icosahedron()
    nb = [j for f in F if 0 in f for j in f if j != 0]
    nb = sorted(set(nb))
    n0 = V[0] / np.linalg.norm(V[0])
    h = np.mean(V[nb] @ n0)                     # height of the link plane (= 1/sqrt 5)
    W = V.copy()
    W[0] = V[0] - 2 * (V[0] @ n0 - h) * n0
    return V, W, F


# ---------------------------------------------------------------------------
# vertex stars and local injectivity
# ---------------------------------------------------------------------------


def star_faces(F, i):
    return F[(F == i).any(axis=1)]


def projected_star_angles(V, F, i, n):
    """signed angles at x_i of the faces of the star of i, after projecting along the unit vector n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    out = []
    for tri in star_faces(F, i):
        k = list(tri).index(i)
        a, b = V[tri[(k + 1) % 3]] - V[i], V[tri[(k + 2) % 3]] - V[i]
        a, b = a - (a @ n) * n, b - (b @ n) * n
        out.append(np.arctan2(np.cross(a, b) @ n, a @ b))
    return np.array(out)


def star_passes(V, F, i, n, tol=1e-9):
    ang = projected_star_angles(V, F, i, n)
    return bool(np.all(ang > 0) and abs(ang.sum() - 2 * np.pi) < tol)


def _clip(subject, clipper):
    """Sutherland-Hodgman: subject polygon clipped by a convex counterclockwise clipper (2D arrays)."""
    out = [np.asarray(p, float) for p in subject]
    C = [np.asarray(p, float) for p in clipper]
    for k in range(len(C)):
        a, b = C[k], C[(k + 1) % len(C)]
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])  # noqa: E731
        for m in range(len(inp)):
            p, q = inp[m], inp[(m + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append(p + t * (q - p))
    return out


def _poly_area(P):
    if len(P) < 3:
        return 0.0
    P = np.asarray(P)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - P[:, 1] * np.roll(P[:, 0], -1)))


def planar_overlap(T1, T2):
    """area of the intersection of two planar triangles (2D vertex arrays, any orientation)."""
    T1 = np.asarray(T1, float)[:, :2]
    T2 = np.asarray(T2, float)[:, :2]
    if _poly_area(T2) < 0:
        T2 = T2[::-1]
    return abs(_poly_area(_clip(list(T1), list(T2))))


def max_star_overlap(V, F, i):
    """largest overlap area between two faces of the star of i in a planar (z = 0) mesh: 0 iff the closed star
    is embedded (two faces sharing an edge or the vertex only meet in a set of zero area)."""
    S = star_faces(F, i)
    best = 0.0
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            best = max(best, planar_overlap(V[S[a]], V[S[b]]))
    return best


# ---------------------------------------------------------------------------
# normals (Section 38.5)
# ---------------------------------------------------------------------------


def dual_cell_pieces(V, F, kind="barycentric"):
    """(n2, 3) array: area of the part of face f inside the dual cell of its c-th corner.

    barycentric: A_f / 3; circumcentric: (|e_ij|^2 cot theta_k + |e_ik|^2 cot theta_j) / 8 (signed);
    mixed: Meyer et al. (half / quarter of A_f on an obtuse face, circumcentric otherwise)."""
    A = dm.face_areas(V, F)
    if kind == "barycentric":
        return np.repeat(A[:, None] / 3.0, 3, axis=1)
    ang = dm.corner_angles(V, F)
    cot = 1.0 / np.tan(ang)
    P = np.zeros(F.shape)
    for c in range(3):
        i, j, k = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        lij = np.sum((V[i] - V[j]) ** 2, 1)
        lik = np.sum((V[i] - V[k]) ** 2, 1)
        P[:, c] = (lij * cot[:, (c + 2) % 3] + lik * cot[:, (c + 1) % 3]) / 8.0
    if kind == "circumcentric":
        return P
    if kind == "mixed":
        obtuse = ang.max(1) > np.pi / 2
        for f in np.where(obtuse)[0]:
            P[f] = [A[f] / 2 if ang[f, c] > np.pi / 2 else A[f] / 4 for c in range(3)]
        return P
    raise ValueError(kind)


def vector_area(V, F, kind="barycentric"):
    """(n0, 3): discrete vector area A_i = sum over faces f at i of |C_i cap f| N_f."""
    N = dm.face_normals(V, F)
    P = dual_cell_pieces(V, F, kind)
    out = np.zeros((len(V), 3))
    for c in range(3):
        np.add.at(out, F[:, c], P[:, c:c + 1] * N)
    return out


def vertex_normal(V, F, kind):
    """unit vertex normals: 'uniform', 'area', 'angle' (dgmesh) or 'vector-<cell>' (direction of vector_area)."""
    if kind.startswith("vector-"):
        A = vector_area(V, F, kind.split("-", 1)[1])
        return A / np.linalg.norm(A, axis=1, keepdims=True)
    return dm.vertex_normals(V, F, kind=kind)


def angle_deg(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# intrinsic DEC from edge lengths (Section 38.6)
# ---------------------------------------------------------------------------


def edge_lengths(V, F):
    """dict {(i, j) with i < j: |x_j - x_i|}."""
    return {(int(min(a, b)), int(max(a, b))): float(np.linalg.norm(V[a] - V[b]))
            for tri in F for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}


def stars_from_lengths(F, ell, n0):
    """(star0 barycentric, star1 dict, star2, L) computed from edge lengths alone (Heron + cot formula)."""
    L = np.zeros((n0, n0))
    s0 = np.zeros(n0)
    s1 = defaultdict(float)
    s2 = []
    for tri in F:
        e = [ell[(min(tri[(c + 1) % 3], tri[(c + 2) % 3]), max(tri[(c + 1) % 3], tri[(c + 2) % 3]))]
             for c in range(3)]                                   # e[c] = side opposite corner c
        a, b, c = e
        A = 0.25 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))
        s2.append(1.0 / A)
        for k in range(3):
            i, j = tri[(k + 1) % 3], tri[(k + 2) % 3]
            cot = (e[(k + 1) % 3] ** 2 + e[(k + 2) % 3] ** 2 - e[k] ** 2) / (4 * A)
            s1[(min(i, j), max(i, j))] += cot / 2
            s0[tri[k]] += A / 3
    for (i, j), w in s1.items():
        L[i, j] += w
        L[j, i] += w
        L[i, i] -= w
        L[j, j] -= w
    return s0, dict(s1), np.array(s2), L


# ---------------------------------------------------------------------------
# shape from normals (Theorem 38.6.8)
# ---------------------------------------------------------------------------


def normal_constraint_matrix(F, N, n0):
    """rows <y_b - y_a, N_f> = 0 for every face f and every edge (a, b) of f; unknown y = vec(Y) (row-major)."""
    rows = []
    for f, tri in enumerate(F):
        for c in range(3):
            a, b = tri[c], tri[(c + 1) % 3]
            r = np.zeros(3 * n0)
            r[3 * b:3 * b + 3] += N[f]
            r[3 * a:3 * a + 3] -= N[f]
            rows.append(r)
    return np.array(rows)


def similarity_error(X, Y):
    """min over c > 0, t of max_i |c X_i + t - Y_i| / diam(Y) (no rotation)."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    c = np.sum(Xc * Yc) / np.sum(Xc * Xc)
    diam = np.max(np.linalg.norm(Yc, axis=1)) * 2
    return float(np.max(np.linalg.norm(c * Xc - Yc, axis=1)) / diam), float(c)


def rigid_error(X, Y, allow_reflection=False):
    """min over rotations Q (and translations) of max_i |Q X_i + t - Y_i|."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Wt = np.linalg.svd(Xc.T @ Yc)
    Q = (U @ Wt).T
    if not allow_reflection and np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = (U @ Wt).T
    return float(np.max(np.linalg.norm(Xc @ Q.T - Yc, axis=1)))


# ---------------------------------------------------------------------------
# verify helpers
# ---------------------------------------------------------------------------


def run_printed(page, block_id, ns=None):
    """run a code block of a page (tools/dgcode.py) and return (namespace, printed text)."""
    import contextlib
    import io

    import dgcode
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(page, block_id, ns)
    return ns, buf.getvalue()


def printed_block(page, block_id):
    import dgcode
    return dgcode.code_block(page, block_id)
