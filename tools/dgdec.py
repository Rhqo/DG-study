"""Discrete exterior calculus (DEC), discrete curves, and derivatives of geometric quantities (Track 2).

This module is the canonical implementation of the discrete conventions of Track 2 (GUIDELINES 2.5), in the same
way that tools/dgsym.py implements the smooth conventions of GUIDELINES 6. Mesh construction, combinatorics, angles,
areas and the cotan matrix come from tools/dgmesh.py; this module builds on them and does not duplicate them.

Import it (tools/ is on PYTHONPATH for every figure, verify and animation script) with::

    import dgdec as dd

Self-test: ``python3 tools/dgdec.py`` prints PASS/FAIL lines and exits 1 on a failure (run by tools/check_all.sh).

Conventions (GUIDELINES 2.5; changing them needs the user's approval)
-----------------------------------------------------------------------
- Meshes: V is an (n0, d) float array of vertex positions x_i (d = 2 or 3), F an (n2, 3) int array of faces.
  Counts are n0, n1, n2 and chi = n0 - n1 + n2.
- Simplices of an abstract complex are sorted tuples of vertex indices; a complex is a frozenset of them.
- Orientation: an oriented simplex is a tuple (v_0, ..., v_k); even permutations give the same orientation.
  The induced boundary orientation is d(v_0 ... v_k) = sum_m (-1)^m (v_0 ... v_m-hat ... v_k). For a triangle this
  is the cyclic order (ij + jk + ki); in R^2 and R^3 it is the "outward normal first" (Stokes) orientation of
  GUIDELINES 6.4.
- Edges: edge {i, j} with i < j is oriented i -> j. edge_list(F) returns the edges in lexicographic order; this
  order indexes every discrete 1-form (and hodge1, dual edge lengths, ...).
  Polygonal curves are the one exception: there edge i runs from vertex i to vertex i + 1 (mod n), i.e. along the
  curve, also for the closing edge n-1 -> 0 of a closed curve (curve_edges, curve_d0).
- Faces: face (i, j, k) is oriented by its row order in F. On a closed mesh the faces face outward (dgmesh builds
  them this way; dgmesh.signed_volume > 0).
- Exterior derivative: d0 (n1 x n0) and d1 (n2 x n1) are the signed incidence matrices, (d0 u)_ij = u_j - u_i and
  (d1 a)_f = sum of a over the three edges of f with sign +1 when the edge orientation agrees with the cyclic order
  of f. Boundary operators are the transposes: boundary1 = d0^T, boundary2 = d1^T. d1 d0 = 0.
- de Rham map: hat(alpha)_sigma = integral of alpha over the oriented simplex sigma (derham0/1/2).
- Diagonal Hodge stars: star0 = diag(dual cell areas), star1 = diag(|e*| / |e|) = diag((cot alpha + cot beta) / 2),
  star2 = diag(1 / face area). Then d0^T star1 d0 = -L, where L = dgmesh.cotan_laplacian(V, F) is the cotan matrix
  with L_ij = (cot alpha_ij + cot beta_ij) / 2 and L_ii = -sum_j L_ij. -L is positive semidefinite (it is the P1
  stiffness matrix), and with M = star0, M^{-1} L ~ Delta = div grad (GUIDELINES 6.5), i.e.
  Delta ~ -star0^{-1} d0^T star1 d0.
- Curvature: W = -dN, H = tr(W) / 2, so H = -1/r on a sphere of radius r with outward N, and the mean curvature
  normal is Delta x = 2 H N (GUIDELINES 2.5, 6.3). Torsion: b' = -tau n, so a right-handed helix has tau > 0
  (GUIDELINES 6.2). Signed plane curvature uses n_s = J t (t rotated by +90 degrees), counterclockwise positive.

Area gradient and mean curvature (derived and finite-difference checked below; see area_gradient):
    grad_{x_i} Area = (1/2) sum_{j ~ i} (cot alpha_ij + cot beta_ij) (x_i - x_j) = -(L x)_i,
so the area gradient is exactly minus the cotan matrix applied to the positions (no extra factor). Since
(L x)_i ~ A_i (Delta x)(x_i) = 2 H_i N_i A_i, the area gradient approximates -2 H N A_i, the discrete form of the
first variation dArea = -int 2 H <N, dx> dA. On the unit sphere with outward N (H = -1): (L x)_i ~ -2 A_i x_i and
grad Area ~ +2 A_i N_i. Pointwise values need the mixed (Voronoi) area A_i; see mean_curvature_normal.
(Cr Exercise 5.4, printed p. 88 of the 2025-01-29 edition, prints the sum with (p_j - p_i): the opposite sign.)

Contents
--------
1. simplicial complexes: simplex, faces, generate_complex, complex_from_faces, is_complex, dimension, k_simplices,
   closure, star, link, is_pure, boundary_simplices, interior, vertex_link_type, is_simplicial_surface
2. orientation: permutation_sign, same_orientation, induced_boundary, induced_orientation, relative_orientation
3. chains and cochains: edge_list, d0_from_edges, d0, d1, boundary1, boundary2, curve_edges, curve_d0
4. de Rham map: gauss_legendre_01, triangle_quadrature, two_form_vector, derham0, derham1, derham2
5. Whitney interpolation: barycentric_coordinates, barycentric_gradients, whitney_basis, whitney_interpolate,
   whitney_at_barycenters
6. dual complex and Hodge stars: circumcenters, primal_edge_lengths, dual_edge_lengths, dual_areas, hodge0, hodge1,
   hodge2, hodge_inverse, laplacian_psd
7. halfedge mesh: HalfedgeMesh, permutation_cycles
8. discrete curves: curve_edge_vectors, curve_edge_lengths, curve_tangents, curve_length, curve_dual_lengths,
   turning_angles, curvature_turning_angle, curvature_length_variation, curvature_steiner,
   curvature_osculating_circle, DISCRETE_CURVATURES, curve_vertex_normals, length_gradient, vertex_binormals,
   torsion_angles, space_curve_curvature_torsion, rotation_between, parallel_transport_frames, bishop_holonomy
9. derivatives of geometric quantities: edge_length_gradient, triangle_area_gradients, angle_gradient,
   corner_angle_gradients, area_gradient, signed_volume_gradient, mean_curvature_normal, mean_curvature

Everything is dense numpy (no scipy, GUIDELINES appendix E); meshes are small (up to a few thousand vertices).
The module limits BLAS threads to 4 (this machine freezes under long all-core loads).
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import itertools  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

import dgmesh as dm  # noqa: E402


def _as3d(V):
    """Positions as an (n, 3) float array (planar input gets z = 0), for the dgmesh functions that use cross."""
    V = np.asarray(V, float)
    if V.shape[-1] == 2:
        V = np.concatenate([V, np.zeros(V.shape[:-1] + (1,))], -1)
    return V


def _cross2(a, b):
    """z-component of the cross product of planar vectors (signed area of the parallelogram)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# ---------------------------------------------------------------------------
# 1. abstract simplicial complexes
# ---------------------------------------------------------------------------


def simplex(vertices):
    """A simplex as a sorted tuple of distinct vertex indices (an int i means the 0-simplex (i,))."""
    if isinstance(vertices, (int, np.integer)):
        vertices = (vertices,)
    s = tuple(sorted(int(v) for v in vertices))
    if not s or len(set(s)) != len(s):
        raise ValueError(f"a simplex needs one or more distinct vertices, got {vertices!r}")
    return s


def faces(s, proper=False):
    """All nonempty faces of the simplex s (sorted tuples), s itself included unless proper=True."""
    s = simplex(s)
    top = len(s) - 1 if proper else len(s)
    return [f for m in range(1, top + 1) for f in itertools.combinations(s, m)]


def _simplex_set(S):
    """A collection of simplices as a set of sorted tuples. An int i means {(i,)}; otherwise S is an iterable whose
    elements are simplices (ints or tuples). Note: (0, 1) means the two vertices {0}, {1}; the edge is [(0, 1)]."""
    if isinstance(S, (int, np.integer)):
        return {(int(S),)}
    return {simplex(s) for s in S}


def generate_complex(simplices):
    """The simplicial complex generated by the given simplices: the set of all their faces (their closure)."""
    K = set()
    for s in simplices:
        K.update(faces(s))
    return frozenset(K)


def complex_from_faces(F):
    """The 2-complex generated by the triangles of a mesh face array F."""
    return generate_complex(tuple(int(v) for v in tri) for tri in np.asarray(F))


def is_complex(K):
    """True if every face of every simplex of K is again in K."""
    K = _simplex_set(K)
    return all(f in K for s in K for f in faces(s, proper=True))


def dimension(K):
    """Largest degree k of a simplex in K (a k-simplex has k + 1 vertices); -1 for the empty complex."""
    return max((len(s) - 1 for s in K), default=-1)


def k_simplices(K, k):
    """The k-simplices of K in lexicographic order."""
    return sorted(s for s in K if len(s) == k + 1)


def closure(K, S):
    """Cl(S): the smallest subcomplex of K containing the simplices S (all faces of the simplices of S)."""
    S = _simplex_set(S)
    if not S <= set(K):
        raise ValueError("closure: S is not a subset of K")
    return frozenset(f for s in S for f in faces(s))


def star(K, S):
    """St(S): all simplices of K that contain some simplex of S as a face. In general not a subcomplex."""
    S = [set(s) for s in _simplex_set(S)]
    return frozenset(t for t in K if any(s <= set(t) for s in S))


def link(K, S):
    """Lk(S) = Cl(St(S)) minus St(Cl(S)) [Cr 2.2]. For a vertex of a surface: the loop (or path) of opposite edges."""
    return closure(K, star(K, S)) - star(K, closure(K, S))


def is_pure(K, k=None):
    """True if K has dimension k (default: dimension(K)) and every simplex of K is a face of some k-simplex."""
    k = dimension(K) if k is None else k
    if dimension(K) != k:
        return False
    return set(K) <= set(generate_complex(k_simplices(K, k)))


def boundary_simplices(K):
    """bd(K) of a pure k-complex K (k >= 1): the closure of the (k-1)-simplices that are faces of exactly one
    k-simplex [Cr 2.2]. For a pure complex this is the same as "proper faces of exactly one simplex"."""
    k = dimension(K)
    if k < 1 or not is_pure(K, k):
        raise ValueError("boundary_simplices needs a pure complex of dimension >= 1")
    count = defaultdict(int)
    for s in k_simplices(K, k):
        for f in itertools.combinations(s, k):
            count[f] += 1
    return closure(K, [f for f, c in count.items() if c == 1])


def interior(K):
    """int(K) = K minus bd(K) for a pure complex K."""
    return frozenset(K) - boundary_simplices(K)


def vertex_link_type(K, i):
    """'cycle', 'path' or 'other' for the link of vertex i in a 2-complex K.

    The edges of Lk(i) are classified by dgmesh.link_type (a single closed loop, a single simple path, or anything
    else); a link with an isolated vertex or a triangle is 'other'."""
    Lk = link(K, i)
    edges = [s for s in Lk if len(s) == 2]
    in_edges = {v for e in edges for v in e}
    if any(len(s) > 2 for s in Lk) or any(len(s) == 1 and s[0] not in in_edges for s in Lk):
        return "other"
    return dm.link_type(edges)


def is_simplicial_surface(K):
    """True if K is a simplicial surface, possibly with boundary [Cr 2.3]: a pure 2-complex in which the link of every
    vertex is a single loop (interior vertex) or a single path (boundary vertex). For a face array F the same test is
    dgmesh.is_manifold(F), which is faster."""
    if dimension(K) != 2 or not is_pure(K, 2):
        return False
    return all(vertex_link_type(K, s[0]) in ("cycle", "path") for s in K if len(s) == 1)


# ---------------------------------------------------------------------------
# 2. orientation
# ---------------------------------------------------------------------------


def permutation_sign(seq, ref=None):
    """Sign (+1 or -1) of the permutation that rearranges ``ref`` (default: sorted(seq)) into ``seq``."""
    seq = [int(v) for v in seq]
    ref = sorted(seq) if ref is None else [int(v) for v in ref]
    if len(set(seq)) != len(seq) or sorted(seq) != sorted(ref):
        raise ValueError(f"{seq} is not a permutation of {ref}")
    pos = {v: n for n, v in enumerate(ref)}
    p = [pos[v] for v in seq]
    inversions = sum(1 for a in range(len(p)) for b in range(a + 1, len(p)) if p[a] > p[b])
    return -1 if inversions % 2 else 1


def same_orientation(s, t):
    """True if the oriented simplices s and t (tuples on the same vertex set) differ by an even permutation."""
    return permutation_sign(t, s) == 1


def induced_boundary(s):
    """Boundary chain of the oriented simplex s = (v_0, ..., v_k), k >= 1, as a list of (sign, face):
    d s = sum_m (-1)^m (v_0, ..., v_m-hat, ..., v_k).
    Example: d(i, j) = j - i, d(i, j, k) = (j, k) - (i, k) + (i, j)."""
    s = tuple(int(v) for v in s)
    if len(s) < 2:
        return []
    return [((-1) ** m, s[:m] + s[m + 1:]) for m in range(len(s))]


def induced_orientation(s, face):
    """+1 if the oriented codimension-1 face ``face`` (a tuple) carries the boundary orientation induced by the
    oriented simplex s, -1 if it carries the opposite one. Example: the edge (i, j) of (i, j, k) gives +1, (j, i) -1;
    for an edge (i, j) the vertex j gives +1 and i gives -1 [Cr 2.2.1]."""
    for sign, f in induced_boundary(s):
        if set(f) == set(int(v) for v in face):
            return sign * permutation_sign(face, f)
    raise ValueError(f"{face} is not a codimension-1 face of {s}")


def relative_orientation(s, t):
    """Relative orientation of two oriented k-simplices (k >= 1) that share exactly k vertices: +1 if they are
    consistently oriented (they induce opposite orientations on the shared face, like two faces of an oriented
    mesh across an edge, or the edges ij and jk), -1 otherwise (ijk and ijl, or ij and kj)."""
    shared = set(s) & set(t)
    if len(s) != len(t) or len(s) < 2 or len(shared) != len(s) - 1:
        raise ValueError("relative_orientation needs two k-simplices that share a (k-1)-face")
    face = tuple(v for v in s if v in shared)
    return -induced_orientation(s, face) * induced_orientation(t, face)


# ---------------------------------------------------------------------------
# 3. chains and cochains
# ---------------------------------------------------------------------------


def edge_list(F):
    """(E, index): E is the (n1, 2) array of edges [i, j] with i < j in lexicographic order (dgmesh.edges), oriented
    i -> j, and index is the dict {(i, j): row}. Every discrete 1-form on the mesh is indexed by this order."""
    E = dm.edges(np.asarray(F, int))
    return E, {(int(i), int(j)): n for n, (i, j) in enumerate(E)}


def d0_from_edges(E, n0):
    """Signed incidence matrix (n1 x n0) of oriented edges E (rows [tail, head]): (d0 u)_e = u_head - u_tail."""
    E = np.asarray(E, int)
    D = np.zeros((len(E), n0))
    D[np.arange(len(E)), E[:, 0]] -= 1.0
    D[np.arange(len(E)), E[:, 1]] += 1.0
    return D


def d0(F, n0=None):
    """Discrete exterior derivative on 0-forms (n1 x n0): (d0 u)_ij = u_j - u_i for the edge i -> j (i < j).
    This is the exact discrete Stokes theorem for an edge: hat(df)_ij = f(x_j) - f(x_i). n0 defaults to max(F) + 1."""
    F = np.asarray(F, int)
    n0 = int(F.max()) + 1 if n0 is None else n0
    E, _ = edge_list(F)
    return d0_from_edges(E, n0)


def d1(F):
    """Discrete exterior derivative on 1-forms (n2 x n1): (d1 a)_f = sum over the edges of f of +-a_e, with + when the
    edge orientation (i < j) agrees with the cyclic order of the face row (i.e. with the induced boundary
    orientation of f). This is the discrete Stokes theorem for a triangle: integral_f d(alpha) = integral_df alpha."""
    F = np.asarray(F, int)
    E, index = edge_list(F)
    D = np.zeros((len(F), len(E)))
    for f, tri in enumerate(F):
        for c in range(3):
            a, b = int(tri[c]), int(tri[(c + 1) % 3])
            D[f, index[(min(a, b), max(a, b))]] += 1.0 if a < b else -1.0
    return D


def boundary1(F, n0=None):
    """Boundary operator on 1-chains, d0^T (n0 x n1): the boundary of the edge i -> j is j - i."""
    return d0(F, n0).T


def boundary2(F):
    """Boundary operator on 2-chains, d1^T (n1 x n2): the boundary of a face is its three edges with induced signs."""
    return d1(F).T


def curve_edges(n, closed):
    """Edges of a polygonal curve with n vertices: rows [i, i + 1], plus [n - 1, 0] if closed. Each edge is oriented
    along the curve (for the closing edge this is n-1 -> 0, not the i < j rule of meshes)."""
    E = [[i, i + 1] for i in range(n - 1)]
    if closed:
        E.append([n - 1, 0])
    return np.array(E, int).reshape(-1, 2)


def curve_d0(n, closed):
    """d0 of a polygonal curve (n_edges x n): (d0 u)_i = u_{i+1} - u_i along the curve (indices mod n if closed).
    For a closed curve every column sums to zero (each vertex is the head of one edge and the tail of one)."""
    return d0_from_edges(curve_edges(n, closed), n)


# ---------------------------------------------------------------------------
# 4. de Rham map
# ---------------------------------------------------------------------------


def gauss_legendre_01(n):
    """n-point Gauss-Legendre nodes and weights on [0, 1] (exact for polynomials of degree <= 2n - 1)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def triangle_quadrature(n=4):
    """(B, w): barycentric points B (m, 3) and weights w (m,) with sum(w) = 1, so that the integral of g over a
    triangle of area A is about A * sum_q w_q g(B_q . corners).

    Collapsed (Duffy) product of two n-point Gauss-Legendre rules, (s, t) = (u, (1 - u) v): exact for polynomials of
    total degree <= 2n - 2 (default n = 4: degree 6, 16 points)."""
    u, wu = gauss_legendre_01(n)
    v, wv = gauss_legendre_01(n)
    U, W = np.meshgrid(u, v, indexing="ij")
    s, t = U.ravel(), ((1.0 - U) * W).ravel()
    w = 2.0 * (np.outer(wu, wv) * (1.0 - U)).ravel()
    return np.stack([1.0 - s - t, s, t], 1), w


def two_form_vector(W):
    """Component vector (w_yz, w_zx, w_xy) of a 2-form on R^3 given as an antisymmetric 3x3 matrix W with
    omega(u, v) = u^T W v. This vector is the representation used by derham2: omega(u, v) = <w, u x v>."""
    W = np.asarray(W, float)
    return np.array([W[1, 2], W[2, 0], W[0, 1]])


def derham0(f, V):
    """de Rham map of a 0-form (function) f: its values at the vertices. f takes one point (a length-d array)."""
    return np.array([float(f(x)) for x in np.asarray(V, float)])


def derham1(alpha, V, E, order=6):
    """de Rham map of a 1-form: hat(alpha)_e = integral over the oriented edge e = [i, j] (i -> j) of alpha.

    alpha(x) returns the covector at x as a length-d array (alpha = sum_a alpha_a(x) dx^a). With gamma(t) =
    x_i + t (x_j - x_i), the integral is int_0^1 alpha(gamma(t)) . (x_j - x_i) dt, computed with ``order``-point
    Gauss-Legendre (exact when alpha is polynomial of degree <= 2 order - 1). Reversing a row of E flips the sign.
    E is edge_list(F)[0] for a mesh or curve_edges(n, closed) for a curve."""
    V = np.asarray(V, float)
    t, w = gauss_legendre_01(order)
    out = np.zeros(len(E))
    for n, (i, j) in enumerate(np.asarray(E, int)):
        e = V[j] - V[i]
        out[n] = sum(wq * float(np.dot(np.asarray(alpha(V[i] + tq * e), float), e)) for tq, wq in zip(t, w))
    return out


def derham2(omega, V, F, order=4):
    """de Rham map of a 2-form: hat(omega)_f = integral over the oriented triangle f = (i, j, k) of omega.

    Representation of omega(x):
    - d = 2: a scalar density rho(x) for omega = rho dx^dy (dx^dy means dx wedge dy). Then
      omega(e1, e2) = rho det[e1 e2], positive on counterclockwise triangles.
    - d = 3: the component vector w(x) = (w_yz, w_zx, w_xy) in the basis (dy^dz, dz^dx, dx^dy), so that
      omega(u, v) = <w, u x v>; the integral is the flux of w through f with normal (x_j - x_i) x (x_k - x_i). In this
      representation the exterior derivative of the 1-form (a_1, a_2, a_3) is curl a (two_form_vector converts an
      antisymmetric matrix).
    The triangle is parametrized orientation-preservingly by x_i + s (x_j - x_i) + t (x_k - x_i) over the reference
    triangle and integrated with triangle_quadrature(order) (exact to total degree 2 order - 2; default 6).
    Reversing the row order of a face flips the sign."""
    V = np.asarray(V, float)
    d = V.shape[1]
    B, w = triangle_quadrature(order)
    out = np.zeros(len(F))
    for f, (i, j, k) in enumerate(np.asarray(F, int)):
        e1, e2 = V[j] - V[i], V[k] - V[i]
        P = B @ V[[i, j, k]]
        if d == 2:
            vals = np.array([float(omega(p)) for p in P])
            out[f] = 0.5 * float(_cross2(e1, e2)) * float(w @ vals)
        elif d == 3:
            c = np.cross(e1, e2)
            vals = np.array([float(np.dot(np.asarray(omega(p), float), c)) for p in P])
            out[f] = 0.5 * float(w @ vals)
        else:
            raise ValueError("derham2 supports points in R^2 or R^3")
    return out


# ---------------------------------------------------------------------------
# 5. Whitney interpolation
# ---------------------------------------------------------------------------


def barycentric_coordinates(P, tri):
    """Barycentric coordinates (..., 3) of point(s) P (..., d) with respect to the triangle tri (3, d). In R^3 the
    point is first projected orthogonally onto the plane of the triangle. Rows sum to 1."""
    tri = np.asarray(tri, float)
    E = np.stack([tri[1] - tri[0], tri[2] - tri[0]], 1)            # (d, 2)
    st = (np.asarray(P, float) - tri[0]) @ E @ np.linalg.inv(E.T @ E)
    return np.concatenate([1.0 - st.sum(-1, keepdims=True), st], -1)


def barycentric_gradients(tri):
    """(3, d) gradients of lambda_0, lambda_1, lambda_2 of the triangle tri (3, d), tangent to its plane. Equivalent
    to grad lambda_i = N x (x_k - x_j) / (2 A) for (i, j, k) in cyclic order. They sum to zero."""
    tri = np.asarray(tri, float)
    E = np.stack([tri[1] - tri[0], tri[2] - tri[0]], 1)
    g12 = (E @ np.linalg.inv(E.T @ E)).T                           # rows: grad lambda_1, grad lambda_2
    return np.vstack([-g12.sum(0), g12])


def whitney_basis(bary, grads, a, b):
    """Whitney 1-form phi_ab = lambda_a d(lambda_b) - lambda_b d(lambda_a) of a triangle, as a covector (..., d), at the
    point with barycentric coordinates ``bary`` (..., 3); a, b are local corner indices and grads =
    barycentric_gradients(tri). Its integral along the edge a -> b is 1, along the other two edges 0."""
    bary = np.asarray(bary, float)
    return bary[..., a, None] * grads[b] - bary[..., b, None] * grads[a]


def whitney_interpolate(V, F, alpha_hat, f, bary, index=None):
    """Whitney interpolant of the discrete 1-form alpha_hat (indexed like edge_list) inside face f, at the point(s)
    with barycentric coordinates ``bary`` (3,) or (m, 3): sum over the edges i < j of f of alpha_hat_ij phi_ij.
    Returns the covector (d,) or (m, d). Its tangential component is continuous across edges, and
    derham1 of the interpolant gives back alpha_hat (de Rham o Whitney = id, checked in the self-test)."""
    V = np.asarray(V, float)
    if index is None:
        _, index = edge_list(F)
    tri = [int(v) for v in F[f]]
    g = barycentric_gradients(V[tri])
    out = 0.0
    for a, b in ((0, 1), (1, 2), (2, 0)):
        i, j = tri[a], tri[b]
        if i < j:
            out = out + alpha_hat[index[(i, j)]] * whitney_basis(bary, g, a, b)
        else:
            out = out + alpha_hat[index[(j, i)]] * whitney_basis(bary, g, b, a)
    return out


def whitney_at_barycenters(V, F, alpha_hat):
    """(n2, d) value of the Whitney interpolant of alpha_hat at every face barycenter (convenient for drawing)."""
    _, index = edge_list(F)
    c = np.full(3, 1.0 / 3.0)
    return np.array([whitney_interpolate(V, F, alpha_hat, f, c, index) for f in range(len(F))])


# ---------------------------------------------------------------------------
# 6. dual complex and diagonal Hodge stars
# ---------------------------------------------------------------------------


def circumcenters(V, F):
    """(n2, d) circumcenters of the faces, from the barycentric weights a^2 (b^2 + c^2 - a^2) etc. (a = side opposite
    corner 0, ...). The circumcenter lies outside an obtuse triangle (one weight is negative)."""
    V = np.asarray(V, float)
    P = V[np.asarray(F, int)]
    a2 = np.sum((P[:, 1] - P[:, 2]) ** 2, 1)
    b2 = np.sum((P[:, 2] - P[:, 0]) ** 2, 1)
    c2 = np.sum((P[:, 0] - P[:, 1]) ** 2, 1)
    W = np.stack([a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2), c2 * (a2 + b2 - c2)], 1)
    W /= W.sum(1, keepdims=True)
    return np.einsum("fc,fcd->fd", W, P)


def primal_edge_lengths(V, F):
    """(n1,) edge lengths |e| in edge_list order."""
    V = np.asarray(V, float)
    E, _ = edge_list(F)
    return np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)


def dual_edge_lengths(V, F):
    """(n1,) signed lengths |e*| of the circumcentric dual edges, in edge_list order.

    For each face f containing e, h_{f,e} = (|e| / 2) cot(theta), theta the angle of f opposite e, is the signed
    distance from the circumcenter of f to the midpoint of e (positive when the circumcenter lies on the same side of
    e as f). |e*| sums h over the one or two faces: |e*| = |e| (cot alpha + cot beta) / 2. It is negative exactly when
    alpha + beta > pi (a non-Delaunay edge); a boundary edge gets one term."""
    return hodge1(V, F) * primal_edge_lengths(V, F)


def dual_areas(V, F, kind="barycentric"):
    """(n0,) areas of the dual cells.

    - "barycentric": one third of the area of the faces around i (dgmesh.mass_barycentric). Always positive.
    - "circumcentric": the signed circumcentric (Voronoi) cell, (1/8) sum over the faces at i of
      (|x_i x_j|^2 cot theta_k + |x_i x_k|^2 cot theta_j). Equal to the Voronoi area on non-obtuse meshes; parts of it
      are negative around obtuse triangles. The cells still sum to the total area.
    - "mixed": the mixed Voronoi area of Meyer et al. (dgmesh.mass_voronoi). Always positive."""
    V3 = _as3d(V)
    F = np.asarray(F, int)
    if kind == "barycentric":
        return dm.mass_barycentric(V3, F)
    if kind == "mixed":
        return dm.mass_voronoi(V3, F)
    if kind == "circumcentric":
        ang = dm.corner_angles(V3, F)
        cot = 1.0 / np.tan(ang)
        A = np.zeros(len(V3))
        for c in range(3):
            i, j, k = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
            lij = np.sum((V3[i] - V3[j]) ** 2, 1)
            lik = np.sum((V3[i] - V3[k]) ** 2, 1)
            np.add.at(A, i, (lij * cot[:, (c + 2) % 3] + lik * cot[:, (c + 1) % 3]) / 8.0)
        return A
    raise ValueError(f"unknown dual area kind {kind!r}")


def _diag(s, matrix):
    return np.diag(s) if matrix else s


def hodge0(V, F, kind="barycentric", matrix=False):
    """star0 = diag(|dual cell| / |vertex|) = diag(dual_areas(V, F, kind)) (|vertex| = 1). It maps a primal 0-form
    (point values) to a dual 2-form (integrals over dual cells); it is the lumped mass matrix M. Returns the diagonal
    (n0,), or the matrix if matrix=True. Default "barycentric"; "circumcentric" is the exact orthogonal-dual star."""
    return _diag(dual_areas(V, F, kind), matrix)


def hodge1(V, F, matrix=False):
    """star1 = diag(|e*| / |e|) = diag((cot alpha + cot beta) / 2) in edge_list order (one cot on a boundary edge),
    the circumcentric (cotan) Hodge star (from dgmesh.cotan_weights). Entries are negative on non-Delaunay edges, so
    star1 is then indefinite, although d0^T star1 d0 = -L stays positive semidefinite."""
    E, _ = edge_list(F)
    w = dm.cotan_weights(_as3d(V), np.asarray(F, int))
    return _diag(np.array([w[(int(i), int(j))] for i, j in E]), matrix)


def hodge2(V, F, matrix=False):
    """star2 = diag(|dual vertex| / |face|) = diag(1 / area) (n2,): a primal 2-form (integral over a face) to the
    dual 0-form (the average density, a value at the dual vertex)."""
    return _diag(1.0 / dm.face_areas(_as3d(V), np.asarray(F, int)), matrix)


def hodge_inverse(star, matrix=False, tol=1e-14):
    """Inverse of a diagonal Hodge star given by its diagonal (dual -> primal). Raises if an entry is (near) zero,
    which happens for star1 on an edge with cot alpha + cot beta = 0 (alpha + beta = pi)."""
    s = np.asarray(star, float)
    if s.ndim == 2:
        s = np.diag(s)
    if np.any(np.abs(s) < tol):
        raise ZeroDivisionError("a diagonal Hodge star entry is zero; its inverse does not exist")
    return _diag(1.0 / s, matrix)


def laplacian_psd(V, F):
    """d0^T star1 d0 (n0 x n0), the positive semidefinite cotan Laplacian. It equals -dgmesh.cotan_laplacian(V, F)
    (= -L) up to rounding, on any mesh (Delaunay or not), and u^T (d0^T star1 d0) u = integral |grad u|^2 for the
    piecewise-linear interpolant of u (dgmesh.dirichlet_energy). The approximation of Delta = div grad is
    -star0^{-1} d0^T star1 d0 = M^{-1} L."""
    V = np.asarray(V, float)
    D = d0(F, len(V))
    return D.T @ (hodge1(V, F)[:, None] * D)


# ---------------------------------------------------------------------------
# 7. halfedge mesh
# ---------------------------------------------------------------------------


def permutation_cycles(perm, subset=None):
    """Cycles (lists of indices) of the permutation array perm, restricted to the indices in ``subset`` (a perm-closed
    set; default all). Faces are the cycles of next, vertices those of twin o prev (or Crane's next o twin)."""
    perm = np.asarray(perm, int)
    todo = set(range(len(perm))) if subset is None else set(int(h) for h in subset)
    out = []
    while todo:
        h0 = min(todo)
        cyc, h = [], h0
        while True:
            cyc.append(h)
            todo.discard(h)
            h = int(perm[h])
            if h == h0:
                break
        out.append(cyc)
    return out


class HalfedgeMesh:
    """Halfedge connectivity of a consistently oriented manifold triangle mesh F, with or without boundary [Cr 2.5].

    Halfedges (all arrays have length nh = 3 n2 + number of boundary edges):
    - interior halfedge h = 3 f + c runs from F[f, c] to F[f, (c + 1) % 3] (the face's cyclic order), face[h] = f;
    - one boundary halfedge per boundary edge, numbered 3 n2, 3 n2 + 1, ...: it runs opposite to the interior
      halfedge of that edge and has face[h] = -1 ("filling each hole with one polygon", as in [Cr 2.5]).
    Arrays: twin, next, prev, vertex (the TAIL of h, the vertex it starts from), head (= vertex[twin]), edge (row of
    edge_list(F)), face. next of a boundary halfedge is the boundary halfedge that starts at its head, so the cycles of
    next are the faces (face >= 0) and the boundary loops (face = -1); next^3 = id on interior halfedges.
    vertex_halfedge[v] is one outgoing halfedge (-1 for an unused index): for a boundary vertex the outgoing interior
    halfedge along the boundary (its twin is a boundary halfedge), so that outgoing(v) visits the one-ring in order
    from one boundary neighbor to the other; for an interior vertex the outgoing halfedge of the first face listing v.
    face_halfedge[f] = 3 f, edge_halfedge[e] = the halfedge of edge e oriented i -> j (i < j), the DEC orientation.

    Order around a vertex: outgoing(v) turns counterclockwise (seen from the side the face normals point to) by
    h -> twin(prev(h)); Crane's vertex orbit next(twin(h)) turns the other way (clockwise).
    """

    def __init__(self, F, n0=None):
        F = np.asarray(F, int)
        if not dm.oriented_consistently(F):
            raise ValueError("HalfedgeMesh: the faces are not consistently oriented (or an edge has > 2 faces)")
        if not dm.is_manifold(F):
            raise ValueError("HalfedgeMesh: the mesh is not a manifold (a vertex link is not a loop or a path)")
        self.F = F
        self.n2 = len(F)
        self.n0 = int(F.max()) + 1 if n0 is None else int(n0)
        E, index = edge_list(F)
        self.n1 = len(E)
        ni = 3 * self.n2
        tail = list(F.reshape(-1))
        head = list(F[:, [1, 2, 0]].reshape(-1))
        directed = {(int(a), int(b)): h for h, (a, b) in enumerate(zip(tail, head))}
        twin = [-1] * ni
        for h in range(ni):
            g = directed.get((int(head[h]), int(tail[h])))
            if g is None:                       # boundary edge: add the opposite boundary halfedge
                twin.append(h)
                twin[h] = len(tail)
                tail.append(head[h])
                head.append(tail[h])
            else:
                twin[h] = g
        nh = len(tail)
        nxt = [3 * (h // 3) + (h % 3 + 1) % 3 for h in range(ni)] + [-1] * (nh - ni)
        start = {int(tail[h]): h for h in range(ni, nh)}   # one outgoing boundary halfedge per boundary vertex
        for h in range(ni, nh):
            nxt[h] = start[int(head[h])]
        self.nh = nh
        self.twin = np.array(twin, int)
        self.next = np.array(nxt, int)
        self.prev = np.empty(nh, int)
        self.prev[self.next] = np.arange(nh)
        self.vertex = np.array(tail, int)
        self.head = np.array(head, int)
        self.face = np.concatenate([np.arange(ni) // 3, -np.ones(nh - ni, int)])
        self.edge = np.array([index[(min(int(a), int(b)), max(int(a), int(b)))] for a, b in zip(tail, head)], int)
        self.is_boundary_halfedge = self.face < 0
        self.boundary_vertex = np.zeros(self.n0, bool)
        self.boundary_vertex[self.vertex[self.is_boundary_halfedge]] = True
        self.vertex_halfedge = -np.ones(self.n0, int)
        for h in range(ni - 1, -1, -1):          # lowest face index wins for interior vertices
            self.vertex_halfedge[self.vertex[h]] = h
        for h in range(ni, nh):                  # boundary vertex: interior outgoing halfedge whose twin is boundary
            self.vertex_halfedge[self.vertex[self.twin[h]]] = self.twin[h]
        self.face_halfedge = 3 * np.arange(self.n2)
        self.edge_halfedge = np.empty(self.n1, int)
        for h in range(nh):
            if self.vertex[h] < self.head[h]:
                self.edge_halfedge[self.edge[h]] = h

    def outgoing(self, v):
        """Outgoing halfedges of v in counterclockwise order, starting at vertex_halfedge[v] (h -> twin(prev(h)))."""
        h0 = int(self.vertex_halfedge[v])
        if h0 < 0:
            return []
        out, h = [h0], int(self.twin[self.prev[h0]])
        while h != h0:
            out.append(h)
            h = int(self.twin[self.prev[h]])
        return out

    def vertex_one_ring(self, v):
        """Neighbors of v in counterclockwise order (for a boundary vertex: the link path from end to end)."""
        return [int(self.head[h]) for h in self.outgoing(v)]

    def vertex_faces(self, v):
        """Faces around v in counterclockwise order (boundary halfedges skipped)."""
        return [int(self.face[h]) for h in self.outgoing(v) if self.face[h] >= 0]

    def face_vertices(self, f):
        """The three vertices of face f in its orientation (= F[f]), read off the next cycle."""
        h = int(self.face_halfedge[f])
        return [int(self.vertex[h]), int(self.vertex[self.next[h]]), int(self.vertex[self.next[self.next[h]]])]

    def boundary_loops(self):
        """Boundary loops as lists of vertices, each in the order of its boundary halfedges (next cycles with face -1),
        which is opposite to the boundary orientation induced by the faces."""
        hb = np.where(self.is_boundary_halfedge)[0]
        return [[int(self.vertex[h]) for h in cyc] for cyc in permutation_cycles(self.next, hb)]

    def euler_characteristic(self):
        """chi = n0 - n1 + n2 counted from the halfedge structure: vertices = cycles of twin o prev, edges = nh / 2,
        faces = cycles of next with face >= 0."""
        rot = self.twin[self.prev]
        n0 = len(permutation_cycles(rot))
        n2 = len(permutation_cycles(self.next, np.where(~self.is_boundary_halfedge)[0]))
        return n0 - self.nh // 2 + n2


# ---------------------------------------------------------------------------
# 8. discrete curves
# ---------------------------------------------------------------------------


def curve_edge_vectors(X, closed):
    """Edge vectors e_i = x_{i+1} - x_i (n - 1 of them, or n for a closed curve, the last one x_0 - x_{n-1})."""
    X = np.asarray(X, float)
    return np.roll(X, -1, 0) - X if closed else X[1:] - X[:-1]


def curve_edge_lengths(X, closed):
    """Edge lengths l_i = |e_i|."""
    return np.linalg.norm(curve_edge_vectors(X, closed), axis=1)


def curve_tangents(X, closed):
    """Unit tangents t_i = e_i / l_i, one per edge."""
    e = curve_edge_vectors(X, closed)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def curve_length(X, closed):
    """Total length L = sum of the edge lengths."""
    return float(curve_edge_lengths(X, closed).sum())


def curve_dual_lengths(X, closed):
    """Vertex dual lengths lbar_i = (l_{i-1} + l_i) / 2 (half of each adjacent edge); the endpoints of an open curve
    get half of their one edge. They sum to L."""
    ell = curve_edge_lengths(X, closed)
    if closed:
        return 0.5 * (np.roll(ell, 1) + ell)
    out = np.zeros(len(ell) + 1)
    out[:-1] += 0.5 * ell
    out[1:] += 0.5 * ell
    return out


def _vertex_edge_pairs(X, closed):
    """(incoming edge vector e_{i-1}, outgoing e_i) per vertex, NaN at the endpoints of an open curve."""
    e = curve_edge_vectors(X, closed)
    if closed:
        return np.roll(e, 1, 0), e
    nan = np.full((1, e.shape[1]), np.nan)
    return np.vstack([nan, e]), np.vstack([e, nan])


def turning_angles(X, closed):
    """Turning (exterior) angle theta_i at every vertex: the angle from e_{i-1} to e_i.

    Plane curve (d = 2): signed, in (-pi, pi), counterclockwise positive (atan2 of cross and dot). For a simple closed
    counterclockwise curve the angles sum to 2 pi (turning number 1). Space curve (d = 3): unsigned, in [0, pi).
    Open curve: NaN at the two endpoints."""
    a, b = _vertex_edge_pairs(X, closed)
    dot = np.sum(a * b, 1)
    if a.shape[1] == 2:
        return np.arctan2(_cross2(a, b), dot)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), dot)


def curvature_turning_angle(X, closed):
    """kappa^A_i = theta_i (integrated curvature at vertex i) [CMU-DDG L1, "Discrete Curvature (Turning Angle)"].
    Divide by curve_dual_lengths for a pointwise value. Exactly preserves total curvature
    (sum = 2 pi * turning number)."""
    return turning_angles(X, closed)


def curvature_length_variation(X, closed):
    """kappa^B_i = 2 sin(theta_i / 2) (integrated) [CMU-DDG L1, "Discrete Curvature (Length Variation)"]: the
    magnitude of the length gradient, grad_{x_i} L = -kappa^B_i n_i (see length_gradient)."""
    return 2.0 * np.sin(0.5 * turning_angles(X, closed))


def curvature_steiner(X, closed):
    """kappa^C_i = 2 tan(theta_i / 2) (integrated) [CMU-DDG L1, "Discrete Curvature (Steiner Formula)"]: the length
    change per unit offset when the offset edges are extended until they meet. Unbounded as |theta| -> pi."""
    return 2.0 * np.tan(0.5 * turning_angles(X, closed))


def curvature_osculating_circle(X, closed):
    """kappa^D_i = 1 / r_i = 2 sin(theta_i) / w_i, w_i = |x_{i+1} - x_{i-1}|: the reciprocal radius of the circle
    through x_{i-1}, x_i, x_{i+1} [CMU-DDG L1, "Discrete Curvature (Osculating Circle)"]. POINTWISE (not integrated);
    signed like theta for plane curves. NaN at the endpoints of an open curve."""
    X = np.asarray(X, float)
    if closed:
        w = np.linalg.norm(np.roll(X, -1, 0) - np.roll(X, 1, 0), axis=1)
    else:
        w = np.full(len(X), np.nan)
        w[1:-1] = np.linalg.norm(X[2:] - X[:-2], axis=1)
    return 2.0 * np.sin(turning_angles(X, closed)) / w


DISCRETE_CURVATURES = {
    "A": curvature_turning_angle,       # integrated: theta
    "B": curvature_length_variation,    # integrated: 2 sin(theta/2)
    "C": curvature_steiner,             # integrated: 2 tan(theta/2)
    "D": curvature_osculating_circle,   # pointwise:  2 sin(theta)/w
}


def curve_vertex_normals(X, closed):
    """Unit vertex normals of a plane curve: n_i = J tbar_i, tbar_i = (t_{i-1} + t_i) / |t_{i-1} + t_i| the angle
    bisector of the two edge tangents and J the rotation by +90 degrees (the study set's n_s = J t). For a
    counterclockwise circle n_i points to the center. Endpoints of an open curve use their one edge."""
    T = curve_tangents(X, closed)
    if T.shape[1] != 2:
        raise ValueError("curve_vertex_normals is for plane curves")
    if closed:
        tb = np.roll(T, 1, 0) + T
    else:
        tb = np.vstack([T[:1], T[:-1] + T[1:], T[-1:]])
    tb /= np.linalg.norm(tb, axis=1, keepdims=True)
    return np.stack([-tb[:, 1], tb[:, 0]], 1)


def length_gradient(X, closed):
    """(n, d) gradient of the total length L with respect to every vertex position.

    grad_{x_i} L = t_{i-1} - t_i (each edge contributes -t at its tail and +t at its head). Its magnitude is
    2 |sin(theta_i / 2)| = |kappa^B_i|; for a plane curve grad_{x_i} L = -kappa^B_i n_i with n_i = curve_vertex_normals,
    the discrete version of the first variation dL = -int kappa_s <n_s, dgamma> ds. Endpoints of an open curve:
    grad_{x_0} L = -t_0, grad_{x_{n-1}} L = t_{n-2}."""
    T = curve_tangents(X, closed)
    if closed:
        return np.roll(T, 1, 0) - T
    G = np.zeros((len(T) + 1, T.shape[1]))
    G[:-1] -= T
    G[1:] += T
    return G


def vertex_binormals(X, closed):
    """(n, 3) unit binormals b_i = (e_{i-1} x e_i) / |e_{i-1} x e_i| of a space curve: the normal of the plane through
    x_{i-1}, x_i, x_{i+1} (the discrete osculating plane). NaN at open endpoints and where theta_i = 0."""
    a, b = _vertex_edge_pairs(X, closed)
    c = np.cross(a, b)
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nrm > 1e-14, c / nrm, np.nan)


def torsion_angles(X, closed):
    """Integrated discrete torsion psi_i on every edge i (from x_i to x_{i+1}): the signed angle from b_i to b_{i+1},
    measured about t_i by the right-hand rule. Both binormals are orthogonal to e_i, so this is a rotation about t_i.
    With b' = -tau n (GUIDELINES 6.2) the binormal turns about t by +tau ds, so psi_i ~ tau l_i and a right-handed
    helix has psi > 0. NaN where a binormal is undefined: on the first and last edge of an open curve (their end
    vertices have no binormal) and next to a vertex with theta = 0."""
    B = vertex_binormals(X, closed)
    T = curve_tangents(X, closed)
    B1 = np.roll(B, -1, 0) if closed else B[1:]
    B0 = B if closed else B[:-1]
    return np.arctan2(np.sum(np.cross(B0, B1) * T, 1), np.sum(B0 * B1, 1))


def space_curve_curvature_torsion(X, closed):
    """(kappa, tau): pointwise discrete curvature kappa_i = theta_i / lbar_i at the vertices (turning angle over dual
    length) and torsion tau_i = psi_i / l_i on the edges. For the helix (a, b) they converge to a / (a^2 + b^2) and
    b / (a^2 + b^2) at second order in the step (self-test)."""
    kappa = turning_angles(X, closed) / curve_dual_lengths(X, closed)
    tau = torsion_angles(X, closed) / curve_edge_lengths(X, closed)
    return kappa, tau


def rotation_between(t0, t1):
    """The minimal rotation R (3 x 3) with R t0 = t1 for unit vectors t0, t1: the rotation about t0 x t1 by the angle
    between them (identity if t1 = t0). Undefined for t1 = -t0."""
    t0, t1 = np.asarray(t0, float), np.asarray(t1, float)
    v = np.cross(t0, t1)
    c = float(np.dot(t0, t1))
    if c <= -1.0 + 1e-12:
        raise ValueError("rotation_between: opposite vectors, the minimal rotation is not unique")
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def parallel_transport_frames(X, closed=False, u0=None):
    """Discrete parallel-transport (Bishop) frames of a space curve: (T, U, W), each (n_edges, 3).

    T are the edge tangents; U[0] = u0 made orthogonal to T[0] and unit (default: a vector orthogonal to T[0]), and
    U[i+1] = rotation_between(T[i], T[i+1]) U[i]; W = T x U. The frame does not twist about the tangent; relative to
    the discrete Frenet binormals it turns by -psi_i on edge i (the discrete form of the Bishop frame turning by
    -tau ds relative to the Frenet frame)."""
    T = curve_tangents(X, closed)
    if T.shape[1] != 3:
        raise ValueError("parallel_transport_frames is for space curves")
    if u0 is None:
        u0 = np.eye(3)[int(np.argmin(np.abs(T[0])))]
    u = np.asarray(u0, float) - np.dot(u0, T[0]) * T[0]
    U = np.zeros_like(T)
    U[0] = u / np.linalg.norm(u)
    for i in range(len(T) - 1):
        U[i + 1] = rotation_between(T[i], T[i + 1]) @ U[i]
        U[i + 1] -= np.dot(U[i + 1], T[i + 1]) * T[i + 1]
        U[i + 1] /= np.linalg.norm(U[i + 1])
    return T, U, np.cross(T, U)


def bishop_holonomy(X):
    """For a closed space curve: the signed angle (about t_0, right-hand rule) from U[0] to the vector obtained by
    transporting it once around the curve. It equals -sum_i psi_i (mod 2 pi) (self-test) and 0 for a plane curve."""
    T, U, _ = parallel_transport_frames(X, closed=True)
    un = rotation_between(T[-1], T[0]) @ U[-1]
    return float(np.arctan2(np.dot(np.cross(U[0], un), T[0]), np.dot(U[0], un)))


# ---------------------------------------------------------------------------
# 9. derivatives of geometric quantities (Cr appendix A topics; derived and checked here)
# ---------------------------------------------------------------------------


def edge_length_gradient(a, b):
    """(grad_a l, grad_b l) for l = |b - a|: (-uhat, uhat), uhat = (b - a) / l. Moving a point along the segment
    changes the length at unit rate; moving it orthogonally does not change it to first order."""
    u = np.asarray(b, float) - np.asarray(a, float)
    uh = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return -uh, uh


def triangle_area_gradients(V, F):
    """(n2, 3, d): G[f, c] = gradient of the area A_f with respect to the corner x_{F[f, c]}:
    (1/2) N_f x (x_k - x_j) for (i, j, k) the face read cyclically from corner c. This is the opposite edge rotated by
    +90 degrees in the face plane (pointing toward x_i) times half its length: base / 2 = d(area)/d(height)."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    V3 = _as3d(V)
    N = dm.face_normals(V3, F)
    P = V3[F]
    G = np.stack([0.5 * np.cross(N, P[:, (c + 2) % 3] - P[:, (c + 1) % 3]) for c in range(3)], 1)
    return G[..., :V.shape[1]]


def angle_gradient(a, b, c):
    """Gradients (g_a, g_b, g_c) of the interior angle alpha in (0, pi) at a of the triangle (a, b, c), in R^2 or R^3.

    With u = b - a, v = c - a and N = u x v / |u x v| (alpha is the rotation from u to v about N):
    g_b = -(N x u) / |u|^2 (N x u is u turned toward v, which closes the angle), g_c = (N x v) / |v|^2, and
    g_a = -(g_b + g_c) (a rigid translation does not change alpha). Accepts stacked inputs (..., d)."""
    a, b, c = (np.asarray(p, float) for p in (a, b, c))
    d = a.shape[-1]
    a3, b3, c3 = _as3d(a), _as3d(b), _as3d(c)
    u, v = b3 - a3, c3 - a3
    n = np.cross(u, v)
    N = n / np.linalg.norm(n, axis=-1, keepdims=True)
    gb = -np.cross(N, u) / np.sum(u * u, -1, keepdims=True)
    gc = np.cross(N, v) / np.sum(v * v, -1, keepdims=True)
    return -(gb + gc)[..., :d], gb[..., :d], gc[..., :d]


def corner_angle_gradients(V, F):
    """(n2, 3, 3, d): G[f, c, k] = gradient of the corner angle theta_{f,c} (dgmesh.corner_angles) with respect to
    the position of vertex F[f, k]."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    G = np.zeros((len(F), 3, 3, V.shape[1]))
    for c in range(3):
        j, k = (c + 1) % 3, (c + 2) % 3
        ga, gb, gc = angle_gradient(V[F[:, c]], V[F[:, j]], V[F[:, k]])
        G[:, c, c], G[:, c, j], G[:, c, k] = ga, gb, gc
    return G


def area_gradient(V, F):
    """(n0, d) gradient of the total surface area with respect to every vertex position.

    grad_{x_i} Area = sum_{f ~ i} grad_{x_i} A_f = (1/2) sum_{j ~ i} (cot alpha_ij + cot beta_ij) (x_i - x_j)
                    = -(L x)_i,  L = dgmesh.cotan_laplacian (L_ij = (cot alpha + cot beta) / 2, L_ii = -sum_j L_ij).
    The middle form follows by writing (1/2) N x (x_k - x_j) in each face through the cotangents of the angles at
    x_j and x_k. Since (L x)_i ~ 2 H N A_i (Delta x = 2 H N), grad Area ~ -2 H N A_i; on the unit sphere with outward
    N (H = -1) it points outward. Finite-difference checked in the self-test."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    G = triangle_area_gradients(V, F)
    out = np.zeros_like(V)
    for c in range(3):
        np.add.at(out, F[:, c], G[:, c])
    return out


def signed_volume_gradient(V, F):
    """(n0, 3) gradient of dgmesh.signed_volume (sum over faces of x_i . (x_j x k) / 6) with respect to every vertex:
    grad_{x_i} Vol = (1/6) sum over the faces (i, j, k) at i (read cyclically from i) of x_j x x_k.
    For a closed mesh every vertex link is a loop, the terms x_i x (x_k - x_j) cancel, and this equals
    (1/3) sum_{f ~ i} A_f N_f (translation invariant), the discrete form of dVol = int <N, dx> dA."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    out = np.zeros_like(V)
    for c in range(3):
        np.add.at(out, F[:, c], np.cross(V[F[:, (c + 1) % 3]], V[F[:, (c + 2) % 3]]) / 6.0)
    return out


def mean_curvature_normal(V, F, kind="mixed"):
    """(n0, 3) discrete mean curvature normal (M^{-1} L x)_i ~ (Delta x)(x_i) = 2 H N, with M = hodge0(V, F, kind).
    Equal to -area_gradient / dual area. On the unit sphere with outward N it is about -2 x_i (H = -1).

    The default is the mixed Voronoi area: the integrated quantity (L x)_i is a Voronoi-cell integral, and dividing it
    by the barycentric area does not converge pointwise at irregular vertices (on the icosphere |H + 1| stays near
    0.14 at the 12 valence-5 vertices for every level, while the mixed area gives errors below 1e-4; self-test)."""
    V = np.asarray(V, float)
    return -area_gradient(V, F) / dual_areas(V, F, kind)[:, None]


def mean_curvature(V, F, kind="mixed"):
    """(n0,) vertex mean curvature H_i = <(M^{-1} L x)_i, N_i> / 2 with N_i the area-weighted vertex normal
    (dgmesh.vertex_normals(kind="area"), outward on a closed outward-oriented mesh). H = -1/r on a sphere of radius r
    (GUIDELINES 6.3). See mean_curvature_normal for the choice of M. dgmesh.mean_curvature_dihedral is the Steiner
    (edge bending angle) variant."""
    N = dm.vertex_normals(np.asarray(V, float), np.asarray(F, int), kind="area")
    return 0.5 * np.sum(mean_curvature_normal(V, F, kind) * N, 1)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


class _Poly:
    """A polynomial in d variables, sum_m c_m x^{e_m} (exponent tuples), for exact de Rham / Stokes tests."""

    def __init__(self, terms):
        self.terms = [(float(c), tuple(int(k) for k in e)) for c, e in terms]

    @classmethod
    def random(cls, rng, d, deg):
        exps = [e for e in itertools.product(range(deg + 1), repeat=d) if sum(e) <= deg]
        return cls([(rng.normal(), e) for e in exps])

    def __call__(self, x):
        return sum(c * np.prod([x[a] ** e[a] for a in range(len(e))]) for c, e in self.terms)

    def diff(self, a):
        out = []
        for c, e in self.terms:
            if e[a] > 0:
                e2 = list(e)
                e2[a] -= 1
                out.append((c * e[a], tuple(e2)))
        return _Poly(out)


def _cylinder_mesh(nu=10, nv=3):
    """An open cylinder (annulus, two boundary loops), faces oriented outward."""
    V, F = [], []
    for j in range(nv):
        for i in range(nu):
            t = 2 * np.pi * i / nu
            V.append([np.cos(t), np.sin(t), j / (nv - 1)])
    for j in range(nv - 1):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            c, d = a + nu, b + nu
            F += [[a, b, d], [a, d, c]]
    return np.array(V), np.array(F, int)


def _selftest():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dgcheck import check, close, summary

    rng = np.random.default_rng(20261018)
    ico_V, ico_F = dm.icosphere(2)
    tor_V, tor_F = dm.torus_mesh(2.0, 0.8, 12, 16)
    grid_V3, grid_F = dm.grid_mesh(np.linspace(0, 1, 6), np.linspace(0, 1.2, 7))
    grid_V = grid_V3[:, :2]
    cyl_V, cyl_F = _cylinder_mesh()
    meshes = {"icosphere(2)": (ico_V, ico_F), "torus T_{2,0.8}": (tor_V, tor_F), "grid 6x7": (grid_V, grid_F),
              "cylinder": (cyl_V, cyl_F)}

    # ---------------------------------------------------------------- 1. simplicial complexes
    fan = [(0, k, k % 6 + 1) for k in range(1, 7)]        # vertex 0 surrounded by the hexagon 1..6
    K = generate_complex(fan)
    check("complex: hexagonal fan has 7 vertices, 12 edges, 6 triangles",
          [len(k_simplices(K, k)) for k in range(3)] == [7, 12, 6])
    check("complex: generated set is a pure simplicial 2-complex", is_complex(K) and is_pure(K, 2))
    check("complex: removing an edge breaks the face condition", not is_complex(K - {(0, 1)}))
    check("complex: a dangling vertex makes it impure", not is_pure(K | {(9,)}, 2))
    hexagon = generate_complex([(k, k % 6 + 1) for k in range(1, 7)])
    check("Lk(vertex 0) is the hexagon (6 edges + 6 vertices)", link(K, 0) == hexagon)
    check("St(vertex 0) = vertex + 6 edges + 6 triangles; Cl(St(0)) = K",
          len(star(K, 0)) == 13 and closure(K, star(K, 0)) == K)
    check("Lk(edge 01) = the two opposite vertices {2}, {6}", link(K, [(0, 1)]) == {(2,), (6,)})
    check("bd(fan) = hexagon, int(fan) = St(0)", boundary_simplices(K) == hexagon and interior(K) == star(K, 0))
    Kico = complex_from_faces(dm.icosahedron()[1])
    check("bd(icosahedron) is empty", boundary_simplices(Kico) == frozenset())
    ok = True
    for i in range(12):
        lk = link(Kico, i)
        lk_dm = {tuple(sorted(e)) for e in dm.vertex_link(dm.icosahedron()[1], i)}
        ok &= {s for s in lk if len(s) == 2} == lk_dm
    check("Lk(i) on the icosahedron = dgmesh.vertex_link (opposite edges)", ok)
    bowtie = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (0, 5, 6), (0, 6, 7), (0, 7, 5)]
    book = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
    for name, F_, expect in (("icosahedron", dm.icosahedron()[1], True), ("grid", grid_F, True),
                             ("two fans at a vertex", np.array(bowtie), False),
                             ("three triangles on an edge", np.array(book), False)):
        Kx = complex_from_faces(F_)
        check(f"simplicial surface test: {name} -> {expect} (agrees with dgmesh.is_manifold)",
              is_simplicial_surface(Kx) == expect == dm.is_manifold(np.asarray(F_)))
    check("vertex link types on the grid: corner/side vertices 'path', inner 'cycle'",
          vertex_link_type(complex_from_faces(grid_F), 0) == "path"
          and vertex_link_type(complex_from_faces(grid_F), 8) == "cycle")

    # ---------------------------------------------------------------- 2. orientation
    check("permutation sign: (0,1,2) +1, (1,0,2) -1, (1,2,0) +1, (3,2,1,0) +1, (2,1,0) -1",
          [permutation_sign(p) for p in ((0, 1, 2), (1, 0, 2), (1, 2, 0), (3, 2, 1, 0), (2, 1, 0))]
          == [1, -1, 1, 1, -1])
    check("same orientation: (p1,p2,p3) ~ (p2,p3,p1), not ~ (p2,p1,p3) [Cr 4.8.2]",
          same_orientation((1, 2, 3), (2, 3, 1)) and not same_orientation((1, 2, 3), (2, 1, 3)))
    i, j, k, l_ = 0, 1, 2, 3
    check("relative orientation [Cr 2.2.1]: ij~jk consistent, ij~kj not, ijk~jil consistent, ijk~ijl not",
          [relative_orientation(*p) for p in (((i, j), (j, k)), ((i, j), (k, j)),
                                              ((i, j, k), (j, i, l_)), ((i, j, k), (i, j, l_)))] == [1, -1, 1, -1])
    check("induced orientation: (i,j) of (i,j,k) +1, (j,i) -1; vertex j of (i,j) +1, i -1",
          [induced_orientation((i, j, k), (i, j)), induced_orientation((i, j, k), (j, i)),
           induced_orientation((i, j), (j,)), induced_orientation((i, j), (i,))] == [1, -1, 1, -1])
    chain = defaultdict(int)                               # boundary of boundary of a tetrahedron
    for s1, f1 in induced_boundary((0, 1, 2, 3)):
        for s2, f2 in induced_boundary(f1):
            chain[tuple(sorted(f2))] += s1 * s2 * permutation_sign(f2)
    check("induced boundary: dd(tetrahedron) = 0", all(v == 0 for v in chain.values()))
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.8]])
    ok = True
    for sgn, (a, b) in induced_boundary((0, 1, 2)):
        a, b = (a, b) if sgn > 0 else (b, a)
        mid, tang = 0.5 * (tri[a] + tri[b]), tri[b] - tri[a]
        outward = mid - tri.mean(0)
        ok &= _cross2(outward, tang) > 0
    check("induced boundary of a ccw triangle in R^2 is the Stokes orientation (outward normal first)", ok)
    D1 = d1(ico_F)
    E_ico, idx_ico = edge_list(ico_F)
    check("d1 entries = induced orientation of the edge (i<j) in the face",
          all(D1[f, idx_ico[(min(a, b), max(a, b))]] == induced_orientation(tuple(ico_F[f]), (min(a, b), max(a, b)))
              for f in range(len(ico_F)) for a, b in itertools.combinations(ico_F[f], 2)))
    ef = dm.edge_faces(ico_F)
    check("icosphere: faces across every edge are consistently oriented (relative orientation +1)",
          all(relative_orientation(tuple(ico_F[f]), tuple(ico_F[g])) == 1 for f, g in ef.values()))

    # ---------------------------------------------------------------- 3. chains and cochains
    for name, (V, F) in meshes.items():
        D0, D1 = d0(F, len(V)), d1(F)
        check(f"d1 d0 = 0 on {name}", np.array_equal(D1 @ D0, np.zeros((len(F), len(V)))))
        check(f"boundary1 = d0^T, boundary2 = d1^T, d0 1 = 0 on {name}",
              np.array_equal(boundary1(F, len(V)), D0.T) and np.array_equal(boundary2(F), D1.T)
              and not np.any(D0 @ np.ones(len(V))))
    for name, (V, F), b1 in (("icosphere(2)", meshes["icosphere(2)"], 0), ("torus", meshes["torus T_{2,0.8}"], 2),
                             ("grid", meshes["grid 6x7"], 0), ("cylinder", meshes["cylinder"], 1)):
        D0, D1 = d0(F, len(V)), d1(F)
        betti1 = D0.shape[0] - np.linalg.matrix_rank(D0) - np.linalg.matrix_rank(D1)
        check(f"first Betti number n1 - rank d0 - rank d1 = {b1} on {name}", betti1 == b1)
    check("curve_d0: closed curve columns sum to 0, open curve has n - 1 rows, d0 1 = 0",
          not np.any(np.ones(7) @ curve_d0(7, True)) and curve_d0(7, False).shape == (6, 7)
          and not np.any(curve_d0(7, False) @ np.ones(7)))

    # ---------------------------------------------------------------- 4. de Rham map and discrete Stokes
    B, w = triangle_quadrature(4)
    from math import factorial
    err = max(abs(float(w @ (B[:, 1] ** p * B[:, 2] ** q)) / 2 - factorial(p) * factorial(q) / factorial(p + q + 2))
              for p in range(7) for q in range(7 - p))
    check(f"triangle_quadrature(4) is exact to total degree 6 (max error {err:.1e})", err < 1e-15)
    for name, (V, F) in (("icosphere(2)", (ico_V, ico_F)), ("torus", (tor_V, tor_F)), ("grid (R^2)", (grid_V, grid_F))):
        dim = V.shape[1]
        f = _Poly.random(rng, dim, 3)
        df = lambda x, f=f: np.array([f.diff(a)(x) for a in range(dim)])  # noqa: E731
        E, _ = edge_list(F)
        close(f"discrete Stokes, 0-forms: d0 R(f) = R(df) on {name} (cubic f)", d0(F, len(V)) @ derham0(f, V),
              derham1(df, V, E), tol=1e-11)
    for name, (V, F) in (("planar grid (R^2)", (grid_V, grid_F)), ("icosphere(2) (R^3)", (ico_V, ico_F))):
        dim = V.shape[1]
        P = [_Poly.random(rng, dim, 3) for _ in range(dim)]
        alpha = lambda x, P=P: np.array([p(x) for p in P])  # noqa: E731
        if dim == 2:
            dalpha = lambda x, P=P: P[1].diff(0)(x) - P[0].diff(1)(x)  # noqa: E731
        else:
            def dalpha(x, P=P):                            # d alpha = curl a
                return np.array([P[2].diff(1)(x) - P[1].diff(2)(x), P[0].diff(2)(x) - P[2].diff(0)(x),
                                 P[1].diff(0)(x) - P[0].diff(1)(x)])
        E, _ = edge_list(F)
        close(f"discrete Stokes, 1-forms: d1 R(alpha) = R(d alpha) on {name} (cubic coefficients)",
              d1(F) @ derham1(alpha, V, E), derham2(dalpha, V, F), tol=1e-11)
    E, _ = edge_list(grid_F)
    a1 = lambda x: np.array([x[1] ** 2, x[0] * x[1]])  # noqa: E731
    close("de Rham map respects orientation: reversed edges / faces flip the sign",
          np.concatenate([derham1(a1, grid_V, E[:, ::-1]), derham2(lambda x: 1.0 + x[0], grid_V, grid_F[:, ::-1])]),
          -np.concatenate([derham1(a1, grid_V, E), derham2(lambda x: 1.0 + x[0], grid_V, grid_F)]), tol=1e-14)
    close("derham2 of dx^dy over the grid = its area 1.2 (ccw faces)", derham2(lambda x: 1.0, grid_V, grid_F).sum(),
          1.2, tol=1e-13)
    close("star2 R(3 dx^dy) = 3 on every face (hodge2 = 1 / area)", hodge2(grid_V, grid_F)
          * derham2(lambda x: 3.0, grid_V, grid_F), np.full(len(grid_F), 3.0), tol=1e-13)
    close("star2 R(omega) = <w, N> for a constant 2-form w on the icosphere (flux density)", hodge2(ico_V, ico_F)
          * derham2(lambda x: np.array([0.3, -1.0, 2.0]), ico_V, ico_F),
          dm.face_normals(ico_V, ico_F) @ np.array([0.3, -1.0, 2.0]), tol=1e-12)
    W = np.array([[0.0, 1.0, -2.0], [-1.0, 0.0, 3.0], [2.0, -3.0, 0.0]])
    u_, v_ = rng.normal(size=3), rng.normal(size=3)
    close("two_form_vector: u^T W v = <w, u x v>", u_ @ W @ v_, two_form_vector(W) @ np.cross(u_, v_), tol=1e-13)

    # ---------------------------------------------------------------- 5. Whitney interpolation
    for name, (V, F) in (("icosphere(1)", dm.icosphere(1)), ("grid (R^2)", (grid_V, grid_F))):
        E, index = edge_list(F)
        ahat = rng.normal(size=len(E))
        back = np.full(len(E), np.nan)
        worst = 0.0
        for f in range(len(F)):
            tri_ = V[F[f]]

            def form(x, f=f, tri_=tri_):
                return whitney_interpolate(V, F, ahat, f, barycentric_coordinates(x, tri_), index)
            for a, b in ((0, 1), (1, 2), (2, 0)):
                ii, jj = sorted((int(F[f, a]), int(F[f, b])))
                val = derham1(form, V, np.array([[ii, jj]]))[0]
                e = index[(ii, jj)]
                if not np.isnan(back[e]):
                    worst = max(worst, abs(back[e] - val))      # same value from the other face
                back[e] = val
        close(f"de Rham o Whitney = identity on {name}", back, ahat, tol=1e-12)
        check(f"Whitney interpolant has continuous tangential component across edges on {name}", worst < 1e-12)
    V, F = dm.icosphere(1)
    u = rng.normal(size=len(V))
    E, index = edge_list(F)
    bc = rng.dirichlet(np.ones(3), size=5)
    ok = all(np.allclose(whitney_interpolate(V, F, d0(F, len(V)) @ u, f, bc, index),
                         np.ones((5, 1)) * (u[F[f]] @ barycentric_gradients(V[F[f]]))) for f in range(len(F)))
    check("Whitney(d0 u) = grad of the P1 interpolant of u (Whitney commutes with d)", ok)
    E, _ = edge_list(grid_F)
    close("Whitney interpolation reproduces a constant 1-form (2 dx - dy on the grid, at all barycenters)",
          whitney_at_barycenters(grid_V, grid_F, derham1(lambda x: np.array([2.0, -1.0]), grid_V, E)),
          np.tile([2.0, -1.0], (len(grid_F), 1)), tol=1e-12)

    # ---------------------------------------------------------------- 6. dual complex and Hodge stars
    for name, (V, F) in meshes.items():
        C = circumcenters(V, F)
        R = np.linalg.norm(C[:, None, :] - np.asarray(V, float)[F], axis=2)
        check(f"circumcenters are equidistant from the three corners on {name}", np.ptp(R, axis=1).max() < 1e-12)
        A_tot = dm.face_areas(_as3d(V), F).sum()
        for kind in ("barycentric", "circumcentric", "mixed"):
            close(f"dual areas ({kind}) sum to the total area on {name}", dual_areas(V, F, kind).sum(), A_tot, 1e-12)
        L = dm.cotan_laplacian(_as3d(V), F)
        Lp = laplacian_psd(V, F)
        close(f"laplacian_psd = d0^T star1 d0 = -L on {name}", Lp, -L, tol=1e-12)
        check(f"L 1 = 0, L symmetric, -L PSD on {name}",
              np.abs(L.sum(1)).max() < 1e-12 and np.abs(L - L.T).max() < 1e-13
              and np.linalg.eigvalsh(-L).min() > -1e-10)
        uu = rng.normal(size=len(V))
        close(f"u^T (d0^T star1 d0) u = Dirichlet energy of the P1 interpolant on {name}", uu @ Lp @ uu,
              dm.dirichlet_energy(_as3d(V), F, uu), tol=1e-10)
        # dual edge = sum over faces of the signed distance from the circumcenter to the edge midpoint
        E, index = edge_list(F)
        geo = np.zeros(len(E))
        for f, t3 in enumerate(np.asarray(F)):
            for c in range(3):
                a, b, o = int(t3[c]), int(t3[(c + 1) % 3]), int(t3[(c + 2) % 3])
                m = 0.5 * (V[a] + V[b])
                e = V[b] - V[a]
                toward = (V[o] - m) - np.dot(V[o] - m, e) / np.dot(e, e) * e
                geo[index[(min(a, b), max(a, b))]] += np.dot(C[f] - m, toward / np.linalg.norm(toward))
        close(f"dual_edge_lengths = signed circumcenter-to-midpoint distances on {name}", dual_edge_lengths(V, F),
              geo, tol=1e-12)
    for name, (V, F), chi in (("icosphere(2)", (ico_V, ico_F), 2), ("torus", (tor_V, tor_F), 0)):
        close(f"sum of angle defects = 2 pi chi on {name} (chi = {chi})", dm.angle_defect(V, F).sum(),
              2 * np.pi * chi, tol=1e-10)
        check(f"chi = {chi} from the halfedge structure on {name}", HalfedgeMesh(F).euler_characteristic() == chi)
    s = hodge0(ico_V, ico_F)
    close("hodge_inverse(hodge0) * hodge0 = 1; matrix form = diag", np.concatenate([hodge_inverse(s) * s,
          np.diag(hodge0(ico_V, ico_F, matrix=True))]), np.concatenate([np.ones(len(s)), s]), tol=1e-14)
    # non-Delaunay pair: both angles opposite the shared edge are obtuse
    Vn = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.3], [1.0, -0.25]])
    Fn = np.array([[0, 1, 2], [1, 0, 3]])
    En, idxn = edge_list(Fn)
    ang = dm.corner_angles(_as3d(Vn), Fn)
    w01 = hodge1(Vn, Fn)[idxn[(0, 1)]]
    check(f"non-Delaunay edge: alpha + beta = {np.degrees(ang[0, 2] + ang[1, 2]):.1f} deg > 180, "
          f"cotan weight star1 = {w01:.3f} < 0, dual edge length < 0",
          ang[0, 2] + ang[1, 2] > np.pi and w01 < 0 and dual_edge_lengths(Vn, Fn)[idxn[(0, 1)]] < 0)
    check("non-Delaunay pair: d0^T star1 d0 is still positive semidefinite (the stiffness matrix)",
          np.linalg.eigvalsh(laplacian_psd(Vn, Fn)).min() > -1e-12)
    Vs, Fs = dm.icosphere(3)
    lap_z = (dm.cotan_laplacian(Vs, Fs) @ Vs[:, 2]) / hodge0(Vs, Fs, "mixed")
    close("M^{-1} L z ~ Delta z = -2 z on the unit sphere (icosphere(3), relative error < 2%)",
          np.linalg.norm(lap_z + 2 * Vs[:, 2]) / np.linalg.norm(2 * Vs[:, 2]), 0.0, tol=0.02)

    # ---------------------------------------------------------------- 7. halfedge mesh
    for name, (V, F), chi, loops in (("icosphere(2)", (ico_V, ico_F), 2, 0), ("torus", (tor_V, tor_F), 0, 0),
                                     ("grid", (grid_V, grid_F), 1, 1), ("cylinder", (cyl_V, cyl_F), 0, 2)):
        he = HalfedgeMesh(F)
        h = np.arange(he.nh)
        inner = h[~he.is_boundary_halfedge]
        check(f"halfedge {name}: twin o twin = id, twin has no fixed point, edge(twin) = edge",
              np.array_equal(he.twin[he.twin], h) and not np.any(he.twin == h)
              and np.array_equal(he.edge[he.twin], he.edge))
        check(f"halfedge {name}: next o prev = id, next^3 = id on interior faces, face(next) = face",
              np.array_equal(he.next[he.prev], h) and np.array_equal(he.next[he.next[he.next[inner]]], inner)
              and np.array_equal(he.face[he.next], he.face))
        check(f"halfedge {name}: tail(next h) = head(h) = tail(twin h); face_vertices(f) = F[f]",
              np.array_equal(he.vertex[he.next], he.head) and np.array_equal(he.vertex[he.twin], he.head)
              and all(he.face_vertices(f) == list(F[f]) for f in range(len(F))))
        check(f"halfedge {name}: chi = {chi} recovered, {loops} boundary loop(s), edge_halfedge oriented i<j",
              he.euler_characteristic() == chi == dm.euler_characteristic(F) and len(he.boundary_loops()) == loops
              and np.all(he.vertex[he.edge_halfedge] < he.head[he.edge_halfedge]))
        nb = dm.neighbors(F, len(V))
        Fset = {tuple(np.roll(t3, -r)) for t3 in F.tolist() for r in range(3)}
        ok = True
        for v in range(len(V)):
            ring = he.vertex_one_ring(v)
            ok &= sorted(ring) == nb[v]
            pairs = list(zip(ring, ring[1:] + ring[:1]))[: len(ring) - (1 if he.boundary_vertex[v] else 0)]
            ok &= all((v, a, b) in Fset for a, b in pairs)          # consecutive neighbors span a face, in order
            ok &= he.vertex_faces(v) == [int(he.face[g]) for g in he.outgoing(v) if he.face[g] >= 0]
        check(f"halfedge {name}: one-ring = neighbors, counterclockwise (consecutive pairs are faces (v, a, b))", ok)
    try:
        HalfedgeMesh(dm.mobius_mesh()[1])
        check("HalfedgeMesh rejects the Moebius strip (non-orientable)", False)
    except ValueError:
        check("HalfedgeMesh rejects the Moebius strip (non-orientable)", True)

    # ---------------------------------------------------------------- 8. discrete curves
    r = 1.5
    errs = {key: [] for key in "ABCD"}
    ns = [8, 16, 32, 64, 128, 256]
    for n in ns:
        t = 2 * np.pi * np.arange(n) / n
        X = r * np.stack([np.cos(t), np.sin(t)], 1)
        th = turning_angles(X, True)
        if n == 16:
            close("regular 16-gon: sum of turning angles = 2 pi (ccw)", th.sum(), 2 * np.pi, tol=1e-12)
            close("regular 16-gon traversed clockwise: sum = -2 pi", turning_angles(X[::-1], True).sum(), -2 * np.pi,
                  tol=1e-12)
        lbar = curve_dual_lengths(X, True)
        for key, fn in DISCRETE_CURVATURES.items():
            k = fn(X, True) if key == "D" else fn(X, True) / lbar
            errs[key].append(np.max(np.abs(k - 1.0 / r)))
    orders = {key: -np.polyfit(np.log(ns), np.log(errs[key]), 1)[0] for key in "AC"}
    # Observed on the regular n-gon (radius r): kappa^A / lbar and kappa^C / lbar converge to 1/r at order 2.00
    # (errors ~ (pi/n)^2 / (6 r) and (pi/n)^2 / (2 r)); kappa^B / lbar = 1/r and kappa^D = 1/r EXACTLY for every n
    # (the polygon's vertices lie on the circle, so the circumcircle is the circle itself).
    check(f"n-gon: kappa^A/lbar, kappa^C/lbar -> 1/r at order 2 (observed {orders['A']:.2f}, {orders['C']:.2f})",
          all(1.95 < orders[key] < 2.05 for key in "AC"))
    check(f"n-gon: kappa^B/lbar and kappa^D are exactly 1/r (max errors {max(errs['B']):.1e}, {max(errs['D']):.1e})",
          max(errs["B"]) < 1e-12 and max(errs["D"]) < 1e-12)
    # ellipse (2 cos t, sin t) with uniform t: all four converge at order 2 (observed ~2.0, see the PASS line)
    eorders = {}
    ns_e = [64, 128, 256, 512]
    eerr = {key: [] for key in "ABCD"}
    for n in ns_e:
        t = 2 * np.pi * np.arange(n) / n
        X = np.stack([2 * np.cos(t), np.sin(t)], 1)
        ks = 2.0 / (4 * np.sin(t) ** 2 + np.cos(t) ** 2) ** 1.5
        lbar = curve_dual_lengths(X, True)
        for key, fn in DISCRETE_CURVATURES.items():
            k = fn(X, True) if key == "D" else fn(X, True) / lbar
            eerr[key].append(np.max(np.abs(k - ks)))
    for key in "ABCD":
        eorders[key] = np.polyfit(np.log(ns_e), np.log(eerr[key]), 1)[0]
    check("ellipse (2cos t, sin t), uniform t: all four discrete curvatures converge at order 2 (observed "
          + ", ".join(f"{key} {-eorders[key]:.2f}" for key in "ABCD") + ")",
          all(-2.1 < eorders[key] < -1.9 for key in "ABCD"))
    # kappa^D = 1 / circumradius, kappa^C from the extended offset polygon (Steiner)
    t = np.sort(rng.uniform(0, 2 * np.pi, 9))
    X = np.stack([np.cos(t) * (1 + 0.2 * np.cos(3 * t)), np.sin(t) * (1 + 0.2 * np.cos(3 * t))], 1)
    tri_ids = np.stack([np.roll(np.arange(9), 1), np.arange(9), np.roll(np.arange(9), -1)], 1)
    rad = np.linalg.norm(circumcenters(X, tri_ids) - X, axis=1)
    close("kappa^D = signed 1 / circumradius of (x_{i-1}, x_i, x_{i+1})", curvature_osculating_circle(X, True),
          np.sign(turning_angles(X, True)) / rad, tol=1e-10)
    eps = 1e-3
    Tn = curve_tangents(X, True)
    Nout = np.stack([Tn[:, 1], -Tn[:, 0]], 1)            # -J t: outward for a ccw curve
    P0 = X + eps * Nout                                  # offset edge i: P0[i] + s Tn[i]
    off = []
    for i_ in range(9):                                  # corner at vertex i: offset edges i-1 and i meet
        a0, ta, b0, tb = P0[i_ - 1], Tn[i_ - 1], P0[i_], Tn[i_]
        s_ = np.linalg.solve(np.stack([ta, -tb], 1), b0 - a0)
        off.append(a0 + s_[0] * ta)
    close("Steiner: extended eps-offset polygon has length L + eps sum kappa^C", curve_length(np.array(off), True),
          curve_length(X, True) + eps * curvature_steiner(X, True).sum(), tol=1e-12)
    # length gradient
    for closed in (True, False):
        for dim in (2, 3):
            X = rng.normal(size=(7, dim))
            G = length_gradient(X, closed)
            Gfd = np.zeros_like(X)
            for a in range(7):
                for c in range(dim):
                    dX = np.zeros_like(X)
                    dX[a, c] = 1e-6
                    Gfd[a, c] = (curve_length(X + dX, closed) - curve_length(X - dX, closed)) / 2e-6
            close(f"length gradient = finite differences ({'closed' if closed else 'open'}, R^{dim})", G, Gfd,
                  tol=1e-8)
    X = rng.normal(size=(8, 2))
    close("length gradient = -kappa^B n (plane closed curve; |grad| = 2|sin(theta/2)|)", length_gradient(X, True),
          -curvature_length_variation(X, True)[:, None] * curve_vertex_normals(X, True), tol=1e-12)
    # helix (a, b) = (1, 0.3) from dgsym.EXAMPLES: discrete curvature and torsion converge to kappa, tau
    import sympy as sp
    import dgsym
    ex = dgsym.EXAMPLES["helix"]
    (tt,) = ex["coords"]
    sa, sb = ex["params"]
    for bval in (0.3, -0.3):
        pars = {sa: 1.0, sb: bval}
        gamma = sp.lambdify(tt, ex["expr"].subs(pars), "numpy")
        kap_ex = float(ex["expected"]["kappa"].subs(pars))
        tau_ex = float(ex["expected"]["tau"].subs(pars))
        herr = []
        for n in (50, 100, 200, 400):
            ts = np.linspace(0, 4 * np.pi, n)
            X = np.array([np.asarray(gamma(tv), float).ravel() for tv in ts])
            kap, tau = space_curve_curvature_torsion(X, False)
            herr.append((np.nanmax(np.abs(kap - kap_ex)), np.nanmax(np.abs(tau - tau_ex))))
        herr = np.array(herr)
        oks = np.polyfit(np.log([50, 100, 200, 400]), np.log(herr), 1)[0]
        hand = "right" if bval > 0 else "left"
        # Observed: both errors decrease at order 2.00 in the step (kappa and tau of GUIDELINES 7 reproduced).
        check(f"helix (1, {bval}) ({hand}-handed): discrete kappa -> {kap_ex:.4f}, tau -> {tau_ex:.4f} "
              f"(errors {herr[-1, 0]:.1e}, {herr[-1, 1]:.1e}; orders {-oks[0]:.2f}, {-oks[1]:.2f})",
              herr[-1].max() < 1e-4 and np.all(oks < -1.9) and np.sign(np.nanmean(tau)) == np.sign(bval))
    # Bishop frames
    ts = np.linspace(0, 4 * np.pi, 60)
    X = np.stack([np.cos(ts), np.sin(ts), 0.3 * ts], 1)
    T, U, Wf = parallel_transport_frames(X)
    check("Bishop frame is orthonormal with U, W orthogonal to T",
          np.abs(np.sum(T * U, 1)).max() < 1e-12 and np.abs(np.linalg.norm(U, axis=1) - 1).max() < 1e-12
          and np.abs(np.sum(Wf * T, 1)).max() < 1e-12)
    Bv = vertex_binormals(X, False)
    psi = torsion_angles(X, False)
    phi = [np.arctan2(np.dot(np.cross(Bv[i_], U[i_]), T[i_]), np.dot(Bv[i_], U[i_])) for i_ in range(1, len(T))]
    dphi = np.angle(np.exp(1j * np.diff(phi)))
    close("Bishop frame turns by -psi_i relative to the discrete binormals (exact)", dphi, -psi[1:-1], tol=1e-10)
    t = 2 * np.pi * np.arange(80) / 80
    Xc = np.stack([np.cos(t), np.sin(t), 0.6 * np.sin(2 * t)], 1)
    hol = bishop_holonomy(Xc)
    close("closed space curve: Bishop holonomy = -sum psi (mod 2 pi)", np.angle(np.exp(1j * (hol + torsion_angles(
        Xc, True).sum()))), 0.0, tol=1e-10)
    close("closed plane curve: Bishop holonomy = 0", bishop_holonomy(np.stack([np.cos(t), 2 * np.sin(t), 0 * t], 1)),
          0.0, tol=1e-12)

    # ---------------------------------------------------------------- 9. derivatives of geometric quantities
    def fd(fun, X, h=1e-6):
        X = np.asarray(X, float)
        G = np.zeros_like(X)
        for idx in np.ndindex(*X.shape):
            dX = np.zeros_like(X)
            dX[idx] = h
            G[idx] = (fun(X + dX) - fun(X - dX)) / (2 * h)
        return G

    a, b = rng.normal(size=3), rng.normal(size=3)
    ga, gb = edge_length_gradient(a, b)
    close("edge length gradient = finite differences",
          np.concatenate([ga, gb]), fd(lambda P: np.linalg.norm(P[3:] - P[:3]), np.concatenate([a, b])), tol=1e-8)
    for dim in (3, 2):
        P = rng.normal(size=(3, dim))
        Ftri = np.array([[0, 1, 2]])
        G = triangle_area_gradients(P, Ftri)[0]
        close(f"triangle area gradient = finite differences (R^{dim})", G,
              fd(lambda Q: dm.face_areas(_as3d(Q), Ftri)[0], P), tol=1e-8)
        g3 = angle_gradient(P[0], P[1], P[2])
        close(f"corner angle gradient (all three vertices) = finite differences (R^{dim})", np.stack(g3),
              fd(lambda Q: dm.corner_angles(_as3d(Q), Ftri)[0, 0], P), tol=1e-7)
    Vp, Fp = dm.icosphere(1)
    Vp = Vp * (1 + 0.1 * rng.normal(size=(len(Vp), 1))) + 0.02 * rng.normal(size=Vp.shape)
    CG = corner_angle_gradients(Vp, Fp)
    close("corner_angle_gradients (vectorized) = finite differences on a perturbed icosphere(1), face 7",
          CG[7].reshape(3, -1),
          np.stack([fd(lambda Q, c=c: dm.corner_angles(Q, np.array([[0, 1, 2]]))[0, c], Vp[Fp[7]]).ravel()
                    for c in range(3)]), tol=1e-7)
    close("total area gradient = finite differences (perturbed icosphere(1))", area_gradient(Vp, Fp),
          fd(lambda Q: dm.face_areas(Q, Fp).sum(), Vp), tol=1e-7)
    close("total area gradient = -L x exactly (L = dgmesh.cotan_laplacian)", area_gradient(Vp, Fp),
          -dm.cotan_laplacian(Vp, Fp) @ Vp, tol=1e-12)
    close("area gradient on a planar grid = -L x (boundary included)", area_gradient(grid_V, grid_F),
          -dm.cotan_laplacian(grid_V3, grid_F)[:, :] @ grid_V, tol=1e-12)
    close("signed volume gradient = finite differences (perturbed icosphere(1))", signed_volume_gradient(Vp, Fp),
          fd(lambda Q: dm.signed_volume(Q, Fp), Vp), tol=1e-8)
    an = np.zeros_like(Vp)
    for c in range(3):
        np.add.at(an, Fp[:, c], dm.face_cross(Vp, Fp) / 6.0)
    close("closed mesh: volume gradient = (1/3) sum A_f N_f (translation invariant)", signed_volume_gradient(Vp, Fp),
          an, tol=1e-13)
    for rad in (1.0, 2.0):
        Vs, Fs = dm.icosphere(3, r=rad)
        H = mean_curvature(Vs, Fs)
        close(f"mean curvature (mixed areas) on the sphere of radius {rad}, outward N: H = -1/r "
              f"(icosphere(3), max relative error < 1e-4)", np.max(np.abs(H * rad + 1.0)), 0.0, tol=1e-4)
    Vs, Fs = dm.icosphere(3)
    err_b = np.abs(mean_curvature(Vs, Fs, "barycentric") + 1.0)
    val = np.bincount(Fs.ravel())
    # Observed for levels 2-5: max |H + 1| with barycentric areas is 0.134, 0.143, 0.145, 0.146 (at the valence-5
    # vertices, no pointwise convergence) while the mean error falls like h^2; with mixed areas it is < 1e-4.
    check(f"barycentric areas: |H + 1| = {err_b[val == 5].min():.3f} at the 12 valence-5 vertices but "
          f"<= {err_b[val == 6].max():.3f} at valence 6 (icosphere(3); no pointwise convergence at irregular vertices)",
          err_b[val == 5].min() > 0.1 and err_b[val == 6].max() < 0.01)
    gA, gV = area_gradient(Vs, Fs), signed_volume_gradient(Vs, Fs)
    close("Euler homogeneity (exact): sum_i <grad_i Area, x_i> = 2 Area, sum_i <grad_i Vol, x_i> = 3 Vol",
          [np.sum(gA * Vs), np.sum(gV * Vs)], [2 * dm.face_areas(Vs, Fs).sum(), 3 * dm.signed_volume(Vs, Fs)],
          tol=1e-12)
    Nv = dm.vertex_normals(Vs, Fs, "area")
    close("unit sphere: <grad_i Area, N_i> ~ -2H A_i = 2 A_i (mixed area), <grad_i Vol, N_i> ~ A_i (barycentric)",
          [np.max(np.abs(np.sum(gA * Nv, 1) / dual_areas(Vs, Fs, "mixed") - 2.0)),
           np.max(np.abs(np.sum(gV * Nv, 1) / dual_areas(Vs, Fs) - 1.0))], [0.0, 0.0], tol=0.01)
    check("unit sphere: grad Area points outward at every vertex (H = -1 < 0), mean curvature normal ~ -2 x",
          np.all(np.sum(gA * Vs, 1) > 0) and np.max(np.abs(mean_curvature_normal(Vs, Fs) + 2 * Vs)) < 0.02)

    summary()


if __name__ == "__main__":
    _selftest()
