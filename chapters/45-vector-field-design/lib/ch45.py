"""Chapter 45 (Vector Field Decomposition and Design) helpers. Used by the figure and verify scripts of Chapter 45;
not a shared module (tools/dgmesh.py and tools/dgdec.py are the shared ones). Dense numpy only (no scipy).

Conventions (Section 45.1, Conventions box; GUIDELINES 2.5, 6.6):
- V: (n0, 3) vertex positions, F: (n2, 3) faces oriented outward on closed meshes. Edges [i, j] with i < j are oriented
  i -> j and indexed by dgdec.edge_list(F). For the edge e = [i, j] the LEFT face f contains the halfedge i -> j and
  the RIGHT face g contains j -> i; the dual edge e* runs g -> f (Remark 36.5.1), so (d1^T w)_e = w_f - w_g and the
  dual exterior derivative on dual 1-forms is tilde d1 = -d0^T (Proposition 36.5.5).
- star1 = dgdec.hodge1 = (cot a + cot b)/2, L = -d0^T star1 d0 (Definition 36.6.6), <a, b>_1 = a^T star1 b.
- Discrete Hodge decomposition: omega = d0 alpha + star1^{-1} d1^T bt + gamma, bt a dual 0-form (one value per face).
  The co-exact part is the discrete d*_2 beta with bt = star2 beta; in the smooth picture d* beta = star dv with
  v = -star beta, i.e. v ~ -bt (Section 45.1).
- A unit tangent field on faces is an angle psi_f measured counterclockwise (about N_f) from the reference direction
  r_f = unit(x_{F[f,1]} - x_{F[f,0]}): X_f = cos psi_f r_f + sin psi_f N_f x r_f.
- Levi-Civita transport across e* (from g to f) is the unfolding map P_e of (44.3.4): it fixes the edge direction and
  sends N_g x e to N_f x e. rho_e (in (-pi, pi]) is the angle of P_e(r_g) measured from r_f, so a vector at angle psi
  in g arrives at angle psi + rho_e in f.
- A discrete connection is a dual 1-form xi (one number per oriented dual edge): crossing e* in its direction
  rotates by rho_e + xi_e, crossing it backwards by -(rho_e + xi_e). xi = 0 is the discrete Levi-Civita connection.
- Holonomy along a dual cycle c* (coefficients in R^{n1}, +1 when e* is traversed g -> f) is c*.(rho + xi) mod 2 pi.
  Around the dual cell of vertex i (counterclockwise, Remark 36.5.1) this is (tilde d1 (rho + xi))_i and for xi = 0
  it equals the angle defect delta_i mod 2 pi (Theorem 45.3.10). The curvature of xi is kappa = delta - d0^T xi.
- Trivial connection with indices k: -d0^T xi = -delta + 2 pi k on dual cells, c*_a.(rho + xi) = 2 pi n_a on the
  dual generator loops (Section 45.4).

Contents
- wrap, run_printed, left_right_faces, unit_cube, flat_torus, corner_data, stars_from_corners
- transport_angles (embedded meshes), transport_angles_flat (flat torus), face_frames3, face_vectors
- solve_closed (L alpha = b on a closed mesh), hodge_decomposition
- tree_cotree, primal_loop, dual_loop, generators, harmonic_basis
- trivial_connection, parallel_field, field_indices, connection_from_field
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

from collections import deque  # noqa: E402

import numpy as np  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def wrap(x):
    """angle(s) wrapped to (-pi, pi]."""
    y = np.mod(np.asarray(x, float) + np.pi, TWO_PI) - np.pi
    return np.where(y <= -np.pi, y + TWO_PI, y) + 0.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def run_printed(page, block, out_block, namespace=None):
    """Run <pre id=block> of the page with dgcode.run_block; return (namespace, printed output equals the text of
    <pre id=out_block> line by line (trailing spaces ignored), the printed text). Same as lib/ch42.run_printed."""
    import contextlib
    import io

    import dgcode
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(page, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(page, out_block)), buf.getvalue()


def left_right_faces(F):
    """(E, left, right): edge_list order; left[e] is the face containing the halfedge i -> j of e = [i, j] (i < j),
    right[e] the face containing j -> i (-1 on a boundary edge). The dual edge e* runs right -> left."""
    F = np.asarray(F, int)
    E, idx = dd.edge_list(F)
    left = -np.ones(len(E), int)
    right = -np.ones(len(E), int)
    for f, t in enumerate(F):
        for c in range(3):
            a, b = int(t[c]), int(t[(c + 1) % 3])
            e = idx[(min(a, b), max(a, b))]
            if a < b:
                left[e] = f
            else:
                right[e] = f
    return E, left, right


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def unit_cube():
    """Unit cube [0, 1]^3, each square split by one diagonal (8 vertices, 18 edges, 12 faces), faces outward.
    Vertex index 4x + 2y + z sits at (x, y, z). Every corner has delta = pi/2 (Example 40.1.2(b)); star1 = 1 on the
    12 cube edges (two 45-degree angles) and 0 on the 6 diagonals (two right angles)."""
    V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    F = []
    for q in quads:
        F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    F = np.array(F, int)
    if dm.signed_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


class FlatTorus:
    """Equilateral triangulation of the flat torus R^2 / (W Z x H Z), W = n s, H = rows (sqrt 3 / 2) s (rows even).
    Vertex index r n + c sits at P = (s (c + (r mod 2)/2), s r sqrt(3)/2) in the fundamental rectangle. Every face is
    counterclockwise in the plane; C[f] holds its three corners in unwrapped plane coordinates, so that the edge vectors
    of a face are the true displacements. Intrinsically flat: every angle is 60 degrees, star1 = 1/sqrt 3, delta = 0."""

    def __init__(self, n=12, rows=12, s=None):
        assert rows % 2 == 0 and n >= 3 and rows >= 4
        s = 1.0 / n if s is None else s
        h = s * np.sqrt(3) / 2
        self.n, self.rows, self.s, self.W, self.H = n, rows, s, n * s, rows * h
        vid = lambda r, c: (r % rows) * n + (c % n)  # noqa: E731
        pos = lambda r, c: np.array([s * (c + (r % 2) / 2), h * r])  # noqa: E731  (unwrapped)
        self.P = np.array([pos(r, c) for r in range(rows) for c in range(n)])
        F, C = [], []
        for r in range(rows):
            for c in range(n):
                if r % 2 == 0:
                    tris = [((r, c), (r, c + 1), (r + 1, c)), ((r, c + 1), (r + 1, c + 1), (r + 1, c))]
                else:
                    tris = [((r, c), (r, c + 1), (r + 1, c + 1)), ((r, c), (r + 1, c + 1), (r + 1, c))]
                for t in tris:
                    F.append([vid(*q) for q in t])
                    C.append([pos(*q) for q in t])
        self.F = np.array(F, int)
        self.C = np.array(C, float)
        # unwrapped (2D) displacement of every edge [i, j], i < j, from x_i to x_j
        E, idx = dd.edge_list(self.F)
        self.E = E
        self.edge_vec = np.zeros((len(E), 2))
        for f, t in enumerate(self.F):
            for a in range(3):
                i, j = int(t[a]), int(t[(a + 1) % 3])
                d = self.C[f, (a + 1) % 3] - self.C[f, a]
                if i < j:
                    self.edge_vec[idx[(i, j)]] = d
                else:
                    self.edge_vec[idx[(j, i)]] = -d

    def centroids(self):
        return self.C.mean(1)


def corner_data(C):
    """Per-face corner angles (n2, 3) and areas (n2,) from corner coordinates C (n2, 3, d), d = 2 or 3."""
    C = np.asarray(C, float)
    ang = np.zeros(C.shape[:2])
    for c in range(3):
        a = C[:, (c + 1) % 3] - C[:, c]
        b = C[:, (c + 2) % 3] - C[:, c]
        cosang = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        ang[:, c] = np.arccos(np.clip(cosang, -1, 1))
    e1, e2 = C[:, 1] - C[:, 0], C[:, 2] - C[:, 0]
    if C.shape[-1] == 2:
        area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    else:
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    return ang, area


def stars_from_corners(F, C):
    """(star0 barycentric, star1 cotan, star2 = 1/area) from per-face corner coordinates; same formulas as
    dgdec.hodge0/1/2 (Definition 36.5.14), usable for the flat torus where no global positions exist."""
    F = np.asarray(F, int)
    E, idx = dd.edge_list(F)
    ang, area = corner_data(C)
    s1 = np.zeros(len(E))
    s0 = np.zeros(int(F.max()) + 1)
    for f, t in enumerate(F):
        for c in range(3):
            i, j = int(t[(c + 1) % 3]), int(t[(c + 2) % 3])
            s1[idx[(min(i, j), max(i, j))]] += 0.5 / np.tan(ang[f, c])
            s0[int(t[c])] += area[f] / 3.0
    return s0, s1, 1.0 / area


# ---------------------------------------------------------------------------
# transport angles, frames, vectors
# ---------------------------------------------------------------------------


def face_frames3(V, F):
    """(r, b, N) per face: r_f = unit(x_{F[f,1]} - x_{F[f,0]}), N_f the unit normal, b_f = N_f x r_f."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    N = dm.face_normals(V, F)
    r = unit(V[F[:, 1]] - V[F[:, 0]])
    return r, np.cross(N, r), N


def transport_angles(V, F):
    """rho_e for every edge (edge_list order) of a closed embedded mesh: the angle, measured counterclockwise in the
    left face f from r_f, of the Levi-Civita transport P_e(r_g) of the right face's reference direction (44.3.4)."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    E, f, g = left_right_faces(F)
    r, b, N = face_frames3(V, F)
    eh = unit(V[E[:, 1]] - V[E[:, 0]])
    c1 = np.einsum("ij,ij->i", r[g], eh)
    c2 = np.einsum("ij,ij->i", r[g], np.cross(N[g], eh))
    P = c1[:, None] * eh + c2[:, None] * np.cross(N[f], eh)
    return np.arctan2(np.einsum("ij,ij->i", P, b[f]), np.einsum("ij,ij->i", P, r[f]))


def transport_angles_flat(F, C):
    """rho_e on a flat mesh given by unwrapped plane corner coordinates C (n2, 3, 2): P_e is the identity of the plane,
    so rho_e = angle(r_g) - angle(r_f), wrapped."""
    F = np.asarray(F, int)
    E, f, g = left_right_faces(F)
    d = C[:, 1] - C[:, 0]
    th = np.arctan2(d[:, 1], d[:, 0])
    return wrap(th[g] - th[f])


def face_vectors(V, F, psi):
    """unit vectors X_f = cos psi_f r_f + sin psi_f N_f x r_f (n2, 3)."""
    r, b, _ = face_frames3(V, F)
    return np.cos(psi)[:, None] * r + np.sin(psi)[:, None] * b


def whitney_barycenters(F, C, omega):
    """(n2, d) Whitney interpolant (Section 36.3) of the discrete 1-form omega at every face barycenter, computed from
    per-face corner coordinates C (n2, 3, d): (1/3) sum over the face's edges a -> b of omega_ab (grad l_b - grad l_a).
    For an embedded mesh pass C = V[F]; it agrees with dgdec.whitney_at_barycenters."""
    F = np.asarray(F, int)
    _, idx = dd.edge_list(F)
    out = np.zeros((len(F), C.shape[-1]))
    for f, t in enumerate(F):
        g = dd.barycentric_gradients(C[f])
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(t[a]), int(t[b])
            w = omega[idx[(i, j)]] if i < j else -omega[idx[(j, i)]]
            out[f] += w * (g[b] - g[a]) / 3.0
    return out


def face_vectors_flat(C, psi):
    """unit plane vectors (n2, 2) of the angles psi on a flat mesh with corner coordinates C."""
    d = C[:, 1] - C[:, 0]
    th = np.arctan2(d[:, 1], d[:, 0]) + psi
    return np.stack([np.cos(th), np.sin(th)], -1)


# ---------------------------------------------------------------------------
# Hodge decomposition (Section 45.1)
# ---------------------------------------------------------------------------


def solve_closed(A, b, a=None):
    """Solve the symmetric positive semidefinite system A x = b whose kernel is the constants (b must sum to 0),
    normalized by a^T x = 0: (A + a a^T) x = b (Proposition 42.6.1 with A = -L). a defaults to 1/sqrt(n)."""
    n = len(b)
    a = np.full(n, 1.0 / np.sqrt(n)) if a is None else np.asarray(a, float)
    return np.linalg.solve(A + np.outer(a, a), b)


def hodge_decomposition(D0, D1, s1, omega):
    """omega = D0 alpha + star1^{-1} D1^T bt + gamma on a closed connected mesh with star1 > 0 (Theorem 45.1.12,
    Proposition 45.1.13). Returns (alpha, bt, gamma); alpha and bt have mean 0."""
    alpha = solve_closed(D0.T @ (s1[:, None] * D0), D0.T @ (s1 * omega))
    bt = solve_closed(D1 @ (D1.T / s1[:, None]), D1 @ omega)
    gamma = omega - D0 @ alpha - (D1.T @ bt) / s1
    return alpha, bt, gamma


GL_T, GL_W = np.polynomial.legendre.leggauss(6)
GL_T, GL_W = (GL_T + 1) / 2, GL_W / 2


def flat_torus_example(ft, a=0.05, b=0.03, c=(0.3, 0.1)):
    """The 1-form omega = du + star dv + c1 dx + c2 dy of Example 45.1.15 on the flat torus ft (FlatTorus), with
    u = a sin(2 pi x/W) cos(2 pi y/H), v = b cos(2 pi x/W + 0.4) sin(4 pi y/H); star dv = (J grad v)^flat with
    J(p, q) = (-q, p). Returns (omega (n1,), u at the vertices, v at the face centroids, h = R_1(c1 dx + c2 dy)).
    The du part is d0 u exactly; the star dv part is integrated with 6-point Gauss-Legendre along each edge."""
    W, H = ft.W, ft.H
    U = lambda x, y: a * np.sin(2 * np.pi * x / W) * np.cos(2 * np.pi * y / H)  # noqa: E731
    Vf = lambda x, y: b * np.cos(2 * np.pi * x / W + 0.4) * np.sin(4 * np.pi * y / H)  # noqa: E731

    def grad_v(x, y):
        return np.stack([-b * 2 * np.pi / W * np.sin(2 * np.pi * x / W + 0.4) * np.sin(4 * np.pi * y / H),
                         b * 4 * np.pi / H * np.cos(2 * np.pi * x / W + 0.4) * np.cos(4 * np.pi * y / H)], -1)

    P0, d = ft.P[ft.E[:, 0]], ft.edge_vec
    co = np.zeros(len(d))
    for t, w in zip(GL_T, GL_W):
        p = P0 + t * d
        g = grad_v(p[:, 0], p[:, 1])
        co += w * np.einsum("ij,ij->i", np.stack([-g[:, 1], g[:, 0]], -1), d)
    P1 = P0 + d
    h = c[0] * d[:, 0] + c[1] * d[:, 1]
    omega = U(P1[:, 0], P1[:, 1]) - U(P0[:, 0], P0[:, 1]) + co + h
    cen = ft.centroids()
    return omega, U(ft.P[:, 0], ft.P[:, 1]), Vf(cen[:, 0], cen[:, 1]), h


def sphere_example(V, F):
    """omega = R_1(dz + star dx) on a mesh inscribed in the unit sphere (Example 45.1.14): the exact part is d0 z
    exactly; star dx = (J grad x)^flat, J = N x . with N = p/|p|, so (J grad x)(p) = (p/|p|) x e_x, integrated with
    6-point Gauss-Legendre along each chord. Smooth answer: u = z, v = x, i.e. bt ~ -x."""
    E, _ = dd.edge_list(F)
    a, b = V[E[:, 0]], V[E[:, 1]]
    ex = np.array([1.0, 0.0, 0.0])
    co = np.zeros(len(E))
    for t, w in zip(GL_T, GL_W):
        p = a + t * (b - a)
        co += w * np.einsum("ij,ij->i", np.cross(unit(p), ex), b - a)
    return V[E[:, 1], 2] - V[E[:, 0], 2] + co


# ---------------------------------------------------------------------------
# tree-cotree, generators, harmonic basis (Section 45.2)
# ---------------------------------------------------------------------------


def tree_cotree(F, root_vertex=0, root_face=0):
    """Tree-cotree decomposition of a closed connected oriented mesh (Definition 45.2.9): BFS spanning tree T of the
    primal graph, then BFS spanning tree T* of the dual graph using only dual edges of edges not in T.
    Returns dict(T=set of edges, Ts=set of edges whose dual is in T*, rest=list of the leftover edges,
    vpar={vertex: (parent vertex, edge)}, fpar={face: (parent face, edge)})."""
    F = np.asarray(F, int)
    E, left, right = left_right_faces(F)
    n0, n2 = int(F.max()) + 1, len(F)
    nbr = [[] for _ in range(n0)]
    for e, (i, j) in enumerate(E):
        nbr[i].append((int(j), e))
        nbr[j].append((int(i), e))
    T, vpar, bag = set(), {root_vertex: (-1, -1)}, deque([root_vertex])
    while bag:
        i = bag.popleft()
        for j, e in nbr[i]:
            if j not in vpar:
                vpar[j] = (i, e)
                T.add(e)
                bag.append(j)
    fnbr = [[] for _ in range(n2)]
    for e in range(len(E)):
        if e not in T:
            fnbr[left[e]].append((int(right[e]), e))
            fnbr[right[e]].append((int(left[e]), e))
    Ts, fpar, bag = set(), {root_face: (-1, -1)}, deque([root_face])
    while bag:
        f = bag.popleft()
        for g, e in fnbr[f]:
            if g not in fpar:
                fpar[g] = (f, e)
                Ts.add(e)
                bag.append(g)
    assert len(vpar) == n0 and len(fpar) == n2, "mesh not connected"
    rest = [e for e in range(len(E)) if e not in T and e not in Ts]
    return dict(T=T, Ts=Ts, rest=rest, vpar=vpar, fpar=fpar, E=E, left=left, right=right)


def _vertex_path_to_root(tc, v):
    out = []
    while tc["vpar"][v][0] >= 0:
        p, e = tc["vpar"][v]
        out.append((v, p, e))       # step v -> p along edge e
        v = p
    return out


def _face_path_to_root(tc, f):
    out = []
    while tc["fpar"][f][0] >= 0:
        p, e = tc["fpar"][f]
        out.append((f, p, e))       # step f -> p across dual edge e*
        f = p
    return out


def primal_loop(tc, e):
    """1-cycle c (R^{n1}, integer) of the leftover edge e = [i, j]: c_e = +1 (i -> j), then the T-path from j to i."""
    E = tc["E"]
    c = np.zeros(len(E))
    i, j = int(E[e, 0]), int(E[e, 1])
    c[e] += 1.0
    pj, pi = _vertex_path_to_root(tc, j), _vertex_path_to_root(tc, i)
    for a, b, ed in pj:             # j -> root
        c[ed] += 1.0 if E[ed, 0] == a else -1.0
    for a, b, ed in pi:             # root -> i is the reverse of i -> root
        c[ed] -= 1.0 if E[ed, 0] == a else -1.0
    return c


def dual_loop(tc, e):
    """dual 1-cycle c* (R^{n1}, integer) of the leftover edge e: c*_e = +1 (e* from right(e) to left(e)), then the
    T*-path from left(e) back to right(e). Coefficient +1 means e'* is traversed right(e') -> left(e')."""
    E, left = tc["E"], tc["left"]
    c = np.zeros(len(E))
    f, g = int(left[e]), int(tc["right"][e])
    c[e] += 1.0
    pf, pg = _face_path_to_root(tc, f), _face_path_to_root(tc, g)
    for a, b, ed in pf:             # f -> root: step a -> b across ed
        c[ed] += 1.0 if left[ed] == b else -1.0
    for a, b, ed in pg:             # root -> g is the reverse of g -> root
        c[ed] -= 1.0 if left[ed] == b else -1.0
    return c


def generators(F, **kw):
    """(tc, primal cycles (m, n1), dual cycles (m, n1)) for the m = 2g leftover edges of tree_cotree."""
    tc = tree_cotree(F, **kw)
    if not tc["rest"]:
        z = np.zeros((0, len(tc["E"])))
        return tc, z, z.copy()
    return tc, np.array([primal_loop(tc, e) for e in tc["rest"]]), np.array([dual_loop(tc, e) for e in tc["rest"]])


def harmonic_basis(D0, s1, cstars):
    """gamma_a = harmonic part of the closed 1-form omega_a = c*_a: gamma_a = omega_a - D0 alpha_a with
    D0^T star1 D0 alpha_a = D0^T star1 omega_a (one factorization for all a). Rows of the result."""
    A = D0.T @ (s1[:, None] * D0)
    n = A.shape[0]
    a = np.full(n, 1.0 / np.sqrt(n))
    K = np.linalg.cholesky(A + np.outer(a, a))
    out = []
    for w in np.atleast_2d(cstars):
        rhs = D0.T @ (s1 * w)
        alpha = np.linalg.solve(K.T, np.linalg.solve(K, rhs))
        out.append(w - D0 @ alpha)
    return np.array(out).reshape(-1, D0.shape[0])


# ---------------------------------------------------------------------------
# trivial connections (Section 45.4)
# ---------------------------------------------------------------------------


def trivial_connection(D0, s1, delta, rho, k, cstars=None, gammas=None, n=None):
    """The trivial connection of Theorem 45.4.10: xi = star1 D0 alpha + sum_b z_b star1 gamma_b with
    L alpha = -delta + 2 pi k and P z = v - C* star1 D0 alpha, P_ab = c*_a . star1 gamma_b,
    v_a = -wrap(c*_a . rho) + 2 pi n_a. Returns xi (R^{n1})."""
    u = -np.asarray(delta, float) + TWO_PI * np.asarray(k, float)
    assert abs(u.sum()) < 1e-8, "the indices must add up to the Euler characteristic"
    A = D0.T @ (s1[:, None] * D0)                      # = -L
    alpha = solve_closed(A, -u)                         # L alpha = u
    xi = s1 * (D0 @ alpha)
    if cstars is not None and len(cstars):
        cstars, gammas = np.atleast_2d(cstars), np.atleast_2d(gammas)
        n = np.zeros(len(cstars)) if n is None else np.asarray(n, float)
        v = -wrap(cstars @ rho) + TWO_PI * n
        P = cstars @ (s1[:, None] * gammas.T)
        z = np.linalg.solve(P, v - cstars @ xi)
        xi = xi + s1 * (gammas.T @ z)
    return xi


def parallel_field(F, rho, xi, f0=0, psi0=0.0):
    """psi on faces from psi[f0] = psi0 by transport across dual edges (BFS): crossing e* right -> left adds
    rho_e + xi_e. For a trivial connection the result does not depend on the BFS order (Proposition 45.4.3)."""
    E, left, right = left_right_faces(F)
    n2 = len(F)
    fnbr = [[] for _ in range(n2)]
    for e in range(len(E)):
        fnbr[right[e]].append((int(left[e]), e, +1.0))
        fnbr[left[e]].append((int(right[e]), e, -1.0))
    psi = np.full(n2, np.nan)
    psi[f0] = psi0
    bag = deque([f0])
    while bag:
        f = bag.popleft()
        for g, e, sgn in fnbr[f]:
            if np.isnan(psi[g]):
                psi[g] = psi[f] + sgn * (rho[e] + xi[e])
                bag.append(g)
    return wrap(psi)


def connection_from_field(F, rho, psi):
    """the dual 1-form xi_e = wrap(psi_left - psi_right - rho_e) for which psi is parallel (Proposition 45.4.5)."""
    E, left, right = left_right_faces(F)
    return wrap(psi[left] - psi[right] - rho)


def field_indices(D0, delta, rho, psi, F):
    """index of the face field psi at every vertex (Definition 45.4.6): (delta - D0^T xi^psi) / 2 pi, rounded."""
    xi = connection_from_field(F, rho, psi)
    k = (np.asarray(delta, float) - D0.T @ xi) / TWO_PI
    return np.rint(k).astype(int), np.abs(k - np.rint(k)).max()


def holonomy(cstar, rho, xi=None):
    """holonomy (mod 2 pi, in (-pi, pi]) of the connection rho + xi along the dual cycle cstar."""
    t = rho if xi is None else rho + xi
    return float(wrap(cstar @ t))
