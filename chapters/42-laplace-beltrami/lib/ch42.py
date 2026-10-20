"""Helpers for Chapter 42 (The Laplace-Beltrami Operator). Dense numpy only (no scipy, GUIDELINES appendix E).

Conventions are those of tools/dgmesh.py and tools/dgdec.py: L = dgmesh.cotan_laplacian (L_ij = (cot a + cot b)/2,
L_ii = -sum_j L_ij, -L positive semidefinite), M a diagonal mass matrix (dual areas), M^{-1} L ~ Delta = div grad.

Contents
- galerkin_mass(V, F): the consistent (Galerkin) P1 mass matrix M^G_ij = int phi_i phi_j dA.
- stiffness(V, F): the P1 stiffness matrix a_ij = int <grad phi_i, grad phi_j> dA, assembled face by face.
- whitney1_mass(V, F): the Galerkin mass matrix of Whitney 1-forms, M1_ee' = int <phi_e, phi_e'> dA.
- cholesky / chol_solve: dense Cholesky factor and two triangular solves (numpy has no triangular solver).
- solve_closed_poisson(L, m, rho): L u = M (rho - mean), area-weighted mean of u equal to 0.
- jittered_sphere(level, s, seed): icosphere with tangential noise, projected back to the unit sphere.
- disk_mesh(n): grid_mesh on [-1, 1]^2 restricted to the faces whose centroid lies in the unit disk.
- boundary_load(V, F, g, order): b_i = int over the boundary of g phi_i ds (Gauss-Legendre on each boundary edge).
- mass(V, F, kind): diagonal mass "barycentric" | "circumcentric" | "mixed" (dgdec.dual_areas).
- implicit_mcf(V, F, h, steps, ...): semi-implicit mean curvature flow (M - h L) x' = M x.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402


def mass(V, F, kind="mixed"):
    """Diagonal of the lumped mass matrix: dual cell areas of the given kind (dgdec.dual_areas)."""
    return dd.dual_areas(V, F, kind)


def stiffness(V, F):
    """P1 stiffness matrix a_ij = int <grad phi_i, grad phi_j> dA (n0 x n0), face by face (Lemma 42.3.2).
    V is (n0, 3); planar meshes have z = 0."""
    n = len(V)
    A = np.zeros((n, n))
    for tri in F:
        P = V[tri]
        G = dd.barycentric_gradients(P)                          # (3, 3) gradients of the hat functions on tri
        area = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
        A[np.ix_(tri, tri)] += area * G @ G.T
    return A


def galerkin_mass(V, F):
    """Consistent P1 mass matrix M^G_ij = int phi_i phi_j dA: (A_f / 12) (1 + delta_ij) on each face f
    (Proposition 42.3.8). V is (n0, 3)."""
    n = len(V)
    M = np.zeros((n, n))
    local = (np.ones((3, 3)) + np.eye(3)) / 12.0
    for tri, a in zip(F, dm.face_areas(V, F)):
        M[np.ix_(tri, tri)] += a * local
    return M


def whitney1_mass(V, F, order=4):
    """Galerkin mass matrix of the Whitney 1-forms, M1_ee' = int <phi_e, phi_e'> dA, in dgdec.edge_list order.
    The integrand is quadratic on each face; the degree-4 triangle rule of dgdec is exact for it."""
    E, index = dd.edge_list(F)
    M1 = np.zeros((len(E), len(E)))
    bary, w = dd.triangle_quadrature(order)
    for tri in F:
        tri = [int(t) for t in tri]
        P = V[tri]
        area = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
        g = dd.barycentric_gradients(P)
        ids, phis = [], []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = tri[a], tri[b]
            sign = 1.0 if i < j else -1.0
            ids.append(index[(min(i, j), max(i, j))])
            phis.append(sign * dd.whitney_basis(bary, g, a, b))       # (q, d) at the quadrature points
        for p, ep in zip(ids, phis):
            for q, eq in zip(ids, phis):
                M1[p, q] += area * np.sum(w * np.sum(ep * eq, 1))
    return M1


def cholesky(A):
    """Lower-triangular C with A = C C^T (A symmetric positive definite)."""
    return np.linalg.cholesky(A)


def chol_solve(C, b):
    """Solve C C^T x = b by forward and back substitution (C lower triangular). b may be (n,) or (n, k)."""
    b = np.array(b, float)
    one = b.ndim == 1
    y = b.reshape(len(b), -1).copy()
    n = len(C)
    for i in range(n):                                   # C y = b
        y[i] = (y[i] - C[i, :i] @ y[:i]) / C[i, i]
    x = y
    for i in range(n - 1, -1, -1):                       # C^T x = y
        x[i] = (x[i] - C[i + 1:, i] @ x[i + 1:]) / C[i, i]
    return x[:, 0] if one else x


def solve_closed_poisson(L, m, rho):
    """On a connected closed mesh: remove the mean of rho, solve L u = M (rho - rho_bar) with (-L + a a^T) u = -b,
    a = m (the vertex areas), so that a^T u = 0 (Proposition 42.6.1). Returns (u, rho_bar)."""
    rho_bar = (m @ rho) / m.sum()
    b = m * (rho - rho_bar)
    u = np.linalg.solve(-L + np.outer(m, m), -b)
    return u, rho_bar


def jittered_sphere(level, s=0.25, seed=42):
    """icosphere(level) whose vertices are moved tangentially by s * (mean edge length) * N(0, 1) and projected back
    to the unit sphere (seed fixed). The result has obtuse triangles and non-Delaunay edges for s ~ 0.3."""
    V, F = dm.icosphere(level)
    rng = np.random.default_rng(seed)
    ell = dd.primal_edge_lengths(V, F).mean()
    D = rng.normal(size=V.shape)
    D -= np.sum(D * V, 1, keepdims=True) * V                   # tangential part
    W = V + s * ell * D
    return W / np.linalg.norm(W, axis=1, keepdims=True), F


def disk_mesh(n):
    """grid_mesh on [-1, 1]^2 with n x n squares (alternating diagonals), keeping the faces whose centroid lies in the
    unit disk. Returns (V, F) with vertices renumbered (dgmesh.submesh)."""
    xs = np.linspace(-1.0, 1.0, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    c = V[F].mean(1)
    Vs, Fs, _ = dm.submesh(V, F, np.sum(c[:, :2] ** 2, 1) < 1.0)
    return Vs, Fs


def boundary_load(V, F, g, order=4):
    """b_i = int over the boundary of g phi_i ds, for g(p, n) a function of the point p and the outward unit
    conormal n of the boundary edge (planar mesh, counterclockwise faces). Gauss-Legendre with `order` points."""
    b = np.zeros(len(V))
    t, w = dd.gauss_legendre_01(order)
    for a, c in dm.boundary_edges(F):
        # orient the edge as the face traverses it (counterclockwise faces: the outward normal is on the right)
        for tri in F:
            k = list(tri)
            if a in k and c in k:
                ia = k.index(a)
                if k[(ia + 1) % 3] != c:
                    a, c = c, a
                break
        e = V[c] - V[a]
        ell = np.linalg.norm(e)
        nrm = np.array([e[1], -e[0], 0.0])[:len(V[a])] / ell      # right of a -> c is outside
        for tk, wk in zip(t, w):
            p = V[a] + tk * e
            gv = g(p, nrm)
            b[a] += wk * ell * gv * (1 - tk)
            b[c] += wk * ell * gv * tk
    return b


def implicit_mcf(V, F, h, steps, kind="mixed", fixed_L=None, record=None):
    """Semi-implicit mean curvature flow: (M - h L) x^{k+1} = M x^k with L, M of the current mesh
    (fixed_L: use this L at every step, e.g. the initial one, as in [Kazhdan12]). Returns the list of positions."""
    X = [V.copy()]
    for _ in range(steps):
        Vk = X[-1]
        L = dm.cotan_laplacian(Vk, F) if fixed_L is None else fixed_L
        m = mass(Vk, F, kind)
        X.append(np.linalg.solve(np.diag(m) - h * L, m[:, None] * Vk))
    return X


def run_printed(page, block, out_block, namespace=None):
    """Run <pre id=block> of the page with dgcode.run_block; return (namespace, printed output equals the text of
    <pre id=out_block> line by line (trailing spaces ignored), the printed text). Same as lib/ch36.run_printed."""
    import contextlib
    import io

    import dgcode
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(page, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(page, out_block)), buf.getvalue()


def circle_mean(u, p, eps, n=720):
    """Mean of u(x, y) over the circle of radius eps about p (trapezoidal rule, spectrally accurate)."""
    t = 2 * np.pi * np.arange(n) / n
    return np.mean(u(p[0] + eps * np.cos(t), p[1] + eps * np.sin(t)))
