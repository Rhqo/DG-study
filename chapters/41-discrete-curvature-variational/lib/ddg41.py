"""Helpers for Chapter 41 (Discrete Curvature II: The Variational Viewpoint).

The shared implementations stay in tools/dgmesh.py and tools/dgdec.py (signs, cotan matrix, area and volume
gradients). This module only adds what the figures and verify scripts of Chapter 41 need:

- running the printed code blocks of a page and capturing their output (run_printed, printed_block)
- central-difference gradients (central_gradient)
- the bending angle of a hinge and its four gradients (Proposition 41.1.8)
- totals and curvature normals on a mesh: vector_area, total_mean_curvature, gauss_curvature_normal
- sphere-inscribed vertex normals [Max99] and discrete principal curvatures (Section 41.3)
- meshes: a subdivided cube, tori with a chosen diagonal pattern, a torus with extraordinary vertices (edge flips
  followed by subdivision and projection), a jittered sphere, a saddle grid
- exact curvatures of the torus of revolution (GUIDELINES 7, outward normal)
- explicit curvature-flow steps and the discrete Willmore energy (Section 41.5)

Conventions follow GUIDELINES 2.5 and 6.3: W = -dN, H = tr(W) / 2, H = -1/r on a sphere with outward N,
(L x)_i ~ 2 H N A_i, grad Area = -(L x)_i. The time step is h.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import contextlib  # noqa: E402
import io  # noqa: E402

import numpy as np  # noqa: E402

import dgmesh as dm  # noqa: E402

# ---------------------------------------------------------------------------
# printed code blocks
# ---------------------------------------------------------------------------


def run_printed(page, block_id, ns=None):
    """run a code block of a page (tools/dgcode.py) and return (namespace, printed text)."""
    import dgcode
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(page, block_id, ns)
    return ns, buf.getvalue()


def printed_block(page, block_id):
    import dgcode
    return dgcode.code_block(page, block_id)


# ---------------------------------------------------------------------------
# finite differences
# ---------------------------------------------------------------------------


def central_gradient(f, X, eps=1e-6):
    """Central-difference gradient of the scalar function f at the array X (any shape): (f(X + eps e) - f(X - eps e))
    / (2 eps) for every coordinate direction e."""
    X = np.array(X, float)
    G = np.zeros_like(X)
    flat, gflat = X.reshape(-1), G.reshape(-1)
    for a in range(flat.size):
        old = flat[a]
        flat[a] = old + eps
        fp = f(X)
        flat[a] = old - eps
        fm = f(X)
        flat[a] = old
        gflat[a] = (fp - fm) / (2 * eps)
    return G


# ---------------------------------------------------------------------------
# a hinge: the bending angle and its gradients (Proposition 41.1.8)
# ---------------------------------------------------------------------------


def bending_angle(xi, xj, xk, xl):
    """Bending angle theta of the edge ij between f = (i, j, k) (traverses i -> j) and g = (j, i, l) (Definition
    39.5.5): N_g = R(e_hat, theta) N_f, theta = atan2(<N_f x N_g, e_hat>, <N_f, N_g>)."""
    nf = np.cross(xj - xi, xk - xi)
    ng = np.cross(xi - xj, xl - xj)
    nf, ng = nf / np.linalg.norm(nf), ng / np.linalg.norm(ng)
    e = (xj - xi) / np.linalg.norm(xj - xi)
    return np.arctan2(np.dot(np.cross(nf, ng), e), np.dot(nf, ng))


def bending_angle_gradients(xi, xj, xk, xl):
    """(g_i, g_j, g_k, g_l) of Proposition 41.1.8: g_k = -l N_f / (2 A_f), g_l = -l N_g / (2 A_g),
    g_i = -(1 - s_k) g_k - (1 - s_l) g_l, g_j = -s_k g_k - s_l g_l with s = <x - x_i, e_hat> / l."""
    xi, xj, xk, xl = (np.asarray(p, float) for p in (xi, xj, xk, xl))
    ell = np.linalg.norm(xj - xi)
    e = (xj - xi) / ell
    cf, cg = np.cross(xj - xi, xk - xi), np.cross(xi - xj, xl - xj)      # 2 A_f N_f, 2 A_g N_g
    gk = -ell * cf / np.dot(cf, cf)
    gl = -ell * cg / np.dot(cg, cg)
    sk, sl = np.dot(xk - xi, e) / ell, np.dot(xl - xi, e) / ell
    return -(1 - sk) * gk - (1 - sl) * gl, -sk * gk - sl * gl, gk, gl


# ---------------------------------------------------------------------------
# totals and curvature normals on a mesh
# ---------------------------------------------------------------------------


def vector_area(V, F):
    """Barycentric discrete vector area A_i = (1/3) sum_{f ~ i} A_f N_f (Proposition 38.5.5) = grad V (Theorem 41.2.3)."""
    out = np.zeros_like(np.asarray(V, float))
    cr = dm.face_cross(V, F)
    for c in range(3):
        np.add.at(out, F[:, c], cr / 6.0)
    return out


def half_sum_ltheta(V, F):
    """(1/2) sum over interior edges of l_ij theta_ij (= -total mean curvature of Definition 40.4.14)."""
    return 0.5 * sum(t * ell for t, ell in dm.edge_bending_angles(V, F).values())


def total_mean_curvature(V, F):
    """sum_ij H_ij = -(1/2) sum l_ij theta_ij (Definition 40.4.14, the sign of GUIDELINES 6.3)."""
    return -half_sum_ltheta(V, F)


def gauss_curvature_normal(V, F):
    """(K N)_i = (1/2) sum_j theta_ij (x_i - x_j) / l_ij (Definition 40.3.9)."""
    V = np.asarray(V, float)
    out = np.zeros_like(V)
    for (i, j), (t, ell) in dm.edge_bending_angles(V, F).items():
        out[i] += 0.5 * t * (V[i] - V[j]) / ell
        out[j] += 0.5 * t * (V[j] - V[i]) / ell
    return out


def sphere_inscribed_normals(V, F):
    """Unit sphere-inscribed vertex normals [Max99] (Definition 41.3.6): the direction of
    sum over the faces (i, j, k) at i of (e_j x e_k) / (|e_j|^2 |e_k|^2), e_j = x_j - x_i, e_k = x_k - x_i."""
    V = np.asarray(V, float)
    out = np.zeros_like(V)
    for c in range(3):
        i, j, k = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        a, b = V[j] - V[i], V[k] - V[i]
        w = np.cross(a, b) / (np.sum(a * a, 1) * np.sum(b * b, 1))[:, None]
        np.add.at(out, i, w)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def principal_curvatures(H, K):
    """kappa_1 >= kappa_2 from H and K: H +- sqrt(H^2 - K), with H^2 - K clamped at 0. Returns (k1, k2, clamped)."""
    H, K = np.asarray(H, float), np.asarray(K, float)
    disc = H * H - K
    clamped = disc < 0
    s = np.sqrt(np.maximum(disc, 0.0))
    return H + s, H - s, clamped


def quadric_fit_K(V, F):
    """Gaussian curvature from a least-squares fit w = a u^2 + b u v + c v^2 to the one-ring in the frame of the
    angle-weighted vertex normal (the method of Remark 40.1.15): K = 4 a c - b^2."""
    V = np.asarray(V, float)
    Nv = dm.vertex_normals(V, F, kind="angle")
    nb = dm.neighbors(F, len(V))
    K = np.zeros(len(V))
    for i in range(len(V)):
        n = Nv[i]
        t = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        t /= np.linalg.norm(t)
        s = np.cross(n, t)
        D = V[list(nb[i])] - V[i]
        uu, vv, ww = D @ t, D @ s, D @ n
        (a, b, c), *_ = np.linalg.lstsq(np.c_[uu * uu, uu * vv, vv * vv], ww, rcond=None)
        K[i] = 4 * a * c - b * b
    return K


def willmore_energy(V, F, m=None):
    """Discrete Willmore energy sum_i |(L x)_i|^2 / (4 A_i) ~ int H^2 dA (A_i: mixed Voronoi unless m is given)."""
    m = dm.mass_voronoi(V, F) if m is None else m
    Lx = dm.cotan_apply(V, F, V)
    return float(np.sum(np.sum(Lx * Lx, 1) / (4 * m)))


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def cube_mesh(n=4, center=(0.5, 0.5, 0.5), side=1.0):
    """The cube of the given side, each square face cut into n x n squares and every small square into two triangles
    (the same diagonal on a face), faces oriented outward. Vertices on shared edges are merged."""
    pts, F, index = [], [], {}

    def vid(p):
        key = tuple(np.round(p, 12))
        if key not in index:
            index[key] = len(pts)
            pts.append(p)
        return index[key]

    c = np.asarray(center, float)
    t = np.linspace(-0.5, 0.5, n + 1) * side
    for axis in range(3):
        a, b = [q for q in range(3) if q != axis]
        for s in (0.5 * side, -0.5 * side):
            grid = np.empty((n + 1, n + 1), int)
            for p in range(n + 1):
                for q in range(n + 1):
                    x = np.zeros(3)
                    x[axis], x[a], x[b] = s, t[p], t[q]
                    grid[p, q] = vid(c + x)
            for p in range(n):
                for q in range(n):
                    v00, v10, v11, v01 = grid[p, q], grid[p + 1, q], grid[p + 1, q + 1], grid[p, q + 1]
                    F += [[v00, v10, v11], [v00, v11, v01]]
    V = np.array(pts)
    F = np.array(F, int)
    for f, (i, j, k) in enumerate(F):                            # orient every face away from the center
        nrm = np.cross(V[j] - V[i], V[k] - V[i])
        if nrm @ ((V[i] + V[j] + V[k]) / 3 - c) < 0:
            F[f] = [i, k, j]
    return V, F


def torus_point(u, v, R=2.0, r=0.8):
    return np.stack([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)], -1)


def torus_uv(P, R=2.0):
    """(u, v) of points on (or near) the torus of revolution T_{R,r}."""
    P = np.asarray(P, float)
    v = np.arctan2(P[..., 1], P[..., 0])
    u = np.arctan2(P[..., 2], np.hypot(P[..., 0], P[..., 1]) - R)
    return u, v


def torus_project(P, R=2.0, r=0.8):
    """Closest point on T_{R,r} (through the tube center circle)."""
    u, v = torus_uv(P, R)
    return torus_point(u, v, R, r)


def torus_exact(P, R=2.0, r=0.8):
    """Exact (u, K, H, kappa_1, kappa_2) at points of T_{R,r} with the outward normal (GUIDELINES 7):
    K = cos u / (r (R + r cos u)), H = -(R + 2 r cos u) / (2 r (R + r cos u)), kappa_1 >= kappa_2 the larger and the
    smaller of -1/r and -cos u / (R + r cos u)."""
    u, _ = torus_uv(P, R)
    rho = R + r * np.cos(u)
    K = np.cos(u) / (r * rho)
    H = -(R + 2 * r * np.cos(u)) / (2 * r * rho)
    a, b = -np.ones_like(u) / r, -np.cos(u) / rho
    return u, K, H, np.maximum(a, b), np.minimum(a, b)


def torus_pattern(R, r, nu, nv, pattern="same"):
    """torus_mesh(R, r, nu, nv) (pattern "same": every vertex has valence 6) or the checkerboard 4-8 pattern
    ("alternate": valence 4 and 8 alternate). nu, nv must be even for "alternate". Faces outward."""
    if pattern == "same":
        return dm.torus_mesh(R, r, nu, nv)
    u = 2 * np.pi * np.arange(nu) / nu
    v = 2 * np.pi * np.arange(nv) / nv
    U, W = np.meshgrid(u, v, indexing="ij")
    V = torus_point(U, W, R, r).reshape(-1, 3)
    idx = lambda i, j: (i % nu) * nv + (j % nv)  # noqa: E731
    F = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F += [[a, d, b], [b, d, c]] if (i + j) % 2 == 0 else [[a, d, c], [a, c, b]]
    F = np.array(F, int)
    if dm.signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def subdivided_torus(level, R=2.0, r=0.8, nu=8, nv=16):
    """torus_mesh(R, r, nu, nv) refined `level` times by midpoint subdivision, every new vertex projected to T_{R,r}.
    Returns (V, F, cls): cls[i] = 1 for the original vertices, 2 for vertices on an original edge, 3 for vertices
    inside an original face (the size of the smallest original simplex that contains the vertex)."""
    V, F = dm.torus_mesh(R, r, nu, nv)
    support = [frozenset([i]) for i in range(len(V))]
    for _ in range(level):
        n = len(V)
        cache = {}
        for tri in F:                                 # the same midpoint order as dgmesh.subdivide
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(a, b), max(a, b))
                if key not in cache:
                    cache[key] = len(support)
                    support.append(support[a] | support[b])
        V, F = dm.subdivide(V, F)
        V = np.asarray(V, float)
        V[n:] = torus_project(V[n:], R, r)
    return V, F, np.array([len(sset) for sset in support])


def flip_edge(F, i, j):
    """Flip the interior edge ij of a consistently oriented mesh: (i, j, k), (j, i, l) -> (k, l, j), (l, k, i)."""
    F = np.array(F, int)
    f = g = None
    for n, tri in enumerate(F):
        t = list(tri)
        for c in range(3):
            if t[c] == i and t[(c + 1) % 3] == j:
                f, k = n, t[(c + 2) % 3]
            if t[c] == j and t[(c + 1) % 3] == i:
                g, l = n, t[(c + 2) % 3]
    if f is None or g is None:
        raise ValueError("edge not interior")
    F[f], F[g] = [k, l, j], [l, k, i]
    return F


def torus_extraordinary(level, R=2.0, r=0.8, nu=8, nv=16):
    """A torus mesh with extraordinary vertices: torus_mesh(R, r, nu, nv), then two edge flips (each turns two valence-6
    vertices into valence 5 and two into valence 7), then `level` midpoint subdivisions with every new vertex projected
    to T_{R,r}. The original vertices keep their positions and valences; returns (V, F, ids) with ids = the indices of
    the four valence-5 and four valence-7 vertices (unchanged by subdivision)."""
    V, F = dm.torus_mesh(R, r, nu, nv)
    flips = []
    for (i0, j0) in ((1, 2), (5, 10)):                           # two well separated quads
        a, d = i0 * nv + j0, i0 * nv + j0 + 1
        b = (i0 + 1) * nv + j0
        flips.append((b, d))                                      # the diagonal b-d of torus_mesh's quad
    for b, d in flips:
        F = flip_edge(F, b, d) if _has_edge(F, b, d) else flip_edge(F, d, b)
    for _ in range(level):
        n_old = len(V)
        V, F = dm.subdivide(V, F)
        V = np.asarray(V, float)
        V[n_old:] = torus_project(V[n_old:], R, r)
    deg = np.array([len(s) for s in dm.neighbors(F, len(V))])
    ids = np.where(deg != 6)[0]
    return V, F, ids


def _has_edge(F, i, j):
    for tri in F:
        t = list(tri)
        for c in range(3):
            if t[c] == i and t[(c + 1) % 3] == j:
                return True
    return False


def jittered_sphere(level, s=0.25, seed=7):
    """icosphere(level) with every vertex moved tangentially by s * (mean edge length) * N(0, 1) and projected back to
    the unit sphere (seed fixed)."""
    V, F = dm.icosphere(level)
    rng = np.random.default_rng(seed)
    E = dm.edges(F)
    ell = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    D = rng.normal(size=V.shape)
    D -= np.sum(D * V, 1, keepdims=True) * V
    W = V + s * ell * D
    return W / np.linalg.norm(W, axis=1, keepdims=True), F


def bumpy_revolution(nu=24, nv=48):
    """The surface of revolution with profile (rho(t), z(t)) = (2 + 0.8 cos t + 0.2 cos 2t, 0.6 sin t) on the
    nu x nv grid with the connectivity of torus_mesh; returns (V, F, N) with the exact unit normals N (outward)."""
    t = 2 * np.pi * np.arange(nu) / nu
    w = 2 * np.pi * np.arange(nv) / nv
    T, Wv = np.meshgrid(t, w, indexing="ij")
    rho, z = 2 + 0.8 * np.cos(T) + 0.2 * np.cos(2 * T), 0.6 * np.sin(T)
    V = np.stack([rho * np.cos(Wv), rho * np.sin(Wv), z], -1).reshape(-1, 3)
    _, F = dm.torus_mesh(2, 0.8, nu, nv)
    if dm.signed_volume(V, F) < 0:
        F = F[:, ::-1]
    tt, ww = T.reshape(-1), Wv.reshape(-1)
    drho, dz = -0.8 * np.sin(tt) - 0.4 * np.sin(2 * tt), 0.6 * np.cos(tt)
    N = np.c_[dz * np.cos(ww), dz * np.sin(ww), -drho]
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    if np.sum(N * dm.vertex_normals(V, F, kind="area")) < 0:
        N = -N
    return V, F, N


def saddle_grid(n=17, a=0.5, pattern="alternate"):
    """grid_mesh over [-a, a]^2 lifted to the saddle z = x^2 - y^2 (K = -4, H = 0 at the origin)."""
    s = np.linspace(-a, a, n)
    V, F = dm.grid_mesh(s, s, pattern=pattern)
    V[:, 2] = V[:, 0] ** 2 - V[:, 1] ** 2
    return V, F


def mean_edge_length(V, F):
    E = dm.edges(F)
    return float(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean())


# ---------------------------------------------------------------------------
# explicit curvature flows (Section 41.5)
# ---------------------------------------------------------------------------


def mcf_velocity(V, F):
    """(M^{-1} L x)_i ~ Delta x = 2 H N: minus the area gradient divided by the mixed area."""
    return dm.cotan_apply(V, F, V) / dm.mass_voronoi(V, F)[:, None]


def gcf_velocity(V, F):
    """-(K N)_i / A_i ~ -K N: minus the gradient of (1/2) sum l theta divided by the mixed area."""
    return -gauss_curvature_normal(V, F) / dm.mass_voronoi(V, F)[:, None]


def explicit_flow(V, F, velocity, h, steps, record=None):
    """Forward Euler x <- x + h v(x) for `steps` steps; record(k, V) is called before every step and at the end."""
    V = np.array(V, float)
    for k in range(steps):
        if record is not None:
            record(k, V)
        V = V + h * velocity(V, F)
    if record is not None:
        record(steps, V)
    return V


def largest_eigenvalue(V, F, iters=200, seed=0):
    """lambda_max of M^{-1}(-L) (power iteration on M^{-1/2}(-L)M^{-1/2}); the explicit MCF step is stable for
    h <= 2 / lambda_max (Proposition 37.6.5)."""
    m = dm.mass_voronoi(V, F)
    s = 1 / np.sqrt(m)
    x = np.random.default_rng(seed).normal(size=len(V))
    lam = 0.0
    for _ in range(iters):
        y = -s * dm.cotan_apply(V, F, (s * x)[:, None])[:, 0]
        lam = float(np.dot(x, y) / np.dot(x, x))
        x = y / np.linalg.norm(y)
    return lam


def willmore_gradient_fd(V, F, eps=1e-6):
    """Central-difference gradient of willmore_energy, computed locally: moving x_i changes (L x)_j and A_j only for j
    in the closed one-ring of i, so each difference is taken on the faces touching that one-ring (exact up to
    rounding, much faster than central_gradient on the whole mesh)."""
    V = np.array(V, float)
    nbr = [set() for _ in range(len(V))]
    vf = [[] for _ in range(len(V))]
    for k, f in enumerate(F):
        for a in range(3):
            nbr[f[a]].update((f[(a + 1) % 3], f[(a + 2) % 3]))
            vf[f[a]].append(k)
    G = np.zeros_like(V)
    for i in range(len(V)):
        ring = sorted(nbr[i] | {i})
        faces = sorted({k for j in ring for k in vf[j]})
        Fs = F[faces]
        used, inv = np.unique(Fs, return_inverse=True)
        Fl, Vl = inv.reshape(Fs.shape), V[used].copy()
        sel = np.searchsorted(used, ring)
        il = int(np.searchsorted(used, i))

        def local(X):
            Lx = dm.cotan_apply(X, Fl, X)[sel]
            return float(np.sum(np.sum(Lx * Lx, 1) / (4 * dm.mass_voronoi(X, Fl)[sel])))
        for c in range(3):
            Vl[il, c] += eps
            fp = local(Vl)
            Vl[il, c] -= 2 * eps
            fm = local(Vl)
            Vl[il, c] += eps
            G[i, c] = (fp - fm) / (2 * eps)
    return G


def torus_willmore_gradient(u, R=2.0, r=0.8):
    """Smooth G = Delta H + 2 H (H^2 - K) on T_{R,r} with the outward normal (Definition 41.5.15); the L^2 gradient of
    int H^2 dA is G N. Closed form of Delta H = (1 / (r rho)) d/du((rho / r) dH/du), rho = R + r cos u."""
    c, s = np.cos(u), np.sin(u)
    rho = R + r * c
    H = -(R + 2 * r * c) / (2 * r * rho)
    K = c / (r * rho)
    # dH/du = R s / (2 rho^2), so (rho / r) dH/du = R s / (2 r rho) and its u-derivative is
    dq = R * (c * rho + r * s * s) / (2 * r * rho ** 2)
    return dq / (r * rho) + 2 * H * (H * H - K)


def roughness(V):
    """Standard deviation over mean of the distances of the vertices from their centroid (0 on a round sphere)."""
    d = np.linalg.norm(V - V.mean(0), axis=1)
    return float(d.std() / d.mean())


def noisy_sphere(level=3, amp=0.05, seed=5):
    """icosphere(level) with every vertex scaled by 1 + amp N(0, 1) (the noisy sphere of Section 42.6 for amp 0.05)."""
    V, F = dm.icosphere(level)
    return V * (1 + amp * np.random.default_rng(seed).normal(size=(len(V), 1))), F
