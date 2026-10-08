"""Triangle-mesh helpers shared by the figure and verify scripts of Chapter 31.

Conventions (Section 31.1, Conventions box):
- V: (n0, 3) float array of vertex positions x_i, F: (n2, 3) int array of faces (i, j, k).
  A face (i, j, k) is positively oriented; its normal is (x_j - x_i) x (x_k - x_i) / |...|.
- Laplacian sign follows the study set: Delta = div grad. The cotan matrix L returned here has
  L_ij = (cot alpha_ij + cot beta_ij) / 2 for an edge ij and L_ii = -sum_j L_ij, so -L is positive
  semidefinite and M^{-1} L approximates Delta. Many codes use the opposite sign (L_code = -L).
- Euler characteristic chi = n0 - n1 + n2 (GUIDELINES 6.3; we do not write V - E + F).

Import from a figure or verify script with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import dgmesh as dm

The module limits BLAS threads to 4 (this machine freezes under long all-core loads).
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

from collections import defaultdict, deque  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def icosahedron():
    """Regular icosahedron inscribed in the unit sphere, faces oriented outward."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], int)
    return V, F


def subdivide(V, F):
    """Split every triangle into four through the edge midpoints (no smoothing)."""
    V = [np.asarray(v, float) for v in V]
    cache = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            cache[key] = len(V)
            V.append(0.5 * (V[a] + V[b]))
        return cache[key]

    out = []
    for i, j, k in F:
        a, b, c = mid(i, j), mid(j, k), mid(k, i)
        out += [[i, a, c], [a, j, b], [c, b, k], [a, b, c]]
    return np.array(V), np.array(out, int)


def icosphere(level, r=1.0):
    """Icosahedron subdivided ``level`` times, vertices projected to the sphere of radius r."""
    V, F = icosahedron()
    for _ in range(level):
        V, F = subdivide(V, F)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return r * V, F


def torus_mesh(R, r, nu, nv):
    """Torus of revolution T_{R,r} (GUIDELINES 7) sampled on the (u, v) grid, faces oriented outward.

    Vertex index i*nv + j sits at u = 2 pi i / nu (around the tube), v = 2 pi j / nv (around the z-axis).
    """
    u = 2 * np.pi * np.arange(nu) / nu
    v = 2 * np.pi * np.arange(nv) / nv
    U, W = np.meshgrid(u, v, indexing="ij")
    V = np.stack([(R + r * np.cos(U)) * np.cos(W), (R + r * np.cos(U)) * np.sin(W), r * np.sin(U)], -1).reshape(-1, 3)
    idx = lambda i, j: (i % nu) * nv + (j % nv)  # noqa: E731
    F = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            # x_u x x_v points inward, so (i,j) -> (i,j+1) -> (i+1,j) is outward
            F += [[a, d, b], [b, d, c]]
    F = np.array(F, int)
    if signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def grid_mesh(xs, ys, pattern="alternate"):
    """Planar grid triangulation over the tensor grid xs x ys (z = 0).

    pattern: "same" (every quad cut by the same diagonal) or "alternate" (4-8 / checkerboard diagonals).
    Vertex index i*len(ys) + j sits at (xs[i], ys[j]).
    """
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            if pattern == "same" or (i + j) % 2 == 0:
                F += [[a, b, c], [a, c, d]]
            else:
                F += [[a, b, d], [b, c, d]]
    return V, np.array(F, int)


def mobius_mesh(n=12, w=0.35, R=1.0):
    """Moebius strip with n quads (2n triangles) along the center circle; the strip is not orientable.

    Vertices 0..n-1 are on one edge (s = -w), n..2n-1 on the other (s = +w). Faces are listed with a
    locally consistent order except across the gluing seam, where no consistent choice exists.
    """
    V = []
    for side in (-1, 1):
        for k in range(n):
            t = 2 * np.pi * k / n
            s = side * w
            V.append([(R + s * np.cos(t / 2)) * np.cos(t), (R + s * np.cos(t / 2)) * np.sin(t), s * np.sin(t / 2)])
    V = np.array(V)
    F = []
    for k in range(n):
        a, b = k, n + k
        if k < n - 1:
            c, d = k + 1, n + k + 1
        else:  # the twist: after a full turn the two edges are swapped
            c, d = n + 0, 0
        F += [[a, c, b], [b, c, d]]
    return V, np.array(F, int)


# ---------------------------------------------------------------------------
# combinatorics
# ---------------------------------------------------------------------------


def edge_faces(F):
    """dict {(i, j) with i < j: [face indices]}."""
    ef = defaultdict(list)
    for f, (i, j, k) in enumerate(F):
        for a, b in ((i, j), (j, k), (k, i)):
            ef[(min(a, b), max(a, b))].append(f)
    return ef


def edges(F):
    return np.array(sorted(edge_faces(F).keys()), int)


def counts(F):
    """(n0, n1, n2) of the mesh (n0 counts referenced vertices)."""
    n0 = len(np.unique(F))
    n1 = len(edge_faces(F))
    return n0, n1, len(F)


def euler_characteristic(F):
    n0, n1, n2 = counts(F)
    return n0 - n1 + n2


def boundary_edges(F):
    return [e for e, fs in edge_faces(F).items() if len(fs) == 1]


def vertex_faces(F):
    vf = defaultdict(list)
    for f, tri in enumerate(F):
        for v in tri:
            vf[int(v)].append(f)
    return vf


def vertex_link(F, i, vf=None):
    """Edges of the link of vertex i: for each face (i, a, b) the opposite edge {a, b}.

    ``vf`` (from vertex_faces) avoids scanning all faces."""
    cand = vf[i] if vf is not None else range(len(F))
    link = []
    for f in cand:
        tri = list(F[f])
        if i in tri:
            k = tri.index(i)
            link.append((tri[(k + 1) % 3], tri[(k + 2) % 3]))
    return link


def link_type(link):
    """Classify a vertex link (list of edges) as 'cycle', 'path', or 'other' (non-manifold)."""
    if not link:
        return "other"
    deg = defaultdict(int)
    adj = defaultdict(set)
    for a, b in link:
        deg[a] += 1
        deg[b] += 1
        adj[a].add(b)
        adj[b].add(a)
    if any(d > 2 for d in deg.values()):
        return "other"
    # connectivity of the link graph
    start = next(iter(adj))
    seen, dq = {start}, deque([start])
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                dq.append(y)
    if len(seen) != len(adj):
        return "other"
    ones = sum(1 for d in deg.values() if d == 1)
    if ones == 0:
        return "cycle"
    if ones == 2:
        return "path"
    return "other"


def manifold_report(F):
    """Return dict with non-manifold edges (>2 faces), boundary edges, and non-manifold vertices."""
    ef = edge_faces(F)
    bad_edges = [e for e, fs in ef.items() if len(fs) > 2]
    bnd = [e for e, fs in ef.items() if len(fs) == 1]
    vf = vertex_faces(F)
    bad_verts = [int(i) for i in np.unique(F) if link_type(vertex_link(F, int(i), vf)) == "other"]
    return dict(nonmanifold_edges=bad_edges, boundary_edges=bnd, nonmanifold_vertices=bad_verts)


def is_manifold(F):
    r = manifold_report(F)
    return not r["nonmanifold_edges"] and not r["nonmanifold_vertices"]


def oriented_consistently(F):
    """True if every edge shared by two faces is traversed in opposite directions by them."""
    seen = defaultdict(int)
    for i, j, k in F:
        for a, b in ((i, j), (j, k), (k, i)):
            seen[(a, b)] += 1
    return all(c == 1 for c in seen.values())


def orient(F):
    """Try to make the orientation consistent by flipping faces (breadth-first over shared edges).

    Returns (F_oriented, ok, conflict_edge). ok is False if a conflict is found (non-orientable).
    """
    F = np.array(F, int).copy()
    ef = edge_faces(F)
    nbr = defaultdict(list)
    for e, fs in ef.items():
        for a in fs:
            for b in fs:
                if a != b:
                    nbr[a].append((b, e))

    def directed(f):
        i, j, k = F[f]
        return {(i, j), (j, k), (k, i)}

    done = np.zeros(len(F), bool)
    for s in range(len(F)):
        if done[s]:
            continue
        done[s] = True
        dq = deque([s])
        while dq:
            f = dq.popleft()
            for g, (a, b) in nbr[f]:
                # f traverses the shared edge in some direction; g must traverse it the other way
                df = (a, b) if (a, b) in directed(f) else (b, a)
                want_g = (df[1], df[0])
                if not done[g]:
                    if want_g not in directed(g):
                        F[g] = F[g][::-1]
                    done[g] = True
                    dq.append(g)
                elif want_g not in directed(g):
                    return F, False, (a, b)
    return F, True, None


def components(F):
    """Connected components of faces (sharing an edge). Returns list of face-index arrays."""
    ef = edge_faces(F)
    nbr = defaultdict(set)
    for fs in ef.values():
        for a in fs:
            nbr[a].update(fs)
    lab = -np.ones(len(F), int)
    out = []
    for s in range(len(F)):
        if lab[s] >= 0:
            continue
        lab[s] = len(out)
        comp, dq = [s], deque([s])
        while dq:
            f = dq.popleft()
            for g in nbr[f]:
                if lab[g] < 0:
                    lab[g] = len(out)
                    comp.append(g)
                    dq.append(g)
        out.append(np.array(comp))
    return out


def neighbors(F, n0=None):
    n0 = n0 if n0 is not None else int(F.max()) + 1
    nb = [set() for _ in range(n0)]
    for i, j, k in F:
        nb[i] |= {j, k}
        nb[j] |= {i, k}
        nb[k] |= {i, j}
    return [sorted(s) for s in nb]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def face_cross(V, F):
    """Unnormalized face normals (x_j - x_i) x (x_k - x_i); length = 2 * area."""
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def face_normals(V, F):
    c = face_cross(V, F)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def face_areas(V, F):
    return 0.5 * np.linalg.norm(face_cross(V, F), axis=1)


def signed_volume(V, F):
    """Signed enclosed volume (positive iff a closed mesh is oriented outward)."""
    return np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0


def corner_angles(V, F):
    """(n2, 3) interior angle of each face at each of its three corners."""
    out = np.zeros(F.shape, float)
    for c in range(3):
        p = V[F[:, c]]
        a = V[F[:, (c + 1) % 3]] - p
        b = V[F[:, (c + 2) % 3]] - p
        cosang = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        out[:, c] = np.arccos(np.clip(cosang, -1.0, 1.0))
    return out


def vertex_normals(V, F, kind="angle"):
    """Unit vertex normals: kind = 'uniform' (mean of unit face normals), 'area' (sum of face cross
    products, i.e. area-weighted), 'angle' (unit face normals weighted by the corner angle)."""
    n = len(V)
    out = np.zeros((n, 3))
    if kind == "uniform":
        w = face_normals(V, F)
        for c in range(3):
            np.add.at(out, F[:, c], w)
    elif kind == "area":
        w = face_cross(V, F)
        for c in range(3):
            np.add.at(out, F[:, c], w)
    elif kind == "angle":
        fn = face_normals(V, F)
        ang = corner_angles(V, F)
        for c in range(3):
            np.add.at(out, F[:, c], fn * ang[:, c:c + 1])
    else:
        raise ValueError(kind)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def angle_sums(V, F):
    s = np.zeros(len(V))
    ang = corner_angles(V, F)
    for c in range(3):
        np.add.at(s, F[:, c], ang[:, c])
    return s


def boundary_vertices(F):
    b = set()
    for i, j in boundary_edges(F):
        b |= {i, j}
    return sorted(b)


def angle_defect(V, F):
    """delta_i = 2 pi - sum of corner angles at i (interior vertices); boundary vertices get
    pi - sum (the turning angle of the boundary polygon), so that sum = 2 pi chi in both cases."""
    s = angle_sums(V, F)
    d = 2 * np.pi - s
    for i in boundary_vertices(F):
        d[i] = np.pi - s[i]
    return d


def mass_barycentric(V, F):
    """A_i = one third of the area of the faces around i."""
    a = face_areas(V, F)
    m = np.zeros(len(V))
    for c in range(3):
        np.add.at(m, F[:, c], a / 3.0)
    return m


def mass_voronoi(V, F):
    """Mixed Voronoi area of Meyer et al. [Meyer03]: Voronoi area in non-obtuse triangles; for an obtuse
    triangle, half its area to the obtuse corner and a quarter to the other two."""
    ang = corner_angles(V, F)
    area = face_areas(V, F)
    m = np.zeros(len(V))
    for f, (tri, a3) in enumerate(zip(F, ang)):
        if a3.max() > np.pi / 2:
            for c in range(3):
                m[tri[c]] += area[f] / 2 if a3[c] > np.pi / 2 else area[f] / 4
            continue
        for c in range(3):
            i, j, k = tri[c], tri[(c + 1) % 3], tri[(c + 2) % 3]
            # Voronoi part of corner i: (|x_i x_j|^2 cot(angle at k) + |x_i x_k|^2 cot(angle at j)) / 8
            lij = np.sum((V[i] - V[j]) ** 2)
            lik = np.sum((V[i] - V[k]) ** 2)
            m[i] += (lij / np.tan(a3[(c + 2) % 3]) + lik / np.tan(a3[(c + 1) % 3])) / 8.0
    return m


def cotan_weights(V, F):
    """dict {(i, j) with i < j: w_ij = (cot alpha + cot beta) / 2} (one cot for a boundary edge)."""
    ang = corner_angles(V, F)
    w = defaultdict(float)
    for f, tri in enumerate(F):
        for c in range(3):
            i, j = tri[(c + 1) % 3], tri[(c + 2) % 3]  # edge opposite corner c
            w[(min(i, j), max(i, j))] += 0.5 / np.tan(ang[f, c])
    return dict(w)


def cotan_laplacian(V, F):
    """Dense cotan matrix L (Delta sign): L_ij = w_ij, L_ii = -sum_j w_ij. -L is PSD."""
    n = len(V)
    L = np.zeros((n, n))
    for (i, j), w in cotan_weights(V, F).items():
        L[i, j] += w
        L[j, i] += w
        L[i, i] -= w
        L[j, j] -= w
    return L


def uniform_laplacian(F, n0=None):
    """Uniform ('umbrella') Laplacian as in PyTorch3D: L_ii = -1, L_ij = 1/deg(i). Not symmetric."""
    nb = neighbors(F, n0)
    n = len(nb)
    L = -np.eye(n)
    for i, js in enumerate(nb):
        for j in js:
            L[i, j] = 1.0 / len(js)
    return L


def dirichlet_energy(V, F, u):
    """Exact integral of |grad u|^2 over the mesh for the piecewise-linear interpolant of u."""
    total = 0.0
    for tri in F:
        p = V[tri]
        e1, e2 = p[1] - p[0], p[2] - p[0]
        n = np.cross(e1, e2)
        A2 = np.linalg.norm(n)
        nhat = n / A2
        # gradients of the three hat functions: (nhat x opposite edge) / (2A)
        g = 0
        for c in range(3):
            opp = p[(c + 2) % 3] - p[(c + 1) % 3]
            g = g + u[tri[c]] * np.cross(nhat, opp) / A2
        total += 0.5 * A2 * np.dot(g, g)
    return total


def edge_bending_angles(V, F):
    """Signed bending (exterior dihedral) angle theta_e for every interior edge.

    theta_e is the angle between the unit normals of the two faces, positive when the edge is convex
    with respect to the normals (the surface bends away from N, like every edge of a convex polyhedron
    with outward normals). Returns dict {(i, j): (theta, length)} with i < j.
    """
    fn = face_normals(V, F)
    out = {}
    for (i, j), fs in edge_faces(F).items():
        if len(fs) != 2:
            continue
        f1, f2 = fs
        n1, n2 = fn[f1], fn[f2]
        e = V[j] - V[i]
        ell = np.linalg.norm(e)
        # orient e as it is traversed by f1
        t1 = list(F[f1])
        a = t1.index(i)
        if t1[(a + 1) % 3] != j:
            e = -e
        s = np.dot(np.cross(n1, n2), e / ell)
        theta = np.arctan2(s, np.dot(n1, n2))
        out[(i, j)] = (theta, ell)
    return out


def mean_curvature_dihedral(V, F, mass=None):
    """Vertex mean curvature H_i = -(1 / (4 A_i)) sum_{j ~ i} theta_ij l_ij  (convention W = -dN,
    so H < 0 on a sphere with outward normals). Uses barycentric A_i unless ``mass`` is given."""
    A = mass_barycentric(V, F) if mass is None else mass
    s = np.zeros(len(V))
    for (i, j), (th, ell) in edge_bending_angles(V, F).items():
        s[i] += th * ell
        s[j] += th * ell
    return -s / (4.0 * A)


# ---------------------------------------------------------------------------
# spectral
# ---------------------------------------------------------------------------


def generalized_eigs(L, m, k=None):
    """Solve -L phi = lambda M phi (M = diag(m)) for the smallest eigenvalues.

    Returns (lam, Phi) with Phi^T M Phi = I. Dense; only for small meshes.
    """
    d = 1.0 / np.sqrt(m)
    A = -(d[:, None] * L * d[None, :])
    A = 0.5 * (A + A.T)
    lam, U = np.linalg.eigh(A)
    Phi = d[:, None] * U
    if k is not None:
        lam, Phi = lam[:k], Phi[:, :k]
    return lam, Phi


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def shade(V, F, base_rgb, light=(0.4, -0.5, 0.8), ambient=0.45):
    """Simple Lambert shading of a single base color, returns (n2, 4) RGBA."""
    n = face_normals(V, F)
    lv = np.asarray(light, float)
    lv /= np.linalg.norm(lv)
    k = ambient + (1 - ambient) * np.abs(n @ lv)
    rgb = np.clip(np.asarray(base_rgb, float)[None, :] * k[:, None], 0, 1)
    return np.concatenate([rgb, np.ones((len(F), 1))], 1)


def draw_mesh(ax, V, F, facecolors, edgecolor="#555555", lw=0.25, alpha=1.0, zorder=1):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pc = Poly3DCollection(V[F], facecolors=facecolors, edgecolors=edgecolor, linewidths=lw, alpha=alpha,
                          zorder=zorder)
    ax.add_collection3d(pc)
    return pc


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ---------------------------------------------------------------------------
# level sets
# ---------------------------------------------------------------------------

# Kuhn (Freudenthal) split of the unit cube into 6 tetrahedra along the diagonal 000 -> 111.
# Corner c = (dx, dy, dz) has index dx + 2 dy + 4 dz. The split is consistent across neighbouring cubes.
_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def marching_tetrahedra(fval, xs, ys, zs):
    """Piecewise-linear zero level set of samples fval[i, j, k] at (xs[i], ys[j], zs[k]).

    Every cube is split into six tetrahedra; on each tetrahedron f is interpolated linearly, so the
    zero set inside it is one triangle or one quadrilateral (two triangles). Vertices sit on grid edges
    at the linear-interpolation root and are shared between neighbouring cells, so the result is an
    indexed mesh. Faces are oriented so that normals point toward f > 0. Samples equal to 0 are moved
    to +1e-12 (a standard tie-break that keeps the output a manifold).
    """
    f = np.array(fval, float)
    f[f == 0] = 1e-12
    nx, ny, nz = f.shape
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1)
    verts, vid, faces = [], {}, []

    def gid(i, j, k):
        return (i * ny + j) * nz + k

    def ev(a, b):
        key = (min(a, b), max(a, b))
        if key not in vid:
            ia, ib = np.unravel_index(a, f.shape), np.unravel_index(b, f.shape)
            fa, fb = f[ia], f[ib]
            t = fa / (fa - fb)
            vid[key] = len(verts)
            verts.append(P[ia] + t * (P[ib] - P[ia]))
        return vid[key]

    corners = [(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    sign = f > 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                cube = sign[i:i + 2, j:j + 2, k:k + 2]
                if cube.all() or not cube.any():
                    continue
                g = [gid(i + c[0], j + c[1], k + c[2]) for c in corners]
                for tet in _KUHN:
                    ids = [g[c] for c in tet]
                    s = [bool(sign.flat[t]) for t in ids]
                    pos = [t for t, si in zip(ids, s) if si]
                    neg = [t for t, si in zip(ids, s) if not si]
                    if not pos or not neg:
                        continue
                    if len(pos) == 1 or len(neg) == 1:
                        a = pos[0] if len(pos) == 1 else neg[0]
                        others = neg if len(pos) == 1 else pos
                        tri = [ev(a, o) for o in others]
                        polys = [tri]
                    else:
                        p0, p1 = pos
                        n0_, n1_ = neg
                        q = [ev(p0, n0_), ev(p0, n1_), ev(p1, n1_), ev(p1, n0_)]
                        polys = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
                    cpos = np.mean([P[np.unravel_index(t, f.shape)] for t in pos], 0)
                    cneg = np.mean([P[np.unravel_index(t, f.shape)] for t in neg], 0)
                    for tri in polys:
                        a, b, c = (verts[t] for t in tri)
                        if np.dot(np.cross(b - a, c - a), cpos - cneg) < 0:
                            tri = [tri[0], tri[2], tri[1]]
                        faces.append(tri)
    V = np.array(verts)
    F = np.array(faces, int)
    # No triangle is removed here. After the tie-break the vertices of a triangle never coincide exactly, but a root
    # can lie numerically very close to a grid point, so nearly degenerate (sliver, even near-zero-area) triangles
    # may remain in the output; callers that need to can filter them.
    return V, F


# ---------------------------------------------------------------------------
# 2D drawing of 3D schematic objects (orthographic camera, painter's algorithm)
# ---------------------------------------------------------------------------


class Ortho:
    """Orthographic camera looking from direction (elev, azim) in degrees, like mplot3d's view_init."""

    def __init__(self, elev=20, azim=-60):
        e, a = np.radians(elev), np.radians(azim)
        self.view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # toward the viewer
        up = np.array([0.0, 0.0, 1.0])
        self.right = np.cross(up, self.view)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.view, self.right)

    def __call__(self, P):
        P = np.asarray(P, float)
        return np.stack([P @ self.right, P @ self.up], -1)

    def depth(self, P):
        return np.asarray(P, float) @ self.view


def torus_sdf(P, center, R, r):
    q = P - np.asarray(center, float)
    return np.sqrt((np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2) - R) ** 2 + q[..., 2] ** 2) - r


def smooth_min(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def genus2_mesh(h=0.08, c=0.95, R=1.0, r=0.35, k=0.15):
    """Genus-2 surface: smooth union of two tori of revolution centred at (+-c, 0, 0), extracted with
    marching tetrahedra on a grid of spacing h. Returns (V, F)."""
    xs = np.arange(-c - R - r - 2 * h, c + R + r + 2 * h + 1e-9, h)
    ys = np.arange(-R - r - 2 * h, R + r + 2 * h + 1e-9, h)
    zs = np.arange(-r - 3 * h, r + 3 * h + 1e-9, h)
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1)
    f = smooth_min(torus_sdf(P, (-c, 0, 0), R, r), torus_sdf(P, (c, 0, 0), R, r), k)
    return marching_tetrahedra(f, xs, ys, zs)


def draw_mesh_2d(ax, V, F, cam, facecolors, edgecolor="#555555", lw=0.3, rasterized=False, zorder=1,
                 cull=False):
    """Draw a mesh with an orthographic camera (painter's algorithm) as a 2D PolyCollection.

    facecolors: (n2, 4) RGBA or a single color. With cull=True, faces whose normal points away from
    the viewer are skipped (closed, outward-oriented meshes only)."""
    from matplotlib.collections import PolyCollection
    depth = cam.depth(V[F].mean(1))
    order = np.argsort(depth)
    if cull:
        n = face_cross(V, F)
        order = order[(n[order] @ cam.view) > 0]
    fc = np.asarray(facecolors)
    if fc.ndim == 2 and len(fc) == len(F):
        fc = fc[order]
    pc = PolyCollection(cam(V[F[order]]), facecolors=fc, edgecolors=edgecolor, linewidths=lw, zorder=zorder)
    pc.set_rasterized(rasterized)
    ax.add_collection(pc)
    P = cam(V)
    return P


# ---------------------------------------------------------------------------
# plotly helpers for the interactive versions (GUIDELINES 12.7)
# ---------------------------------------------------------------------------


def plotly_mesh(V, F, color="#D0D0D0", intensity=None, colorscale=None, cmin=None, cmax=None, name="mesh",
                showscale=False, opacity=1.0, hovertemplate=None, colorbar_title=None):
    import plotly.graph_objects as go
    kw = dict(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], name=name, opacity=opacity,
              flatshading=True, lighting=dict(ambient=0.55, diffuse=0.6, specular=0.05),
              hovertemplate=hovertemplate or (name + "<extra></extra>"))
    if intensity is None:
        kw["color"] = color
    else:
        kw.update(intensity=intensity, colorscale=colorscale or "RdBu_r", cmin=cmin, cmax=cmax,
                  showscale=showscale)
        if colorbar_title:
            kw["colorbar"] = dict(title=colorbar_title)
        if len(intensity) == len(F):
            kw["intensitymode"] = "cell"
    return go.Mesh3d(**kw)


def plotly_edges(V, F, color="#555555", width=1.5, name="edges"):
    import plotly.graph_objects as go
    E = edges(F)
    xs, ys, zs = [], [], []
    for a, b in E:
        xs += [V[a, 0], V[b, 0], None]
        ys += [V[a, 1], V[b, 1], None]
        zs += [V[a, 2], V[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width), hoverinfo="skip",
                        name=name, showlegend=False)


def plotly_arrows(P, D, color="#D55E00", width=5, cone=0.06, name="vectors"):
    import plotly.graph_objects as go
    P, D = np.asarray(P, float), np.asarray(D, float)
    xs, ys, zs = [], [], []
    for p, d in zip(P, D):
        q = p + d
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    tips = P + D
    return [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width), hoverinfo="skip",
                         showlegend=False, name=name),
            go.Cone(x=tips[:, 0], y=tips[:, 1], z=tips[:, 2], u=D[:, 0], v=D[:, 1], w=D[:, 2], anchor="tip",
                    sizemode="absolute", sizeref=cone, colorscale=[[0, color], [1, color]], showscale=False,
                    hoverinfo="skip", name=name)]


def plotly_save(traces, script_file, title, div_id):
    import pathlib
    import plotly.graph_objects as go
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                      margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                      title=dict(text=title, x=0.5, font=dict(size=13)))
    out = pathlib.Path(script_file).with_name(pathlib.Path(script_file).stem + "-interactive.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id=div_id, config={"displaylogo": False})
    print(f"saved {out}")
    return out


def cotan_apply(V, F, X):
    """(L X) for the cotan Laplacian without forming the matrix (works for large meshes)."""
    ang = corner_angles(V, F)
    out = np.zeros_like(np.asarray(X, float))
    X = np.asarray(X, float)
    for c in range(3):
        i, j = F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        w = 0.5 / np.tan(ang[:, c])
        d = (X[j] - X[i]) * (w[:, None] if X.ndim == 2 else w)
        np.add.at(out, i, d)
        np.add.at(out, j, -d)
    return out


def dumbbell_sdf(P, rho=0.05):
    """Two balls of radius 0.5 at (-0.8, 0, 0), (0.8, 0, 0) joined by a slightly tilted rod of radius rho
    (axis from (-0.8, 0, -0.07) to (0.8, 0, 0.07)). Exact distance for each piece, union by min."""
    A = np.array([-0.8, 0.0, -0.07])
    B = np.array([0.8, 0.0, 0.07])
    s1 = np.linalg.norm(P - np.array([-0.8, 0.0, 0.0]), axis=-1) - 0.5
    s2 = np.linalg.norm(P - np.array([0.8, 0.0, 0.0]), axis=-1) - 0.5
    d = B - A
    t = np.clip(((P - A) @ d) / (d @ d), 0, 1)
    rod = np.linalg.norm(P - (A + t[..., None] * d), axis=-1) - rho
    return np.minimum(np.minimum(s1, s2), rod)


def dumbbell_mesh(h):
    """Marching tetrahedra of dumbbell_sdf on a grid of spacing h whose y- and z-lines avoid the rod axis
    (grid values +-h/2, +-3h/2, ... in y and z)."""
    xs = np.arange(-1.4, 1.4 + 1e-9, h) + 0.0137
    n = int(np.ceil(1.4 / h))
    n += n % 2
    ys = (np.arange(n) - (n - 1) / 2) * h
    P = np.stack(np.meshgrid(xs, ys, ys, indexing="ij"), -1)
    return marching_tetrahedra(dumbbell_sdf(P), xs, ys, ys)


# ---------------------------------------------------------------------------
# parametrization (Section 31.6)
# ---------------------------------------------------------------------------


def submesh(V, F, face_mask):
    """Faces F[face_mask] with their vertices renumbered: returns (V_sub, F_sub, old_index_of_new)."""
    Fs = F[face_mask]
    used = np.unique(Fs)
    new = -np.ones(len(V), int)
    new[used] = np.arange(len(used))
    return V[used], new[Fs], used


def boundary_loop(F):
    """The boundary vertices of a disk-like mesh in order, following the face orientation."""
    nxt = {}
    directed = set()
    for a, b, c in F:
        directed.update(((a, b), (b, c), (c, a)))
    for a, b in directed:
        if (b, a) not in directed:
            nxt[a] = b
    start = min(nxt)
    loop = [start]
    while True:
        v = nxt[loop[-1]]
        if v == start:
            break
        loop.append(v)
    assert len(loop) == len(nxt), "the boundary is not a single loop"
    return np.array(loop)


def fixed_boundary_map(V, F, weights="uniform"):
    """Tutte / convex-combination ('uniform') or discrete harmonic ('cotan') map of a disk-like mesh,
    boundary placed on the unit circle by arc length (counterclockwise). Returns UV (n0, 2)."""
    n = len(V)
    loop = boundary_loop(F)
    seg = np.linalg.norm(V[np.roll(loop, -1)] - V[loop], axis=1)
    t = 2 * np.pi * np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum()
    UV = np.zeros((n, 2))
    UV[loop] = np.stack([np.cos(t), np.sin(t)], 1)
    W = np.zeros((n, n))
    if weights == "uniform":
        for i, js in enumerate(neighbors(F, n)):
            W[i, list(js)] = 1.0
    else:
        for (i, j), w in cotan_weights(V, F).items():
            W[i, j] = W[j, i] = w
    is_b = np.zeros(n, bool)
    is_b[loop] = True
    inn = np.where(~is_b)[0]
    A = np.diag(W[inn].sum(1)) - W[np.ix_(inn, inn)]
    rhs = W[np.ix_(inn, np.where(is_b)[0])] @ UV[is_b]
    UV[inn] = np.linalg.solve(A, rhs)
    return UV


def face_frames(V, F):
    """Per face: local 2D coordinates (3, 2) of its corners in an orthonormal frame (e1, e2) of its plane,
    (e1, e2, N) right-handed, so a counterclockwise UV triangle means an orientation-preserving map."""
    P = V[F]
    e1 = P[:, 1] - P[:, 0]
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    N = face_normals(V, F)
    e2 = np.cross(N, e1)
    Q = P - P[:, :1]
    return np.stack([(Q * e1[:, None]).sum(-1), (Q * e2[:, None]).sum(-1)], -1)


def lscm(V, F, pins, pin_uv):
    """Least squares conformal map [Levy02]: minimise sum_T A_T * 1/2 ((u_x - v_y)^2 + (u_y + v_x)^2) over
    piecewise-linear (u, v) with the UV of the vertices `pins` fixed to `pin_uv`. Returns UV (n0, 2).
    Unknowns are ordered (u_0..u_{n-1}, v_0..v_{n-1}); the normal equations are assembled directly."""
    n = len(V)
    X = face_frames(V, F)
    A = 0.5 * np.abs(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]))
    E1 = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]], -1)            # (n2, 2, 2): columns = edges
    Gi = np.linalg.inv(E1).transpose(0, 2, 1) @ np.array([[-1.0, 1, 0], [-1, 0, 1]])   # (n2, 2, 3)
    s = np.sqrt(A / 2)[:, None]
    gx, gy = Gi[:, 0] * s, Gi[:, 1] * s                                    # (n2, 3)
    idx_u, idx_v = F, F + n
    # row 1: u_x - v_y ; row 2: u_y + v_x  (coefficients on (u_k), (v_k))
    rows = [(np.concatenate([gx, -gy], 1), np.concatenate([idx_u, idx_v], 1)),
            (np.concatenate([gy, gx], 1), np.concatenate([idx_u, idx_v], 1))]
    C = np.zeros((2 * n, 2 * n))
    for coef, idx in rows:
        for a in range(6):
            for b in range(6):
                np.add.at(C, (idx[:, a], idx[:, b]), coef[:, a] * coef[:, b])
    fixed = np.zeros(2 * n, bool)
    val = np.zeros(2 * n)
    for p, uv in zip(pins, pin_uv):
        fixed[[p, n + p]] = True
        val[[p, n + p]] = uv
    free = ~fixed
    sol = np.linalg.solve(C[np.ix_(free, free)], -C[np.ix_(free, fixed)] @ val[fixed])
    out = val.copy()
    out[free] = sol
    return np.stack([out[:n], out[n:]], 1)


def uv_distortion(V, F, UV):
    """Per face: singular values s1 >= s2 of the Jacobian J of the affine map UV -> surface triangle (3 x 2),
    and the signed UV area (negative = flipped triangle)."""
    X = face_frames(V, F)
    s = np.zeros((len(F), 2))
    det = np.zeros(len(F))
    for f, (tri, x) in enumerate(zip(F, X)):
        U = np.stack([UV[tri[1]] - UV[tri[0]], UV[tri[2]] - UV[tri[0]]], 1)   # 2 x 2 (UV edges)
        P = np.stack([x[1] - x[0], x[2] - x[0]], 1)                         # 2 x 2 (surface edges, local)
        det[f] = 0.5 * np.linalg.det(U)
        if abs(det[f]) < 1e-15:
            s[f] = (np.inf, np.inf)
            continue
        J = P @ np.linalg.inv(U)
        s[f] = np.linalg.svd(J, compute_uv=False)
    return s, det
