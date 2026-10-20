"""Helpers for Chapter 40 (Discrete Curvature I: The Integral Viewpoint).

Shared by the figure and verify scripts of this chapter. Mesh construction, angles, areas, bending angles and the
cotan matrix come from tools/dgmesh.py and tools/dgdec.py (the canonical conventions of GUIDELINES 2.5):

- faces (i, j, k) are positively oriented, N_f = (x_j - x_i) x (x_k - x_i) / |...|, closed meshes face outward;
- the bending angle theta_ij is dgmesh.edge_bending_angles (Definition 39.5.5): positive at a convex edge;
- L = dgmesh.cotan_laplacian (L_ij = (cot a + cot b) / 2, L_ii = -sum_j L_ij), (L x)_i ~ 2 H N A_i, H = -1/r on an
  outward sphere of radius r.

Everything here is a small, self-contained helper used to *check* the statements of the chapter independently of the
code printed in the pages (for example the dual-cell boundary integrals are computed segment by segment, and the
signed area of a spherical polygon by solid angles). Candidates for tools/: link_cycle, faces_around,
spherical_polygon_area, dual_cell_boundary, point_triangle_distance.

Run from the project root with PYTHONPATH=tools.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import contextlib  # noqa: E402
import io  # noqa: E402

import numpy as np  # noqa: E402

import dgmesh as dm  # noqa: E402

# ---------------------------------------------------------------------------
# printed code blocks (tools/dgcode.py)
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
# meshes
# ---------------------------------------------------------------------------


def orient_outward(V, F, center=None):
    """Flip faces of a mesh that is star-shaped with respect to ``center`` so that the normals point away from it."""
    V = np.asarray(V, float)
    c = V.mean(0) if center is None else np.asarray(center, float)
    F = np.array(F, int).copy()
    for f, (i, j, k) in enumerate(F):
        n = np.cross(V[j] - V[i], V[k] - V[i])
        if n @ ((V[i] + V[j] + V[k]) / 3 - c) < 0:
            F[f] = [i, k, j]
    return F


def tetrahedron():
    """Regular tetrahedron with vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) (edge 2 sqrt 2), outward faces."""
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    F = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    return V, orient_outward(V, F, np.zeros(3))


def octahedron():
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    return V, orient_outward(V, F, np.zeros(3))


def unit_cube(diagonals="avoid"):
    """Unit cube [0, 1]^3 split into 12 triangles, outward faces.

    Every square is cut by one of its diagonals. diagonals = "avoid": the diagonal that misses the corners (0, 0, 0) and
    (1, 1, 1) (so these two corners have exactly three triangles, one per square); "max": the diagonal through the
    vertex with the largest coordinate sum (and the opposite one); "alt": "max" on the squares x, y, z = 1 and "avoid"
    on the squares x, y, z = 0."""
    V = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    idx = {tuple(v): n for n, v in enumerate(V)}
    F = []
    for axis in range(3):
        for s in (1.0, 0.0):
            a, b = [c for c in range(3) if c != axis]
            sq = []
            for pa, pb in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[a], p[b] = s, pa, pb
                sq.append(idx[tuple(p)])
            through_max = {"avoid": False, "max": True, "alt": s == 1.0}[diagonals]
            if through_max:
                F += [[sq[0], sq[1], sq[2]], [sq[0], sq[2], sq[3]]]
            else:
                F += [[sq[1], sq[2], sq[3]], [sq[1], sq[3], sq[0]]]
    return V, orient_outward(V, F, np.full(3, 0.5))


def star_mesh(link, apex=(0.0, 0.0, 0.0)):
    """Vertex 0 at ``apex``, link vertices 1..m, faces (0, k, k+1) (closing up). The star is oriented so that the
    link runs counterclockwise seen from +z when the link is listed counterclockwise."""
    link = np.asarray(link, float)
    m = len(link)
    V = np.vstack([np.asarray(apex, float), link])
    F = np.array([[0, k + 1, (k + 1) % m + 1] for k in range(m)], int)
    return V, F


STAR_ANGLES = np.radians([10.0, 65.0, 130.0, 185.0, 250.0, 305.0])
STAR_RADII = np.array([1.0, 0.85, 1.05, 0.9, 1.1, 0.8])
STAR_HEIGHTS = np.array([0.1, -0.12, 0.05, 0.0, -0.06, 0.08])


def example_star(apex_height=0.55):
    """The convex valence-6 star of Figures 40.1.1 and 40.1.2: x_i = (0, 0, apex_height), link points at the angles
    STAR_ANGLES with radii STAR_RADII and heights STAR_HEIGHTS (a convex vertex with delta_i = 0.8363)."""
    link = np.c_[STAR_RADII * np.cos(STAR_ANGLES), STAR_RADII * np.sin(STAR_ANGLES), STAR_HEIGHTS]
    return star_mesh(link, (0.0, 0.0, apex_height))


def perturbed_icosahedron(amp=0.18, seed=3):
    """dgmesh.icosahedron() with vertex k scaled by 1 + amp u_k, u_k uniform in [-1, 1] (seeded): a convex polyhedron
    with twelve different angle defects (Figure 40.2.2)."""
    V, F = dm.icosahedron()
    r = 1 + amp * np.random.default_rng(seed).uniform(-1, 1, len(V))
    return V * r[:, None], F


def saddle_star(h=0.35, m=6):
    """x_i = 0 with m link points (cos t_k, sin t_k, h (-1)^k), t_k = 2 pi k / m: a saddle vertex (delta_i < 0)."""
    t = 2 * np.pi * np.arange(m) / m
    return star_mesh(np.c_[np.cos(t), np.sin(t), h * (-1.0) ** np.arange(m)])


def jitter(V, s, seed=0):
    """Move every vertex by a seeded random vector of size at most s (componentwise uniform in [-s, s])."""
    rng = np.random.default_rng(seed)
    return V + rng.uniform(-s, s, V.shape)


def saddle_grid(n=9, a=1.0, pattern="alternate"):
    """grid_mesh over [-a, a]^2 lifted to z = x^2 - y^2 (running example: K = -4, H = 0 at the origin)."""
    s = np.linspace(-a, a, n)
    V, F = dm.grid_mesh(s, s, pattern)
    V[:, 2] = V[:, 0] ** 2 - V[:, 1] ** 2
    return V, F


def icosphere_cap(level, theta0):
    """Faces of icosphere(level) whose three vertices have colatitude <= theta0, renumbered (V_sub, F_sub)."""
    V, F = dm.icosphere(level)
    col = np.arccos(np.clip(V[:, 2], -1, 1))
    keep = np.all(col[F] <= theta0 + 1e-12, axis=1)
    Vs, Fs, _ = dm.submesh(V, F, keep)
    return Vs, Fs


# ---------------------------------------------------------------------------
# vertex stars
# ---------------------------------------------------------------------------


def link_cycle(F, i):
    """Link vertices j_1, ..., j_m of the interior vertex i in orientation order: the faces at i are (i, j_k, j_{k+1})."""
    nxt = {}
    for f in np.asarray(F).tolist():
        if i in f:
            a = f.index(i)
            nxt[f[(a + 1) % 3]] = f[(a + 2) % 3]
    seq = [min(nxt)]
    while nxt[seq[-1]] != seq[0]:
        seq.append(nxt[seq[-1]])
    if len(seq) != len(nxt):
        raise ValueError(f"vertex {i} is not an interior manifold vertex")
    return seq


def faces_around(F, i):
    """Face indices f_k = (i, j_k, j_{k+1}) (as rows of F, read cyclically) in the order of link_cycle."""
    F = np.asarray(F)
    seq = link_cycle(F, i)
    out = []
    for j, jn in zip(seq, seq[1:] + seq[:1]):
        for f, t in enumerate(F.tolist()):
            if i in t:
                a = t.index(i)
                if t[(a + 1) % 3] == j and t[(a + 2) % 3] == jn:
                    out.append(f)
                    break
    return out


def is_convex_vertex(V, F, i, tol=1e-12):
    """<N_f, x_j - x_i> < 0 for every face f at i and every link vertex j not in f."""
    N = dm.face_normals(V, F)
    seq = link_cycle(F, i)
    for f in faces_around(F, i):
        for j in seq:
            if j not in F[f] and N[f] @ (V[j] - V[i]) >= -tol:
                return False
    return True


# ---------------------------------------------------------------------------
# spherical polygons
# ---------------------------------------------------------------------------


def solid_angle(a, b, c):
    """Signed area of the spherical triangle (a, b, c) of unit vectors (Van Oosterom-Strackee), in (-2 pi, 2 pi)."""
    num = np.dot(a, np.cross(b, c))
    den = 1.0 + a @ b + b @ c + c @ a
    return 2.0 * np.arctan2(num, den)


def spherical_polygon_area(P, center=None):
    """Signed area of the closed spherical polygon with unit vertices P[0], ..., P[m-1] (great arcs between
    neighbours), summed over the fan of triangles from ``center`` (default: the normalised mean of P). Positive when
    the polygon turns counterclockwise around the center seen from outside the sphere."""
    P = np.asarray(P, float)
    c = P.sum(0) if center is None else np.asarray(center, float)
    c = c / np.linalg.norm(c)
    return float(sum(solid_angle(c, P[k], P[(k + 1) % len(P)]) for k in range(len(P))))


def girard_area(angles):
    """Area of a convex spherical polygon with the given interior angles: sum - (m - 2) pi."""
    return float(np.sum(angles) - (len(angles) - 2) * np.pi)


def spherical_interior_angles(P):
    """Interior angles of a convex spherical polygon traversed counterclockwise (seen from outside)."""
    P = np.asarray(P, float)
    m = len(P)
    out = []
    for k in range(m):
        p, a, b = P[k], P[k - 1], P[(k + 1) % m]
        ta = a - (a @ p) * p
        tb = b - (b @ p) * p
        out.append(np.arccos(np.clip(ta @ tb / (np.linalg.norm(ta) * np.linalg.norm(tb)), -1, 1)))
    return np.array(out)


def spherical_triangle_vector_area(a, b, c, n=60):
    """integral of the position over the spherical triangle (a, b, c): subdivide the flat triangle n x n, project the
    small triangles to the sphere and add (centroid direction) x (signed solid angle)."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    tot = np.zeros(3)
    P = lambda s, t: (1 - s - t) * a + s * b + t * c  # noqa: E731
    U = lambda p: p / np.linalg.norm(p)  # noqa: E731
    for s in range(n):
        for t in range(n - s):
            tris = [((s, t), (s + 1, t), (s, t + 1))]
            if s + t + 2 <= n:
                tris.append(((s + 1, t), (s + 1, t + 1), (s, t + 1)))
            for tri in tris:
                q = [U(P(x / n, y / n)) for x, y in tri]
                w = solid_angle(*q)
                tot += w * U(q[0] + q[1] + q[2])
    return tot


# ---------------------------------------------------------------------------
# dual cells and boundary integrals (Section 40.3)
# ---------------------------------------------------------------------------


def dual_cell_boundary(V, F, i, kind="circumcentric"):
    """The closed path dC_i of the interior vertex i, positively oriented: m_{i j_1} -> c_{f_1} -> m_{i j_2} -> ...
    Returns (points, face_of_segment): segment s goes from points[s] to points[s + 1] (cyclically) inside face
    face_of_segment[s]. kind: 'circumcentric' (dual vertex at the circumcenter) or 'barycentric'."""
    import dgdec as dd
    V = np.asarray(V, float)
    seq = link_cycle(F, i)
    fs = faces_around(F, i)
    cen = dd.circumcenters(V, F) if kind == "circumcentric" else V[F].mean(1)
    pts, seg_face = [], []
    for j, f in zip(seq, fs):
        pts += [0.5 * (V[i] + V[j]), cen[f]]
        seg_face += [f, f]
    return np.array(pts), seg_face


def boundary_integral_N_cross_dx(V, F, i, kind="circumcentric"):
    """oint over dC_i of N x dx with N = N_f on the part of dC_i inside f (the crossings of edges add nothing)."""
    N = dm.face_normals(V, F)
    pts, sf = dual_cell_boundary(V, F, i, kind)
    tot = np.zeros(3)
    for s in range(len(pts)):
        tot += np.cross(N[sf[s]], pts[(s + 1) % len(pts)] - pts[s])
    return tot


def boundary_integral_N_cross_dN(V, F, i):
    """oint over dC_i of N x dN when N is constant in faces and runs along the short great arc from N_{f_k} to
    N_{f_{k+1}} where dC_i crosses the edge i j_{k+1}: each arc contributes (arc length) x (unit rotation axis)."""
    N = dm.face_normals(V, F)
    fs = faces_around(F, i)
    tot = np.zeros(3)
    for k in range(len(fs)):
        a, b = N[fs[k]], N[fs[(k + 1) % len(fs)]]
        ax = np.cross(a, b)
        s = np.linalg.norm(ax)
        if s < 1e-15:
            continue
        tot += np.arctan2(s, a @ b) * ax / s
    return tot


def curvature_normals(V, F):
    """(area normal (barycentric vector area), (L x)_i, (K N)_i) for every vertex, from the printed definitions."""
    V = np.asarray(V, float)
    A = dm.face_cross(V, F) / 2.0
    AN = np.zeros_like(V)
    for c in range(3):
        np.add.at(AN, F[:, c], A / 3.0)
    Lx = dm.cotan_laplacian(V, F) @ V
    KN = np.zeros_like(V)
    for (i, j), (th, ell) in dm.edge_bending_angles(V, F).items():
        KN[i] += 0.5 * th * (V[i] - V[j]) / ell
        KN[j] += 0.5 * th * (V[j] - V[i]) / ell
    return AN, Lx, KN


def mean_normal_edge_form(V, F):
    """-sum_j l_ij sin(theta_ij / 2) Nbar_ij, Nbar_ij the unit bisector of the two face normals (interior edges)."""
    V = np.asarray(V, float)
    N = dm.face_normals(V, F)
    ef = dm.edge_faces(F)
    out = np.zeros_like(V)
    for (i, j), (th, ell) in dm.edge_bending_angles(V, F).items():
        f, g = ef[(i, j)]
        nb = N[f] + N[g]
        nb /= np.linalg.norm(nb)
        out[i] -= ell * np.sin(th / 2) * nb
        out[j] -= ell * np.sin(th / 2) * nb
    return out


# ---------------------------------------------------------------------------
# cones with rounded tips (Section 40.1)
# ---------------------------------------------------------------------------


def rounded_cone_curvature_in_disk(eps, Theta, a):
    """Total Gaussian curvature inside the geodesic disk of radius eps around the tip of a circular cone of cone angle
    Theta (half-angle alpha, sin alpha = Theta / (2 pi)) whose tip is replaced by a tangent sphere cap of radius a
    (a = 0: the cone itself). Proposition 39.4.6: 2 pi (1 - rho'(eps)), rho' = cos(eps / a) on the cap, sin alpha
    after it."""
    sa = Theta / (2 * np.pi)
    eps = np.asarray(eps, float)
    if a == 0:
        return np.where(eps > 0, 2 * np.pi * (1 - sa), 0.0)
    s0 = a * (np.pi / 2 - np.arcsin(sa))          # arc length of the cap profile
    rp = np.where(eps <= s0, np.cos(np.minimum(eps, s0) / a), sa)
    return 2 * np.pi * (1 - rp)


def rounded_cone_profile(Theta, a, smax, n=400):
    """profile (rho, z) of the rounded cone (apex up, opening downward), arc length from the tip, and K along it."""
    sa = Theta / (2 * np.pi)
    al = np.arcsin(sa)
    s = np.linspace(0, smax, n)
    if a == 0:
        return s * sa, -s * np.cos(al), s, np.zeros_like(s)
    s0 = a * (np.pi / 2 - al)
    phi = np.minimum(s, s0) / a
    rho = np.where(s <= s0, a * np.sin(phi), a * np.cos(al) + (s - s0) * sa)
    z = np.where(s <= s0, a * (np.cos(phi) - 1), a * (np.sin(al) - 1) - (s - s0) * np.cos(al))
    K = np.where(s <= s0, 1 / a ** 2, 0.0)
    return rho, z, s, K


# ---------------------------------------------------------------------------
# quadric fitting (Section 40.1)
# ---------------------------------------------------------------------------


def quadric_fit_K(V, F):
    """Gaussian curvature from a least-squares fit w = a u^2 + b u v + c v^2 to the one-ring, in the frame of the
    angle-weighted vertex normal: K = 4 a c - b^2 (the value at the origin of the fitted graph)."""
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
        u, v, w = D @ t, D @ s, D @ n
        coef, *_ = np.linalg.lstsq(np.c_[u * u, u * v, v * v], w, rcond=None)
        a, b, c = coef
        K[i] = 4 * a * c - b * b
    return K


# ---------------------------------------------------------------------------
# convex polyhedra and their mollifications (Section 40.4)
# ---------------------------------------------------------------------------


def point_triangle_distance(P, a, b, c):
    """Euclidean distance from each row of P to the triangle (a, b, c) (closest point by Voronoi regions)."""
    P = np.asarray(P, float)
    ab, ac = b - a, c - a
    ap = P - a
    d1, d2 = ap @ ab, ap @ ac
    bp = P - b
    d3, d4 = bp @ ab, bp @ ac
    cp = P - c
    d5, d6 = cp @ ab, cp @ ac
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    Q = np.empty_like(P)
    # face region (default)
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vb / denom
        w = vc / denom
    Q[:] = a + v[:, None] * ab + w[:, None] * ac
    # edge bc
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    Q[m] = b + t[m, None] * (c - b)
    # edge ac
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d2 / (d2 - d6)
    Q[m] = a + t[m, None] * ac
    # edge ab
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d1 / (d1 - d3)
    Q[m] = a + t[m, None] * ab
    # vertex regions
    Q[(d1 <= 0) & (d2 <= 0)] = a
    Q[(d3 >= 0) & (d4 <= d3)] = b
    Q[(d6 >= 0) & (d5 <= d6)] = c
    return np.linalg.norm(P - Q, axis=1)


def mollified_volume_grid(V, F, eps, h, chunk=400000, shift=(0.3141, 0.5772, 0.7071), rng=None):
    """Volume of P + eps B for the convex polyhedron P bounded by the outward mesh (V, F), counted on the cubic grid
    of spacing h: a point is in P + eps B iff it is inside P or within eps of a face. The grid is shifted by
    shift * h (irrational-looking offsets) so that it is not aligned with the faces of the polyhedron. With a numpy
    Generator ``rng`` every grid point is moved to a uniform random point of its cell (stratified sampling), which
    removes the aliasing bias of a regular grid. The grid is swept in slabs of constant x so that memory stays small."""
    V = np.asarray(V, float)
    lo, hi = V.min(0) - eps - 2 * h, V.max(0) + eps + 2 * h
    axes = [lo[k] + h * (np.arange(int(np.ceil((hi[k] - lo[k]) / h))) + shift[k]) for k in range(3)]
    N = dm.face_normals(V, F)
    off = np.einsum("fi,fi->f", N, V[F[:, 0]])
    Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
    YZ = np.c_[Y.ravel(), Z.ravel()]
    per = max(1, chunk // len(YZ))
    count = 0
    for s in range(0, len(axes[0]), per):
        xs = axes[0][s:s + per]
        P = np.c_[np.repeat(xs, len(YZ)), np.tile(YZ, (len(xs), 1))]
        if rng is not None:
            P = P + h * rng.uniform(-0.5, 0.5, P.shape)
        inside = np.all(P @ N.T - off <= 0, axis=1)
        d = np.full(len(P), np.inf)
        rest = ~inside
        for f in F:
            d[rest] = np.minimum(d[rest], point_triangle_distance(P[rest], V[f[0]], V[f[1]], V[f[2]]))
        count += int(np.sum(inside | (d <= eps)))
    return count * h ** 3


def steiner_coefficients(V, F):
    """(volume, area, (1/2) sum l theta, sum delta) of a closed outward mesh."""
    th = dm.edge_bending_angles(V, F)
    return (dm.signed_volume(V, F), dm.face_areas(V, F).sum(),
            0.5 * sum(t * l for t, l in th.values()), dm.angle_defect(V, F).sum())


def polygon_minkowski_area_grid(P, eps, h):
    """Area of the plane polygon P (counterclockwise vertices) plus the eps-disk, counted on a grid of spacing h."""
    P = np.asarray(P, float)
    lo, hi = P.min(0) - eps - h, P.max(0) + eps + h
    xs = lo[0] + h * (np.arange(int(np.ceil((hi[0] - lo[0]) / h))) + 0.5)
    ys = lo[1] + h * (np.arange(int(np.ceil((hi[1] - lo[1]) / h))) + 0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Q = np.c_[X.ravel(), Y.ravel()]
    # inside test (crossing number) and distance to the edges
    inside = np.zeros(len(Q), bool)
    d = np.full(len(Q), np.inf)
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        cond = (a[1] > Q[:, 1]) != (b[1] > Q[:, 1])
        xint = a[0] + (Q[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1] + 1e-300)
        inside ^= cond & (Q[:, 0] < xint)
        ab = b - a
        t = np.clip(((Q - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(Q - (a + t[:, None] * ab), axis=1))
    return float(np.sum(inside | (d <= eps)) * h * h)


def mollified_pieces(V, F, eps, n_arc=12, n_sph=8):
    """Triangle meshes of the three kinds of pieces of the boundary of P + eps B for a convex polyhedron (outward mesh):
    returns a list of (V_piece, F_piece, kind) with kind in {'face', 'edge', 'vertex'} (Theorem 40.4.10).
    face: f translated by eps N_f; edge: the cylinder piece x_i + t e_ij + eps R(e_hat, s) N_f, s in [0, theta_ij]
    (skipped when theta_ij = 0); vertex: x_i + eps (spherical polygon of the face normals at i), fan-subdivided."""
    V = np.asarray(V, float)
    N = dm.face_normals(V, F)
    out = []
    for f, t in enumerate(F):
        out.append((V[t] + eps * N[f], np.array([[0, 1, 2]]), "face"))
    ef = dm.edge_faces(F)
    for (i, j), (theta, ell) in dm.edge_bending_angles(V, F).items():
        if theta < 1e-12:
            continue
        fs = ef[(i, j)]
        f = [h for h in fs if any(F[h, c] == i and F[h, (c + 1) % 3] == j for c in range(3))][0]
        e = (V[j] - V[i]) / ell
        P = []
        for s in np.linspace(0, theta, n_arc + 1):
            nn = N[f] * np.cos(s) + np.cross(e, N[f]) * np.sin(s)
            P += [V[i] + eps * nn, V[j] + eps * nn]
        P = np.array(P)
        Fp = []
        for k in range(n_arc):
            a, b, c, d = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
            Fp += [[a, b, d], [a, d, c]]
        out.append((P, np.array(Fp), "edge"))
    for i in range(len(V)):
        Nk = N[faces_around(F, i)]
        c = Nk.sum(0)
        c /= np.linalg.norm(c)
        P, Fp = [], []
        for k in range(len(Nk)):
            a, b = Nk[k], Nk[(k + 1) % len(Nk)]
            idx = {}
            for p in range(n_sph + 1):
                for q in range(n_sph + 1 - p):
                    w = (1 - (p + q) / n_sph) * c + p / n_sph * a + q / n_sph * b
                    idx[p, q] = len(P)
                    P.append(w / np.linalg.norm(w))
            for p in range(n_sph):
                for q in range(n_sph - p):
                    Fp.append([idx[p, q], idx[p + 1, q], idx[p, q + 1]])
                    if p + q + 1 < n_sph:
                        Fp.append([idx[p + 1, q], idx[p + 1, q + 1], idx[p, q + 1]])
        out.append((V[i] + eps * np.array(P), np.array(Fp), "vertex"))
    return out
