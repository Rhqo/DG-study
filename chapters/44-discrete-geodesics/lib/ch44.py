"""Chapter 44 (Geodesics) helpers: shortest paths, straightest geodesics, exp/log/Karcher mean, geodesic ODE,
heat method. Used by the figure and verify scripts of Chapter 44 (not a shared module: tools/dgmesh.py and
tools/dgdec.py are the shared ones).

Conventions (Section 44.1, Conventions box; GUIDELINES 2.5, 6.5):
- V: (n0, 3) vertex positions x_i, F: (n2, 3) faces, positively oriented (outward on closed meshes).
- L = dgmesh.cotan_laplacian (L_ij = (cot a + cot b)/2, L_ii = -sum_j L_ij, -L PSD), M = diag(mass).
- Heat method (Section 44.6): (M - t L) u = delta, X = -grad u/|grad u|, L phi = b with
  b_i = (1/2) sum_{f=(i,j,k)} (cot theta_k <e_ij, X_f> + cot theta_j <e_ik, X_f>), e_ij = x_j - x_i;
  b_i = -sum_f A_f <grad lambda_i, X_f>; phi is shifted so that min phi = 0.
- Discrete geodesic curvature at a vertex (Section 44.3): kappa_g = (theta_r - theta_l)/2, positive when the curve
  turns to the left, i.e. toward N x t (the sign of Definition 39.2.1 / Proposition 39.2.4).
- grad (1/2) d(., y)^2 = -log_x(y) (Section 44.4).

Run from the project root with PYTHONPATH=tools.
"""

import heapq
import itertools

import dgmesh as dm  # noqa: F401  (sets the BLAS thread limit before numpy)
import numpy as np

# ---------------------------------------------------------------------------
# small utilities
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


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def chol_solve(C, b):
    """Solve C C^T x = b for lower-triangular C (forward, then back substitution). b may be (n,) or (n, k)."""
    b = np.asarray(b, float)
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - C[i, :i] @ y[:i]) / C[i, i]
    x = np.zeros_like(b)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - C[i + 1:, i] @ x[i + 1:]) / C[i, i]
    return x


def mean_edge_length(V, F):
    E = dm.edges(F)
    return float(np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1).mean())


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def unit_cube(n=1):
    """Surface of the unit cube [0, 1]^3, every face an n x n grid of squares, each square cut by one diagonal.
    Faces are oriented outward; vertices are shared along the cube edges."""
    pts, idx, F = [], {}, []

    def vid(p):
        key = tuple(np.round(np.asarray(p) * n).astype(int))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(np.asarray(p, float))
        return idx[key]

    s = np.linspace(0, 1, n + 1)
    for axis in range(3):
        for side in (0.0, 1.0):
            a, b = [k for k in range(3) if k != axis]
            for i in range(n):
                for j in range(n):
                    quad = []
                    for (di, dj) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis], p[a], p[b] = side, s[i + di], s[j + dj]
                        quad.append(vid(p))
                    q0, q1, q2, q3 = quad
                    tris = [[q0, q1, q2], [q0, q2, q3]]
                    for t in tris:
                        P = np.array([pts[t[0]], pts[t[1]], pts[t[2]]])
                        nrm = np.cross(P[1] - P[0], P[2] - P[0])
                        out = np.zeros(3)
                        out[axis] = 1.0 if side == 1.0 else -1.0
                        F.append(t if nrm @ out > 0 else t[::-1])
    return np.array(pts), np.array(F, int)


def star_mesh(m, h=0.0, ring=1.0, zig=0.0):
    """A disk-like star: vertex 0 at (0, 0, h), m link vertices on the unit circle at height zig * (-1)^k
    (zig != 0 gives a saddle for m even), faces (0, k, k+1) oriented upward."""
    ang = 2 * np.pi * np.arange(m) / m
    V = [np.array([0.0, 0.0, h])]
    for k, a in enumerate(ang):
        V.append(np.array([ring * np.cos(a), ring * np.sin(a), zig * (-1) ** k]))
    F = [[0, 1 + k, 1 + (k + 1) % m] for k in range(m)]
    return np.array(V), np.array(F, int)


def equilateral_star(m, ell=1.0):
    """Star of m equilateral triangles of side ell around vertex 0, embedded in R^3 (angle sum m pi/3, angle defect
    2 pi - m pi/3). m <= 5: a cone (pyramid-like, link vertices in a plane below the apex); m = 6: flat; m even >= 8:
    a saddle (link vertices alternate up and down)."""
    if m == 6:
        ang = 2 * np.pi * np.arange(6) / 6
        V = np.vstack([[0, 0, 0], np.stack([ell * np.cos(ang), ell * np.sin(ang), 0 * ang], 1)])
    elif m < 6:
        # regular m-gon of side ell at depth -z below the apex, with |apex - link vertex| = ell
        R = ell / (2 * np.sin(np.pi / m))
        z = np.sqrt(ell ** 2 - R ** 2)
        ang = 2 * np.pi * np.arange(m) / m
        V = np.vstack([[0, 0, 0], np.stack([R * np.cos(ang), R * np.sin(ang), -z + 0 * ang], 1)])
    else:
        # saddle (m even, m >= 8): link vertices at angles 2 pi k / m, radius R, heights +-z (alternating), with
        # R^2 + z^2 = ell^2 (spokes) and 4 R^2 sin^2(pi/m) + 4 z^2 = ell^2 (rim edges)
        if m % 2:
            raise ValueError("equilateral saddle star: m must be even")
        s2 = np.sin(np.pi / m) ** 2
        R = ell * np.sqrt(3.0 / (4.0 * (1.0 - s2)))
        z = np.sqrt(ell ** 2 - R ** 2)
        ang = 2 * np.pi * np.arange(m) / m
        V = np.vstack([[0, 0, 0], np.stack([R * np.cos(ang), R * np.sin(ang), z * (-1.0) ** np.arange(m)], 1)])
    F = np.array([[0, 1 + k, 1 + (k + 1) % m] for k in range(m)], int)
    return np.asarray(V, float), F


# ---------------------------------------------------------------------------
# graph shortest paths (Section 44.2)
# ---------------------------------------------------------------------------


def dijkstra(V, F, sources, nbrs=None):
    """Edge-graph distances (edge weight = Euclidean edge length) from the given source vertices, and predecessors."""
    nbrs = nbrs if nbrs is not None else dm.neighbors(F, len(V))
    dist = np.full(len(V), np.inf)
    pred = np.full(len(V), -1)
    heap = []
    for s in np.atleast_1d(sources):
        dist[s] = 0.0
        heap.append((0.0, int(s)))
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for j in nbrs[i]:
            nd = d + float(np.linalg.norm(V[j] - V[i]))
            if nd < dist[j]:
                dist[j], pred[j] = nd, i
                heapq.heappush(heap, (nd, j))
    return dist, pred


def graph_path(pred, target):
    path = [int(target)]
    while pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return path[::-1]


# ---------------------------------------------------------------------------
# exact distance on the unit cube by unfolding (Section 44.2)
# ---------------------------------------------------------------------------

# the six square faces of [0, 1]^3: (axis, side); a point is on face (a, s) if p[a] == s.
CUBE_FACES = [(a, s) for a in range(3) for s in (0.0, 1.0)]


def _cube_face_of(p, tol=1e-12):
    return [k for k, (a, s) in enumerate(CUBE_FACES) if abs(p[a] - s) < tol]


def _cube_face_corners(k):
    a, s = CUBE_FACES[k]
    b, c = [i for i in range(3) if i != a]
    P = []
    for (u, v) in ((0, 0), (1, 0), (1, 1), (0, 1)):
        p = np.zeros(3)
        p[a], p[b], p[c] = s, u, v
        P.append(p)
    return np.array(P)


def _cube_adjacent(k):
    """faces sharing an edge with face k: list of (neighbor face, edge endpoints (2, 3))."""
    a, s = CUBE_FACES[k]
    out = []
    for kk, (aa, ss) in enumerate(CUBE_FACES):
        if aa == a:
            continue
        P = _cube_face_corners(k)
        on = [p for p in P if abs(p[aa] - ss) < 1e-12]
        out.append((kk, np.array(on)))
    return out


def _rotate_about(p, a, b, angle):
    """rotate the point p about the line through a, b by angle (right-hand rule about b - a)."""
    k = unit(b - a)
    v = p - a
    return a + v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * (k @ v) * (1 - np.cos(angle))


def _unfold_across(src, a, b, k):
    """rotate the image src (in the plane of the previous face) by 90 degrees about the line through a, b (a point of
    the shared edge) into the plane of cube face k, on the side of the edge opposite to face k."""
    center = _cube_face_corners(k).mean(0)
    e = unit(b - a)
    w = (center - a) - ((center - a) @ e) * e                  # in-plane direction from the edge into face k
    for ang in (np.pi / 2, -np.pi / 2):
        q = _rotate_about(src, a, b, ang)
        if abs((q - a) @ _cube_normal(k)) < 1e-9 and (q - a) @ w < 1e-12:
            return q
    raise RuntimeError("unfolding failed")


def _cube_normal(k):
    a, s = CUBE_FACES[k]
    n = np.zeros(3)
    n[a] = 1.0 if s == 1.0 else -1.0
    return n


def _in_cone(src, a, b, q, tol=1e-12):
    """is q inside the closed planar cone with apex src spanned by rays src->a and src->b (all in one plane)?"""
    u, v, w = a - src, b - src, q - src
    n = np.cross(u, v)
    if np.linalg.norm(n) < 1e-14:
        return False
    # q in cone iff w = alpha u + beta v with alpha, beta >= 0
    M = np.stack([u, v], 1)
    coef, *_ = np.linalg.lstsq(M, w, rcond=None)
    return coef.min() >= -tol and np.linalg.norm(M @ coef - w) < 1e-9


def _clip_edge_to_cone(src, wa, wb, ea, eb):
    """part of the segment [ea, eb] inside the cone (src; wa, wb), as a segment (pa, pb) or None."""
    # parametrize e(s) = ea + s (eb - ea); inside iff e - src in cone; the cone is convex: solve two half-plane tests
    n = np.cross(wa - src, wb - src)

    def side(p, w):   # sign of orientation of (w - src, p - src) relative to n
        return np.cross(w - src, p - src) @ n

    # inside: side(p, wa) >= 0 and side(wb, p) ... use: cross(wa-src, p-src).n >= 0 and cross(p-src, wb-src).n >= 0
    def f1(s):
        p = ea + s * (eb - ea)
        return np.cross(wa - src, p - src) @ n

    def f2(s):
        p = ea + s * (eb - ea)
        return np.cross(p - src, wb - src) @ n
    lo, hi = 0.0, 1.0
    for f in (f1, f2):
        f0, f1v = f(0.0), f(1.0)       # linear in s
        if f0 < 0 and f1v < 0:
            return None
        if f0 < 0 <= f1v:
            lo = max(lo, f0 / (f0 - f1v))
        elif f1v < 0 <= f0:
            hi = min(hi, f0 / (f0 - f1v))
    # also the points must lie beyond the window segment (in front of src): guaranteed by construction of faces
    if hi - lo < 1e-13:
        return None
    return ea + lo * (eb - ea), ea + hi * (eb - ea)


def cube_windows(p, max_depth=6):
    """All unfolding windows from the point p on the cube surface: list of (face index, source image (3,),
    window endpoints wa, wb (the rays from the image through wa, wb bound the visible cone), face sequence).
    The first entries are the faces containing p with an 'everything visible' window (wa = wb = None)."""
    p = np.asarray(p, float)
    out = []
    stack = []
    for k in _cube_face_of(p):
        out.append((k, p.copy(), None, None, (k,)))
        for kk, edge in _cube_adjacent(k):
            if abs(p[CUBE_FACES[kk][0]] - CUBE_FACES[kk][1]) < 1e-12:
                continue      # p lies on that face too (handled as its own start face)
            stack.append((kk, p.copy(), edge[0], edge[1], k, (k, kk)))
    while stack:
        k, src, wa, wb, prev, seq = stack.pop()
        src2 = _unfold_across(src, wa, wb, k)
        # wa, wb lie on the shared edge, hence are fixed by the rotation
        out.append((k, src2, wa, wb, seq))
        if len(seq) >= max_depth:
            continue
        for kk, edge in _cube_adjacent(k):
            if kk == prev:
                continue
            seg = _clip_edge_to_cone(src2, wa, wb, edge[0], edge[1])
            if seg is None:
                continue
            stack.append((kk, src2, seg[0], seg[1], k, seq + (kk,)))
    return out


def cube_distance(p, Q, max_depth=6, return_second=False):
    """Exact intrinsic distance on the cube surface from p to every point of Q (n, 3) (all on the surface), by
    unfolding: shortest paths on a convex polyhedron pass through no vertex, so the distance is the minimum over
    face sequences of the planar distance in the unfolding, among those whose straight segment stays visible.
    With return_second, also the second smallest candidate coming from a different face sequence (cut locus)."""
    Q = np.atleast_2d(np.asarray(Q, float))
    wins = cube_windows(p, max_depth)
    best = np.full(len(Q), np.inf)
    second = np.full(len(Q), np.inf)
    faces_of_q = [_cube_face_of(q, 1e-9) for q in Q]
    for k, src, wa, wb, seq in wins:
        for n, q in enumerate(Q):
            if k not in faces_of_q[n]:
                continue
            if wa is not None and not _in_cone(src, wa, wb, q, 1e-10):
                continue
            d = float(np.linalg.norm(q - src))
            if d < best[n] - 1e-12:
                second[n] = best[n]
                best[n] = d
            elif d < second[n] and abs(d - best[n]) > 1e-12:
                second[n] = d
    return (best, second) if return_second else best


# ---------------------------------------------------------------------------
# straightest geodesics (Section 44.3)
# ---------------------------------------------------------------------------


class MeshTopo:
    """Edge-to-face and vertex-fan data for tracing on a triangle mesh."""

    def __init__(self, V, F):
        self.V, self.F = np.asarray(V, float), np.asarray(F, int)
        self.ef = dm.edge_faces(self.F)
        self.N = dm.face_normals(self.V, self.F)
        self.vf = dm.vertex_faces(self.F)
        self.ang = dm.corner_angles(self.V, self.F)

    def across(self, f, a, b):
        fs = self.ef[(min(a, b), max(a, b))]
        others = [g for g in fs if g != f]
        return others[0] if others else None

    def fan(self, v):
        """faces around v in counterclockwise order (about N): list of (face, a, b, angle) with face = (v, a, b)."""
        items = []
        for f in self.vf[v]:
            tri = list(self.F[f])
            k = tri.index(v)
            a, b = tri[(k + 1) % 3], tri[(k + 2) % 3]
            items.append((f, a, b, self.ang[f, k]))
        # order: next face starts where the previous ends (b of one = a of the next)
        order = [items[0]]
        while len(order) < len(items):
            nxt = [it for it in items if it[1] == order[-1][2]]
            if not nxt:
                break
            order.append(nxt[0])
        return order

    def angle_sum(self, v):
        return sum(it[3] for it in self.fan(v))


def ray_exit(V, tri, p, d, skip_vertex=None):
    """first exit of the ray p + t d (t > 0) from the triangle tri: (t, a, b, s) where the exit point is
    x_a + s (x_b - x_a) on edge (a, b)."""
    best = None
    for c in range(3):
        a, b = tri[(c + 1) % 3], tri[(c + 2) % 3]
        if skip_vertex is not None and skip_vertex in (a, b) and skip_vertex != tri[c]:
            # edges incident to the start vertex are skipped (we start on them)
            continue
        e = V[b] - V[a]
        M = np.stack([d, -e], 1)
        sol, *_ = np.linalg.lstsq(M, V[a] - p, rcond=None)
        t, s = sol
        if np.linalg.norm(M @ sol - (V[a] - p)) > 1e-8:
            continue
        if t > 1e-10 and -1e-9 <= s <= 1 + 1e-9:
            if best is None or t < best[0]:
                best = (t, a, b, min(max(s, 0.0), 1.0))
    return best


def trace_straightest(V, F, f0, p0, d0, length, topo=None, vtol=1e-7, max_steps=100000):
    """Trace the discrete straightest geodesic from p0 (in face f0) in the tangent direction d0 for the given length.
    Returns dict(points (m, 3), faces (m-1,), events list of ('edge', a, b) / ('vertex', v, theta_sum),
    end_face, end_dir). At an interior vertex the exit direction leaves equal angles Theta/2 on both sides."""
    T = topo if topo is not None else MeshTopo(V, F)
    V = T.V
    f, p = int(f0), np.asarray(p0, float)
    d = unit(np.asarray(d0, float) - (np.asarray(d0, float) @ T.N[f]) * T.N[f])
    pts, faces, events = [p.copy()], [], []
    left, at_vertex = float(length), None
    for _ in range(max_steps):
        hit = ray_exit(V, T.F[f], p, d, skip_vertex=at_vertex)
        if hit is None:
            raise RuntimeError("no exit found")
        t, a, b, s = hit
        if t >= left:
            p = p + left * d
            pts.append(p.copy())
            faces.append(f)
            return dict(points=np.array(pts), faces=np.array(faces), events=events, end_face=f, end_dir=d)
        q = p + t * d
        left -= t
        faces.append(f)
        lab = np.linalg.norm(V[b] - V[a])
        if s * lab < vtol or (1 - s) * lab < vtol:              # through a vertex
            v = a if s * lab < vtol else b
            q = V[v].copy()
            pts.append(q)
            fan = T.fan(v)
            Theta = sum(it[3] for it in fan)
            if len(fan) != len(T.vf[v]) or fan[-1][2] != fan[0][1]:
                events.append(("boundary-vertex", v, Theta))
                return dict(points=np.array(pts), faces=np.array(faces), events=events, end_face=f, end_dir=d)
            # angular position of the backward direction -d inside face f
            cum = np.cumsum([0.0] + [it[3] for it in fan])
            k = [it[0] for it in fan].index(f)
            _, a0, _, _ = fan[k]
            e0 = unit(V[a0] - V[v])
            w = -d
            beta = np.arctan2(np.cross(e0, w) @ T.N[f], e0 @ w)
            alpha_in = cum[k] + beta
            alpha_out = (alpha_in + Theta / 2) % Theta
            kk = int(np.searchsorted(cum, alpha_out, side="right") - 1)
            kk = min(max(kk, 0), len(fan) - 1)
            g, a1, _, _ = fan[kk]
            e1 = unit(V[a1] - V[v])
            bet = alpha_out - cum[kk]
            d = np.cos(bet) * e1 + np.sin(bet) * np.cross(T.N[g], e1)
            events.append(("vertex", v, Theta))
            f, p, at_vertex = g, q, v
            continue
        g = T.across(f, a, b)
        if g is None:
            pts.append(q)
            events.append(("boundary", a, b))
            return dict(points=np.array(pts), faces=np.array(faces), events=events, end_face=f, end_dir=d)
        eh = unit(V[b] - V[a])
        d = (d @ eh) * eh + (d @ np.cross(T.N[f], eh)) * np.cross(T.N[g], eh)
        d = unit(d)
        pts.append(q)
        events.append(("edge", a, b))
        f, p, at_vertex = g, q, None
    raise RuntimeError("too many steps")


def locate_face(V, F, p):
    """face whose plane-projected barycentric coordinates of p are all >= -tol, closest plane first."""
    best, bf = None, None
    for f, tri in enumerate(F):
        P = V[tri]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        n /= np.linalg.norm(n)
        dist = abs((p - P[0]) @ n)
        E = np.stack([P[1] - P[0], P[2] - P[0]], 1)
        st = np.linalg.lstsq(E, p - P[0], rcond=None)[0]
        lam = np.array([1 - st.sum(), st[0], st[1]])
        if lam.min() >= -1e-9 and (best is None or dist < best):
            best, bf = dist, f
    return bf


def polyline_length(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


# ---------------------------------------------------------------------------
# sphere: exp, log, Karcher mean (Section 44.4)
# ---------------------------------------------------------------------------


def sphere_exp(p, v):
    """exp_p(v) on the unit sphere: cos|v| p + sin|v| v/|v| ((0.8.5))."""
    p, v = np.asarray(p, float), np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        return p.copy()
    return np.cos(n) * p + np.sin(n) * v / n


def sphere_log(p, q):
    """log_p(q) on the unit sphere for q != -p: theta/sin(theta) (q - cos(theta) p)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    c = float(np.clip(p @ q, -1, 1))
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    return th / np.sin(th) * (q - c * p)


def karcher_mean_sphere(Y, x0, iters=100, tol=1e-13):
    """fixed-point iteration x <- exp_x(mean_i log_x(y_i)); returns (x, list of |step|)."""
    x = unit(x0)
    steps = []
    for _ in range(iters):
        v = np.mean([sphere_log(x, y) for y in Y], axis=0)
        steps.append(float(np.linalg.norm(v)))
        x = sphere_exp(x, v)
        if steps[-1] < tol:
            break
    return x, steps


def sphere_point(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


# ---------------------------------------------------------------------------
# heat method (Section 44.6)
# ---------------------------------------------------------------------------


def face_gradients(V, F, u):
    """gradient of the piecewise-linear interpolant of u on every face: sum_i u_i N x (x_k - x_j)/(2A)."""
    E = [V[F[:, (c + 2) % 3]] - V[F[:, (c + 1) % 3]] for c in range(3)]      # edge opposite corner c
    Nn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    A2 = np.linalg.norm(Nn, axis=1)
    N = Nn / A2[:, None]
    g = sum(u[F[:, c], None] * np.cross(N, E[c]) for c in range(3))
    return g / A2[:, None]


def integrated_divergence(V, F, X):
    """b_i = (1/2) sum_{f=(i,j,k)} (cot theta_k <x_j - x_i, X_f> + cot theta_j <x_k - x_i, X_f>)."""
    cot = 1.0 / np.tan(dm.corner_angles(V, F))
    b = np.zeros(len(V))
    for c in range(3):
        i, j, k = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        val = 0.5 * (cot[:, (c + 2) % 3] * np.einsum("ij,ij->i", V[j] - V[i], X)
                     + cot[:, (c + 1) % 3] * np.einsum("ij,ij->i", V[k] - V[i], X))
        np.add.at(b, i, val)
    return b


def weak_divergence(V, F, X):
    """-sum_f A_f <grad lambda_i, X_f> (the FEM form of the same quantity)."""
    out = np.zeros(len(V))
    A = dm.face_areas(V, F)
    for f, tri in enumerate(F):
        import dgdec as dd
        G = dd.barycentric_gradients(V[tri])
        for c in range(3):
            out[tri[c]] -= A[f] * (G[c] @ X[f])
    return out


class HeatMethod:
    """Prefactored heat method on a mesh (closed or with boundary).

    bc: 'neumann' (L as is), 'dirichlet' (u = 0 on the boundary in step I), or 'average' (mean of the two).
    mass: 'voronoi' or 'barycentric'. t = m_t * (mean edge length)^2.
    """

    def __init__(self, V, F, m_t=1.0, mass="voronoi", bc="neumann"):
        self.V, self.F = V, F
        self.L = dm.cotan_laplacian(V, F)
        self.m = dm.mass_voronoi(V, F) if mass == "voronoi" else dm.mass_barycentric(V, F)
        self.ell = mean_edge_length(V, F)
        self.t = m_t * self.ell ** 2
        n = len(V)
        A = np.diag(self.m) - self.t * self.L
        self.CN = np.linalg.cholesky(A)
        self.bc = bc
        bnd = dm.boundary_vertices(F)
        self.bnd = np.array(bnd, int)
        self.inner = np.setdiff1d(np.arange(n), self.bnd)
        if len(bnd) and bc in ("dirichlet", "average"):
            self.CD = np.linalg.cholesky(A[np.ix_(self.inner, self.inner)])
        self.CP = np.linalg.cholesky(-self.L + np.outer(self.m, self.m))

    def heat(self, sources):
        delta = np.zeros(len(self.V))
        delta[np.atleast_1d(sources)] = 1.0
        uN = chol_solve(self.CN, delta)
        if self.bc == "neumann" or len(self.bnd) == 0:
            return uN
        uD = np.zeros(len(self.V))
        uD[self.inner] = chol_solve(self.CD, delta[self.inner])
        return uD if self.bc == "dirichlet" else 0.5 * (uN + uD)

    def distance(self, sources, return_all=False):
        u = self.heat(sources)
        gu = face_gradients(self.V, self.F, u)
        X = -gu / np.linalg.norm(gu, axis=1, keepdims=True)
        b = integrated_divergence(self.V, self.F, X)
        phi = chol_solve(self.CP, -b)
        phi = phi - phi.min()
        if return_all:
            return phi, u, X, b
        return phi


def mixed_errors(phi, exact, area):
    """(max abs error, area-weighted mean abs error)."""
    e = np.abs(phi - exact)
    return float(e.max()), float(area @ e / area.sum())


# ---------------------------------------------------------------------------
# geodesic ODE (Section 44.5)
# ---------------------------------------------------------------------------


def surface_geodesic(X, coords, params, x0, v0, ts):
    """integrate the geodesic equation of the parametrized surface X(coords) (sympy Matrix) with dgsym.christoffel
    and dgnum.rk4; returns (c(t) (n, 2), c'(t) (n, 2), gamma(t) (n, 3))."""
    import sympy as sp

    import dgnum
    import dgsym
    Xs = sp.Matrix(X).subs(params)
    g = dgsym.induced_metric(Xs, coords)
    G = dgsym.christoffel(g, coords)
    flat = [G[k][i][j] for k in range(2) for i in range(2) for j in range(2)]
    Gam = sp.lambdify(coords, flat, modules="numpy")
    Xf = sp.lambdify(coords, list(Xs), modules="numpy")

    def rhs(t, y):
        Gk = np.array([np.broadcast_to(v, ()) for v in Gam(y[0], y[1])], float).reshape(2, 2, 2)
        return np.concatenate([y[2:], -np.einsum("kij,i,j->k", Gk, y[2:], y[2:])])
    Y = dgnum.rk4(rhs, np.concatenate([x0, v0]), ts)
    P = np.array([np.array(Xf(*y[:2]), float).ravel() for y in Y])
    return Y[:, :2], Y[:, 2:], P


def torus_xyz(u, v, R=2.0, r=0.8):
    return np.stack([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)], -1)


# ---------------------------------------------------------------------------
# medial axis (Section 44.2)
# ---------------------------------------------------------------------------


def polygon_closest(P, poly):
    """for each point of P (n, 2): (distance to the boundary of the closed polygon poly (m, 2), closest points per
    edge (n, m, 2), distances per edge (n, m))."""
    P = np.atleast_2d(P)
    A = np.asarray(poly, float)
    B = np.roll(A, -1, 0)
    AB = B - A
    t = np.einsum("nmk,mk->nm", P[:, None, :] - A[None], AB) / np.einsum("mk,mk->m", AB, AB)
    t = np.clip(t, 0, 1)
    C = A[None] + t[..., None] * AB[None]
    D = np.linalg.norm(P[:, None, :] - C, axis=2)
    return D.min(1), C, D


def ellipse_closest(P, a=2.0, b=1.0, n=20000):
    """distance from points P (n, 2) to the ellipse (a cos t, b sin t) and the number of (well separated) closest
    points within 1e-9 (dense sampling refined by Newton)."""
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    E = np.stack([a * np.cos(ts), b * np.sin(ts)], 1)
    out_d, out_k = [], []
    for p in np.atleast_2d(P):
        D = np.linalg.norm(E - p, axis=1)
        cand = []
        for i in np.argsort(D)[:200]:
            t = ts[i]
            for _ in range(30):     # Newton on f(t) = <e(t) - p, e'(t)>
                e = np.array([a * np.cos(t), b * np.sin(t)])
                e1 = np.array([-a * np.sin(t), b * np.cos(t)])
                e2 = -e
                f = (e - p) @ e1
                fp = e1 @ e1 + (e - p) @ e2
                if abs(fp) < 1e-14:
                    break
                t -= f / fp
            e = np.array([a * np.cos(t), b * np.sin(t)])
            cand.append((float(np.linalg.norm(e - p)), e))
        dmin = min(c[0] for c in cand)
        pts = [c[1] for c in cand if c[0] < dmin + 1e-9]
        uniq = []
        for q in pts:
            if all(np.linalg.norm(q - w) > 1e-6 for w in uniq):
                uniq.append(q)
        out_d.append(dmin)
        out_k.append(len(uniq))
    return np.array(out_d), np.array(out_k)


if __name__ == "__main__":
    V, F = unit_cube(2)
    print(len(V), len(F), dm.euler_characteristic(F), dm.signed_volume(V, F))
    print(cube_distance(np.array([0.5, 0.5, 0.0]), np.array([[0.5, 0.5, 1.0], [1, 1, 1.0]])))


def cube_distance_vec(p, Q, max_depth=6):
    """Vectorized cube_distance: (best, second) for points Q (n, 3) on the cube surface. 'second' is the smallest
    candidate from another visible unfolding whose source image differs from the best one (used to locate the cut
    locus: best ~ second)."""
    Q = np.atleast_2d(np.asarray(Q, float))
    wins = cube_windows(p, max_depth)
    face = np.full(len(Q), -1)
    for k, (a, s) in enumerate(CUBE_FACES):
        on = np.abs(Q[:, a] - s) < 1e-9
        face[(face < 0) & on] = k
    best = np.full(len(Q), np.inf)
    second = np.full(len(Q), np.inf)
    best_src = np.full((len(Q), 3), np.nan)
    for k, src, wa, wb, seq in wins:
        idx = np.where(face == k)[0]
        if len(idx) == 0:
            continue
        W = Q[idx] - src
        d = np.linalg.norm(W, axis=1)
        if wa is not None:
            u, v = wa - src, wb - src
            a, _ = CUBE_FACES[k]
            ax = [i for i in range(3) if i != a]
            M = np.array([[u[ax[0]], v[ax[0]]], [u[ax[1]], v[ax[1]]]])
            if abs(np.linalg.det(M)) < 1e-14:
                continue
            coef = np.linalg.solve(M, W[:, ax].T).T
            ok = (coef >= -1e-10).all(1)
        else:
            ok = np.ones(len(idx), bool)
        for n_, dd, okk in zip(idx, d, ok):
            if not okk:
                continue
            if dd < best[n_] - 1e-12:
                if np.isfinite(best[n_]) and np.linalg.norm(best_src[n_] - src) > 1e-9:
                    second[n_] = best[n_]
                best[n_], best_src[n_] = dd, src
            elif dd < second[n_] and np.linalg.norm(best_src[n_] - src) > 1e-9:
                second[n_] = dd
    return best, second


def cone_distance(r1, a1, r2, a2, Theta):
    """intrinsic distance on the infinite polyhedral cone (one vertex, total angle Theta) between the points with
    polar coordinates (r, alpha), alpha in [0, Theta) measured at the vertex: unfold whichever way around is
    shorter; if both angular separations exceed pi the shortest path runs through the vertex (length r1 + r2)."""
    D = np.abs(np.asarray(a1) - np.asarray(a2)) % Theta
    D = np.minimum(D, Theta - D)
    straight = np.sqrt(np.maximum(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(np.minimum(D, np.pi)), 0))
    return np.where(D < np.pi, straight, r1 + r2), D


def polygon_medial_samples(poly, h, tol_factor=1.0):
    """grid points (spacing h) inside the closed polygon whose distance to the boundary is attained (to within
    tol_factor * h) at two different local minima of the distance along the boundary: a sampled medial axis.
    Candidates are the feet on edges that fall inside the edge, and the vertices at which both adjacent edges have
    their foot (a reflex vertex seen from the point). A foot clamped to a vertex that the next edge already beats is
    not a local minimum and is ignored. Returns (points (k, 2), radius r = distance to the boundary)."""
    from matplotlib.path import Path as MPath
    A = np.asarray(poly, float)
    B = np.roll(A, -1, 0)
    AB = B - A
    lo, hi = A.min(0), A.max(0)
    xs = np.arange(lo[0] + h / 2, hi[0], h)
    ys = np.arange(lo[1] + h / 2, hi[1], h)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    P = P[MPath(A).contains_points(P)]
    out_p, out_r = [], []
    for chunk in np.array_split(P, max(1, len(P) // 4000)):
        t = np.einsum("nmk,mk->nm", chunk[:, None, :] - A[None], AB) / np.einsum("mk,mk->m", AB, AB)
        C = A[None] + np.clip(t, 0, 1)[..., None] * AB[None]
        D = np.linalg.norm(chunk[:, None, :] - C, axis=2)
        cand_edge = np.where((t > 0) & (t < 1), D, np.inf)
        # vertex i joins edge i-1 (ending at A[i]) and edge i (starting at A[i])
        vert = (np.roll(t, 1, axis=1) >= 1) & (t <= 0)
        dv = np.linalg.norm(chunk[:, None, :] - A[None], axis=2)
        cand_vert = np.where(vert, dv, np.inf)
        allc = np.concatenate([cand_edge, cand_vert], 1)
        srt = np.sort(allc, 1)
        dmin = D.min(1)
        m = np.isfinite(srt[:, 1]) & ((srt[:, 1] - srt[:, 0]) < tol_factor * h) & (np.abs(srt[:, 0] - dmin) < 1e-12)
        out_p.append(chunk[m])
        out_r.append(dmin[m])
    return np.vstack(out_p), np.concatenate(out_r)


def cone_ray(psi, Theta, tmax, r0=1.0, n=400):
    """the straightest geodesic from p = (r0, alpha = 0) on the infinite polyhedral cone of total angle Theta, leaving
    p at angle psi in a planar unfolding around p (psi = pi aims at the vertex). Returns (r(t), alpha(t)) for
    t in [0, tmax] with alpha unwrapped (alpha mod Theta is the point on the cone). The ray aimed at the vertex
    continues at alpha = Theta / 2 (equal angles on both sides)."""
    t = np.linspace(0.0, tmax, n)
    if abs(np.cos(psi) + 1) < 1e-15:
        r = np.abs(r0 - t)
        alpha = np.where(t <= r0, 0.0, Theta / 2)
        return r, alpha
    P = np.stack([r0 + t * np.cos(psi), t * np.sin(psi)], 1)
    r = np.linalg.norm(P, axis=1)
    alpha = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    return r, alpha


def cone_meeting_point(beta, delta, r0=1.0):
    """two straightest geodesics from p (distance r0 from a cone vertex with angle defect delta > 0) leaving at angles
    pi -+ beta (0 < beta < delta/2) meet behind the vertex: returns (their common length t, the distance s of the
    meeting point from the vertex). t = r0 sin(delta/2)/sin(delta/2 - beta), s = r0 sin(beta)/sin(delta/2 - beta)."""
    return r0 * np.sin(delta / 2) / np.sin(delta / 2 - beta), r0 * np.sin(beta) / np.sin(delta / 2 - beta)
