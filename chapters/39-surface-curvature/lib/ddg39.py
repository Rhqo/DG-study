"""Chapter 39 helpers (Curvature of Surfaces; Track 2, GUIDELINES 2.5).

Import from a figure or verify script of this chapter with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import ddg39

The canonical implementations are tools/dgsym.py (smooth conventions and EXAMPLES), tools/dgmesh.py (meshes, face
normals, corner angles, bending angles) and tools/dgdec.py. This module adds what Chapter 39 needs:

- numeric parametrizations (vectorized): sphere, torus (2, 0.8), cylinder, saddle, ellipsoid (2, 1.5, 1), the bumpy
  sphere rho = 1 + a cos 4t, helicoid, catenoid;
- numeric fundamental forms by central differences: Gauss map, g, h, W = g^{-1} h, principal curvatures and
  directions (39.1, 39.2);
- curves on surfaces: Darboux frame, normal and geodesic curvature by finite differences (39.2);
- principal line fields near an umbilic of a Monge-form graph (39.2);
- surfaces of revolution: K from the profile, cap integrals (39.4);
- mesh reconstruction from edge lengths and bending angles, vertex holonomy (39.5);
- verify helpers run_printed / printed_block.

Conventions (GUIDELINES 6.3): W = -dN, H = tr W / 2, N = x_u x x_v / |x_u x x_v| unless stated. The torus is
T_{2, 0.8} with x_u x x_v inward; the sphere uses (theta, phi) with x_theta x x_phi outward; the saddle graph has
the upward normal. Bending angles follow dgmesh.edge_bending_angles (positive at a convex edge for outward normals).
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402

import dgmesh as dm  # noqa: E402

R_TORUS, r_TORUS = 2.0, 0.8
SADDLE_Q = (0.5, -0.4)
ELLIPSOID = (2.0, 1.5, 1.0)

# ---------------------------------------------------------------------------
# numeric parametrizations: x(u, v) returns an array of shape (3,) + shape of u
# ---------------------------------------------------------------------------


def sphere(th, ph, r=1.0):
    return np.array([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th) + 0 * ph])


def torus(u, v, R=R_TORUS, r=r_TORUS):
    return np.array([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u) + 0 * v])


def cylinder(u, v, r=1.0):
    return np.array([r * np.cos(u) + 0 * v, r * np.sin(u) + 0 * v, v + 0 * u])


def saddle(u, v):
    return np.array([u + 0 * v, v + 0 * u, u ** 2 - v ** 2])


def ellipsoid(th, ph, a=ELLIPSOID[0], b=ELLIPSOID[1], c=ELLIPSOID[2]):
    return np.array([a * np.sin(th) * np.cos(ph), b * np.sin(th) * np.sin(ph), c * np.cos(th) + 0 * ph])


def ellipsoid_K(p, a=ELLIPSOID[0], b=ELLIPSOID[1], c=ELLIPSOID[2]):
    """Gaussian curvature of x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 at the point p (closed form)."""
    x, y, z = p
    return 1.0 / (a * b * c * (x * x / a ** 4 + y * y / b ** 4 + z * z / c ** 4)) ** 2


def bumpy_rho(t, a=0.2):
    return 1.0 + a * np.cos(4 * t)


def bumpy_sphere(t, v, a=0.2):
    """x = rho(t) (sin t cos v, sin t sin v, cos t), rho = 1 + a cos 4t (a polynomial in cos t: smooth at the poles)."""
    rho = bumpy_rho(t, a)
    return np.array([rho * np.sin(t) * np.cos(v), rho * np.sin(t) * np.sin(v), rho * np.cos(t) + 0 * v])


def bumpy_profile(t, a=0.2):
    rho = bumpy_rho(t, a)
    return rho * np.sin(t), rho * np.cos(t)


def torus_profile(t, R=R_TORUS, r=r_TORUS):
    return R + r * np.cos(t), r * np.sin(t)


def helicoid(u, v, c=1.0):
    return np.array([v * np.cos(u), v * np.sin(u), c * u + 0 * v])


def catenoid(u, v, c=1.0):
    return np.array([c * np.cosh(v / c) * np.cos(u), c * np.cosh(v / c) * np.sin(u), v + 0 * u])


# ---------------------------------------------------------------------------
# numeric fundamental forms (central differences)
# ---------------------------------------------------------------------------


def partials(f, u, v, h=1e-4):
    return (f(u + h, v) - f(u - h, v)) / (2 * h), (f(u, v + h) - f(u, v - h)) / (2 * h)


def second_partials(f, u, v, h=1e-4):
    fuu = (f(u + h, v) - 2 * f(u, v) + f(u - h, v)) / h ** 2
    fvv = (f(u, v + h) - 2 * f(u, v) + f(u, v - h)) / h ** 2
    fuv = (f(u + h, v + h) - f(u + h, v - h) - f(u - h, v + h) + f(u - h, v - h)) / (4 * h * h)
    return fuu, fuv, fvv


def unit_normal(x, u, v, h=1e-4):
    xu, xv = partials(x, u, v, h)
    n = np.cross(xu, xv, axis=0)
    return n / np.linalg.norm(n, axis=0)


def forms(x, u, v, h=1e-4):
    """(g, hh, N, xu, xv) at a single point (u, v) of a numeric parametrization; hh = (L M; M N) (8.5.2)."""
    xu, xv = partials(x, u, v, h)
    xuu, xuv, xvv = second_partials(x, u, v, h)
    n = np.cross(xu, xv)
    n = n / np.linalg.norm(n)
    g = np.array([[xu @ xu, xu @ xv], [xv @ xu, xv @ xv]])
    hh = np.array([[xuu @ n, xuv @ n], [xuv @ n, xvv @ n]])
    return g, hh, n, xu, xv


def weingarten(x, u, v, h=1e-4):
    g, hh, n, xu, xv = forms(x, u, v, h)
    return np.linalg.solve(g, hh)


def principal(x, u, v, h=1e-4):
    """(k1, k2, X1, X2, g): principal curvatures k1 >= k2 and g-orthonormal eigenvectors X1, X2 in the parameter
    plane (2-vectors) at (u, v)."""
    g, hh, n, xu, xv = forms(x, u, v, h)
    # generalized symmetric eigenproblem hh X = k g X via the Cholesky factor of g
    C = np.linalg.cholesky(g)
    Ci = np.linalg.inv(C)
    lam, Y = np.linalg.eigh(Ci @ hh @ Ci.T)
    order = np.argsort(-lam)
    lam, Y = lam[order], Y[:, order]
    X = Ci.T @ Y
    return lam[0], lam[1], X[:, 0], X[:, 1], g


def KH(x, u, v, h=1e-4):
    W = weingarten(x, u, v, h)
    return np.linalg.det(W), 0.5 * np.trace(W)


def KH_grid(x, U, V, h=1e-4):
    """K, H, area density sqrt(EG - F^2) on arrays U, V (vectorized central differences)."""
    xu, xv = partials(x, U, V, h)
    xuu, xuv, xvv = second_partials(x, U, V, h)
    n = np.cross(xu, xv, axis=0)
    dA = np.linalg.norm(n, axis=0)
    n = n / dA
    E, F, G = (xu * xu).sum(0), (xu * xv).sum(0), (xv * xv).sum(0)
    L, M, N = (xuu * n).sum(0), (xuv * n).sum(0), (xvv * n).sum(0)
    det = E * G - F * F
    return (L * N - M * M) / det, (E * N - 2 * F * M + G * L) / (2 * det), dA


# ---------------------------------------------------------------------------
# curves on surfaces (Section 39.2)
# ---------------------------------------------------------------------------


def darboux(x, c, s, h=1e-4):
    """kappa, kappa_n, kappa_g of the curve gamma = x o c at parameter s (any regular parameter).

    Darboux frame (t, N x t, N) with N the Gauss map of x along the curve; kappa_n = <t', N>, kappa_g = <t', N x t>
    where ' is d/ds_arc."""
    gam = lambda s_: x(*c(s_))  # noqa: E731
    g1 = (gam(s + h) - gam(s - h)) / (2 * h)
    g2 = (gam(s + h) - 2 * gam(s) + gam(s - h)) / h ** 2
    speed = np.linalg.norm(g1)
    t = g1 / speed
    acc = (g2 - (g2 @ t) * t) / speed ** 2          # dt/ds_arc
    N = unit_normal(x, *c(s))
    kn = acc @ N
    kg = acc @ np.cross(N, t)
    return np.linalg.norm(acc), kn, kg, t, N


# ---------------------------------------------------------------------------
# Monge graphs near an umbilic (Section 39.2)
# ---------------------------------------------------------------------------


def monge(a, b, c, d, k=1.0):
    """f = k/2 (x^2 + y^2) + (a x^3 + 3 b x^2 y + 3 c x y^2 + d y^3) / 6; returns derivs(x, y) -> (fu, fv, fuu, fuv,
    fvv). The origin is an umbilic with k1 = k2 = k (upward normal)."""
    def derivs(x, y):
        fu = k * x + (a * x * x + 2 * b * x * y + c * y * y) / 2
        fv = k * y + (b * x * x + 2 * c * x * y + d * y * y) / 2
        fuu = k + a * x + b * y
        fuv = b * x + c * y
        fvv = k + c * x + d * y
        return fu, fv, fuu, fuv, fvv
    return derivs


def graph_W(fu, fv, fuu, fuv, fvv):
    g = np.array([[1 + fu * fu, fu * fv], [fu * fv, 1 + fv * fv]])
    w = np.sqrt(1 + fu * fu + fv * fv)
    hh = np.array([[fuu, fuv], [fuv, fvv]]) / w
    return g, hh


def graph_principal(derivs, x, y):
    """(k1, k2, X1, X2) of the graph at (x, y); X1, X2 are unit (Euclidean) directions in the xy-plane."""
    g, hh = graph_W(*derivs(x, y))
    C = np.linalg.cholesky(g)
    Ci = np.linalg.inv(C)
    lam, Y = np.linalg.eigh(Ci @ hh @ Ci.T)
    order = np.argsort(-lam)
    X = Ci.T @ Y[:, order]
    X = X / np.linalg.norm(X, axis=0)
    return lam[order][0], lam[order][1], X[:, 0], X[:, 1]


def line_field_index(derivs, rad=0.05, n=720, which=0):
    """index of the principal line field (which = 0: k1, 1: k2) around the origin (a multiple of 1/2)."""
    ang = []
    for t in np.linspace(0, 2 * np.pi, n + 1):
        X = graph_principal(derivs, rad * np.cos(t), rad * np.sin(t))[2 + which]
        ang.append(np.arctan2(X[1], X[0]))
    tot = 0.0
    for a0, a1 in zip(ang[:-1], ang[1:]):
        tot += (a1 - a0 + np.pi / 2) % np.pi - np.pi / 2
    return tot / (2 * np.pi)


def trace_line(derivs, p0, which=0, step=0.004, nmax=4000, box=1.0, stop_rad=0.0, direction=1.0):
    """integrate a curvature line of the graph from p0 (RK2 on the line field, sign kept continuous)."""
    p = np.array(p0, float)
    prev = None
    pts = [p.copy()]
    for _ in range(nmax):
        X = graph_principal(derivs, *p)[2 + which]
        if prev is None:
            X = direction * X
        elif X @ prev < 0:
            X = -X
        mid = p + 0.5 * step * X
        Xm = graph_principal(derivs, *mid)[2 + which]
        if Xm @ X < 0:
            Xm = -Xm
        p = p + step * Xm
        prev = Xm
        pts.append(p.copy())
        if np.max(np.abs(p)) > box or (stop_rad > 0 and np.linalg.norm(p) < stop_rad):
            break
    return np.array(pts)


def separatrix_directions(derivs, which=0, rad=0.02, n=3600):
    """polar angles t (radians) near which the line field is radial on the circle of radius rad."""
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = []
    for t in ts:
        X = graph_principal(derivs, rad * np.cos(t), rad * np.sin(t))[2 + which]
        cr, dt = X[0] * np.sin(t) - X[1] * np.cos(t), X[0] * np.cos(t) + X[1] * np.sin(t)
        s.append(2 * cr * dt)          # sin of twice the angle between X and the radial direction (sign-free)
    s = np.array(s)
    out = []
    for k in range(n):
        if s[k] == 0 or s[k] * s[(k + 1) % n] < 0:
            t = ts[k]
            X = graph_principal(derivs, rad * np.cos(t), rad * np.sin(t))[2 + which]
            if abs(X[0] * np.cos(t) + X[1] * np.sin(t)) > 0.7:
                out.append(t)
    return out


# ---------------------------------------------------------------------------
# surfaces of revolution (Section 39.4)
# ---------------------------------------------------------------------------


def profile_derivs(profile, t, h=1e-5):
    (rp, zp), (rm, zm), (r0, z0) = profile(t + h), profile(t - h), profile(t)
    return r0, z0, (rp - rm) / (2 * h), (zp - zm) / (2 * h), (rp - 2 * r0 + rm) / h ** 2, (zp - 2 * z0 + zm) / h ** 2


def revolution_K(profile, t):
    """K of the surface of revolution of the profile (r(t), z(t)) (independent of the normal)."""
    r, z, r1, z1, r2, z2 = profile_derivs(profile, t)
    return (r1 * z2 - r2 * z1) * z1 / (r * (r1 * r1 + z1 * z1) ** 2)


def cap_terms(profile, t0, n=4000):
    """(integral of K dA over t in [0, t0], integral of kappa_g ds over the parallel t = t0) for a profile with
    r(0) = 0; the parallel is traversed with v increasing (positive for N = x_t x x_v / |...|)."""
    t = (np.arange(n) + 0.5) * t0 / n
    r, z, r1, z1, r2, z2 = profile_derivs(profile, t)
    K = (r1 * z2 - r2 * z1) * z1 / (r * (r1 * r1 + z1 * z1) ** 2)
    area = 2 * np.pi * np.sum(K * r * np.hypot(r1, z1)) * t0 / n
    _, _, r1b, z1b, _, _ = profile_derivs(profile, t0)
    return area, 2 * np.pi * r1b / np.hypot(r1b, z1b)


# ---------------------------------------------------------------------------
# mesh reconstruction (Section 39.5)
# ---------------------------------------------------------------------------


def rotate(v, a, t):
    """rotate v about the unit axis a by the angle t (right-hand rule, Rodrigues)."""
    return v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * np.dot(a, v) * (1 - np.cos(t))


def apex(xi, xj, n, lik, ljk):
    """the point x_k with |x_k - x_i| = lik, |x_k - x_j| = ljk for which (x_i, x_j, x_k) has unit normal n."""
    lij = np.linalg.norm(xj - xi)
    e = (xj - xi) / lij
    a = (lik ** 2 - ljk ** 2 + lij ** 2) / (2 * lij)
    return xi + a * e + np.sqrt(max(lik ** 2 - a ** 2, 0.0)) * np.cross(n, e)


def reconstruct(F, ell, theta, order=None):
    """vertex positions from edge lengths ell and bending angles theta (dicts on sorted edges), gluing faces by a
    depth-first search from face 0, which is placed with x_i = 0, x_j on the positive x-axis, normal e_3.
    Returns (X, sequence of faces in the order they were placed, set of tree edges)."""
    F = np.asarray(F)
    lng = lambda a, b: ell[min(a, b), max(a, b)]  # noqa: E731
    face = {(int(f[c]), int(f[(c + 1) % 3])): g for g, f in enumerate(F) for c in range(3)}
    i, j, k = (int(a) for a in F[0])
    X = {i: np.zeros(3), j: np.array([lng(i, j), 0.0, 0.0])}
    N = {0: np.array([0.0, 0.0, 1.0])}
    X[k] = apex(X[i], X[j], N[0], lng(i, k), lng(j, k))
    seq, tree, stack = [0], set(), [0]
    while stack:
        f = stack.pop()
        for c in range(3):
            a, b = int(F[f][c]), int(F[f][(c + 1) % 3])
            g = face.get((b, a))
            if g is None or g in N:
                continue
            e = (X[b] - X[a]) / np.linalg.norm(X[b] - X[a])
            N[g] = rotate(N[f], e, theta[min(a, b), max(a, b)])
            m = [int(w) for w in F[g] if w != a and w != b][0]
            if m not in X:
                X[m] = apex(X[b], X[a], N[g], lng(b, m), lng(a, m))
            seq.append(g)
            tree.add((min(a, b), max(a, b)))
            stack.append(g)
    n0 = int(F.max()) + 1
    return np.array([X[v] for v in range(n0)]), seq, tree


def star_cycle(F, i):
    """link vertices j_1, ..., j_m of an interior vertex i in orientation order: faces (i, j_k, j_{k+1})."""
    nxt = {}
    for f in F:
        f = [int(a) for a in f]
        if i in f:
            c = f.index(i)
            nxt[f[(c + 1) % 3]] = f[(c + 2) % 3]
    j0 = next(iter(nxt))
    seq = [j0]
    while nxt[seq[-1]] != j0:
        seq.append(nxt[seq[-1]])
    return seq


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def corner_angle(ell, i, j, k):
    """angle at i of the triangle (i, j, k) from the three edge lengths (law of cosines)."""
    lng = lambda a, b: ell[min(a, b), max(a, b)]  # noqa: E731
    a, b, c = lng(i, j), lng(i, k), lng(j, k)
    return np.arccos(np.clip((a * a + b * b - c * c) / (2 * a * b), -1.0, 1.0))


def vertex_holonomy(F, ell, theta, i):
    """prod_k Rz(alpha_k) Rx(-theta_{i j_{k+1}}) around the interior vertex i (identity if the data come from an
    immersion)."""
    seq = star_cycle(F, i)
    m = len(seq)
    P = np.eye(3)
    for k in range(m):
        j, jn = seq[k], seq[(k + 1) % m]
        P = P @ Rz(corner_angle(ell, i, j, jn)) @ Rx(-theta[min(i, jn), max(i, jn)])
    return P


def lengths_angles(V, F):
    """edge lengths on all edges and bending angles on interior edges, keyed by sorted Python-int pairs."""
    b = dm.edge_bending_angles(V, F)
    ell = {(int(i), int(j)): float(np.linalg.norm(V[j] - V[i])) for (i, j) in dm.edge_faces(F)}
    theta = {(int(i), int(j)): float(t) for (i, j), (t, _) in b.items()}
    return ell, theta


def rigid_error(X, Y):
    """max |R X_k + c - Y_k| for the best proper rigid motion (Kabsch, det R = +1)."""
    cx, cy = X.mean(0), Y.mean(0)
    Hm = (X - cx).T @ (Y - cy)
    U, S, Vt = np.linalg.svd(Hm)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    return float(np.max(np.linalg.norm((X - cx) @ Rm.T + cy - Y, axis=1)))


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
