"""Chapter 36 (Discrete Exterior Calculus) helpers for the figure and verify scripts.

The DEC itself (d0, d1, de Rham map, Whitney forms, circumcenters, Hodge stars, cotan Laplacian) is the canonical
implementation in tools/dgdec.py and tools/dgmesh.py. This module adds only what the chapter needs on top of it:
the small meshes used in the text, the dual orientation of Section 36.5, and drawing helpers for discrete forms.

Conventions (GUIDELINES 2.5): edge {i, j} with i < j is oriented i -> j and numbered by dgdec.edge_list; a face
(i, j, k) is oriented by its row order; planar meshes are counterclockwise, closed meshes outward.
Dual orientation (Section 36.5): the dual edge e* of e = [i, j] runs from the face with relative orientation
o(f, e) = -1 (on the right of i -> j) to the face with o(f, e) = +1 (on the left), i.e. along e rotated by +90
degrees; dual faces i* turn counterclockwise. Then the dual exterior derivatives are d1^T and -d0^T.

Run from the project root with ``PYTHONPATH=tools``; scripts add this folder to sys.path.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

# ---------------------------------------------------------------------------
# small meshes
# ---------------------------------------------------------------------------


def grid(n, a=1.0, pattern="alternate"):
    """Planar grid of the square [0, a]^2 with (n + 1) x (n + 1) vertices: (V (m, 2), F). Vertex (n + 1) i + j sits
    at (a i / n, a j / n); faces are counterclockwise (dgmesh.grid_mesh)."""
    xs = np.linspace(0.0, a, n + 1)
    V, F = dm.grid_mesh(xs, xs, pattern)
    return V[:, :2].copy(), F


def reference_triangle():
    """The triangle [(0, 0), (1, 0), (0, 1)] (counterclockwise)."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])


def equilateral_triangle(h=1.0):
    """The equilateral triangle with side h, counterclockwise, base on the x-axis."""
    return np.array([[0.0, 0.0], [h, 0.0], [0.5 * h, 0.5 * np.sqrt(3.0) * h]]), np.array([[0, 1, 2]])


def kite():
    """Two triangles for hand computations (Section 36.4): x_0 = (0, 0), x_1 = (2, 0), x_2 = (1, 1.4),
    x_3 = (2.2, 1.6), faces (0, 1, 2) and (1, 3, 2), both counterclockwise. Edges 01, 02, 12, 13, 23."""
    V = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.4], [2.2, 1.6]])
    return V, np.array([[0, 1, 2], [1, 3, 2]])


def tetrahedron():
    """The boundary of the regular tetrahedron of Example 33.5.2 (same vertices and faces as lib/ch33.py):
    faces (0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2), oriented outward."""
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0)
    F = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    assert dm.signed_volume(V, F) > 0
    return V, F


def annulus(nr=4, nt=16, r0=1.0, r1=2.0):
    """Polar triangulation of the annulus r0 <= r <= r1 (nr rings of vertices, nt per ring), counterclockwise.
    Vertex nt i + j sits at radius r_i, angle 2 pi j / nt."""
    rs = np.linspace(r0, r1, nr)
    ts = 2 * np.pi * np.arange(nt) / nt
    V = np.array([[r * np.cos(t), r * np.sin(t)] for r in rs for t in ts])
    idx = lambda i, j: i * nt + j % nt  # noqa: E731
    F = []
    for i in range(nr - 1):
        for j in range(nt):
            F += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return V, np.array(F, int)


def equilateral_patch(n=2, h=1.0):
    """The triangular lattice of spacing h restricted to the hexagon of lattice radius n around the origin:
    3 n (n + 1) + 1 vertices, 6 n^2 equilateral triangles (counterclockwise). Vertex 0 is the origin."""
    a1, a2 = np.array([h, 0.0]), np.array([0.5 * h, 0.5 * np.sqrt(3.0) * h])
    pts, key = [], {}
    order = sorted(((p, q) for p in range(-n, n + 1) for q in range(-n, n + 1) if max(abs(p), abs(q), abs(p + q)) <= n),
                   key=lambda pq: (max(abs(pq[0]), abs(pq[1]), abs(pq[0] + pq[1])), pq))
    for p, q in order:
        key[(p, q)] = len(pts)
        pts.append(p * a1 + q * a2)
    F = []
    for p in range(-n - 1, n + 1):
        for q in range(-n - 1, n + 1):
            for tri in (((p, q), (p + 1, q), (p, q + 1)), ((p + 1, q), (p + 1, q + 1), (p, q + 1))):
                if all(t in key for t in tri):
                    F.append([key[t] for t in tri])
    assert len(F) == 6 * n * n
    return np.array(pts), np.array(F, int)


def non_delaunay_pair():
    """The non-Delaunay pair of the dgdec self-test: x_0 = (0, 0), x_1 = (2, 0), x_2 = (1, 0.3), x_3 = (1, -0.25),
    faces (0, 1, 2) and (1, 0, 3). Both angles opposite the shared edge 01 are obtuse."""
    return np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.3], [1.0, -0.25]]), np.array([[0, 1, 2], [1, 0, 3]])


# ---------------------------------------------------------------------------
# chains (Section 36.2)
# ---------------------------------------------------------------------------


def boundary_of_simplex(s):
    """The boundary chain of the oriented simplex s as a dict {sorted face: coefficient}, the face orientation being
    the sorted order (each term (-1)^m [v_0 .. v_m-hat .. v_k] is rewritten with the sign of the sorting)."""
    out = {}
    for sign, f in dd.induced_boundary(s):
        key = tuple(sorted(f))
        out[key] = out.get(key, 0) + sign * dd.permutation_sign(f, key)
    return {k: v for k, v in out.items() if v}


def boundary_of_chain(chain):
    """Linear extension of boundary_of_simplex to a chain {sorted simplex: coefficient}."""
    out = {}
    for s, c in chain.items():
        for f, v in boundary_of_simplex(s).items():
            out[f] = out.get(f, 0) + c * v
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# dual orientation and the circumcentric dual (Section 36.5)
# ---------------------------------------------------------------------------


def left_right_faces(F):
    """(n1, 2) array of (right face, left face) for every edge in edge_list order: the left face f has
    d1[f, e] = +1 (it lies on the left of i -> j for a counterclockwise planar mesh), the right face d1[f, e] = -1.
    A missing face (boundary edge) is -1."""
    D1 = dd.d1(F)
    out = -np.ones((D1.shape[1], 2), int)
    for f, e in zip(*np.nonzero(D1)):
        out[e, 0 if D1[f, e] < 0 else 1] = f
    return out


def dual_d0(F):
    """Dual exterior derivative on dual 0-forms (values on faces) with the orientation of Section 36.5: d1^T."""
    return dd.d1(F).T


def dual_d1(F, n0=None):
    """Dual exterior derivative on dual 1-forms (values on dual edges): -d0^T."""
    return -dd.d0(F, n0).T


def dual_edge_polylines(V, F):
    """For every edge (edge_list order) the circumcentric dual edge as three points (start, midpoint, end), oriented
    from the right face to the left face; a boundary edge starts or ends at the edge midpoint."""
    V = np.asarray(V, float)
    E, _ = dd.edge_list(F)
    C = dd.circumcenters(V, F)
    lr = left_right_faces(F)
    out = []
    for e, (i, j) in enumerate(E):
        m = 0.5 * (V[i] + V[j])
        start = C[lr[e, 0]] if lr[e, 0] >= 0 else m
        end = C[lr[e, 1]] if lr[e, 1] >= 0 else m
        out.append(np.array([start, m, end]))
    return out


def signed_polygon_area(P):
    """Signed area (counterclockwise positive) of a planar polygon P (m, 2)."""
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def cross2(a, b):
    """z-component of the cross product of planar vectors."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# ---------------------------------------------------------------------------
# verify helper
# ---------------------------------------------------------------------------


def run_printed(page, block, out_block, namespace=None):
    """Run <pre id=block> of the page with dgcode.run_block; return (namespace, printed output equals the text of
    <pre id=out_block> line by line (trailing spaces ignored), the printed text)."""
    import contextlib
    import io

    import dgcode
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(page, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(page, out_block)), buf.getvalue()


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def draw_mesh2d(ax, V, F, face="#F3F3F3", edge="#A0A0A0", lw=0.8, zorder=0):
    """Light faces and gray edges of a planar mesh."""
    from matplotlib.patches import Polygon
    for tri in np.asarray(F):
        ax.add_patch(Polygon(np.asarray(V)[tri], closed=True, facecolor=face, edgecolor=edge, lw=lw, zorder=zorder))


def edge_arrow(ax, p, q, color, lw=1.6, frac=(0.18, 0.82), head=9, zorder=4):
    """An arrow along the segment p -> q, drawn on the middle part [frac0, frac1] of the segment."""
    from matplotlib.patches import FancyArrowPatch
    p, q = np.asarray(p, float), np.asarray(q, float)
    a, b = p + frac[0] * (q - p), p + frac[1] * (q - p)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def normal_offset(p, q, d):
    """The point at distance d to the left of the midpoint of p -> q (planar)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    t = (q - p) / np.linalg.norm(q - p)
    return 0.5 * (p + q) + d * np.array([-t[1], t[0]])
