"""Helpers for Chapter 43 (Conformal Geometry). Not shared with other chapters.

Conventions (Section 43.1, Conventions box):
  * a map to the plane is a complex vector z in C^{n0} (one value per vertex), z = u + i v;
  * 1-form Hodge star (star alpha)(X) = -alpha(J X), so holomorphic  <=>  star dz = -i dz (Section 43.4);
  * E_D(z) = -1/2 z^H L z (L = dgmesh.cotan_laplacian, Delta sign), signed area
    A(z) = 1/2 sum_{(i, j) in boundary} Im(conj(z_i) z_j) with the boundary loop of dgmesh.boundary_loop
    (it follows the face orientation, i.e. the Stokes orientation), E_C = E_D - A = z^H EC z.
The Implementation boxes of Sections 43.2 and 43.5 contain their own (shorter) versions of these functions; the
verify scripts check that both agree.
"""

import contextlib
import io

import numpy as np

import dgmesh as dm


# ---------------------------------------------------------------------------------------------------------- pages
def run_printed(page, block, out_block, namespace=None):
    """Run <pre id=block> of the page with dgcode.run_block; return (namespace, printed output equals the text of
    <pre id=out_block> line by line (trailing spaces ignored), the printed text)."""
    import dgcode
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(page, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(page, out_block)), buf.getvalue()


# ---------------------------------------------------------------------------------------------------------- meshes
def hemisphere(level=3):
    """Southern hemisphere of dgmesh.icosphere(level): faces whose centroid has z < 0 (outward normals).
    Returns (V, F). Disk topology (chi = 1), one boundary loop near the equator."""
    V, F = dm.icosphere(level)
    keep = V[F].mean(1)[:, 2] < 0
    Vh, Fh, _ = dm.submesh(V, F, keep)
    return Vh, Fh


def torus_piece(nu=24, nv=48, R=2.0, r=0.8):
    """Piece |u| < pi/2 (outer half of the tube), 0 < v < pi/2 of dgmesh.torus_mesh(R, r, nu, nv) (face centroids).
    Returns (V, F, u, v) with the torus parameters (u, v) of the kept vertices."""
    V, F = dm.torus_mesh(R, r, nu, nv)
    c = V[F].mean(1)
    uc = np.arctan2(c[:, 2], np.hypot(c[:, 0], c[:, 1]) - R)
    vc = np.arctan2(c[:, 1], c[:, 0])
    keep = (np.abs(uc) < np.pi / 2) & (vc > 0) & (vc < np.pi / 2)
    Vp, Fp, _ = dm.submesh(V, F, keep)
    u = np.arctan2(Vp[:, 2], np.hypot(Vp[:, 0], Vp[:, 1]) - R)
    v = np.arctan2(Vp[:, 1], Vp[:, 0])
    return Vp, Fp, u, v


def torus_w(u, R=2.0, r=0.8):
    """Conformal coordinate of the torus of revolution: dw = r du / (R + r cos u), w(0) = 0 (|u| < pi)."""
    k = np.sqrt(R * R - r * r)
    return 2 * r / k * np.arctan(np.sqrt((R - r) / (R + r)) * np.tan(np.asarray(u) / 2))


def bump_height(x, y):
    """Height of the running height field of Section 43.5: z = exp(-3 ((x - 0.4)^2 + y^2)) on [-1, 1]^2."""
    return np.exp(-3 * ((x - 0.4) ** 2 + y ** 2))


def height_field(n=16):
    """grid_mesh on [-1, 1]^2 with n x n cells lifted to z = bump_height(x, y). Faces are counterclockwise seen from
    +z, so the unit normal points up (positive z component)."""
    xs = np.linspace(-1, 1, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    V = V.copy()
    V[:, 2] = bump_height(V[:, 0], V[:, 1])
    return V, F


def stereo_conj(V):
    """conj(sigma) for the stereographic projection from the north pole, sigma(x) = (x1 + i x2) / (1 - x3).
    On the southern hemisphere with outward normal, conj(sigma) is holomorphic (Example 43.3.6)."""
    return ((V[:, 0] + 1j * V[:, 1]) / (1 - V[:, 2])).conj()


# ---------------------------------------------------------------------------------------------------------- energy
def boundary_edges_oriented(F):
    """Oriented boundary edges (i, j) in the order of dgmesh.boundary_loop (face orientation)."""
    loop = dm.boundary_loop(F)
    return list(zip(loop, np.roll(loop, -1)))


def area_matrix(F, n):
    """Hermitian matrix sA with z^H sA z = signed area = 1/2 sum_{(i, j)} Im(conj(z_i) z_j) (boundary edges)."""
    A = np.zeros((n, n), complex)
    for i, j in boundary_edges_oriented(F):
        A[i, j] += -0.25j
        A[j, i] += 0.25j
    return A


def conformal_energy_matrix(V, F):
    """Hermitian EC with E_C(z) = z^H EC z = -1/2 z^H L z - signed area (Definition 43.5.2)."""
    return -0.5 * dm.cotan_laplacian(V, F).astype(complex) - area_matrix(F, len(V))


def dirichlet(V, F, z):
    return float(-0.5 * (np.conj(z) @ dm.cotan_laplacian(V, F) @ z).real)


def signed_area(F, z):
    return float(sum(0.5 * (np.conj(z[i]) * z[j]).imag for i, j in boundary_edges_oriented(F)))


def face_signed_areas(F, z):
    """signed area of each image triangle (positive = counterclockwise = orientation-preserving)."""
    a, b, c = z[F[:, 0]], z[F[:, 1]], z[F[:, 2]]
    return 0.5 * (np.conj(b - a) * (c - a)).imag


def face_cr_energy(V, F, z):
    """per face A_f * 1/2 ((u_x - v_y)^2 + (u_y + v_x)^2) in the face frame of dgmesh.face_frames."""
    X = dm.face_frames(V, F)
    out = np.zeros(len(F))
    for f, (tri, x) in enumerate(zip(F, X)):
        P = np.stack([x[1] - x[0], x[2] - x[0]], 1)
        dz = np.array([z[tri[1]] - z[tri[0]], z[tri[2]] - z[tri[0]]])
        D = np.stack([dz.real, dz.imag]) @ np.linalg.inv(P)     # rows (u_x, u_y), (v_x, v_y)
        ux, uy, vx, vy = D[0, 0], D[0, 1], D[1, 0], D[1, 1]
        out[f] = 0.5 * abs(np.linalg.det(P)) * 0.5 * ((ux - vy) ** 2 + (uy + vx) ** 2)
    return out


def dz_singular_values(V, F, z):
    """per face singular values t1 >= t2 of the differential of the affine map surface triangle -> image triangle
    (the reciprocals of dgmesh.uv_distortion, which measures the map image -> surface), and the image signed area."""
    s, det = dm.uv_distortion(V, F, np.c_[np.real(z), np.imag(z)])
    return 1 / s[:, 1], 1 / s[:, 0], det


def quasi_conformal(V, F, z):
    """sigma_1 / sigma_2 per face (the same for a map and its inverse)."""
    s, _ = dm.uv_distortion(V, F, np.c_[np.real(z), np.imag(z)])
    return s[:, 0] / s[:, 1]


# ---------------------------------------------------------------------------------------------------------- solvers
def spectral_conformal(V, F, b, iters=80, seed=0):
    """Inverse power iteration for min E_C subject to <z, 1>_B = 0, |z|_B = 1, B = diag(b) (Section 43.5).
    Returns (z, lambda, residual |EC z - lambda B z|)."""
    EC = conformal_energy_matrix(V, F)
    K = EC + np.outer(b, b)
    y = np.random.default_rng(seed).standard_normal((len(V), 2)) @ np.array([1, 1j])
    for _ in range(iters):
        y = np.linalg.solve(K, b * y)
        y -= (b @ y) / b.sum()
        y /= np.sqrt((b * np.abs(y) ** 2).sum())
    lam = float((np.conj(y) @ EC @ y).real)
    return y, lam, float(np.linalg.norm(EC @ y - lam * b * y))


def pencil_eigs(EC, b, k=6):
    """The k smallest eigenvalues lambda of EC z = lambda B z on {<z, 1>_B = 0} with eigenvectors (B-normalized),
    computed densely through K = EC + b b^T: B z = mu K z, lambda = 1 / mu (Theorem 43.5.8)."""
    K = EC + np.outer(b, b)
    Rc = np.linalg.cholesky(K)
    Ri = np.linalg.inv(Rc)
    mu, W = np.linalg.eigh(Ri @ np.diag(b).astype(complex) @ Ri.conj().T)
    Z = Ri.conj().T @ W
    keep = []
    for j in np.argsort(-mu):
        nz = np.sqrt((b * np.abs(Z[:, j]) ** 2).sum())
        if mu[j] > 1e-14 and abs(b @ Z[:, j]) < 1e-8 * nz * np.sqrt(b.sum()):
            keep.append(j)
        if len(keep) == k:
            break
    Z = Z[:, keep]
    Z = Z / np.sqrt((b[:, None] * np.abs(Z) ** 2).sum(0))
    return 1 / mu[keep], Z


def best_fit(z, w, b):
    """complex a minimizing |z - a w|_B (w is first made B-orthogonal to 1); returns (a, relative error)."""
    w = w - (b @ w) / b.sum()
    a = (np.conj(w) @ (b * z)) / (np.conj(w) @ (b * w))
    err = np.sqrt((b * np.abs(z - a * w) ** 2).sum() / (b * np.abs(z) ** 2).sum())
    return a, float(err)


def farthest_boundary_pair(V, F):
    """the two boundary vertices that are farthest apart in R^3 (the usual LSCM pins, Section 31.6)."""
    loop = dm.boundary_loop(F)
    P = V[loop]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    a, c = np.unravel_index(D.argmax(), D.shape)
    return int(loop[a]), int(loop[c])


def lscm_complex(V, F, pins, values):
    """minimize E_C with z fixed at the pins: solve the free block of EC (Proposition 43.5.6)."""
    EC = conformal_energy_matrix(V, F)
    n = len(V)
    free = np.setdiff1d(np.arange(n), pins)
    z = np.zeros(n, complex)
    z[list(pins)] = values
    z[free] = np.linalg.solve(EC[np.ix_(free, free)], -EC[np.ix_(free, list(pins))] @ z[list(pins)])
    return z


def normalize_similarity(z, i, j):
    """apply the similarity that sends z_i -> 0 and z_j -> 1 (for comparing parametrizations)."""
    return (z - z[i]) / (z[j] - z[i])
