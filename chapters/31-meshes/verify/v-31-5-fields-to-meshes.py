"""Verification for Section 31.5 (From Fields to Meshes and Back).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/verify/v-31-5-fields-to-meshes.py``
"""

import pathlib
import runpy
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

from dgcheck import check, close, summary, sym_equal  # noqa: E402

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"

# ---------------------------------------------------------------- linear interpolation on an edge (30.2.9)
f0, f1, x0, x1, t = sp.symbols("f0 f1 x0 x1 t", real=True)
root = x0 + f0 / (f0 - f1) * (x1 - x0)
sym_equal("Text: (30.2.9) is the zero of the linear interpolant of (x0, f0), (x1, f1)",
          (f0 + (f1 - f0) * (root - x0) / (x1 - x0)), 0)
# the zero set of a linear function on a tetrahedron is a triangle (1 vs 3) or a quadrilateral (2 vs 2)
check("Text: Kuhn split: 6 tetrahedra fill the cube, consistent on shared faces (sphere meshes are closed)",
      len(dm._KUHN) == 6)

# ---------------------------------------------------------------- Proposition 31.5.1: manifold output
for h in (0.2, 0.1, 0.05):
    xs = np.arange(-1.3, 1.3 + 1e-9, h) + 0.0137
    G = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1)
    V, F = dm.marching_tetrahedra(np.linalg.norm(G, axis=-1) - 1, xs, xs, xs)
    check(f"Prop 31.5.1: unit sphere SDF, h = {h}: closed, manifold, consistently oriented, chi = 2",
          dm.is_manifold(F) and not dm.boundary_edges(F) and dm.oriented_consistently(F) and dm.euler_characteristic(F) == 2)
# an open surface: the grid boundary cuts it (a plane z = 0.013)
xs = np.linspace(-1, 1, 9)
G = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1)
V, F = dm.marching_tetrahedra(G[..., 2] - 0.013, xs, xs, xs)
check("Prop 31.5.1: a plane cut by the grid box gives a disk (chi = 1) with one boundary loop",
      dm.euler_characteristic(F) == 1 and dm.is_manifold(F) and len(dm.boundary_edges(F)) > 0)

# ---------------------------------------------------------------- Figure 31.5.1: thin structures
for h, comps_, chi_ in ((0.04, 1, 2), (0.06, 1, 2), (0.08, 2, 4), (0.10, 2, 4), (0.14, 2, 4)):
    V, F = dm.dumbbell_mesh(h)
    c = dm.components(F)
    check(f"Fig 31.5.1: dumbbell, h = {h}: {comps_} component(s), chi = {chi_}",
          len(c) == comps_ and dm.euler_characteristic(F) == chi_ and dm.is_manifold(F))
V, F = dm.dumbbell_mesh(0.08)
ext = sorted((V[np.unique(F[c])][:, 0].min(), V[np.unique(F[c])][:, 0].max()) for c in dm.components(F))
check(f"Fig 31.5.1: h = 0.08: the rod breaks between x = {ext[0][1]:.3f} and {ext[1][0]:.3f}",
      round(ext[0][1], 2) == -0.11 and round(ext[1][0], 2) == 0.11)
# why: the nearest grid line to the rod axis is at distance >= h/2 in y
check("Text: with y-lines at +-h/2 the rod (radius 0.05) is missed entirely once h/2 >= 0.05 + 0 (h = 0.14: 0.07)",
      0.14 / 2 > 0.05)

# ---------------------------------------------------------------- Figure 31.5.2: accuracy
rows = []
for h in (0.2, 0.14, 0.1, 0.07, 0.05):
    xs = np.arange(-1.3, 1.3 + 1e-9, h) + 0.0137
    G = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1)
    V, F = dm.marching_tetrahedra(np.linalg.norm(G, axis=-1) - 1, xs, xs, xs)
    r = np.linalg.norm(V, axis=1)
    fn = dm.face_normals(V, F)
    cen = V[F].mean(1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    m = dm.mass_voronoi(V, F)
    nc = ((dm.cotan_apply(V, F, V) / m[:, None]) * (V / r[:, None])).sum(1)
    rows.append((h, np.abs(r - 1).max(), np.arccos(np.clip((fn * cen).sum(1), -1, 1)).mean(), np.median(np.abs(nc + 2)),
                 abs((nc * m).sum() / m.sum() + 2), (np.degrees(dm.corner_angles(V, F)).min(1) < 5).mean(),
                 dm.face_areas(V, F).sum()))
R = np.array(rows)
slope = lambda col: np.polyfit(np.log(R[:, 0]), np.log(R[:, col]), 1)[0]  # noqa: E731
check(f"Fig 31.5.2: vertex position error ~ h^{slope(1):.2f} (0.0115 at h = 0.2, 0.00092 at 0.05)",
      1.8 < slope(1) < 2.2 and round(R[0, 1], 4) == 0.0115 and round(R[-1, 1], 5) == 0.00092)
check(f"Fig 31.5.2: face normal error ~ h^{slope(2):.2f} (mean {np.degrees(R[0, 2]):.1f} deg at 0.2, "
      f"{np.degrees(R[-1, 2]):.1f} deg at 0.05)", 0.85 < slope(2) < 1.15 and round(np.degrees(R[0, 2]), 1) == 4.1
      and round(np.degrees(R[-1, 2]), 1) == 1.0)
check(f"Fig 31.5.2: pointwise 2H error (median) stays near 1: {np.round(R[:, 3], 2)}",
      (R[:, 3] > 0.9).all() and (R[:, 3] < 1.2).all())
check(f"Fig 31.5.2: integrated 2H error ~ h^{slope(4):.2f} ({R[0, 4]:.4f} -> {R[-1, 4]:.5f})", 1.8 < slope(4) < 2.2)
check(f"Fig 31.5.2: fraction of faces with an angle < 5 deg stays ~5.7%: {np.round(R[:, 5], 3)}",
      (np.abs(R[:, 5] - 0.057) < 0.006).all())
check(f"Text: the area converges: {R[0, 6]:.3f} (h = 0.2) and {R[-1, 6]:.3f} (h = 0.05) vs 4 pi = 12.566",
      abs(R[-1, 6] - 4 * np.pi) < abs(R[0, 6] - 4 * np.pi) < 0.2)

# ---------------------------------------------------------------- Figure 31.5.3: Poisson reconstruction
x = sp.symbols("x", real=True)
a_, b_ = sp.Rational(-1, 2), sp.Rational(3, 5)
chi = sp.Heaviside(x - a_) - sp.Heaviside(x - b_)
sym_equal("Text: (1D) chi' = delta(x - a) - delta(x - b): +inward normal at a, -(+x) at b",
          sp.diff(chi, x), sp.DiracDelta(x - a_) - sp.DiracDelta(x - b_))
ns = runpy.run_path(str(FIG / "fig-31-5-3-poisson.py"))
check(f"Fig 31.5.3: IoU with correct normals and all samples = {ns['iou'](ns['chi_f'], ns['iso_f']):.3f}",
      round(ns["iou"](ns["chi_f"], ns["iso_f"]), 3) == 0.999)
check(f"Fig 31.5.3(b): with a gap IoU = {ns['iou'](ns['chi_g'], ns['iso_g']):.3f}",
      round(ns["iou"](ns["chi_g"], ns["iso_g"]), 3) == 0.995)
check(f"Fig 31.5.3(c): with flipped normals IoU = {ns['iou'](ns['chi_x'], ns['iso_x']):.3f}",
      round(ns["iou"](ns["chi_x"], ns["iso_x"]), 3) == 0.853)
chi_f = ns["chi_f"]
check(f"Fig 31.5.3: chi jumps by {chi_f.max() - chi_f.min():.2f} ~ 1 across the curve", abs(chi_f.max() - chi_f.min() - 1) < 0.05)
# least squares: the Poisson solution minimizes sum |grad chi - V|^2 (perturbations increase it), on a small grid
n = 32
hh = 1.0 / n
rng = np.random.default_rng(0)
Vx, Vy = rng.normal(size=(n, n)), rng.normal(size=(n, n))
k = 2 * np.pi * np.fft.fftfreq(n, d=hh)
KX, KY = np.meshgrid(k, k, indexing="ij")
K2 = KX ** 2 + KY ** 2
K2[0, 0] = 1
sol_hat = -(1j * KX * np.fft.fft2(Vx) + 1j * KY * np.fft.fft2(Vy)) / K2
sol_hat[0, 0] = 0


def energy(c_hat):
    gx = np.real(np.fft.ifft2(1j * KX * c_hat)) - Vx
    gy = np.real(np.fft.ifft2(1j * KY * c_hat)) - Vy
    return (gx ** 2 + gy ** 2).sum()


E0 = energy(sol_hat)
ok = all(energy(sol_hat + 0.05 * np.fft.fft2(rng.normal(size=(n, n)))) > E0 for _ in range(10))
check("Text: the spectral Poisson solution minimizes sum |grad chi - V|^2 (10 random perturbations)", ok)

# ---------------------------------------------------------------- TSDF thin wall (insight box, 1D model)
def fuse(w, tau, xs_):
    """Two cameras on the two sides of the wall [0, w]; a voxel is updated when sdf > -tau, with value
    min(1, sdf / tau) and a running average of weight 1 (Open3D UniformTSDFVolume::Integrate, used by the 2DGS
    bounded extraction through ScalableTSDFVolume)."""
    num = np.zeros_like(xs_)
    den = np.zeros_like(xs_)
    for sdf in (0.0 - xs_, xs_ - w):           # left camera sees the surface at 0, right camera at w
        m_ = sdf > -tau
        num[m_] += np.clip(sdf[m_] / tau, -1, 1)
        den[m_] += 1
    D = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    s = np.sign(D)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return xs_[idx] + 0.5 * (xs_[1] - xs_[0])


xs_ = np.linspace(-0.1, 0.1, 200001) + 3.3e-7    # no grid point exactly on a surface
tau = 0.02
for w, expect in ((0.01, (-0.01, 0.02)), (0.005, (-0.015, 0.02)), (0.03, (0.0, 0.03))):
    zc = fuse(w, tau, xs_)
    check(f"Insight box: wall of thickness {w}, truncation {tau}: fused surfaces at {np.round(zc, 4)} "
          f"(thickness {zc[-1] - zc[0]:.3f})", len(zc) == 2 and np.allclose(zc, expect, atol=2e-6))
w_, t_ = sp.symbols("w tau", positive=True)
sym_equal("Insight box: for w < tau the fused thickness is 2 tau - w", t_ - (w_ - t_), 2 * t_ - w_)
check("Text: 2DGS bounded defaults: voxel = depth_trunc / 1024, sdf_trunc = 5 voxels; paper: 0.004 and 0.02",
      np.isclose(5 * 0.004, 0.02))

# ---------------------------------------------------------------- Figure 31.5.4: Gaussian bound to a face
ns4 = runpy.run_path(str(FIG / "fig-31-5-4-gaussian-on-face.py"))
asp = ns4["asp"]
check(f"Fig 31.5.4: axis ratio rest {asp[0]:.2f}, affine {asp[1]:.2f}, similarity {asp[2]:.2f}",
      [round(v, 2) for v in asp] == [2.92, 7.14, 2.92])
close("Fig 31.5.4: centre offset between the two carried Gaussians = 0.052",
      round(np.linalg.norm(ns4["mu_aff"] - ns4["mu_sim"]), 3), 0.052, tol=1e-12)
A = ns4["A"]
SIG0 = ns4["SIG0"]
# the affine image of the 1-sigma ellipse is the 1-sigma ellipse of A Sigma A^T (Proposition 29.1.3)
tt = np.linspace(0, 2 * np.pi, 50)
w0, U0 = np.linalg.eigh(SIG0)
pts = U0 @ (np.sqrt(w0)[:, None] * np.stack([np.cos(tt), np.sin(tt), 0 * tt]))
Sa = A @ SIG0 @ A.T
mahal = np.einsum("ij,ij->j", A @ pts, np.linalg.solve(Sa, A @ pts))
close("Prop 31.5.4(a), Fig 31.5.4: A maps the rest 1-sigma ellipse onto the 1-sigma ellipse of A Sigma A^T", mahal, np.ones(50), tol=1e-10)
for ang_, sc_, sh_ in ((0.4, 1.3, (0.5, -0.2)), (-1.1, 0.7, (2.0, 1.0))):
    Q_ = np.array([[np.cos(ang_), -np.sin(ang_), 0], [np.sin(ang_), np.cos(ang_), 0], [0, 0, 1]]) * sc_
    Xs_ = ns4["X0"] @ Q_.T + np.array([*sh_, 0])
    Rq, kq, Tq = ns4["face_frame"](Xs_)
    Aq = ns4["affine_of"](ns4["X0"], Xs_)
    ok_ = np.allclose((kq ** 2 * Rq @ ns4["sig_loc"] @ Rq.T)[:2, :2], (Aq @ SIG0 @ Aq.T)[:2, :2]) and np.allclose(
        kq * Rq @ ns4["mu_loc"] + Tq, ns4["BARY"] @ Xs_)
    check(f"Prop 31.5.4(b): under a similarity (angle {ang_}, scale {sc_}) the two bindings agree (in-plane)", ok_)
# Prop 31.5.4(b): a mean off the face plane (normal offset c) is moved by c in the affine binding, by s c in the similarity one
ang_, sc_ = 0.4, 1.3
Q_ = np.array([[np.cos(ang_), -np.sin(ang_), 0], [np.sin(ang_), np.cos(ang_), 0], [0, 0, 1]]) * sc_
Xs_ = ns4["X0"] @ Q_.T + np.array([0.5, -0.2, 0])
Rr, kr, Tr = ns4["face_frame"](ns4["X0"])
Rq, kq, Tq = ns4["face_frame"](Xs_)
Aq = ns4["affine_of"](ns4["X0"], Xs_)
c_off = 0.3
mu_off = Tr + np.array([0.1, 0.2, 0]) + c_off * Rr[:, 1]           # R[:, 1] is the face normal
mu_loc_off = Rr.T @ (mu_off - Tr) / kr
sim_ = kq * Rq @ mu_loc_off + Tq
aff_ = Xs_[0] + Aq @ (mu_off - ns4["X0"][0])
close("Prop 31.5.4(b): off the face plane the two means differ by (s - 1) c along the normal",
      np.linalg.norm(sim_ - aff_), (sc_ - 1) * c_off, tol=1e-12)
# similarity binding is not the closest similarity to the in-plane A (Frobenius norm)
M2 = A[:2, :2]
a2, b2 = (M2[0, 0] + M2[1, 1]) / 2, (M2[1, 0] - M2[0, 1]) / 2
R1b, k1b, _ = ns4["face_frame"](ns4["X1"])
ga = (k1b / kr) * (R1b @ Rr.T)[:2, :2]                             # GaussianAvatars: (k'/k) R' R^T
d_best = np.linalg.norm(M2 - np.array([[a2, -b2], [b2, a2]]))
d_ga = np.linalg.norm(M2 - ga)
check(f"Text: distance from the in-plane A to the closest similarity {d_best:.2f}, to GaussianAvatars' similarity {d_ga:.2f}",
      round(d_best, 2) == 0.58 and round(d_ga, 2) == 0.70)
R0, k0, _ = ns4["face_frame"](ns4["X0"])
check(f"Text: GaussianAvatars k = (|x1 - x0| + height)/2 = {k0:.2f} for the rest face (edge 2, height 1.6)",
      np.isclose(k0, 1.8))

# ---------------------------------------------------------------- Figure 31.5.5: silhouette gradient
ns5 = runpy.run_path(str(FIG / "fig-31-5-5-silhouette-gradient.py"))
ps = np.linspace(0.05, 2.95, 581) + 1e-4
c_aa = np.array([ns5["antialiased"](p)[1] for p in ps])
c_ps = np.array([ns5["point_sampled"](p)[1] for p in ps])
g_aa = np.gradient(c_aa, ps)
check("Fig 31.5.5: antialiased colour = exact coverage for a vertical edge",
      all(np.allclose(ns5["antialiased"](p), ns5["exact"](p)) for p in ps))
check("Fig 31.5.5: d c_1 / d p = 1 on (0.55, 1.45) and 0 outside (0.45, 1.55)",
      np.allclose(g_aa[(ps > 0.55) & (ps < 1.45)], 1, atol=1e-6) and np.allclose(g_aa[(ps < 0.45) | (ps > 1.55)], 0, atol=1e-6))
check("Fig 31.5.5: the point-sampled colour is piecewise constant (zero derivative except at the jump p = 1)",
      np.allclose(np.gradient(c_ps, ps)[np.abs(ps - 1) > 0.02], 0))
check("Fig 31.5.5: at p = 1.3 the antialiased row is (1, 0.8, 0, 0)", np.allclose(ns5["antialiased"](1.3), [1, 0.8, 0, 0]))

# ---------------------------------------------------------------- Proposition 31.5.2 on a mesh: G^T A G = -L
def gradient_operator(V, F):
    """Per-face gradient of the piecewise-linear interpolant: (3 n2) x n0 matrix."""
    n0 = len(V)
    G = np.zeros((3 * len(F), n0))
    N = dm.face_normals(V, F)
    A2 = 2 * dm.face_areas(V, F)
    for f, (i, j, k) in enumerate(F):
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            # grad of the hat function of vertex a on face f: N x (x_c - x_b) / (2 A)
            G[3 * f:3 * f + 3, a] = np.cross(N[f], V[c] - V[b]) / A2[f]
    return G


Vt, Ft = dm.torus_mesh(1.0, 0.4, 14, 9)
Gt = gradient_operator(Vt, Ft)
At = np.repeat(dm.face_areas(Vt, Ft), 3)
close("Prop 31.5.2: on a torus mesh (126 vertices) G^T A G = -L (cotan Laplacian)", Gt.T @ (At[:, None] * Gt),
      -dm.cotan_laplacian(Vt, Ft), tol=1e-10)
# a gradient field is recovered exactly: V = G u0  ->  -L u = G^T A V has the solution u0 (+ const)
u0 = np.sin(Vt[:, 0] * 2) + Vt[:, 2]
rhs = Gt.T @ (At * (Gt @ u0))
u = np.linalg.lstsq(-dm.cotan_laplacian(Vt, Ft), rhs, rcond=None)[0]
close("Prop 31.5.2: for an exact gradient field the least-squares solution is u0 up to a constant",
      u - u.mean(), u0 - u0.mean(), tol=1e-8)
# a rotated (curl) field contributes nothing: V = N x G u0 per face  ->  G^T A V = 0
Nt = np.repeat(dm.face_normals(Vt, Ft), 1, axis=0)
Vrot = np.cross(Nt, (Gt @ u0).reshape(-1, 3)).reshape(-1)
close("Text: a rotated gradient field (pure curl part) has zero divergence G^T A V = 0", Gt.T @ (At * Vrot),
      np.zeros(len(Vt)), tol=1e-10)

# ---------------------------------------------------------------- exercises
# Exercise 31.5.1: sign-change edges of a tetrahedron with k positive corners
check("Exercise 31.5.1: k positive corners of a tetrahedron -> k (4 - k) sign-change edges: 0, 3, 4, 3, 0",
      [k * (4 - k) for k in range(5)] == [0, 3, 4, 3, 0])
xs = np.array([0.0, 1.0])
G1 = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1)
for vals, nf in (((0.4, -1, -1, -1), 1), ((0.4, 0.3, -1, -1), 2)):
    f = -np.ones((2, 2, 2))
    # corners of the first Kuhn tetrahedron (0,0,0), (1,0,0), (1,1,0), (1,1,1)
    for (a, b, c), v in zip(((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)), vals):
        f[a, b, c] = v
    others = [(0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1)]
    for o in others:
        f[o] = -1.0
    Vq, Fq = dm.marching_tetrahedra(f, xs, xs, xs)
    check(f"Exercise 31.5.1: corner signs {vals} in a single cube give a disk-like patch "
          f"({len(Fq)} triangles, chi = {dm.euler_characteristic(Fq)})", dm.euler_characteristic(Fq) == 1 and dm.is_manifold(Fq))
# Exercise 31.5.2 (TSDF): limit w -> 0 gives thickness 2 tau
zc = fuse(1e-4, 0.02, xs_)
check(f"Exercise 31.5.2: a very thin wall (w = 1e-4) becomes {zc[-1] - zc[0]:.4f} ~ 2 tau = 0.04 thick",
      abs((zc[-1] - zc[0]) - 0.0399) < 2e-5)
# Exercise 31.5.3: all normals flipped -> chi -> -chi, the region is the complement; scaling V does not move it
chi_all, iso_all, _, _ = ns["poisson"](ns["TH"], np.ones(len(ns["TH"]), bool))
close("Exercise 31.5.3: flipping every normal gives -chi", chi_all, -chi_f, tol=1e-9)
inside_flip = chi_all > iso_all
inside = chi_f > ns["iso_f"]
check("Exercise 31.5.3: ... and the extracted region is exactly the complement (same curve, reversed normal)",
      np.array_equal(inside_flip, ~inside))
# Exercise 31.5.4: a shear parallel to the edge x0 x1 does not change GaussianAvatars' R and k
face_frame = ns4["face_frame"]
X0 = ns4["X0"]
Xsh = X0.copy()
Xsh[2] = X0[2] + np.array([0.8, 0, 0])
R1, k1, T1 = face_frame(Xsh)
R0, k0, T0 = face_frame(X0)
check("Exercise 31.5.4: shearing x_2 along the edge x_0 x_1 keeps R and k (only T moves by 0.8/3)",
      np.allclose(R1, R0) and np.isclose(k1, k0) and np.allclose(T1 - T0, [0.8 / 3, 0, 0]))
Ash = ns4["affine_of"](X0, Xsh)
w_ = np.linalg.eigvalsh((Ash @ SIG0 @ Ash.T)[:2, :2])
check(f"Exercise 31.5.4: ... while the affine Gaussian's axis ratio changes from 2.92 to {np.sqrt(w_[1] / w_[0]):.2f}",
      abs(np.sqrt(w_[1] / w_[0]) - 2.917) > 0.3)
# Exercise 31.5.5 and Proposition 31.5.5: d(coverage)/dp = length of the edge inside the pixel


def coverage(theta, p):
    """Area of the unit pixel [0,1]^2 inside the half-plane <x, n> < p, n = (cos theta, sin theta)."""
    n = np.array([np.cos(theta), np.sin(theta)])
    poly = [np.array(v, float) for v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    out = []
    for a_, b_ in zip(poly, poly[1:] + poly[:1]):
        da, db = a_ @ n - p, b_ @ n - p
        if da <= 0:
            out.append(a_)
        if da * db < 0:
            out.append(a_ + da / (da - db) * (b_ - a_))
    if len(out) < 3:
        return 0.0
    P_ = np.array(out)
    return 0.5 * abs(np.dot(P_[:, 0], np.roll(P_[:, 1], -1)) - np.dot(P_[:, 1], np.roll(P_[:, 0], -1)))


def chord(theta, p):
    n = np.array([np.cos(theta), np.sin(theta)])
    t = np.array([-n[1], n[0]])
    s_ = np.linspace(-3, 3, 600001)
    pts = p * n[None, :] + s_[:, None] * t[None, :]
    inside_ = (pts >= 0).all(1) & (pts <= 1).all(1)
    return inside_.sum() * (s_[1] - s_[0])


ok = True
for theta, p in ((0.0, 0.3), (np.pi / 4, np.sqrt(2) / 2), (0.3, 0.7), (1.1, 0.4), (2.0, 0.1)):
    d = (coverage(theta, p + 1e-6) - coverage(theta, p - 1e-6)) / 2e-6
    ok &= abs(d - chord(theta, p)) < 1e-4
check("Prop 31.5.5: d(coverage)/dp equals the length of the edge inside the pixel (5 edge directions)", ok)
d45 = (coverage(np.pi / 4, np.sqrt(2) / 2 + 1e-6) - coverage(np.pi / 4, np.sqrt(2) / 2 - 1e-6)) / 2e-6
close("Exercise 31.5.5: diagonal edge through the pixel centre: d(coverage)/dp = sqrt 2", d45, np.sqrt(2), tol=1e-5)

# ---------------------------------------------------------------- numbers quoted in the text
TH = ns["TH"]
check("Fig 31.5.3: the gap removes 20 samples, the flip reverses 23 normals",
      ((TH > 0.4) & (TH < 1.2)).sum() == 20 and ((TH > 2.4) & (TH < 3.3)).sum() == 23)
tr_in, dA = ns["true_in"], ns["dx"] ** 2
area_true = tr_in.sum() * dA
fi_g = ((ns["chi_g"] > ns["iso_g"]) & ~tr_in).sum() * dA
mi_g = (~(ns["chi_g"] > ns["iso_g"]) & tr_in).sum() * dA
fi_x = ((ns["chi_x"] > ns["iso_x"]) & ~tr_in).sum() * dA
mi_x = (~(ns["chi_x"] > ns["iso_x"]) & tr_in).sum() * dA
check(f"Text: true area {area_true:.2f}; gap: wrongly inside {fi_g:.4f}, missed {mi_g:.3f}; "
      f"flip: wrongly inside {fi_x:.2f}, missed {mi_x:.2f}",
      round(area_true, 2) == 3.19 and round(fi_g, 4) == 0.0025 and round(mi_g, 3) == 0.013
      and round(fi_x, 2) == 0.28 and round(mi_x, 2) == 0.23)
# where the flipped part puts inside and outside: just outside the curve -> inside, just inside -> outside
xs2 = ns["xs"]


def chi_at(chi, x, y):
    return chi[np.argmin(abs(xs2 - x)), np.argmin(abs(xs2 - y))]


check("Text: next to the flipped normals the outside point (-1.1, 0.3) is classified inside and the inside point "
      "(-0.6, 0.3) outside", chi_at(ns["chi_x"], -1.1, 0.3) > ns["iso_x"] > chi_at(ns["chi_x"], -0.6, 0.3))
# thin rod: covering radius of the (y, z) lattice is h / sqrt 2
for h, caught in ((0.04, True), (0.08, False), (0.14, False)):
    check(f"Text: h = {h}: h / sqrt 2 = {h / np.sqrt(2):.3f} {'<' if caught else '>'} rho = 0.05",
          (h / np.sqrt(2) < 0.05) == caught)
check("Text: h = 0.08, the rod axis at x ~ 0 passes through a cell centre: nearest sample at distance 0.057 > 0.05",
      np.isclose(np.hypot(0.04, 0.04), 0.0566, atol=1e-4))
# face affine map: singular values, area ratio, k ratio; centroid is carried identically by both bindings
sv = np.linalg.svd(A[:2, :2])[1]
check(f"Text: in-plane singular values of A = {np.round(sv, 3)}, area ratio {np.linalg.det(A[:2, :2]):.3f}, "
      f"k'/k = {ns4['face_frame'](ns4['X1'])[1] / 1.8:.3f}",
      np.allclose(np.round(sv, 3), [1.369, 0.555]) and round(np.linalg.det(A[:2, :2]), 3) == 0.759)
X1 = ns4["X1"]
R1_, k1_, T1_ = ns4["face_frame"](X1)
check("Prop 31.5.4(c): a Gaussian at the centroid (mu_loc = 0) is carried to the new centroid by both bindings",
      np.allclose(T1_, X1.mean(0)) and np.allclose(A @ (X0.mean(0) - X0[0]) + X1[0], X1.mean(0)))

summary()
