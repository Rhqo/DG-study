"""Verification for Section 43.5 (Discrete Conformal Parametrization).

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/verify/v-43-5-discrete-conformal.py``
"""

import pathlib
import sys

import dgcode
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch43  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "43-5-discrete-conformal.html"
rng = np.random.default_rng(5)
crand = lambda n: rng.standard_normal(n) + 1j * rng.standard_normal(n)  # noqa: E731

Vh, Fh = ch43.hemisphere(3)
n = len(Vh)
L = dm.cotan_laplacian(Vh, Fh)
EC = ch43.conformal_energy_matrix(Vh, Fh)
m = dm.mass_barycentric(Vh, Fh)
loop = dm.boundary_loop(Fh)
bd = np.isin(np.arange(n), loop) * 1.0
check("setting: hemisphere(3) is a disk (chi = 1, one boundary loop, consistently oriented), 337 vertices, 624 faces",
      dm.euler_characteristic(Fh) == 1 and dm.oriented_consistently(Fh) and n == 337 and len(Fh) == 624
      and len(loop) == len(dm.boundary_vertices(Fh)))

# ---------------------------------------------------------------- Proposition 43.5.1
z = crand(n)
ED = -0.5 * (np.conj(z) @ L @ z).real
close("Proposition 43.5.1(a): -z^H L z / 2 = E_D(u) + E_D(v) (exact Dirichlet energy of W0 z)",
      ED, 0.5 * (dm.dirichlet_energy(Vh, Fh, z.real) + dm.dirichlet_energy(Vh, Fh, z.imag)), 1e-9)
check("Proposition 43.5.1(a): z^H L z is real (L real symmetric)", abs((np.conj(z) @ L @ z).imag) < 1e-9)
fa = ch43.face_signed_areas(Fh, z)
close("Proposition 43.5.1(b,c): sum of face signed areas = (1/2) sum over oriented boundary edges of Im(conj z_i z_j)",
      fa.sum(), ch43.signed_area(Fh, z), 1e-9)
P2 = np.array([[0, 0], [1, 0], [0.3, 0.8]])
zt = P2[:, 0] + 1j * P2[:, 1]
close("Proposition 43.5.1(b): Im(conj(z_j - z_i)(z_k - z_i))/2 = shoelace area", 0.5 * (np.conj(zt[1] - zt[0]) * (zt[2] - zt[0])).imag,
      0.5 * np.linalg.det(np.c_[P2[1] - P2[0], P2[2] - P2[0]]), 1e-15)

# ---------------------------------------------------------------- Definition 43.5.2, Proposition 43.5.3
Am = ch43.area_matrix(Fh, n)
check("Definition 43.5.2: A and EC are Hermitian, z^H A z = signed area, z^H EC z = E_D - A",
      np.allclose(Am, Am.conj().T) and np.allclose(EC, EC.conj().T) and abs(np.conj(z) @ Am @ z - ch43.signed_area(Fh, z)) < 1e-9
      and abs(np.conj(z) @ EC @ z - (ED - ch43.signed_area(Fh, z))) < 1e-9)
close("Proposition 43.5.3(a): E_C(z) = sum_f A_f ((u_x - v_y)^2 + (u_y + v_x)^2)/2", (np.conj(z) @ EC @ z).real,
      ch43.face_cr_energy(Vh, Fh, z).sum(), 1e-9)
t1, t2, det = ch43.dz_singular_values(Vh, Fh, z)
A_f = dm.face_areas(Vh, Fh)
perface = np.where(det >= 0, 0.5 * (t1 - t2) ** 2, 0.5 * (t1 + t2) ** 2) * A_f
close("Proposition 43.5.3(a): face terms A_f (s1 -+ s2)^2 / 2 by orientation", perface.sum(), ch43.face_cr_energy(Vh, Fh, z).sum(), 1e-8)
ev = np.linalg.eigvalsh(EC)
check("Proposition 43.5.3(b): EC is positive semidefinite", ev.min() > -1e-12)
a, c = 0.7 - 1.3j, 2 + 0.5j
check("Proposition 43.5.3(c): EC 1 = 0 and E_C(a z + c) = |a|^2 E_C(z)",
      np.abs(EC @ np.ones(n)).max() < 1e-12
      and abs((np.conj(a * z + c) @ EC @ (a * z + c)).real - abs(a) ** 2 * (np.conj(z) @ EC @ z).real) < 1e-8)
interior = np.setdiff1d(np.arange(n), loop)
defect = dm.angle_defect(Vh, Fh)[interior]
check("Proposition 43.5.3(d): the hemisphere has interior defects not in 2 pi Z and a one-dimensional kernel",
      np.all(np.abs(defect) > 1e-3) and ev[0] < 1e-12 and ev[1] > 1e-6)

# ---------------------------------------------------------------- Example 43.5.4 (planar)
xs = np.linspace(0, 1, 7)
Vg, Fg = dm.grid_mesh(xs, xs)
ECg = ch43.conformal_energy_matrix(Vg, Fg)
pg = Vg[:, 0] + 1j * Vg[:, 1]
close("Example 43.5.4: planar 6x6 grid: E_C(p) ~ 0 and E_C(conj p) = 2 (twice the area)",
      ((np.conj(pg) @ ECg @ pg).real, (pg @ ECg @ np.conj(pg)).real), (0.0, 2.0), 1e-12)
evg = np.linalg.eigvalsh(ECg)
check("Example 43.5.4, Exercise 43.5.3: the planar kernel is 2-dimensional (span{1, p})",
      evg[1] < 1e-12 and evg[2] > 1e-6 and np.abs(ECg @ pg).max() < 1e-12)

# ---------------------------------------------------------------- Looking back: dgmesh.lscm = pinned minimisation of EC
p, q = ch43.farthest_boundary_pair(Vh, Fh)
UV = dm.lscm(Vh, Fh, [p, q], [(0, 0), (1, 0)])
zl = ch43.lscm_complex(Vh, Fh, [p, q], [0, 1])
check("Looking back (31.6): dgmesh.lscm equals the pinned minimiser of the discrete conformal energy (1e-12)",
      np.abs(UV[:, 0] + 1j * UV[:, 1] - zl).max() < 1e-12)

# ---------------------------------------------------------------- Non-example 43.5.5, Proposition 43.5.6
one = np.ones(n) / np.sqrt(m.sum())
check("Non-example 43.5.5: the unit constant has E_C = 0", abs(np.conj(one) @ EC @ one) < 1e-14)
R_ = np.setdiff1d(np.arange(n), [p, q])
check("Proposition 43.5.6: E_RR is positive definite", np.linalg.eigvalsh(EC[np.ix_(R_, R_)]).min() > 1e-8)
a2, c2 = 1.7 * np.exp(0.6j), -0.4 + 2j
zl2 = ch43.lscm_complex(Vh, Fh, [p, q], [a2 * 0 + c2, a2 * 1 + c2])
check("Proposition 43.5.6: pinning a z_p + c, a z_q + c moves the whole solution by the same similarity",
      np.abs(zl2 - (a2 * zl + c2)).max() < 1e-10)

# ---------------------------------------------------------------- Theorem 43.5.8 (B = M)
lam_d, Zd = ch43.pencil_eigs(EC, m, 3)
zM, lamM, resM = ch43.spectral_conformal(Vh, Fh, m)
d = 1 / np.sqrt(m)
lam_all = np.linalg.eigvalsh(d[:, None] * EC * d[None, :])
check("Theorem 43.5.8: eigenvalues 0 = lambda_0 < lambda_1 (M^{-1/2} EC M^{-1/2}), matching the dense pencil",
      abs(lam_all[0]) < 1e-12 and abs(lam_all[1] - lam_d[0]) < 1e-12 and lam_all[1] > 0)
close("Theorem 43.5.8: the iteration reaches lambda_1 and satisfies the constraints", (lamM, m @ zM, m @ np.abs(zM) ** 2), (lam_d[0], 0, 1), 1e-10)
ph = np.vdot(Zd[:, 0], m * zM)
check("Theorem 43.5.8: the minimiser is the first nontrivial eigenvector up to a phase",
      abs(abs(ph) - 1) < 1e-8 and np.abs(zM - ph * Zd[:, 0]).max() < 1e-7)
# a random admissible z does not beat lambda_1
okR = True
for _ in range(10):
    y = crand(n)
    y -= (m @ y) / m.sum()
    y /= np.sqrt(m @ np.abs(y) ** 2)
    okR &= (np.conj(y) @ EC @ y).real >= lamM - 1e-12
check("Theorem 43.5.8: random admissible maps have E_C >= lambda_1", okR)

# ---------------------------------------------------------------- Proposition 43.5.9 (B = B_bd)
zB, lamB, resB = ch43.spectral_conformal(Vh, Fh, bd)
check("Proposition 43.5.9(a): the boundary-mass solution is discrete harmonic in the interior",
      np.abs((L @ zB)[interior]).max() < 1e-10)
I_, B_ = interior, loop
S = EC[np.ix_(B_, B_)] - EC[np.ix_(B_, I_)] @ np.linalg.solve(EC[np.ix_(I_, I_)], EC[np.ix_(I_, B_)])
evS, VS = np.linalg.eigh(S)
check("Proposition 43.5.9(b,c): S has kernel 1_bd, the boundary values are its first nontrivial eigenvector, EC z = lambda B z",
      abs(evS[0]) < 1e-12 and abs(evS[1] - lamB) < 1e-10 and np.linalg.norm(EC @ zB - lamB * bd * zB) < 1e-10
      and np.allclose(EC[np.ix_(I_, I_)], -0.5 * L[np.ix_(I_, I_)]))

# ---------------------------------------------------------------- Remark 43.5.10
for bb in (m, bd):
    K = EC + np.outer(bb, bb)
    y = crand(n)
    y -= (bb @ y) / bb.sum()
    x = np.linalg.solve(K, bb * y)
    check("Remark 43.5.10: K = EC + b b^T is positive definite and K x = B y gives EC x = B y, <x, 1>_B = 0",
          np.linalg.eigvalsh(K).min() > 1e-12 and np.abs(EC @ x - bb * y).max() < 1e-9 and abs(bb @ x) < 1e-9)
close("Remark 43.5.10: lambda_1 / lambda_2 = 0.042 for B = M", lam_d[0] / lam_d[1], 0.042, 5e-4)

# ---------------------------------------------------------------- Implementation 43.5
ns, same, printed = ch43.run_printed(PAGE, "code-43-5-1", "out-43-5-1")
check("Implementation 43.5: code-43-5-1 prints exactly out-43-5-1", same, printed)
check("Implementation 43.5: at most 30 lines", dgcode.code_lines(PAGE, "code-43-5-1") <= 30)
zI, lamI, _ = ns["spectral_conformal"](Vh, Fh, m)
check("Implementation 43.5: the printed functions agree with lib/ch43 (matrix and eigenvalue)",
      np.abs(ns["conformal_energy_matrix"](Vh, Fh) - EC).max() < 1e-14 and abs(lamI - lamM) < 1e-12)

# ---------------------------------------------------------------- Conventions box
Erev = -0.5 * L.astype(complex) + Am                   # reversed boundary orientation: + signed area
Krev = Erev + np.outer(m, m)
y = crand(n)
for _ in range(80):
    y = np.linalg.solve(Krev, m * y)
    y -= (m @ y) / m.sum()
    y /= np.sqrt(m @ np.abs(y) ** 2)
check("Conventions: reversing the boundary orientation gives the mirror image (every face flipped)",
      np.all(ch43.face_signed_areas(Fh, y) < 0))
uu, vv = z.real, z.imag
mul = sum(0.5 * (uu[i] * vv[j] - uu[j] * vv[i]) for i, j in ch43.boundary_edges_oriented(Fh))
crn = sum(-0.25j * (np.conj(z[i]) * z[j] - np.conj(z[j]) * z[i]) for i, j in ch43.boundary_edges_oriented(Fh))
close("Conventions: Mullen's sum (u_i v_j - u_j v_i)/2 and Crane's -(i/4) sum (conj z_i z_j - conj z_j z_i) equal (43.5.1)",
      (mul, crn.real, crn.imag), (ch43.signed_area(Fh, z), ch43.signed_area(Fh, z), 0.0), 1e-9)

# ---------------------------------------------------------------- Example 43.5.11, Figure 43.5.1
sig = ch43.stereo_conj(Vh)
aM, errM = ch43.best_fit(zM, sig, m)
aB, errB = ch43.best_fit(zB, sig, bd)
qM = ch43.quasi_conformal(Vh, Fh, zM)
cen = (m @ (zM - aM * sig)) / m.sum()
dev = np.abs(zM[loop] - (aM * sig + cen)[loop]).max() / abs(aM)
close("Example 43.5.11: lambda_1 = 2.194e-3, max / median sigma1/sigma2 = 1.099 / 1.051, fit 0.0078, |a| = 0.663",
      (lamM * 1e3, qM.max(), np.median(qM), errM, abs(aM)), (2.194, 1.099, 1.051, 0.0078, 0.663), 6e-4)
check("Example 43.5.11: boundary deviation from the best fit <= 1.8 % of |a|, no flipped face; B_bd fit 0.0118",
      dev < 0.0185 and (ch43.face_signed_areas(Fh, zM) < 0).sum() == 0 and abs(errB - 0.0118) < 6e-5)

# ---------------------------------------------------------------- Figure 43.5.2 (modes) numbers
lv = {}
for level in (2, 3, 4):
    V_, F_ = ch43.hemisphere(level)
    m_ = dm.mass_barycentric(V_, F_)
    lam_, Z_ = ch43.pencil_eigs(ch43.conformal_energy_matrix(V_, F_), m_, 4)
    E_ = dm.edges(F_)
    lv[level] = (lam_, np.mean(np.linalg.norm(V_[E_[:, 0]] - V_[E_[:, 1]], axis=1)),
                 [ch43.best_fit(Z_[:, k], ch43.stereo_conj(V_) ** (k + 1), m_)[1] for k in range(3)])
close("Figure 43.5.2: lambda_1 = 7.56e-3, 2.19e-3, 5.88e-4 and mean edge 0.299, 0.150, 0.075",
      [lv[k][0][0] * 1e3 for k in (2, 3, 4)] + [lv[k][1] for k in (2, 3, 4)], (7.56, 2.19, 0.588, 0.299, 0.150, 0.075), 6e-3)
close("Figure 43.5.2: level 3 lambda_1..4 = 0.0022, 0.0521, 0.2528, 0.7384", lv[3][0], (0.0022, 0.0521, 0.2528, 0.7384), 6e-5)
close("Figure 43.5.2: level-4 distances to conj(sigma)^k = 0.0035, 0.0108, 0.0288", lv[4][2], (0.0035, 0.0108, 0.0288), 6e-5)
r23, r34 = lv[2][0][0] / lv[3][0][0], lv[3][0][0] / lv[4][0][0]
check("Figure 43.5.2: lambda_1 drops by 3.5 to 3.7 per level", 3.4 < r23 < 3.75 and 3.4 < r34 < 3.75)
slopes34 = np.log(lv[3][0][:4] / lv[4][0][:4]) / np.log(lv[3][1] / lv[4][1])
check("Figure 43.5.2: log-log slopes of lambda_1..4 from level 3 to 4 lie in [1.76, 1.91] (approaching 2)",
      bool(np.all((slopes34 > 1.755) & (slopes34 < 1.915))), str(np.round(slopes34, 3)))

# ---------------------------------------------------------------- Example 43.5.12, Figure 43.5.3
for (nu, nv), expect in (((24, 48), (1.648, 14.30, 1.041, 1.074, 93.7)), ((48, 96), (None, 29.6, None, 1.037, 91.9))):
    V_, F_, u_, v_ = ch43.torus_piece(nu, nv)
    lp = dm.boundary_loop(F_)
    UVf = dm.fixed_boundary_map(V_, F_, "cotan")
    zf = UVf[:, 0] + 1j * UVf[:, 1]
    zs = ch43.spectral_conformal(V_, F_, dm.mass_barycentric(V_, F_))[0]
    qf, qs = ch43.quasi_conformal(V_, F_, zf), ch43.quasi_conformal(V_, F_, zs)
    k0 = int(np.argmin((u_ + np.pi / 2) ** 2 + v_ ** 2))
    pos = list(lp).index(k0)
    ang = abs(np.degrees(np.angle((zs[lp[pos - 1]] - zs[k0]) / (zs[lp[(pos + 1) % len(lp)]] - zs[k0]))))
    got = (np.median(qf), qf.max(), np.median(qs), qs.max(), ang)
    tol = (6e-4, 6e-3, 6e-4, 6e-4, 0.06) if nu == 24 else (None, 0.06, None, 6e-4, 0.06)
    close(f"Example 43.5.12 ({nu}, {nv}): fixed median/max, free median/max, corner angle",
          [g for g, e in zip(got, expect) if e is not None], [e for e in expect if e is not None], max(t for t in tol if t))
    check(f"Example 43.5.12 ({nu}, {nv}): no flipped faces in either map",
          (ch43.face_signed_areas(F_, zf) < 0).sum() == 0 and (ch43.face_signed_areas(F_, zs) < 0).sum() == 0)
V_, F_, _, _ = ch43.torus_piece(24, 48)
check("Example 43.5.12: torus_piece(24, 48) has 169 vertices and 288 faces and is a disk",
      len(V_) == 169 and len(F_) == 288 and dm.euler_characteristic(F_) == 1)
close("Example 43.5.12: a circle polygon with 48 boundary vertices has corners of 172.5 deg", 180 - 360 / 48, 172.5, 1e-12)

# ---------------------------------------------------------------- Remark 43.5.13 (length cross ratio)
Pq = rng.standard_normal((4, 3))
uq = rng.standard_normal(4)
L0 = lambda a_, b_: np.linalg.norm(Pq[a_] - Pq[b_])  # noqa: E731
L1 = lambda a_, b_: np.exp((uq[a_] + uq[b_]) / 2) * L0(a_, b_)  # noqa: E731
cr = lambda Lf: Lf(0, 1) * Lf(2, 3) / (Lf(1, 2) * Lf(3, 0))  # noqa: E731
close("Remark 43.5.13: vertex scaling e^{(u_i + u_j)/2} keeps the length cross ratio l_ij l_kl / (l_jk l_li)", cr(L1), cr(L0), 1e-12)

# ---------------------------------------------------------------- Exercise 43.5.1
V1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
F1 = np.array([[0, 1, 2]])
L1m = dm.cotan_laplacian(V1, F1)
A1 = ch43.area_matrix(F1, 3)
check("Exercise 43.5.1: L = [[-1, 1/2, 1/2], [1/2, -1/2, 0], [1/2, 0, -1/2]]",
      np.allclose(L1m, [[-1, 0.5, 0.5], [0.5, -0.5, 0], [0.5, 0, -0.5]]))
check("Exercise 43.5.1: A = (i/4)[[0, -1, 1], [1, 0, -1], [-1, 1, 0]]", np.allclose(A1, 0.25j * np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])))
E1 = -0.5 * L1m - A1
p1 = np.array([0, 1, 1j])
close("Exercise 43.5.1: E_C(p) = 0, E_C(conj p) = 1", ((np.conj(p1) @ E1 @ p1).real, (p1 @ E1 @ np.conj(p1)).real), (0.0, 1.0), 1e-14)

# ---------------------------------------------------------------- Exercise 43.5.2
zd = zB.copy()
zd[interior] += crand(len(interior))
close("Exercise 43.5.2(a): moving interior values does not change the signed area", ch43.signed_area(Fh, zd), ch43.signed_area(Fh, zB), 1e-9)
check("Exercise 43.5.2(b): EC has no interior entries of A (EC_II = -L_II / 2, EC_I,bd = -L_I,bd / 2)",
      np.allclose(EC[np.ix_(I_, I_)], -0.5 * L[np.ix_(I_, I_)]) and np.allclose(EC[np.ix_(I_, B_)], -0.5 * L[np.ix_(I_, B_)]))
check("Exercise 43.5.2(c): B = M and B = B_bd give different eigenvalues and maps", abs(lamM - lamB) > 1e-4 and abs(errM - errB) > 1e-3)

# ---------------------------------------------------------------- Exercise 43.5.3 (planar meshes)
mg = dm.mass_barycentric(Vg, Fg)
dg = 1 / np.sqrt(mg)
lg, Yg = np.linalg.eigh(dg[:, None] * ECg * dg[None, :])
# the 2-dim kernel contains p - mean: the projection of p - mean onto the computed kernel is complete
pc = pg - (mg @ pg) / mg.sum()
Kb = np.c_[dg * Yg[:, 0], dg * Yg[:, 1]]
coef = np.linalg.lstsq(Kb, pc, rcond=None)[0]
check("Exercise 43.5.3: lambda_0 = lambda_1 = 0 on a planar mesh and p - mean lies in the kernel",
      abs(lg[0]) < 1e-12 and abs(lg[1]) < 1e-12 and np.abs(Kb @ coef - pc).max() < 1e-10)
pp, qq = ch43.farthest_boundary_pair(Vg, Fg)
zlg = ch43.lscm_complex(Vg, Fg, [pp, qq], [pg[pp], pg[qq]])
check("Exercise 43.5.3: LSCM reproduces a planar mesh exactly", np.abs(zlg - pg).max() < 1e-12)

# ---------------------------------------------------------------- Exercises 43.5.4 and 43.5.5
ns4, same4, printed4 = ch43.run_printed(PAGE, "code-ex-43-5-4", "out-ex-43-5-4")
check("Exercise 43.5.4: code-ex-43-5-4 prints exactly out-ex-43-5-4", same4, printed4)
ns5, same5, printed5 = ch43.run_printed(PAGE, "code-ex-43-5-5", "out-ex-43-5-5")
check("Exercise 43.5.5: code-ex-43-5-5 prints exactly out-ex-43-5-5", same5, printed5)
Vhf, Fhf = ch43.height_field(16)
check("Exercise 43.5.5: the height field has upward normals and 289 vertices",
      (dm.face_normals(Vhf, Fhf)[:, 2] > 0).all() and len(Vhf) == 289 and dm.euler_characteristic(Fhf) == 1)

summary()
