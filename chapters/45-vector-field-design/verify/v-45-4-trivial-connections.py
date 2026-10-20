"""Verification for Section 45.4 (Vector Field Design with Trivial Connections).

Run from the project root:
``PYTHONPATH=tools python3 chapters/45-vector-field-design/verify/v-45-4-trivial-connections.py``
"""

import pathlib
import sys
import warnings

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch45  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

warnings.simplefilter("ignore")
PAGE = HERE / "45-4-trivial-connections.html"
rng = np.random.default_rng(45)
TWO_PI = 2 * np.pi
W = ch45.wrap


def setup(V, F):
    D0, s1 = dd.d0(F, len(V)), dd.hodge1(V, F)
    return D0, s1, ch45.transport_angles(V, F), dm.angle_defect(V, F)


# ---------------------------------------------------------------- Example 45.4.2
ft = ch45.FlatTorus(12, 12)
D0f = dd.d0(ft.F)
s0f, s1f, s2f = ch45.stars_from_corners(ft.F, ft.C)
rhof = ch45.transport_angles_flat(ft.F, ft.C)
tcf, Cf, Csf = ch45.generators(ft.F)
check("Example 45.4.2(a): on the flat torus Levi-Civita is trivial (cells and both dual loops have holonomy 0 mod 2 pi)",
      np.allclose(W(-D0f.T @ rhof), 0, atol=1e-12) and np.allclose(W(Csf @ rhof), 0, atol=1e-12))
psi_lc = ch45.parallel_field(ft.F, rhof, np.zeros(len(rhof)))
d = ft.C[:, 1] - ft.C[:, 0]
glob = np.arctan2(d[:, 1], d[:, 0]) + psi_lc
check("Example 45.4.2(a): its parallel field is the constant field", np.allclose(W(glob - glob[0]), 0, atol=1e-10))
V, F = dm.icosphere(3)
D0, s1, rho, delta = setup(V, F)
check("Example 45.4.2(b): on icosphere(3) Levi-Civita is not trivial (0 < delta_i < 0.03)",
      delta.min() > 0 and delta.max() < 0.03)

# ---------------------------------------------------------------- Propositions 45.4.3, 45.4.4, 45.4.5
k = np.zeros(len(V), int)
k[[0, 3]] = 1
xi = ch45.trivial_connection(D0, s1, delta, rho, k)
check("Proposition 45.4.3: the BFS field is parallel on every dual edge (path independence)",
      np.abs(W(ch45.connection_from_field(F, rho, ch45.parallel_field(F, rho, xi)) - xi)).max() < 1e-9)
psi_a = ch45.parallel_field(F, rho, xi, f0=0, psi0=0.4)
psi_b = ch45.parallel_field(F, rho, xi, f0=777, psi0=psi_a[777])
check("Proposition 45.4.3: starting from another face (with the same vector there) gives the same field",
      np.allclose(W(psi_a - psi_b), 0, atol=1e-9))
# Proposition 45.4.4 on the torus: random integer dual cycles
Vt, Ft = dm.torus_mesh(2.0, 0.8, 12, 24)
D0t, s1t, rhot, deltat = setup(Vt, Ft)
tct, Ct, Cst = ch45.generators(Ft)
Gt = ch45.harmonic_basis(D0t, s1t, Cst)
kt = np.zeros(len(Vt), int)
kt[5], kt[150] = 1, -1
xit = ch45.trivial_connection(D0t, s1t, deltat, rhot, kt, Cst, Gt, [1, -2])
kap = deltat - D0t.T @ xit
check("Theorem 45.4.10 (torus, k = +-1, n = (1, -2)): curvature 2 pi k and dual-loop holonomies 2 pi n",
      np.allclose(kap, TWO_PI * kt) and np.allclose((Cst @ (rhot + xit)) / TWO_PI, [1, -2]))
ok = True
for trial in range(20):
    y = rng.integers(-2, 3, len(Vt))
    m = rng.integers(-2, 3, 2)
    cst = Cst.T @ m - D0t @ y                    # an integer dual cycle
    hol = cst @ (rhot + xit)
    pred = m @ (Cst @ (rhot + xit)) + y @ kap
    ok = ok and abs(W(hol - pred)) < 1e-8 and abs(W(hol)) < 1e-8
check("Proposition 45.4.4: holonomy of integer dual cycles = sum m_a hol(c*_a) + sum y_i kappa_i, and 0 mod 2 pi", ok)
# decomposition of an arbitrary integer dual cycle: the boundary of a set of cells plus loops
S = (rng.random(len(Vt)) < 0.3).astype(float)
cst = -D0t @ S + Cst[0]
mm = cst @ Ct.T
resid = cst - Cst.T @ mm
yy = np.linalg.lstsq(-D0t, resid, rcond=None)[0]
check("Proposition 45.4.4: c* - sum m_a c*_a is -d0 y with integer y (up to a constant)",
      np.allclose(-D0t @ yy, resid, atol=1e-9) and np.allclose(yy - yy[0], np.rint(yy - yy[0]), atol=1e-8))
# Proposition 45.4.5: every field
psi_r = rng.uniform(-np.pi, np.pi, len(F))
xir = ch45.connection_from_field(F, rho, psi_r)
check("Proposition 45.4.5: xi^psi of a random field is trivial (cells 2 pi Z) and psi is parallel",
      np.allclose((delta - D0.T @ xir) / TWO_PI, np.rint((delta - D0.T @ xir) / TWO_PI), atol=1e-9)
      and np.abs(W(ch45.connection_from_field(F, rho, psi_r) - xir)).max() < 1e-12)
ind_r, frac = ch45.field_indices(D0, delta, rho, psi_r, F)
check("Theorem 45.4.8: a random face field has integer indices summing to chi = 2", frac < 1e-9 and ind_r.sum() == 2)
for name, (Vm, Fm) in (("torus", (Vt, Ft)), ("genus2_mesh(0.2)", dm.genus2_mesh(0.2))):
    D0m, s1m, rhom, deltam = setup(Vm, Fm)
    pr = rng.uniform(-np.pi, np.pi, len(Fm))
    im, fr = ch45.field_indices(D0m, deltam, rhom, pr, Fm)
    check(f"Theorem 45.4.8 ({name}): random field indices sum to chi = {dm.euler_characteristic(Fm)}",
          fr < 1e-8 and im.sum() == dm.euler_characteristic(Fm))

# ---------------------------------------------------------------- Example 45.4.7: rotation about the axis x_0
a = V[0]
r, b, N = ch45.face_frames3(V, F)
cen = V[F].mean(1)
X = np.cross(a, cen)
X -= np.einsum("ij,ij->i", X, N)[:, None] * N
check("Example 45.4.7: a x c_f is nonzero on every face", np.linalg.norm(X, axis=1).min() > 1e-3)
psi_e = np.arctan2(np.einsum("ij,ij->i", X, b), np.einsum("ij,ij->i", X, r))
ind_e, frac = ch45.field_indices(D0, delta, rho, psi_e, F)
check("Example 45.4.7: the rotation field has index +1 at vertices 0 and 3 and 0 at the other 640",
      frac < 1e-9 and {int(i): int(ind_e[i]) for i in np.nonzero(ind_e)[0]} == {0: 1, 3: 1} and int(np.argmin(V @ a)) == 3)

# ---------------------------------------------------------------- Theorem 45.4.9: existence iff sum k = chi
k_bad = np.zeros(len(V))
k_bad[0] = 1
A = np.vstack([-D0.T])
ub = -delta + TWO_PI * k_bad
x = np.linalg.lstsq(A, ub, rcond=None)[0]
check("Theorem 45.4.9: with sum k = 1 != chi the cell constraints have no solution (residual > 0)",
      np.linalg.norm(A @ x - ub) > 0.1 and abs(ub.sum() + TWO_PI) < 1e-9)
Amat = np.vstack([-D0t.T, Cst])
sv = np.linalg.svd(Amat, compute_uv=False)
check("Theorem 45.4.9: rank of (-d0^T; C*^T) is n0 - 1 + 2g on the torus", int((sv > 1e-9).sum()) == len(Vt) - 1 + 2)
check("Theorem 45.4.9: its left kernel is spanned by (1, 0)",
      np.allclose(Amat.T @ np.concatenate([np.ones(len(Vt)), [0, 0]]), 0))

# ---------------------------------------------------------------- Theorem 45.4.10: minimality and uniqueness
for name, (Vm, Fm), km, nm in (("icosphere(3)", (V, F), k, None), ("torus", (Vt, Ft), kt, [1, -2])):
    D0m, s1m, rhom, deltam = setup(Vm, Fm)
    tcm, Cm, Csm = ch45.generators(Fm)
    Gm = ch45.harmonic_basis(D0m, s1m, Csm) if len(Csm) else None
    xim = ch45.trivial_connection(D0m, s1m, deltam, rhom, km, Csm if len(Csm) else None, Gm, nm)
    E0 = xim @ (xim / s1m)
    # the null space of the constraints: dual exact forms d1^T w; adding them only increases the norm
    D1m = dd.d1(Fm)
    better = True
    for trial in range(5):
        wv = rng.normal(size=len(Fm))
        pert = D1m.T @ wv * 0.05
        xp = xim + pert
        better = better and np.allclose(-D0m.T @ xp, -D0m.T @ xim) and (len(Csm) == 0 or np.allclose(Csm @ xp, Csm @ xim))
        better = better and np.isclose(xp @ (xp / s1m), E0 + pert @ (pert / s1m))
    check(f"Theorem 45.4.10 / Exercise 45.4.4 ({name}): dual exact perturbations keep the constraints and add their norm",
          better)
    th = xim / s1m
    check(f"Theorem 45.4.10 ({name}): theta = star1^-1 xi is closed (no co-exact part)", np.abs(D1m @ th).max() < 1e-9)
    if s1m.min() < 1e-12:                        # zero star1 entries (torus_mesh diagonals): Remark 45.4.12
        check(f"Remark 45.4.12 ({name}): xi vanishes on the edges with star1 = 0", np.abs(xim[s1m < 1e-12]).max() < 1e-14)
        continue
    # Lagrange check against a direct KKT solve
    Acon = np.vstack([-D0m.T] + ([Csm] if len(Csm) else []))
    rhs = np.concatenate([-deltam + TWO_PI * km] + ([-W(Csm @ rhom) + TWO_PI * np.asarray(nm)] if len(Csm) else []))
    Wm = np.diag(1 / s1m)
    K = np.block([[2 * Wm, Acon.T], [Acon, np.zeros((len(Acon), len(Acon)))]])
    sol = np.linalg.lstsq(K, np.concatenate([np.zeros(len(s1m)), rhs]), rcond=None)[0]
    check(f"Theorem 45.4.10 ({name}): agrees with the minimum-norm solution of the KKT system", np.allclose(sol[:len(s1m)], xim, atol=1e-8))
# KKT check on the flat torus (all star1 > 0) with generators
D0f12, (s0x, s1x, s2x) = dd.d0(ch45.FlatTorus(12, 12).F), ch45.stars_from_corners(ch45.FlatTorus(12, 12).F, ch45.FlatTorus(12, 12).C)
ftx = ch45.FlatTorus(12, 12)
rx = ch45.transport_angles_flat(ftx.F, ftx.C)
tcx, Cx, Csx = ch45.generators(ftx.F)
Gx = ch45.harmonic_basis(D0f12, s1x, Csx)
kx = np.zeros(len(ftx.P), int)
kx[13], kx[100] = 1, -1
xix = ch45.trivial_connection(D0f12, s1x, np.zeros(len(ftx.P)), rx, kx, Csx, Gx, [2, 1])
Acon = np.vstack([-D0f12.T, Csx])
rhs = np.concatenate([TWO_PI * kx, -W(Csx @ rx) + TWO_PI * np.array([2, 1])])
K = np.block([[2 * np.diag(1 / s1x), Acon.T], [Acon, np.zeros((len(Acon), len(Acon)))]])
sol = np.linalg.lstsq(K, np.concatenate([np.zeros(len(s1x)), rhs]), rcond=None)[0]
check("Theorem 45.4.10 (flat torus, k = +-1, n = (2, 1)): agrees with the minimum-norm solution of the KKT system",
      np.allclose(sol[:len(s1x)], xix, atol=1e-8))
P = Cst @ (s1t[:, None] * Gt.T)
check("Theorem 45.4.10: P is symmetric positive definite on the torus mesh (Remark 45.4.12)",
      np.allclose(P, P.T, atol=1e-10) and np.linalg.eigvalsh((P + P.T) / 2).min() > 0.3)

# ---------------------------------------------------------------- Proposition 45.4.11 and Examples 45.4.13-45.4.15
Vc, Fc = ch45.unit_cube()
D0c, s1c, rhoc, deltac = setup(Vc, Fc)
kc = np.zeros(8, int)
kc[[0, 7]] = 1
xic = ch45.trivial_connection(D0c, s1c, deltac, rhoc, kc)
E, _ = dd.edge_list(Fc)
dist = np.array([bin(i).count("1") for i in range(8)])               # graph distance from x_0 = index popcount
cube_edge = np.abs(s1c - 1) < 1e-12
expect = np.zeros(len(E))
for e, (i, j) in enumerate(E):
    if cube_edge[e] and {dist[i], dist[j]} == {0, 1}:
        expect[e] = np.pi / 2
    if cube_edge[e] and {dist[i], dist[j]} == {2, 3}:
        expect[e] = -np.pi / 2
check("Example 45.4.13: star1 = 1 on the 12 cube edges and 0 on the 6 diagonals",
      cube_edge.sum() == 12 and np.allclose(s1c[~cube_edge], 0, atol=1e-12))
check("Example 45.4.13: xi = pi/2 on the three edges out of x_0, -pi/2 on the three into x_7, 0 elsewhere",
      np.allclose(xic, expect, atol=1e-12))
check("Example 45.4.13 / Exercise 45.4.2: curvature 2 pi at x_0 and x_7, 0 at the other corners",
      np.allclose(deltac - D0c.T @ xic, TWO_PI * kc))
res = {}
for label, sing in (("antipodal", {0: 1, 3: 1}), ("90", {0: 1, int(np.argmin(np.abs(V @ V[0] - np.cos(np.radians(90))))): 1}),
                    ("one +2", {0: 2})):
    km = np.zeros(len(V), int)
    for i_, v_ in sing.items():
        km[i_] = v_
    xim = ch45.trivial_connection(D0, s1, delta, rho, km)
    psim = ch45.parallel_field(F, rho, xim)
    indm, _ = ch45.field_indices(D0, delta, rho, psim, F)
    res[label] = (xim @ (xim / s1), np.abs(xim).max(), np.array_equal(indm, km))
print("   icosphere(3) designs:", {k_: (round(v[0], 3), round(v[1], 3), v[2]) for k_, v in res.items()})
check("Example 45.4.14: energies 43.081, 47.169, 98.770 and max |xi| 1.253, 1.302 (< pi), indices as prescribed",
      [round(res[k_][0], 3) for k_ in ("antipodal", "90", "one +2")] == [43.081, 47.169, 98.770]
      and [round(res[k_][1], 3) for k_ in ("antipodal", "90")] == [1.253, 1.302] and all(v[2] for v in res.values())
      and res["one +2"][1] < np.pi)
# Example 45.4.15: flat torus
ft = ch45.FlatTorus(24, 24)
D0f, (s0f, s1f, s2f) = dd.d0(ft.F), ch45.stars_from_corners(ft.F, ft.C)
rhof = ch45.transport_angles_flat(ft.F, ft.C)
tcf, Cf, Csf = ch45.generators(ft.F)
Gf = ch45.harmonic_basis(D0f, s1f, Csf)
E, left, right = ch45.left_right_faces(ft.F)
_, idx = dd.edge_list(ft.F)
n = ft.n
h = np.zeros(len(E))
row = [2 * c + t for c in range(n) for t in (0, 1)]
for a_, b_ in zip(row, row[1:] + row[:1]):
    e = idx[tuple(sorted(set(ft.F[a_]) & set(ft.F[b_])))]
    h[e] += 1.0 if (right[e] == a_ and left[e] == b_) else -1.0
check("Example 45.4.15: the horizontal face strip h* has (h* . c_1, h* . c_2) = (0, -1)", np.allclose(h @ Cf.T, [0, -1]))
xa = ch45.trivial_connection(D0f, s1f, np.zeros(len(ft.P)), rhof, np.zeros(len(ft.P), int), Csf, Gf, (0, -1))
close("Example 45.4.15(a): holonomy +2 pi along the horizontal strip", h @ (rhof + xa), TWO_PI, 1e-9)
close("Example 45.4.15(a): max |xi| = 0.131", round(np.abs(xa).max(), 3), 0.131, 1e-12)
kf = np.zeros(len(ft.P), int)
kf[6 * n + 6], kf[17 * n + 17] = 1, -1
xb = ch45.trivial_connection(D0f, s1f, np.zeros(len(ft.P)), rhof, kf, Csf, Gf, (0, 0))
psib = ch45.parallel_field(ft.F, rhof, xb)
indb, _ = ch45.field_indices(D0f, np.zeros(len(ft.P)), rhof, psib, ft.F)
check("Example 45.4.15(b): indices +1 at p and -1 at q, max |xi| = 1.160",
      np.array_equal(indb, kf) and round(np.abs(xb).max(), 3) == 1.160)
check("Theorem 45.4.9 (flat torus): a single +1 has no trivial connection (sum k = 1 != 0)",
      np.linalg.norm(-D0f.T @ np.linalg.lstsq(-D0f.T, TWO_PI * np.eye(len(ft.P))[0], rcond=None)[0] - TWO_PI * np.eye(len(ft.P))[0]) > 0.1)

# ---------------------------------------------------------------- Implementation box and coding exercises
ns, same, out = ch45.run_printed(PAGE, "code-45-4-1", "out-45-4-1")
check("code-45-4-1 runs and prints exactly out-45-4-1", same)
check("code-45-4-1 has at most 30 lines", dgcode.code_lines(PAGE, "code-45-4-1") <= 30)
xi_box, _ = ns["trivial_connection"](V, F, k)
check("code-45-4-1 trivial_connection agrees with lib/ch45.trivial_connection (icosphere(3))",
      np.allclose(xi_box, ch45.trivial_connection(D0, s1, delta, rho, k), atol=1e-10))
xi_box_t, _ = ns["trivial_connection"](Vt, Ft, kt, [1, -2])
check("code-45-4-1 trivial_connection agrees with lib/ch45.trivial_connection (torus, n = (1, -2))",
      np.allclose(xi_box_t, xit, atol=1e-10))
for blk in ("code-ex-45-4-5", "code-ex-45-4-6"):
    ns, same, out = ch45.run_printed(PAGE, blk, blk.replace("code", "out"))
    check(f"{blk} runs and prints exactly {blk.replace('code', 'out')}", same)

# Example 45.4.14: the 90-degree vertex is 81; another 90-degree vertex (25) gives a slightly different energy
V, F = dm.icosphere(3)
D0, s1, rho, delta = setup(V, F)
j81 = int(np.argmin(np.abs(V @ V[0] - np.cos(np.radians(90)))))
en = {}
for j in (81, 25):
    km = np.zeros(len(V))
    km[[0, j]] = 1
    xj = ch45.trivial_connection(D0, s1, delta, rho, km)
    en[j] = round(float(xj @ (xj / s1)), 3)
check("Example 45.4.14: vertex 81 (and 25) is at 90 degrees from x_0; energies 47.169 and 47.292",
      j81 == 81 and abs(V[81] @ V[0]) < 1e-12 and abs(V[25] @ V[0]) < 1e-12 and en == {81: 47.169, 25: 47.292})
# the text before Theorem 45.4.10: the solutions form an affine space of dimension n2 - 1 (= rank d1^T)
Amat = np.vstack([-D0t.T, Cst])
check("Theorem 45.4.10 (text): n1 - rank A = n2 - 1 = rank d1^T on the torus mesh",
      Amat.shape[1] - np.linalg.matrix_rank(Amat) == len(Ft) - 1 == np.linalg.matrix_rank(dd.d1(Ft)))

summary()
