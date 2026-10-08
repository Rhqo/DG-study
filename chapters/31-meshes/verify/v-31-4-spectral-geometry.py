"""Verification for Section 31.4 (Spectral Geometry on Meshes).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/verify/v-31-4-spectral-geometry.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

from dgcheck import check, close, summary, sym_equal  # noqa: E402

# ---------------------------------------------------------------- Definition 31.4.1 / Proposition 31.4.2
V, F = dm.icosphere(3)
L = dm.cotan_laplacian(V, F)
m = dm.mass_voronoi(V, F)
lam, Phi = dm.generalized_eigs(L, m)
check("Prop 31.4.2: Phi^T M Phi = I", np.allclose(Phi.T @ (m[:, None] * Phi), np.eye(len(V)), atol=1e-8))
check("Prop 31.4.2: -L phi_k = lambda_k M phi_k", np.abs(-L @ Phi - (m[:, None] * Phi) * lam).max() < 1e-8)
check("Prop 31.4.2: lambda_0 = 0 with a constant eigenvector, all lambda >= 0",
      abs(lam[0]) < 1e-9 and np.std(Phi[:, 0]) < 1e-9 and lam.min() > -1e-9)
k = 7
close("Prop 31.4.2: lambda_k = Rayleigh quotient int|grad phi|^2 / int phi^2", lam[k],
      dm.dirichlet_energy(V, F, Phi[:, k]) / (Phi[:, k] @ (m * Phi[:, k])), tol=1e-9)
u = np.random.default_rng(0).normal(size=len(V))
close("Prop 31.4.2: expansion u = sum <u, phi_k>_M phi_k is exact with all eigenvectors", Phi @ (Phi.T @ (m * u)), u,
      tol=1e-9)

# ---------------------------------------------------------------- Figure 31.4.1: sphere spectrum
for lv, err7 in ((2, 0.268), (3, 0.074)):
    Vs, Fs = dm.icosphere(lv)
    lam_s, _ = dm.generalized_eigs(dm.cotan_laplacian(Vs, Fs), dm.mass_voronoi(Vs, Fs), 64)
    grp = np.array([lam_s[l * l:(l + 1) ** 2].mean() for l in range(1, 8)])
    rel = (grp - np.arange(1, 8) * np.arange(2, 9)) / (np.arange(1, 8) * np.arange(2, 9))
    check(f"Fig 31.4.1: level {lv}: group means below l(l+1), error at l = 7 is {abs(rel[-1]):.3f}",
          (rel < 1e-12).all() and round(abs(rel[-1]), 3) == err7)
    # groups separate: the largest value of group l is below the smallest of group l + 1
    sep = all(lam_s[(l + 1) ** 2 - 1] < lam_s[(l + 1) ** 2] for l in range(0, 7))
    check(f"Fig 31.4.1: level {lv}: the first 64 eigenvalues fall into groups of 1, 3, 5, ..., 15", sep)
    exact64 = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(8)])
    check(f"Fig 31.4.1: level {lv}: every one of the 64 eigenvalues is below its exact value (observation)",
          (lam_s[1:64] < exact64[1:64]).all())
    if lv == 2:
        check("Fig 31.4.1: level 2: lambda_1..3 = 2 to relative 5e-6",
              round(np.abs(lam_s[1:4] / 2 - 1).max(), 6) == 5e-6 or 4e-6 < np.abs(lam_s[1:4] / 2 - 1).max() < 6e-6)
close("Fig 31.4.1: level 3: lambda_1..3 = 2 (relative 4e-7)", lam[1:4], np.full(3, 2.0), tol=2e-6)
close("Fig 31.4.1: level 3: the l = 2 group is 5.966 (5 equal values)", lam[4:9], np.full(5, lam[4]), tol=1e-9)
check("Fig 31.4.1: level 3: l = 2 eigenvalue 5.966", round(lam[4], 3) == 5.966)
g3 = lam[9:16]
check("Text: level 3: the 7 values of l = 3 split into 3 x 11.803 and 4 x 11.851",
      np.allclose(g3[:3], 11.803, atol=5e-4) and np.allclose(g3[3:], 11.851, atol=5e-4))
g4 = lam[16:25]
check("Text: level 3: the 9 values of l = 4 split into 5 x 19.480 and 4 x 19.509 (icosahedral group: 4 + 5)",
      np.allclose(g4[:5], 19.480, atol=5e-4) and np.allclose(g4[5:], 19.509, atol=5e-4))
# the mesh l = 1 eigenspace is (almost) spanned by x, y, z  (Proposition 29.6.1: Y_1m ~ x, y, z)
P1 = Phi[:, 1:4]
coef = P1.T @ (m[:, None] * V)
res = np.sqrt((m[:, None] * (V - P1 @ coef) ** 2).sum() / (m[:, None] * V ** 2).sum())
check(f"Text: x, y, z lie in span(phi_1, phi_2, phi_3) up to {res:.1e} (relative)", 1.0e-4 < res < 1.2e-4)
# a degree-2 spherical harmonic on the mesh: Rayleigh quotient ~ 6
Y = 3 * V[:, 2] ** 2 - 1
close("Text: Rayleigh quotient of 3z^2 - 1 on the level-3 mesh ~ l(l+1) = 6", round(
    dm.dirichlet_energy(V, F, Y) / (Y @ (m * Y)), 2), 5.97, tol=1e-12)
# SH count of 3DGS (Exercise 31.4.1)
check("Exercise 31.4.1: eigenvalues of the unit sphere below 13: 1 + 3 + 5 + 7 = 16 (SH degree <= 3)",
      sum(2 * l + 1 for l in range(10) if l * (l + 1) < 13) == 16)
check("Exercise 31.4.1: on the level-3 mesh also 16 eigenvalues below 13", (lam < 13).sum() == 16)

# ---------------------------------------------------------------- Exercise 31.4.2: scaling
s_ = 2.5
lam2, Phi2 = dm.generalized_eigs(dm.cotan_laplacian(s_ * V, F), dm.mass_voronoi(s_ * V, F), 10)
close("Exercise 31.4.2: scaling by s divides eigenvalues by s^2", lam2[1:10], lam[1:10] / s_ ** 2, tol=1e-9)
check("Exercise 31.4.2: M-normalized eigenfunctions scale by 1/s (up to sign/rotation within groups)",
      np.isclose(np.abs(Phi2[:, 0]).max(), np.abs(Phi[:, 0]).max() / s_))

# ---------------------------------------------------------------- Figure 31.4.2: ellipsoid
VE = V * np.array([1.6, 1.0, 0.7])
lamE, PhiE = dm.generalized_eigs(dm.cotan_laplacian(VE, F), dm.mass_voronoi(VE, F), 16)
check(f"Fig 31.4.2: ellipsoid eigenvalues lambda_1..3 = {lamE[1]:.2f}, {lamE[2]:.2f}, {lamE[3]:.2f}",
      [round(x, 2) for x in lamE[1:4]] == [1.13, 2.02, 2.28])
check("Fig 31.4.2: ellipsoid phi_1, phi_2, phi_3 follow x, y, z (|corr| > 0.95)",
      all(abs(np.corrcoef(PhiE[:, k_], VE[:, ax_])[0, 1]) > 0.95 for k_, ax_ in ((1, 0), (2, 1), (3, 2))))
check("Fig 31.4.2: the first 12 ellipsoid eigenvalues are simple (smallest gap 0.047, between lambda_7 and lambda_8)",
      round(np.diff(lamE[:12]).min(), 3) == 0.047)

# ---------------------------------------------------------------- Figure 31.4.3: low-pass
V0, F0 = dm.icosphere(3)
rng = np.random.default_rng(7)
x, y, z = V0.T
r = 1 + 0.18 * (x * z) + 0.12 * (y * y - z * z) + 0.1 * x
cent = rng.normal(size=(12, 3))
cent /= np.linalg.norm(cent, axis=1, keepdims=True)
for c in cent:
    r += 0.22 * np.exp(-np.sum((V0 - c) ** 2, 1) / (2 * 0.22 ** 2))
VB = V0 * r[:, None]
mB = dm.mass_voronoi(VB, F0)
lamB, PhiB = dm.generalized_eigs(dm.cotan_laplacian(VB, F0), mB)
coefB = PhiB.T @ (mB[:, None] * VB)
c0 = (mB[:, None] * VB).sum(0) / mB.sum()
den = np.sqrt((mB[:, None] * (VB - c0) ** 2).sum())
rec = lambda K: PhiB[:, :K] @ coefB[:K]  # noqa: E731
check("Fig 31.4.3: K = 1 reconstruction is the area-weighted centroid", np.allclose(rec(1), c0, atol=1e-10))
errs = [round(np.sqrt((mB[:, None] * (VB - rec(K)) ** 2).sum()) / den, 4) for K in (4, 25, 100, 200)]
check(f"Fig 31.4.3: relative errors {errs} for K = 4, 25, 100, 200", errs == [0.0728, 0.0399, 0.0059, 0.0014])
check("Fig 31.4.3: the error is non-increasing in K (orthogonal projection)",
      all(np.sqrt((mB[:, None] * (VB - rec(K)) ** 2).sum()) >= np.sqrt((mB[:, None] * (VB - rec(K + 1)) ** 2).sum()) - 1e-12
          for K in range(2, 80)))
# Exercise 31.4.3: on the unit sphere mesh four eigenfunctions reproduce the positions
r4 = Phi[:, :4] @ (Phi[:, :4].T @ (m[:, None] * V))
check("Exercise 31.4.3: unit icosphere: x^(4) = x up to 1.1e-4 (relative)",
      round(np.sqrt((m[:, None] * (V - r4) ** 2).sum() / (m[:, None] * V ** 2).sum()), 5) == 1.1e-4)

# ---------------------------------------------------------------- Figure 31.4.4: heat method
mb = dm.mass_barycentric(V, F)
h = np.mean([np.linalg.norm(V[a] - V[b]) for a, b in dm.edges(F)])
t = h * h
check(f"Fig 31.4.4: mean edge length h = {h:.3f}, t = h^2 = {t:.4f}", round(h, 3) == 0.151 and round(t, 4) == 0.0227)
s = int(np.argmax(V[:, 2]))
delta = np.zeros(len(V))
delta[s] = 1
N = dm.face_normals(V, F)
A = dm.face_areas(V, F)


def heat_method(tt):
    uu = np.linalg.solve(np.diag(mb) - tt * L, delta)
    G = np.zeros((len(F), 3))
    for c_ in range(3):
        e = V[F[:, (c_ + 2) % 3]] - V[F[:, (c_ + 1) % 3]]
        G += uu[F[:, c_], None] * np.cross(N, e) / (2 * A[:, None])
    X = -G / np.linalg.norm(G, axis=1, keepdims=True)
    bb = np.zeros(len(V))
    for c_ in range(3):
        e = V[F[:, (c_ + 2) % 3]] - V[F[:, (c_ + 1) % 3]]
        np.add.at(bb, F[:, c_], 0.5 * (np.cross(N, e) * X).sum(1))
    idx = np.array([i for i in range(len(V)) if i != s])
    ph = np.zeros(len(V))
    ph[idx] = np.linalg.solve(-L[np.ix_(idx, idx)], bb[idx])
    return uu, ph, bb


u, phi, bvec = heat_method(t)
# direction of X = -grad u / |grad u| compared with the exact meridian direction (away from the north pole)
Gu = np.zeros((len(F), 3))
for c_ in range(3):
    e = V[F[:, (c_ + 2) % 3]] - V[F[:, (c_ + 1) % 3]]
    Gu += u[F[:, c_], None] * np.cross(N, e) / (2 * A[:, None])
Xd = -Gu / np.linalg.norm(Gu, axis=1, keepdims=True)
cn = V[F].mean(1)
cn /= np.linalg.norm(cn, axis=1, keepdims=True)
mer = -(np.array([0, 0, 1.0]) - cn[:, 2:3] * cn)          # tangent direction of increasing polar angle
okf = np.linalg.norm(mer, axis=1) > 1e-3
mer = mer[okf] / np.linalg.norm(mer[okf], axis=1, keepdims=True)
Xt = Xd[okf] - (Xd[okf] * cn[okf]).sum(1, keepdims=True) * cn[okf]
Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
dev = np.degrees(np.arccos(np.clip((Xt * mer).sum(1), -1, 1)))
check(f"Fig 31.4.4: X deviates from the meridian by {dev.mean():.1f} deg on average, {dev.max():.1f} deg at most",
      round(dev.mean()) == 5 and round(dev.max()) == 14)
thf = np.arccos(np.clip(cn[okf, 2], -1, 1))
check("Text: the deviation is largest near the source (mean over theta < 0.5 exceeds the mean over theta > 2.6)",
      dev[thf < 0.5].mean() > dev[thf > 2.6].mean())
theta = np.arccos(np.clip(V[:, 2], -1, 1))
err = np.abs(phi - theta)
check(f"Fig 31.4.4: heat method max error {err.max():.3f}, mean {err.mean():.3f}",
      round(err.max(), 3) == 0.050 and round(err.mean(), 3) == 0.021)
check("Fig 31.4.4: the divergence vector sums to zero (Poisson problem solvable)", abs(bvec.sum()) < 1e-10)
check("Fig 31.4.4: the heat u spans more than 8 orders of magnitude", 1e8 < u.max() / u.min() < 1e9)
naive = np.sqrt(np.maximum(-4 * t * np.log(u / u[s]), 0))
check(f"Fig 31.4.4: the magnitude-only estimate is far off at the antipode ({naive[np.argmin(V[:, 2])]:.2f} vs pi)",
      naive[np.argmin(V[:, 2])] < 1.5)
for mult, mean_err in ((0.25, 0.034), (4, 0.017)):
    _, ph_, _ = heat_method(mult * t)
    check(f"Text: t = {mult} h^2 gives mean error {np.abs(ph_ - theta).mean():.3f}",
          round(np.abs(ph_ - theta).mean(), 3) == mean_err)
# direction argument (Exercise 31.4.5): if u = c exp(-d^2 / 4t), then -grad u / |grad u| = grad d / |grad d|
dd, tt_, cc = sp.symbols("d t c", positive=True)
uu = cc * sp.exp(-dd ** 2 / (4 * tt_))
sym_equal("Exercise 31.4.5: d u / d d = -(d / 2t) u < 0, so -grad u points along +grad d", sp.diff(uu, dd),
          -(dd / (2 * tt_)) * uu)

# ---------------------------------------------------------------- heat kernel signature on the sphere
def hks_exact(tt, lmax=300):
    return sum((2 * l + 1) / (4 * np.pi) * np.exp(-l * (l + 1) * tt) for l in range(lmax))


for tt in (0.01, 0.05, 0.1):
    close(f"Text: unit sphere: 4 pi t k_t(x, x) = 1 + t/3 + O(t^2) at t = {tt} (K = 1)", 4 * np.pi * tt * hks_exact(tt),
          1 + tt / 3, tol=2 * tt ** 2)
hks_mesh = lambda tt: (np.exp(-lam * tt) * Phi[s] ** 2).sum()  # noqa: E731
check(f"Text: mesh HKS at t = 0.1: {hks_mesh(0.1):.3f} vs exact {hks_exact(0.1):.3f} (within 4%)",
      abs(hks_mesh(0.1) / hks_exact(0.1) - 1) < 0.04)
check(f"Text: mesh HKS at t = 0.01 < h^2: {hks_mesh(0.01):.2f} vs exact {hks_exact(0.01):.2f} (unreliable)",
      abs(hks_mesh(0.01) / hks_exact(0.01) - 1) > 0.3)

# ---------------------------------------------------------------- Figure 31.4.5: functional maps
a_, b_ = 2.0, 0.7
VP, FP = dm.grid_mesh(np.linspace(0, a_, 41), np.linspace(0, b_, 15), pattern="alternate")


def eigP(W, kk=25):
    mm = dm.mass_voronoi(W, FP)
    l_, P_ = dm.generalized_eigs(dm.cotan_laplacian(W, FP), mm, kk)
    return l_, P_, mm


lamP, PhiP, mP = eigP(VP)
exact = np.sort([np.pi ** 2 * (i * i / a_ ** 2 + j * j / b_ ** 2) for i in range(20) for j in range(10)])[:25]
check("Fig 31.4.5: plate eigenvalues ~ pi^2 (m^2/4 + n^2/0.49): 2.466 (2.467), 9.849 (9.870), 20.058 (20.142)",
      [round(x_, 3) for x_ in lamP[1:4]] == [2.466, 9.849, 20.058] and np.allclose(lamP[1:4], exact[1:4], rtol=0.005))
# the rectangle's Neumann eigenvalues pi^2 (m^2/4 + n^2/0.49) are distinct at low frequency; the first coincidence
# (in units of pi^2/196: 49 m^2 + 400 n^2) is (20, 0) ~ (0, 7) at 100 pi^2, the 122nd and 123rd values
keys = sorted(49 * i * i + 400 * j * j for i in range(80) for j in range(30))
dup = min(k_ for k_, k2 in zip(keys, keys[1:]) if k_ == k2)
check("Fig 31.4.5: first repeated rectangle eigenvalue is 100 pi^2 = (20,0) ~ (0,7), values no. 122 and 123",
      dup == 19600 and keys.index(dup) == 121 and 49 * 400 == 19600 == 400 * 49)
tt2 = np.linspace(0, 3.0, 30001)
zz2 = 0.25 * np.sin(np.pi * tt2)
sarc = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(tt2) ** 2 + np.diff(zz2) ** 2))])
xa = np.interp(VP[:, 0], sarc, tt2)
Wbend = np.stack([xa, VP[:, 1], 0.25 * np.sin(np.pi * xa)], 1)
# isometry check: edge lengths change by less than 0.1%
E_ = dm.edges(FP)
lr = np.linalg.norm(Wbend[E_[:, 0]] - Wbend[E_[:, 1]], axis=1) / np.linalg.norm(VP[E_[:, 0]] - VP[E_[:, 1]], axis=1)
check(f"Fig 31.4.5(a): the bend changes edge lengths by at most {np.abs(lr - 1).max():.1e}", np.abs(lr - 1).max() < 1e-3)
Cs = []
for W in (Wbend, np.c_[1.1 * VP[:, 0], VP[:, 1], VP[:, 2]], np.c_[VP[:, 0] * (1 + 0.3 * VP[:, 0] / a_), VP[:, 1], VP[:, 2]]):
    l_, P_, mm = eigP(W)
    Cs.append(np.abs((P_.T @ (mm[:, None] * PhiP))[:20, :20]))
check("Fig 31.4.5(a): isometric bend: |C| = identity to 1e-3", np.abs(Cs[0] - np.eye(20)).max() < 1e-3)
B = Cs[1]
check("Fig 31.4.5(b): stretch: one entry ~ sqrt(1.1) = 1.049 per row and column, off the diagonal for some",
      ((B > 0.5).sum(0) == 1).all() and ((B > 0.5).sum(1) == 1).all() and np.allclose(B[B > 0.5], np.sqrt(1.1), atol=0.01)
      and (np.diag(B) < 0.5).sum() > 0)
check("Fig 31.4.5(c): non-uniform stretch: 56 entries above 0.1", (Cs[2] > 0.1).sum() == 56)
# HKS is (nearly) isometry invariant: flat vs bent plate, not for the stretch
lb, Pb, _ = eigP(Wbend, 60)
lf, Pf, _ = eigP(VP, 60)
ls_, Ps_, _ = eigP(np.c_[1.1 * VP[:, 0], VP[:, 1], VP[:, 2]], 60)
hk = lambda l_, P_, tt: (np.exp(-l_ * tt)[None, :] * P_ ** 2).sum(1)  # noqa: E731
rel_b = np.abs(hk(lb, Pb, 0.05) / hk(lf, Pf, 0.05) - 1).max()
rel_s = np.abs(hk(ls_, Ps_, 0.05) / hk(lf, Pf, 0.05) - 1).max()
check(f"Text: HKS at t = 0.05 (60 eigenpairs): bend changes it by {rel_b:.1e}, stretch by {rel_s:.2f}",
      rel_b < 1e-3 and rel_s > 0.05)
# Exercise 31.4.4: a square plate has repeated eigenvalues -> 2 x 2 rotation blocks are allowed
lsq = sorted(np.pi ** 2 * (i * i + j * j) for i in range(4) for j in range(4))
check("Exercise 31.4.4: on the unit square lambda(1,0) = lambda(0,1) = pi^2 (a repeated eigenvalue)",
      np.isclose(lsq[1], lsq[2]))

summary()
