"""Verification for Section 29.1 (A Gaussian Is a Metric).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/verify/v-29-1-gaussian-metric.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(291)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rot2(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def random_rotation(rng):
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rr)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# ---------------------------------------------------------------- (29.1.1)-(29.1.3): G(x) = exp(-|x - mu|_Sigma^2 / 2)
R = random_rotation(rng)
s = np.array([2.0, 0.7, 0.1])
Sigma = R @ np.diag(s ** 2) @ R.T
mu = np.array([0.3, -1.0, 2.0])
x = mu + np.array([0.5, 0.2, -0.1])
G = np.exp(-0.5 * (x - mu) @ np.linalg.solve(Sigma, x - mu))
# in the eigenbasis: |v|_Sigma^2 = sum (r_i . v)^2 / s_i^2
v = x - mu
close("(29.1.3): |v|_Sigma^2 = sum <r_i, v>^2 / s_i^2", (v @ np.linalg.solve(Sigma, v)),
      sum((R[:, i] @ v) ** 2 / s[i] ** 2 for i in range(3)), tol=1e-10)
close("(29.1.1): G = exp(-|x-mu|_Sigma^2/2)", G, np.exp(-0.5 * sum((R[:, i] @ v) ** 2 / s[i] ** 2 for i in range(3))),
      tol=1e-12)
# the point mu + k s_i r_i lies on the level set |x - mu|_Sigma = k
for k in (1, 2, 3):
    for i in range(3):
        p = k * s[i] * R[:, i]
        close(f"(29.1.3): mu + {k} s_{i+1} r_{i+1} is on the level set k = {k}", np.sqrt(p @ np.linalg.solve(Sigma, p)), k,
              tol=1e-12)
# G on the level sets k = 1, 2, 3 (table in the text)
close("table: G at k = 1 is 0.607", round(np.exp(-0.5), 3), 0.607, tol=0)
close("table: G at k = 2 is 0.135", round(np.exp(-2.0), 3), 0.135, tol=0)
close("table: G at k = 3 is 0.011", round(np.exp(-4.5), 3), 0.011, tol=0)

# symbolic: the 1-sigma ellipsoid of Sigma = R diag(s^2) R^T has semi-axes s_i along r_i (2D symbolic check)
th = sp.symbols("theta", real=True)
s1, s2 = sp.symbols("s1 s2", positive=True)
Rs = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
Ss = Rs * sp.diag(s1 ** 2, s2 ** 2) * Rs.T
for i, si in enumerate((s1, s2)):
    p = si * Rs[:, i]
    sym_equal(f"(29.1.3) symbolic: p = s_{i+1} r_{i+1} has |p|_Sigma = 1", (p.T * Ss.inv() * p)[0, 0], 1)

# ---------------------------------------------------------------- Figure 29.1.2 / Example: affine image x = mu + R S y
mu2 = np.array([1.0, 0.5])
R2 = rot2(30.0)
s2v = np.array([2.0, 0.7])
Sig2 = R2 @ np.diag(s2v ** 2) @ R2.T
Y = rng.normal(size=(400000, 2))
X = mu2 + Y @ (R2 @ np.diag(s2v)).T
close("(29.1.4): samples of mu + R S y have covariance R S S^T R^T (MC, tol 0.02)", np.cov(X.T), Sig2, tol=0.02)
close("(29.1.4): samples have mean mu (MC, tol 0.01)", X.mean(0), mu2, tol=0.01)
# the unit circle maps onto the 1-sigma ellipse
ang = np.linspace(0, 2 * np.pi, 50)
circ = np.stack([np.cos(ang), np.sin(ang)], 1)
E = circ @ (R2 @ np.diag(s2v)).T
close("Figure 29.1.2: R S (unit circle) is the 1-sigma ellipse", np.einsum("ni,ij,nj->n", E, np.linalg.inv(Sig2), E),
      np.ones(50), tol=1e-12)
# Figure 29.1.2 values
close("Figure 29.1.2: Sigma = [[3.12, 1.52], [1.52, 1.37]]", Sig2, np.array([[3.1225, 1.5199], [1.5199, 1.3675]]), tol=1e-4)

# ---------------------------------------------------------------- Figure 29.1.1: same Euclidean distance, different Mahalanobis
r1, r2v = R2[:, 0], R2[:, 1]
p = mu2 + 2.0 * r1
q = mu2 + 2.0 * r2v
dm = lambda z: np.sqrt((z - mu2) @ np.linalg.solve(Sig2, z - mu2))
close("Figure 29.1.1: |p - mu| = |q - mu| = 2", [np.linalg.norm(p - mu2), np.linalg.norm(q - mu2)], [2, 2], tol=1e-12)
close("Figure 29.1.1: |p - mu|_Sigma = 1", dm(p), 1.0, tol=1e-12)
close("Figure 29.1.1: |q - mu|_Sigma = 2/0.7 = 2.857", dm(q), 2 / 0.7, tol=1e-12)
close("Figure 29.1.1: G(p) = 0.607", round(np.exp(-0.5 * dm(p) ** 2), 3), 0.607, tol=0)
close("Figure 29.1.1: G(q) = 0.017", round(np.exp(-0.5 * dm(q) ** 2), 3), 0.017, tol=0)

# ---------------------------------------------------------------- (29.1.5)-(29.1.6): transformation law
A = rng.normal(size=(3, 3)) + 2 * np.eye(3)
b = rng.normal(size=3)
SigP = A @ Sigma @ A.T
muP = A @ mu + b
Ainv = np.linalg.inv(A)
# G'(x') = G(A^{-1}(x' - b)) equals the Gaussian with (A mu + b, A Sigma A^T)
xp = rng.normal(size=3)
lhs = np.exp(-0.5 * (Ainv @ (xp - b) - mu) @ np.linalg.solve(Sigma, Ainv @ (xp - b) - mu))
rhs = np.exp(-0.5 * (xp - muP) @ np.linalg.solve(SigP, xp - muP))
close("(29.1.5): G(A^{-1}(x' - b)) is the Gaussian (A mu + b, A Sigma A^T)", lhs, rhs, tol=1e-12)
close("(29.1.6): (A Sigma A^T)^{-1} = A^{-T} Sigma^{-1} A^{-1}", np.linalg.inv(SigP), Ainv.T @ np.linalg.inv(Sigma) @ Ainv,
      tol=1e-8)
vv = rng.normal(size=3)
close("(29.1.6): |A v|_{Sigma'} = |v|_Sigma (A is an isometry)", (A @ vv) @ np.linalg.solve(SigP, A @ vv),
      vv @ np.linalg.solve(Sigma, vv), tol=1e-9)
# projection (non-invertible) of a Gaussian: marginal covariance = P Sigma P^T
P = np.array([[1.0, 0, 0], [0, 1.0, 0]])
Z = rng.normal(size=(300000, 3)) @ (R @ np.diag(s)).T
close("(29.1.5) for a projection P: covariance of P x is P Sigma P^T (MC, tol 0.02)", np.cov((Z @ P.T).T),
      P @ Sigma @ P.T, tol=0.02)

# Exercise 29.1.3: change of units m -> cm (A = 100 I)
SigM = np.diag([0.04, 0.01, 1e-4])
SigCm = 100.0 ** 2 * SigM
check("Exercise 29.1.3: Sigma scales by 10^4", np.allclose(SigCm, np.diag([400.0, 100.0, 1.0])))
check("Exercise 29.1.3: Sigma^{-1} scales by 10^{-4}", np.allclose(np.linalg.inv(SigCm), 1e-4 * np.linalg.inv(SigM)))
vm = np.array([0.1, 0.0, 0.0])
close("Exercise 29.1.3: Mahalanobis distance of 10 cm along r_1 is 0.5 in both units",
      [np.sqrt(vm @ np.linalg.solve(SigM, vm)), np.sqrt((100 * vm) @ np.linalg.solve(SigCm, 100 * vm))], [0.5, 0.5], tol=1e-12)

# ---------------------------------------------------------------- Shortest axis = normal (PCA principle)
# a flat Gaussian: the eigenvector of the smallest eigenvalue is r_3, and it minimizes sum <x_j - c, n>^2 over samples
lam, V = np.linalg.eigh(Sigma)
close("normal: smallest eigenvalue of Sigma is s_3^2", lam[0], s[2] ** 2, tol=1e-12)
close("normal: its eigenvector is +-r_3", abs(V[:, 0] @ R[:, 2]), 1.0, tol=1e-12)
samples = mu + rng.normal(size=(200000, 3)) @ (R @ np.diag(s)).T
C = np.cov(samples.T)
lamC, VC = np.linalg.eigh(C)
check("normal: PCA of samples recovers r_3 within 0.5 degree",
      np.degrees(np.arccos(min(1.0, abs(VC[:, 0] @ R[:, 2])))) < 0.5)
# Rayleigh quotient: n^T C n is minimized by e_min
ns = rng.normal(size=(2000, 3))
ns /= np.linalg.norm(ns, axis=1, keepdims=True)
check("normal: n^T C n >= lambda_min for random unit n", np.all(np.einsum("ni,ij,nj->n", ns, C, ns) >= lamC[0] - 1e-12))

# Exercise 29.1.1: Sigma = R_z(30) diag(4, 1, 0.01) R_z(30)^T
Rz = rot_z(30.0)
SigE = Rz @ np.diag([4.0, 1.0, 0.01]) @ Rz.T
lamE, VE = np.linalg.eigh(SigE)
close("Exercise 29.1.1: semi-axes of the 1-sigma ellipsoid are 2, 1, 0.1", np.sort(np.sqrt(lamE))[::-1], [2.0, 1.0, 0.1],
      tol=1e-12)
close("Exercise 29.1.1: normal is +-e_3", abs(VE[:, 0] @ np.array([0, 0, 1.0])), 1.0, tol=1e-12)
close("Exercise 29.1.1: long axis is (cos30, sin30, 0)", abs(VE[:, 2] @ np.array([np.cos(np.pi / 6), np.sin(np.pi / 6), 0])),
      1.0, tol=1e-12)
w = np.array([0.0, 0.0, 0.3])
close("Exercise 29.1.1: |0.3 e_3|_Sigma = 3", np.sqrt(w @ np.linalg.solve(SigE, w)), 3.0, tol=1e-12)
w2 = 2.0 * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6), 0])
close("Exercise 29.1.1: |2 r_1|_Sigma = 1", np.sqrt(w2 @ np.linalg.solve(SigE, w2)), 1.0, tol=1e-12)

# Exercise 29.1.5: needle Gaussian s = (1, 0.01, 0.01): the two smallest eigenvalues coincide
lamN = np.linalg.eigvalsh(np.diag([1.0, 1e-4, 1e-4]))
check("Exercise 29.1.5: needle has a repeated smallest eigenvalue", abs(lamN[0] - lamN[1]) < 1e-15)

# ---------------------------------------------------------------- Screen footprint (29.1.7): eigenvalues of a 2x2 covariance
a_, b_, c_ = sp.symbols("a b c", real=True)
M2 = sp.Matrix([[a_, b_], [b_, c_]])
mid = (a_ + c_) / 2
det = a_ * c_ - b_ ** 2
ev = sorted(M2.eigenvals().keys(), key=lambda e: sp.default_sort_key(e))
sym_equal("(29.1.7): eigenvalues = mid +- sqrt(mid^2 - det)",
          sp.Matrix(sorted([mid + sp.sqrt(mid ** 2 - det), mid - sp.sqrt(mid ** 2 - det)], key=sp.default_sort_key)),
          sp.Matrix(ev), {a_: (1, 3), b_: (-0.5, 0.5), c_: (1, 3)})

# Figure 29.1.4: 2D Gaussian with 3 sqrt(lam1) = 60 px, 3 sqrt(lam2) = 10 px, rotated by 30 degrees
l1, l2 = 20.0 ** 2, (10.0 / 3) ** 2
Sig2d = rot2(30.0) @ np.diag([l1, l2]) @ rot2(30.0).T
m = 0.5 * np.trace(Sig2d)
dd = np.linalg.det(Sig2d)
lmax = m + np.sqrt(m * m - dd)
close("Figure 29.1.4: lambda_max from (29.1.7) is 400", lmax, 400.0, tol=1e-9)
check("Figure 29.1.4: radius ceil(3 sqrt(lambda_max)) = 60", int(np.ceil(3 * np.sqrt(lmax) - 1e-9)) == 60)
close("Figure 29.1.4: circle / ellipse area ratio = sqrt(l1/l2) = 6", (np.pi * 9 * lmax) / (np.pi * 9 * np.sqrt(l1 * l2)),
      6.0, tol=1e-9)
# tiles: 16-px tiles, Gaussian centered at (130, 100) px. Count tiles in the square bbox vs tiles meeting the 3-sigma ellipse
center = np.array([130.0, 100.0])
T = 16
r = 60
xmin, xmax = int((center[0] - r) // T), int((center[0] + r - 1) // T)
ymin, ymax = int((center[1] - r) // T), int((center[1] + r - 1) // T)
n_square = (xmax - xmin + 1) * (ymax - ymin + 1)
Sinv = np.linalg.inv(Sig2d)
hit = 0
for tx in range(xmin, xmax + 1):
    for ty in range(ymin, ymax + 1):
        gx, gy = np.meshgrid(np.linspace(tx * T, (tx + 1) * T, 33), np.linspace(ty * T, (ty + 1) * T, 33))
        d = np.stack([gx - center[0], gy - center[1]], -1)
        if np.min(np.einsum("...i,ij,...j->...", d, Sinv, d)) <= 9.0:
            hit += 1
print(f"tiles: square {n_square}, ellipse {hit}")
check("Figure 29.1.4: the square bbox touches 64 tiles", n_square == 64)
check("Figure 29.1.4: the 3-sigma ellipse meets 19 tiles", hit == 19)

# Mahalanobis cutoff of the rasterizer: alpha G < 1/255 is skipped -> k^2 <= 2 ln(255 alpha) < 2 ln 255
close("cutoff: sqrt(2 ln 255) = 3.33", round(np.sqrt(2 * np.log(255)), 2), 3.33, tol=0)
check("cutoff: sqrt(2 ln(255 alpha)) < 3.33 for alpha < 1", all(np.sqrt(2 * np.log(255 * al)) < 3.33 for al in (0.5, 0.9, 0.99)))
check("box: aspect ratio 10 -> 90% of the circle lies outside the ellipse", abs((1 - 1 / 10) - 0.9) < 1e-12)

# insight box: first-order perturbation of r_3, delta r_3 = sum_j (r_j^T dS r_3)/(s_3^2 - s_j^2) r_j
dS = 1e-6 * rng.normal(size=(3, 3))
dS = dS + dS.T
lam0, V0 = np.linalg.eigh(Sigma)          # ascending: index 0 is s_3^2
lam1, V1 = np.linalg.eigh(Sigma + dS)
r3, r3n = V0[:, 0], V1[:, 0] * np.sign(V1[:, 0] @ V0[:, 0])
pred = sum((V0[:, j] @ dS @ r3) / (lam0[0] - lam0[j]) * V0[:, j] for j in (1, 2))
close("insight: delta r_3 = sum (r_j^T dSigma r_3)/(s_3^2 - s_j^2) r_j (first order)", r3n - r3, pred, tol=1e-9)

# Exercise 29.1.4: marginal metric vs. block of Sigma^{-1}
SigX = np.array([[1.0, 0, 0.9], [0, 1.0, 0], [0.9, 0, 1.0]])
check("Exercise 29.1.4: example Sigma is SPD", np.all(np.linalg.eigvalsh(SigX) > 0))
close("Exercise 29.1.4: P Sigma P^T = I_2", P @ SigX @ P.T, np.eye(2), tol=1e-15)
close("Exercise 29.1.4: (Sigma^{-1})_{11} = 1/0.19 = 5.26", np.linalg.inv(SigX)[0, 0], 1 / 0.19, tol=1e-12)
Sd = np.diag([4.0, 1.0, 0.01])
close("Exercise 29.1.4: diagonal case, block of Sigma^{-1} = (P Sigma P^T)^{-1}", np.linalg.inv(Sd)[:2, :2],
      np.linalg.inv(P @ Sd @ P.T), tol=1e-12)
# conditional precision = block of the full precision (Schur complement identity)
Sxx, Sxz, Szz = SigX[:2, :2], SigX[:2, 2:], SigX[2:, 2:]
cond = Sxx - Sxz @ np.linalg.inv(Szz) @ Sxz.T
close("Exercise 29.1.4: block of Sigma^{-1} = inverse of the conditional covariance", np.linalg.inv(SigX)[:2, :2],
      np.linalg.inv(cond), tol=1e-12)

summary()
