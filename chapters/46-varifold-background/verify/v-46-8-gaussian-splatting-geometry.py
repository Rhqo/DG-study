"""Verification for Section 46.8 (Gaussian Splatting as Geometry).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-8-gaussian-splatting-geometry.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(468)


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def plane_of(S, k=2):
    w, U = np.linalg.eigh(S)
    V = U[:, -k:]
    return V @ V.T


def rand_sym(n, size):
    E = rng.normal(size=(n, n))
    E = 0.5 * (E + E.T)
    return E * size / np.linalg.norm(E, 2)


# ---------------------------------------------------------------- (46.8.1): the plane of a covariance
for _ in range(5):
    R = rot(rng.normal(size=3), rng.uniform(0, math.pi))
    s = np.sort(rng.uniform(0.01, 0.5, size=3))[::-1]
    S = R @ np.diag(s ** 2) @ R.T
    r3 = R[:, 2]
    P = plane_of(S)
    ok = np.allclose(P, np.eye(3) - np.outer(r3, r3)) and np.allclose(P, np.outer(R[:, 0], R[:, 0]) + np.outer(R[:, 1], R[:, 1]))
    check(f"(46.8.1): top-2 eigenspace projector = r1 r1^T + r2 r2^T = I - r3 r3^T (s = {np.round(s, 3)})", ok)
    check("(46.8.1): P is a rank-2 projection (Proposition 46.3.2)", v.is_projection(P, 2, tol=1e-10))
S_eq = np.diag([0.09, 0.09, 0.0016])
check("(46.8.1): s1 = s2 still gives a well-defined plane", np.allclose(plane_of(S_eq), np.diag([1.0, 1.0, 0.0])))

# Example 46.8.1: a Gaussian on the unit sphere, and the same Gaussian tilted by 5 degrees
mu = np.array([0.0, 0.0, 1.0])
R0 = np.eye(3)
S0 = R0 @ np.diag(np.array([0.32, 0.20, 0.04]) ** 2) @ R0.T
P_true = np.eye(3) - np.outer(mu, mu)
close("Example 46.8.1: Sigma's plane is the tangent plane of the sphere", plane_of(S0), P_true, 1e-12)
Rt = rot([1, 0, 0], math.radians(5))
P_tilt = plane_of(Rt @ S0 @ Rt.T)
close("Example 46.8.1: tilt by 5 degrees: ||P - P_true|| = sin 5 = 0.0872, Frobenius 0.1233",
      [v.op_dist(P_tilt, P_true), v.fro_dist(P_tilt, P_true)], [0.0872, 0.1233], 5e-5)

# ---------------------------------------------------------------- Proposition 46.8.2: sensitivity of the plane
worst_ratio = 0.0
for _ in range(20):
    R = rot(rng.normal(size=3), rng.uniform(0, math.pi))
    s = np.array([0.3, 0.2, rng.uniform(0.01, 0.19)])
    S = R @ np.diag(s ** 2) @ R.T
    E = rand_sym(3, 1.0)
    h = 1e-7
    dP = (plane_of(S + h * E) - plane_of(S - h * E)) / (2 * h)
    # first-order formula: d r3 = sum_j (r_j^T E r3)/(s3^2 - s_j^2) r_j, dP = -(dr3 r3^T + r3 dr3^T)
    r = [R[:, k] for k in range(3)]
    dr3 = sum((r[j] @ E @ r[2]) / (s[2] ** 2 - s[j] ** 2) * r[j] for j in (0, 1))
    dP_formula = -(np.outer(dr3, r[2]) + np.outer(r[2], dr3))
    check("Prop 46.8.2: dP = -(dr3 r3^T + r3 dr3^T) with the first-order eigenvector formula", np.allclose(dP, dP_formula, atol=1e-6))
    close("Prop 46.8.2: ||dP|| = |dr3|", np.linalg.norm(dP_formula, 2), np.linalg.norm(dr3), 1e-12)
    worst_ratio = max(worst_ratio, np.linalg.norm(dP, 2) * (s[1] ** 2 - s[2] ** 2) / np.linalg.norm(E, 2))
check("Prop 46.8.2: ||dP|| <= ||E|| / (s2^2 - s3^2) (20 random cases)", worst_ratio <= 1 + 1e-6, f"max ratio {worst_ratio:.4f}")
# the bound is attained: E = r2 r3^T + r3 r2^T
s = np.array([0.3, 0.2, 0.1])
S = np.diag(s ** 2)
E = np.zeros((3, 3))
E[1, 2] = E[2, 1] = 1.0
h = 1e-7
dP = (plane_of(S + h * E) - plane_of(S - h * E)) / (2 * h)
close("Prop 46.8.2: equality for E = r2 r3^T + r3 r2^T: ||dP|| = 1/(s2^2 - s3^2) = 33.33", np.linalg.norm(dP, 2), 1 / (0.04 - 0.01), 1e-5)
close("Figure 46.8.1(b): bound at s3 = 0.01 is 25.06, at s3 = 0.19 is 256.4", [1 / (0.04 - 0.0001), 1 / (0.04 - 0.0361)], [25.063, 256.41], 5e-3)
close("Figure 46.8.1(a): 2D gaps s1^2 - s2^2 = 0.1575, 0.12, 0.0576, 0.0156",
      [0.16 - m ** 2 for m in (0.05, 0.2, 0.32, 0.38)], [0.1575, 0.12, 0.0576, 0.0156], 1e-12)

# ---------------------------------------------------------------- Examples 46.8.3-46.8.4: opacity is not mass
for a in (0.3, 0.5, 0.9):
    close(f"Example 46.8.3: two copies on one ray: 1 - (1 - a)^2 = 2a - a^2 (a = {a})", 1 - (1 - a) ** 2, 2 * a - a * a, 1e-15)
close("Example 46.8.3: a = 0.5 gives 0.75, not 1", 1 - 0.5 ** 2, 0.75, 1e-15)
s1, s2 = 0.3, 0.15
g = np.linspace(-1.6, 1.6, 481)
GX, GY = np.meshgrid(g, g)
PTS = np.stack([GX, GY], axis=-1)
dA = (g[1] - g[0]) ** 2
SIG = np.diag([s1 ** 2, s2 ** 2])


def foot(mu, S):
    d = PTS - np.asarray(mu)
    return np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, np.linalg.inv(S), d))


def coverage(items, a):
    T = np.ones(GX.shape)
    for mu_, S_ in items:
        T *= 1 - a * foot(mu_, S_)
    return float((1 - T).sum() * dA)


for a in (0.5, 0.9):
    c0 = coverage([((0, 0), SIG)], a)
    close(f"Example 46.8.3: one footprint covers a 2 pi s1 s2 (a = {a})", c0, a * 2 * math.pi * s1 * s2, 1e-6)
    close(f"Example 46.8.3: clone covers (2 - a/2) times as much (a = {a})", coverage([((0, 0), SIG), ((0, 0), SIG)], a) / c0,
          2 - a / 2, 1e-6)
c0 = coverage([((0, 0), SIG)], 0.5)
close("Example 46.8.3: clone ratios 1.75 (a = 0.5) and 1.55 (a = 0.9)", [2 - 0.25, 2 - 0.45], [1.75, 1.55], 1e-15)
SC = SIG / 1.6 ** 2
close("Example 46.8.4: split: sum of 1-sigma ellipse areas 2/1.6^2 = 0.78", 2 / 1.6 ** 2, 0.78125, 1e-15)
close("Example 46.8.4: split at mu +- 0.8 s1 r1: coverage 0.762 of the parent",
      coverage([((0.8 * s1, 0), SC), ((-0.8 * s1, 0), SC)], 0.5) / c0, 0.7623, 5e-4)
draws = []
rr = np.random.default_rng(4682)
for _ in range(400):
    x1, x2 = rr.multivariate_normal([0, 0], SIG, size=2)
    draws.append(coverage([(x1, SC), (x2, SC)], 0.5) / c0)
close("Example 46.8.4: 400 random splits: coverage between 0.684 and 0.781 (mean 0.754)", [min(draws), max(draws), np.mean(draws)],
      [0.6839, 0.7813, 0.7544], 5e-4)
check("Example 46.8.4: no split covers more than two far-apart children (0.78125)", max(draws) <= 2 / 2.56 + 1e-6)
close("Exercise 46.8.3: phi = sqrt2 keeps the sum of areas: 2/phi^2 = 1", 2 / math.sqrt(2) ** 2, 1.0, 1e-15)
close("Exercise 46.8.3: and the expected moment (Example 29.4.4): 1/phi^2 + 1/2 = 1", 1 / 2 + 1 / 2, 1.0, 0)
close("Exercise 46.8.2: a = 0.3: two copies 0.51 (not 0.6); ten copies 1 - 0.7^10 = 0.9718",
      [1 - 0.7 ** 2, 1 - 0.7 ** 10], [0.51, 0.97175], 5e-6)

# ---------------------------------------------------------------- Proposition 46.8.5: refinement conservation
for _ in range(10):
    q = rng.uniform(0.5, 2.0)
    mu0 = rng.normal(size=3)
    P0 = v.plane_proj(rng.normal(size=3))
    k = rng.integers(2, 5)
    w = rng.uniform(0.2, 1.0, size=k)
    qj = q * w / w.sum()
    off = rng.normal(size=(k, 3)) * 0.1
    off -= (qj[:, None] * off).sum(axis=0) / q          # recentre: sum q_j mu_j = q mu
    muj = mu0 + off
    Pj = np.array([v.plane_proj(rng.normal(size=3) * 0.1 + np.linalg.eigh(np.eye(3) - P0)[1][:, -1]) for _ in range(k)])
    check("Prop 46.8.5(a): mass and first moment conserved", abs(qj.sum() - q) < 1e-12 and np.allclose((qj[:, None] * muj).sum(0), q * mu0))
    tests = v.random_lipschitz_tests(8, seed=int(rng.integers(1000)), n=3)
    bound = float(np.sum(qj * (np.linalg.norm(muj - mu0, axis=1) + np.array([v.op_dist(Pk, P0) for Pk in Pj]))))
    gaps = [abs(v.atoms_integral(f, muj, Pj, qj) - q * float(f(mu0[None, :], P0[None])[0])) for f in tests]
    check("Prop 46.8.5(c): |sum q_j phi(mu_j, P_j) - q phi(mu, P)| <= sum q_j (|mu_j - mu| + ||P_j - P||)", max(gaps) <= bound + 1e-12)
# conservative split with inherited tangent: affine test functions do not change
q, mu0, P0 = 1.0, np.zeros(3), v.plane_proj(np.array([0.0, 0.0, 1.0]))
u = np.array([1.0, 0.0, 0.0])
muj, qj = np.array([0.05 * u, -0.05 * u]), np.array([0.5, 0.5])
lin = lambda x: 0.3 + x @ np.array([0.2, -0.5, 0.1])
close("Exercise 46.8.5: conservative split: affine phi unchanged", (qj * lin(muj)).sum() - q * lin(mu0[None])[0], 0.0, 1e-15)
kink = lambda x: np.linalg.norm(x - mu0, axis=-1)
close("Exercise 46.8.5: phi = |x - mu| (kink at mu): change 0.05 = the bound sum q_j |mu_j - mu|",
      [(qj * kink(muj)).sum() - q * kink(mu0[None])[0], (qj * np.linalg.norm(muj - mu0, axis=1)).sum()], [0.05, 0.05], 1e-15)
x0 = np.array([0.0, 0.5, 0.0])
far = lambda x: np.linalg.norm(x - x0, axis=-1)
close("Exercise 46.8.5: phi = |x - x0|, x0 at distance 0.5 across: change sqrt(0.05^2 + 0.5^2) - 0.5 = 0.002494",
      (qj * far(muj)).sum() - q * far(mu0[None])[0], math.sqrt(0.0025 + 0.25) - 0.5, 1e-15)
close("Exercise 46.8.5: second-order estimate (1/2)(1/0.5) sum q_j |mu_j - mu|^2 = 0.0025", 0.5 * (1 / 0.5) * (qj * 0.05 ** 2).sum(), 0.0025, 1e-15)
# Exercise 46.8.4: sum q_j P_j = q P with rank-2 projections forces P_j = P
P_a, P_b = v.plane_proj(np.array([0, 0, 1.0])), v.plane_proj(np.array([0, 0.2, 1.0]))
avg = 0.5 * P_a + 0.5 * P_b
check("Exercise 46.8.4: the average of two different planes is not a projection (it lies inside, not on G_{2,3})",
      not v.is_projection(avg, 2, tol=1e-6))
n = np.array([0, 0, 1.0])
close("Exercise 46.8.4: n^T (sum q_j P_j) n = sum q_j |P_j n|^2 > 0 when some P_j != P",
      n @ avg @ n, 0.5 * np.linalg.norm(P_b @ n) ** 2, 1e-15)

# ---------------------------------------------------------------- Example 46.8.6: Gaussians tiling the unit sphere
def sph(th, ph):
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


# cell-sum and phi* integrals use Gauss-Legendre on the four quarters of each cell: |x - mu_i| has a cone
# kink at the cell centre, and 12 nodes across the whole cell overestimated the integrals by ~0.05% (review).
expect = {4: (32, 0.5412, 0.5210, 13.348, 6.6506, 6.6454), 8: (128, 0.2759, 0.2733, 6.901, 3.3893, 3.3888),
          16: (512, 0.1386, 0.1383, 3.480, 1.7025, 1.7024)}
for nt, (cnt, h_e, eT_e, sup_e, cell_e, low_e) in expect.items():
    npb = 2 * nt
    tb, pb = np.linspace(0, np.pi, nt + 1), np.linspace(0, 2 * np.pi, npb + 1)
    atoms = []
    for a in range(nt):
        for b in range(npb):
            atoms.append((a, b, sph(0.5 * (tb[a] + tb[a + 1]), 0.5 * (pb[b] + pb[b + 1])),
                          (math.cos(tb[a]) - math.cos(tb[a + 1])) * (pb[b + 1] - pb[b])))
    M = np.array([x[2] for x in atoms])
    Q = np.array([x[3] for x in atoms])
    xg12, wg12 = v.gauss_legendre(12)
    xg, wg = np.concatenate([0.5 * xg12, 0.5 + 0.5 * xg12]), np.concatenate([0.5 * wg12, 0.5 * wg12])
    hh = eT = cell = low = 0.0
    for a, b, m_, ar in atoms:
        th, ph = tb[a] + (tb[a + 1] - tb[a]) * xg, pb[b] + (pb[b + 1] - pb[b]) * xg
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        W = np.outer(wg, wg) * (tb[a + 1] - tb[a]) * (pb[b + 1] - pb[b]) * np.sin(TH)
        X = sph(TH, PH)
        dist = np.linalg.norm(X - m_, axis=-1)
        sang = np.sqrt(np.clip(1 - np.einsum("...i,i->...", X, m_) ** 2, 0, None))
        TF, PF = np.meshgrid(np.linspace(tb[a], tb[a + 1], 9), np.linspace(pb[b], pb[b + 1], 9), indexing="ij")
        XF = sph(TF, PF)
        hh = max(hh, np.linalg.norm(XF - m_, axis=-1).max())
        eT = max(eT, np.sqrt(np.clip(1 - np.einsum("...i,i->...", XF, m_) ** 2, 0, None)).max())
        cell += np.sum(W * (dist + sang))
        D = np.linalg.norm(X[..., None, :] - M[None, None, :, :], axis=-1)
        Sg = np.sqrt(np.clip(1 - np.einsum("...i,ki->...k", X, M) ** 2, 0, None))
        low += np.sum(W * np.minimum(1, (D + Sg).min(axis=-1)))
    check(f"Example 46.8.6 (n_theta = {nt}): {cnt} atoms, masses = exact cell areas sum to 4 pi", len(atoms) == cnt and abs(Q.sum() - 4 * math.pi) < 1e-12)
    close(f"Example 46.8.6 (n_theta = {nt}): h, eps_T", [hh, eT], [h_e, eT_e], 5e-4)
    close(f"Example 46.8.6 (n_theta = {nt}): sup-form bound 4 pi (h + eps_T)", 4 * math.pi * (hh + eT), sup_e, 5e-3)
    close(f"Example 46.8.6 (n_theta = {nt}): cell-sum upper bound and phi* lower bound", [cell, low], [cell_e, low_e], 5e-4)
    check(f"Example 46.8.6 (n_theta = {nt}): lower <= upper, within 0.1 percent", low <= cell + 1e-12 and (cell - low) / cell < 1e-3)
# the 1-sigma check on the identity ||P - Q||_op = sin(angle between normals) for planes in R^3
for _ in range(5):
    n1, n2 = rng.normal(size=3), rng.normal(size=3)
    n1, n2 = n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2)
    close("Example 46.8.6: ||P(n1) - P(n2)|| = sqrt(1 - <n1, n2>^2)", v.op_dist(v.plane_proj(n1), v.plane_proj(n2)),
          math.sqrt(1 - (n1 @ n2) ** 2), 1e-12)

# ---------------------------------------------------------------- Exercise 46.8.1
R30 = rot([1, 0, 0], math.radians(30))
S30 = R30 @ np.diag([0.09, 0.04, 0.0016]) @ R30.T
n30 = R30[:, 2]
close("Exercise 46.8.1: normal (0, -sin 30, cos 30) up to sign", abs(n30 @ np.linalg.eigh(S30)[1][:, 0]), 1.0, 1e-12)
close("Exercise 46.8.1: n = (0, -0.5, 0.8660)", n30, [0, -0.5, math.sqrt(3) / 2], 1e-12)
close("Exercise 46.8.1: distance to the horizontal plane sin 30 = 0.5", v.op_dist(plane_of(S30), v.plane_proj(np.array([0, 0, 1.0]))), 0.5, 1e-12)

summary()
