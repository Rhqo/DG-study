"""Figure 30.4.5: which smoothness prior leaves a slanted plane alone? (Section 30.4)

Flatland, normalized image coordinate u in [-0.5, 0.5] (161 pixels). True surface: the plane z = 3 + x (45 deg),
d(u) = 3/(1 - u). Observed depth = d + N(0, 0.06^2) (seed 3044). Denoising by
    min_d  (1/2) sum_i w_i (d_i - d_obs,i)^2 + lambda R(d)
with three L1 priors, solved by iteratively reweighted least squares (300 iterations):
    TV on depth           R = sum |d_{i+1} - d_i|
    2nd order on depth    R = sum |d_{i+1} - 2 d_i + d_{i-1}|
    2nd order on 1/d      R = d_ref^2 sum |rho_{i+1} - 2 rho_i + rho_{i-1}|, rho = 1/d, d_ref = 3 (the depth at u = 0)
                          (solved in rho with data weights w_i = d_obs,i^4, so that the data term matches the depth one
                          to first order; the factor d_ref^2 puts the prior in depth units, since a small depth change
                          delta d changes rho by -delta d/d^2, so that equal lambda means comparable strength)
(a) lambda = 100: signed distance from the true plane of the reconstructed points X_i = d_i (u_i, 1) versus u
    (gray dots: the noisy observation).
(b) rms distance to the plane versus lambda (log-log); gray dashed: the noise level.
Self-checks: the ordering of the three priors at lambda = 100 (also in verify/v-30-4-depth-maps.py).

Run from the project root: ``OMP_NUM_THREADS=4 PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-4-5-smoothness-priors.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
n = 161
u = np.linspace(-0.5, 0.5, n)
dtrue = 3 / (1 - u)
dn = dtrue + 0.06 * np.random.default_rng(3044).normal(size=n)
D1 = np.diff(np.eye(n), axis=0)
D2 = np.diff(np.eye(n), n=2, axis=0)
DREF = 3.0


def irls_l1(y, w, D, lam, it=300, eps=1e-6):
    x = y.copy()
    W = np.diag(w)
    for _ in range(it):
        res = np.abs(D @ x)
        x = np.linalg.solve(W + lam * D.T @ np.diag(1 / np.maximum(res, eps)) @ D, W @ y)
    return x


def dist(d):
    return (u * d - d + 3) / np.sqrt(2)          # signed distance of (u d, d) from the line z = 3 + x


def solve(kind, lam):
    if kind == "tv":
        return irls_l1(dn, np.ones(n), D1, lam)
    if kind == "d2":
        return irls_l1(dn, np.ones(n), D2, lam)
    return 1 / irls_l1(1 / dn, dn ** 4, D2, DREF ** 2 * lam)


kinds = [("tv", "TV on depth", C["normal"]), ("d2", "2nd order on depth", C["tangent"]),
         ("rho2", r"2nd order on inverse depth", C["third"])]
res100 = {k: solve(k, 100.0) for k, _, _ in kinds}
rms = {k: np.sqrt(np.mean(dist(v) ** 2)) for k, v in res100.items()}
noise = np.sqrt(np.mean(dist(dn) ** 2))
assert rms["tv"] > 10 * noise and rms["d2"] > 5 * rms["rho2"] and rms["rho2"] < 0.25 * noise
print("lambda = 100:", {k: round(v, 4) for k, v in rms.items()}, "noise", round(noise, 4))

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 8.0), gridspec_kw=dict(hspace=0.38))
axa.plot(u, dist(dn), "o", color=C["aux"], ms=2.6, alpha=0.7, label="noisy depth")
for k, lab, col in kinds:
    axa.plot(u, dist(res100[k]), color=col, lw=2.2, label=lab + rf" (rms {rms[k]:.3f})")
axa.axhline(0, color=C["main"], lw=0.8)
axa.set_xlim(-0.5, 0.5)
axa.set_ylim(-1.5, 1.25)
axa.set_xlabel(r"normalized image coordinate $u$")
axa.set_ylabel("distance to the true plane")
axa.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="none", fontsize=10.5)
axa.set_title(r"(a) reconstructions at $\lambda = 100$", fontsize=13)

lams = np.geomspace(0.3, 1000, 12)
for k, lab, col in kinds:
    vals = [np.sqrt(np.mean(dist(solve(k, lm)) ** 2)) for lm in lams]
    axb.loglog(lams, vals, "o-", color=col, lw=2.0, ms=4.5, label=lab)
axb.axhline(noise, color=C["aux"], lw=1.2, ls=(0, (5, 3)))
axb.text(0.35, noise * 1.15, "noise level", color=C["aux"], fontsize=11)
axb.set_xlabel(r"regularization weight $\lambda$")
axb.set_ylabel("rms distance to the plane")
axb.legend(loc="upper left", frameon=False, fontsize=10.5)
axb.set_title(r"(b) only the inverse-depth prior gets better as $\lambda$ grows", fontsize=13)
dgfig.save(fig, __file__)
