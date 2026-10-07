"""Figure 29.3.3: size of the EWA error as a function of s/t_z and the off-axis angle (Section 29.3).

Isotropic Gaussian Sigma_c = s^2 I_3 at t0 = (tan theta, 0, 1), restricted to its 3-sigma ball (the standard normal
truncated at |y| <= 3). The mean and covariance of its image under the exact projection phi and under the
linearization phi(t0) + J (t - t0) are computed by deterministic quadrature (Gauss-Legendre in r and cos(polar
angle), uniform in the azimuth), not by Monte Carlo, so the numbers are reproducible to all printed digits.
(a) relative center offset |mean(exact) - mean(linear)| / (largest std of the linear footprint), theta = 15..60 deg.
    Dashed gray: leading-order prediction sqrt(c_2) (s/t_z) sin(theta), c_2 = E[y_1^2] = 0.918.
(b) relative covariance error |Cov(exact) - Cov(linear)|_F / |Cov(linear)|_F, theta = 0, 30, 60 deg.
    Dashed gray: leading-order on-axis prediction 3 c_4 (s/t_z)^2, c_4 = E[y_1^2 y_3^2] / E[y_1^2] = 0.839.
    (For the untruncated normal both constants would be 1.)
Self-checks: the quadrature is converged (a finer grid changes nothing at 1e-6); the curves reproduce the table of
Example 29.3.4 (checked again in verify/v-29-3-splatting-linearization.py).

Run from the project root: ``OMP_NUM_THREADS=4 PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-3-3-ewa-error.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def quad_nodes(nr=120, nc=60, nph=120):
    """nodes y and normalized weights w of the standard normal on R^3 truncated at |y| <= 3."""
    xr, wr = np.polynomial.legendre.leggauss(nr)
    r = 1.5 * (xr + 1.0)
    wr = 1.5 * wr * r ** 2 * np.exp(-r ** 2 / 2)
    xc, wc = np.polynomial.legendre.leggauss(nc)
    ph = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    R_, CT, PH = np.meshgrid(r, xc, ph, indexing="ij")
    ST = np.sqrt(1 - CT ** 2)
    Y = np.stack([R_ * ST * np.cos(PH), R_ * ST * np.sin(PH), R_ * CT], -1).reshape(-1, 3)
    W = (wr[:, None, None] * wc[None, :, None] * np.ones(nph)[None, None, :]).ravel()
    return Y, W / W.sum()


Y, W = quad_nodes()
c2 = float(W @ Y[:, 0] ** 2)                                  # E[y_1^2]
c4 = float(W @ (Y[:, 0] ** 2 * Y[:, 2] ** 2)) / c2            # E[y_1^2 y_3^2] / E[y_1^2]
assert abs(c2 - 0.9178) < 1e-4 and abs(c4 - 0.8388) < 1e-4


def errors(s, thd, Y=Y, W=W):
    t0 = np.array([np.tan(np.radians(thd)), 0.0, 1.0])
    x, y, z = t0
    J = np.array([[1 / z, 0, -x / z ** 2], [0, 1 / z, -y / z ** 2]])
    T = t0 + s * Y
    u = T[:, :2] / T[:, 2:3]
    ul = t0[:2] / t0[2] + s * Y @ J.T

    def moments(v):
        m = W @ v
        d = v - m
        return m, (W[:, None] * d).T @ d

    m, Cv = moments(u)
    ml, Cl = moments(ul)
    return (np.linalg.norm(m - ml) / np.sqrt(np.linalg.eigvalsh(Cl).max()),
            np.linalg.norm(Cv - Cl) / np.linalg.norm(Cl))


# convergence of the quadrature at the hardest point (largest s, largest angle)
Yf, Wf = quad_nodes(200, 80, 160)
assert np.allclose(errors(0.3, 60), errors(0.3, 60, Yf, Wf), rtol=1e-6)

ss = np.geomspace(0.01, 0.3, 14)
cols = {0: C["main"], 15: C["third"], 30: C["accent"], 45: C["normal"], 60: C["tangent"]}
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(wspace=0.32))
res = {th: np.array([errors(s, th) for s in ss]) for th in (0, 15, 30, 45, 60)}
# self-check against the table of Example 29.3.4
for (th, s), (e_c, e_v) in {(30, 0.1): (0.049, 0.034), (60, 0.1): (0.085, 0.058), (0, 0.1): (0.0, 0.026),
                            (0, 0.3): (0.0, 0.37), (60, 0.3): (0.34, 1.25)}.items():
    got = errors(s, th)
    assert abs(got[0] - e_c) <= 0.006 * max(1, 10 * e_c) and abs(got[1] / e_v - 1) < 0.02, (th, s, got)
for th in (15, 30, 45, 60):
    axa.loglog(ss, res[th][:, 0], "o-", color=cols[th], ms=3.5, lw=1.5, label=r"$\theta = %d^\circ$" % th)
    axa.loglog(ss, np.sqrt(c2) * ss * np.sin(np.radians(th)), color=C["aux"], lw=0.8, ls=(0, (3, 2)))
for th in (0, 30, 60):
    axb.loglog(ss, res[th][:, 1], "o-", color=cols[th], ms=3.5, lw=1.5, label=r"$\theta = %d^\circ$" % th)
axb.loglog(ss, 3 * c4 * ss ** 2, color=C["aux"], lw=0.8, ls=(0, (3, 2)), label=r"$3c_4\,(s/t_z)^2$")
for ax in (axa, axb):
    ax.axvline(0.1, color="#BBBBBB", lw=0.6)
    ax.set_xlabel(r"$s/t_z$ (Gaussian size / depth)")
    ax.tick_params(labelsize=9)
    ax.grid(True, which="major", color="#EEEEEE", lw=0.5)
axa.set_ylabel("relative center offset")
axb.set_ylabel("relative covariance error")
axa.set_title("(a) center offset", fontsize=11)
axb.set_title("(b) covariance error", fontsize=11)
axa.legend(fontsize=9, frameon=False, loc="upper left")
axb.legend(fontsize=9, frameon=False, loc="upper left")
axa.text(0.012, 0.0009, r"gray: $\sqrt{c_2}\,(s/t_z)\sin\theta$", fontsize=9, color="#555555")
axa.set_ylim(5e-4, 1)
axb.set_ylim(1e-4, 3)

dgfig.save(fig, __file__)
