"""Figure 28.1.3: numerical accuracy of two implementations of Log on SO(3) in float64 (Section 28.1).

For each angle theta (fixed unit axis n), the rotation Exp(theta n) is computed in 50-digit arithmetic (mpmath) and
rounded to float64. Its exact log is then computed again in 50 digits from the rounded matrix. The plot shows the
relative error |w_hat - w| / |w| of
  * black: the textbook formula theta = arccos((tr R - 1)/2), w = theta/(2 sin theta) (R - R^T)^vee,
  * blue:  the robust route R -> quaternion (Shepperd) -> 2 atan2(|v|, q0) v/|v| with a Taylor series near 0
           (the form used by Sophus).
(a) near theta = 0 (horizontal axis theta), (b) near theta = pi (horizontal axis pi - theta). Errors below 1e-17
are drawn at 1e-17.
Self-check: the robust error stays below 1e-14 everywhere; the textbook formula returns 0 (relative error 1) for
theta < 1e-8; its error exceeds 1e-6 for pi - theta <= 1e-6 and exceeds 1 (completely wrong) for pi - theta <= 1e-8.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-1-3-log-accuracy.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dglie as L  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
mp.mp.dps = 50
N_AXIS = np.array([0.48, -0.6, 0.64])
N_AXIS = N_AXIS / np.linalg.norm(N_AXIS)
FLOOR = 1e-17


def R_exact(th):
    nn = [mp.mpf(float(x)) for x in N_AXIS]
    s = mp.sqrt(sum(x * x for x in nn))
    nn = [x / s for x in nn]
    K = mp.matrix([[0, -nn[2], nn[1]], [nn[2], 0, -nn[0]], [-nn[1], nn[0], 0]])
    return mp.eye(3) + mp.sin(th) * K + (1 - mp.cos(th)) * K * K


def log_hp(Rf):
    """50-digit log of the float64 matrix Rf (angle by atan2, axis from the symmetric part near pi)."""
    R = mp.matrix(Rf.tolist())
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    a = mp.matrix([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    c = (tr - 1) / 2
    sn = mp.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    th = mp.atan2(sn, c)
    if th < mp.mpf("0.5"):
        return np.array([float(x * th / sn) for x in a])
    B = (R + R.T) / 2 - c * mp.eye(3)
    k = max(range(3), key=lambda i: B[i, i])
    v = mp.matrix([B[k, 0], B[k, 1], B[k, 2]])
    v = v / mp.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if sum(v[i] * a[i] for i in range(3)) < 0:
        v = -v
    return np.array([float(x * th) for x in v])


def errors(thetas):
    e_naive, e_rob = [], []
    for th in thetas:
        Rf = np.array(R_exact(th).tolist(), dtype=float)
        wt = log_hp(Rf)
        e_naive.append(np.linalg.norm(L.logSO3_naive(Rf) - wt) / np.linalg.norm(wt))
        e_rob.append(np.linalg.norm(L.logSO3(Rf) - wt) / np.linalg.norm(wt))
    return np.maximum(e_naive, FLOOR), np.maximum(e_rob, FLOOR)


x0 = np.logspace(-12, 0, 97)
n0, r0 = errors([mp.mpf(float(x)) for x in x0])
x1 = np.logspace(-12, 0, 97)
n1, r1 = errors([mp.pi - mp.mpf(float(x)) for x in x1])

assert max(r0.max(), r1.max()) < 1e-14
assert np.all(n0[x0 < 1e-8] > 0.99)
assert np.all(n1[x1 <= 1e-6] > 1e-6) and np.all(n1[x1 <= 1e-8] > 1.0)

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, x, en, er, lab, ttl in [(axs[0], x0, n0, r0, r"$\theta$", r"(a) near $\theta = 0$"),
                                (axs[1], x1, n1, r1, r"$\pi - \theta$", r"(b) near $\theta = \pi$")]:
    ax.loglog(x, en, color=C["main"], lw=1.8, label=r"textbook: $\arccos$ and $(R - R^{\mathsf{T}})^\vee$")
    ax.loglog(x, er, color=C["tangent"], lw=1.8, ls=(0, (5, 2)), label="robust: quaternion + atan2")
    ax.axhline(2.2e-16, color=C["aux"], lw=0.8, ls=":")
    ax.set_xlabel(lab)
    ax.set_title(ttl, fontsize=12)
    ax.set_xlim(1e-12, 1)
    ax.set_ylim(3e-18, 1e9)
    ax.grid(True, which="major", color="#E5E5E5", lw=0.5)
axs[0].set_ylabel(r"relative error $|\hat\omega - \omega| / |\omega|$")
axs[0].text(2e-12, 2e-14, r"machine $\varepsilon \approx 2.2\times 10^{-16}$", fontsize=10, color=C["aux"])
axs[0].annotate(r"$\arccos(1) = 0$: returns $\hat\omega = 0$" + "\n" + r"for $\theta \lesssim 10^{-8}$",
                xy=(3e-10, 1.0), xytext=(1.5e-11, 1e4), fontsize=10.5, arrowprops=dict(arrowstyle="->", lw=0.9))
axs[1].annotate("grows as " + r"$\theta \to \pi$", xy=(1e-6, 4.4e-5), xytext=(1e-4, 1e1), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", lw=0.9))
axs[1].annotate(r"$\arccos(-1) = \pi$, $\sin\hat\theta \approx 10^{-16}$:" + "\n" + r"error $> 10^{3}$",
                xy=(1e-10, 8e5), xytext=(1.5e-12, 1e-11), fontsize=10.5, arrowprops=dict(arrowstyle="->", lw=0.9))
axs[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=10, frameon=True, framealpha=0.95)

dgfig.save(fig, __file__)
