"""Figure 29.3.2: exact pushforward vs. EWA for small/large Gaussians on/off the optical axis (Section 29.3).

Isotropic Gaussian Sigma_c = s^2 I_3 centered at t0 = (tan theta, 0, 1) (depth t_z = 1, f = 1).
Panels: (a) s/t_z = 0.03, theta = 0; (b) s/t_z = 0.03, theta = 45deg; (c) s/t_z = 0.25, theta = 0; (d) s/t_z = 0.25,
theta = 45deg. Axes: (u - u0)/(s/t_z), i.e. image coordinates centered at u0 = phi(t0) in units of the on-axis
footprint std, the same in all panels.
Gray dots: 2500 samples of the 3D Gaussian (truncated at 3 sigma, seed 7) pushed through the exact projection phi.
Blue: EWA 2-sigma ellipse (center u0, covariance J Sigma_c J^T). Orange dashed: exact perspective image (outline) of the
3D 2-sigma ellipsoid, computed from the dual quadric. Orange x: mean of the exact pushforward (from 400k samples).
Self-checks: projected surface points of the 2-sigma ellipsoid lie inside the orange outline and touch it.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-3-2-ewa-grid.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
rng = np.random.default_rng(7)


def phi(T):
    return T[:, :2] / T[:, 2:3]


def Jm(t0):
    x, y, z = t0
    return np.array([[1 / z, 0, -x / z ** 2], [0, 1 / z, -y / z ** 2]])


def silhouette(t0, Sc, k):
    A = np.linalg.inv(Sc) / k ** 2
    Q = np.zeros((4, 4))
    Q[:3, :3] = A; Q[:3, 3] = -A @ t0; Q[3, :3] = -t0 @ A; Q[3, 3] = t0 @ A @ t0 - 1
    Cm = np.linalg.inv(np.linalg.inv(Q)[:3, :3])
    M, m, c = Cm[:2, :2], Cm[:2, 2], Cm[2, 2]
    if np.linalg.eigvalsh(M)[0] < 0:
        M, m, c = -M, -m, -c
    x0 = -np.linalg.solve(M, m)
    kap = m @ np.linalg.solve(M, m) - c
    return x0, kap * np.linalg.inv(M) / k ** 2


def ellipse_pts(c, B, k, n=300):
    w, V = np.linalg.eigh(B)
    t = np.linspace(0, 2 * np.pi, n)
    return c + (np.stack([np.cos(t), np.sin(t)], 1) * (k * np.sqrt(w))) @ V.T


Ybig = rng.normal(size=(1600000, 3))
Ybig = Ybig[(Ybig ** 2).sum(1) <= 9][:400000]
Ydots = Ybig[:2500]
cases = [(0.03, 0), (0.03, 45), (0.25, 0), (0.25, 45)]
fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6), sharex=True, sharey=True)
for ax, (s, thd), lab in zip(axes.ravel(), cases, "abcd"):
    t0 = np.array([np.tan(np.radians(thd)), 0.0, 1.0])
    Sc = s * s * np.eye(3)
    u0 = phi(t0[None])[0]
    sc = s / t0[2]
    U = (phi(t0 + s * Ydots) - u0) / sc
    ax.plot(U[:, 0], U[:, 1], ".", color="#888888", ms=1.6, alpha=0.5, zorder=1, rasterized=True)
    E = Jm(t0) @ Sc @ Jm(t0).T
    ewa = (ellipse_pts(u0, E, 2) - u0) / sc
    ax.plot(*ewa.T, color=C["tangent"], lw=1.8, zorder=3)
    x0, B = silhouette(t0, Sc, 2)
    d = rng.normal(size=(5000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    up = phi(t0 + 2 * s * d)
    val = np.einsum("ni,ij,nj->n", up - x0, np.linalg.inv(B), up - x0)
    assert val.max() <= 4 + 1e-6 and val.max() > 4 - 1e-3
    sil = (ellipse_pts(x0, B, 2) - u0) / sc
    ax.plot(*sil.T, color=C["accent"], lw=1.8, ls=(0, (4, 2.5)), zorder=4)
    mean_ex = (phi(t0 + s * Ybig).mean(0) - u0) / sc
    ax.plot(0, 0, "+", color=C["tangent"], ms=10, mew=1.8, zorder=5)
    ax.plot(*mean_ex, "x", color="#B07700", ms=8, mew=1.8, zorder=6)
    ax.set_title(f"({lab}) " + r"$s/t_z = %.2f$, $\theta = %d^\circ$" % (s, thd), fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 5.3)
    ax.set_ylim(-3.4, 3.4)
    ax.axhline(0, color="#DDDDDD", lw=0.5, zorder=0)
    ax.axvline(0, color="#DDDDDD", lw=0.5, zorder=0)
    ax.tick_params(labelsize=9)
for ax in axes[1]:
    ax.set_xlabel(r"$(u^1 - u_0^1)\,/\,(s/t_z)$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$(u^2 - u_0^2)\,/\,(s/t_z)$")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color="#888888", marker=".", ls="none", ms=6, label="exact samples"),
           Line2D([], [], color=C["tangent"], lw=1.8, label=r"EWA $2\sigma$ ellipse ($+$: $u_0$)"),
           Line2D([], [], color=C["accent"], lw=1.8, ls=(0, (4, 2.5)), label=r"exact image of $2\sigma$ ellipsoid"),
           Line2D([], [], color="#B07700", marker="x", ls="none", ms=7, mew=1.8, label="exact mean")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.subplots_adjust(hspace=0.28, wspace=0.08)

dgfig.save(fig, __file__)
