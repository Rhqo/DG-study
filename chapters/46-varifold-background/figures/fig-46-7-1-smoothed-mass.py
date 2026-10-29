"""Figure 46.7.1: smoothing the mass of a curve and of a point cloud by convolution.

Kernel rho_eps(x) = eps^-2 c rho(|x|/eps), rho(r) = exp(-1/(1 - r^2)) on [0, 1), c normalising the integral over R^2 to 1.
(a), (b) ||V|| * rho_eps for the point cloud of N = 24 equally spaced atoms on the unit circle (mass 2 pi / N each),
eps = 0.15 (beads) and eps = 0.5 (a ring).
(c) eps (||V|| * rho_eps) evaluated on the circle, as a function of the angle, for the same point cloud and eps = 0.15, 0.3,
0.5; the grey line is C_rho = 0.9517, the limit for the smooth circle (Proposition 46.7.2(c)).
(d) (||V|| * rho_eps)(s, 0) for the smooth unit circle across the curve, eps = 0.1, 0.2, 0.4: a bump of width 2 eps and
height about C_rho / eps.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-7-1-smoothed-mass.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
N = 24
t, X, P, m = v.circle_atoms(N)
C_RHO = v.normalisation(v.rho_exp, 2) * v.kernel_constant(v.rho_exp, 1)
assert abs(C_RHO - 0.9517) < 5e-5


def cloud_mass(Y, eps):
    """(||V|| * rho_eps)(y) = sum_i m_i rho_eps(y - x_i) for points Y of shape (k, 2)."""
    return np.array([np.sum(m * v.kernel_eps(y[None, :] - X, eps)) for y in Y])


def circle_mass(Y, eps):
    return np.array([v.integrate_1d(lambda tt: v.kernel_eps(y[None, :] - v.circle(tt), eps), 0, 2 * np.pi, n=32, pieces=256)
                     for y in Y])


# self-checks: total mass preserved, the smooth-circle limit, support of width 2 eps
g = np.linspace(-1.7, 1.7, 341)
GX, GY = np.meshgrid(g, g)
grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
dA = (g[1] - g[0]) ** 2
for eps in (0.15, 0.5):
    tot = cloud_mass(grid, eps).sum() * dA
    assert abs(tot - 2 * math.pi) < 2e-3, tot
peak = circle_mass(np.array([[1.0, 0.0]]), 0.1)[0]
assert abs(0.1 * peak - C_RHO) < 1e-3
assert circle_mass(np.array([[1.0 + 0.1001, 0.0]]), 0.1)[0] == 0.0

fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2), gridspec_kw={"hspace": 0.45, "wspace": 0.5})
gs = np.linspace(-1.6, 1.6, 241)
HX, HY = np.meshgrid(gs, gs)
hgrid = np.stack([HX.ravel(), HY.ravel()], axis=1)
for ax, eps, lab in ((axes[0, 0], 0.15, "(a)"), (axes[0, 1], 0.5, "(b)")):
    Z = cloud_mass(hgrid, eps).reshape(HX.shape)
    ax.imshow(Z, origin="lower", extent=(gs[0], gs[-1], gs[0], gs[-1]), cmap="Blues", interpolation="bilinear", vmin=0)
    ax.plot(X[:, 0], X[:, 1], "o", color=C["main"], ms=2.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(rf"{lab} $\|V\|*\rho_\varepsilon$, $\varepsilon = {eps}$", fontsize=11)

ax = axes[1, 0]
ang = np.linspace(0, 4 * np.pi / N, 201)
on = np.stack([np.cos(ang), np.sin(ang)], axis=1)
for eps, col in ((0.15, C["normal"]), (0.3, C["accent"]), (0.5, C["tangent"])):
    ax.plot(np.degrees(ang), eps * cloud_mass(on, eps), color=col, lw=1.4, label=rf"$\varepsilon = {eps}$")
ax.axhline(C_RHO, color=C["aux"], lw=1.0, ls="--")
ax.text(1.02, C_RHO, r"$C_\rho$", fontsize=10, color=C["aux"], ha="left", va="center",
        transform=ax.get_yaxis_transform())
ax.set_xlabel("angle on the circle (degrees)")
ax.set_ylabel(r"$\varepsilon\,(\|V\|*\rho_\varepsilon)$")
ax.set_xticks([0, 7.5, 15, 22.5, 30])
ax.set_ylim(0, 1.8)
ax.legend(loc="upper center", frameon=False, fontsize=8, ncol=3, handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
ax.set_title("(c) on the circle, N = 24", fontsize=11)

ax = axes[1, 1]
s = np.linspace(0.5, 1.5, 201)
line = np.stack([s, np.zeros_like(s)], axis=1)
for eps, col in ((0.1, C["normal"]), (0.2, C["accent"]), (0.4, C["tangent"])):
    ax.plot(s, circle_mass(line, eps), color=col, lw=1.4, label=rf"$\varepsilon = {eps}$")
ax.set_xlabel(r"$s$  (point $(s, 0)$)")
ax.set_ylabel(r"$\|V\|*\rho_\varepsilon$")
ax.set_xticks([0.5, 1.0, 1.5])
ax.set_ylim(0, 11.5)
ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.0, handletextpad=0.3)
ax.set_title("(d) across the smooth circle", fontsize=11)

dgfig.save(fig, __file__)
print("C_rho", round(C_RHO, 4), "peaks", [round(float(circle_mass(np.array([[1.0, 0.0]]), e)[0]), 3) for e in (0.1, 0.2, 0.4)])
