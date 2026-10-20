"""Figure 43.4.2: Dirichlet energy, signed area, and conformal energy of z_eps(w) = w + eps conj(w) on the unit disk
(Example 43.4.5, Theorem 43.4.6).

(a) E_D = pi (1 + eps^2), signed area A = pi (1 - eps^2), E_C = E_D - A = 2 pi eps^2 for 0 <= eps <= 1.5, computed by
    midpoint quadrature of the integrands on a polar grid (dots) and compared with the formulas (lines).
(b) the image of the unit circle under z_eps for eps = 0.3, 0.7, 1.0, 1.3 with an arrow showing the direction in which
    the image is traversed when the circle is traversed counterclockwise; eps = 1 is the segment [-2, 2]; for eps > 1
    the ellipse is traversed clockwise (A < 0).
Self-check: quadrature agrees with the formulas to 1e-6; A computed as the boundary integral (1/2) Im int conj(z) dz
agrees with the area integral; A(1) = 0.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-4-2-energies.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
nr, nt = 200, 400
r = (np.arange(nr) + 0.5) / nr
t = 2 * np.pi * (np.arange(nt) + 0.5) / nt
Rg, Tg = np.meshgrid(r, t, indexing="ij")
dA = (1 / nr) * (2 * np.pi / nt) * Rg


def energies(eps):
    # dz = dw + eps dwbar: u = (1 + eps) x, v = (1 - eps) y
    ux, uy, vx, vy = 1 + eps, 0.0, 0.0, 1 - eps
    ED = 0.5 * np.sum((ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2) * dA)
    A = np.sum((ux * vy - uy * vx) * dA)
    EC = 0.5 * np.sum(((ux - vy) ** 2 + (uy + vx) ** 2) * dA)
    tb = 2 * np.pi * np.arange(4000) / 4000
    zb = np.exp(1j * tb) + eps * np.exp(-1j * tb)
    dzb = 1j * np.exp(1j * tb) - 1j * eps * np.exp(-1j * tb)
    Ab = 0.5 * np.sum((np.conj(zb) * dzb).imag) * (2 * np.pi / 4000)
    return ED, A, EC, Ab


eps_dots = np.linspace(0, 1.5, 11)
vals = np.array([energies(e) for e in eps_dots])
assert np.allclose(vals[:, 0], np.pi * (1 + eps_dots ** 2), atol=1e-6)
assert np.allclose(vals[:, 1], np.pi * (1 - eps_dots ** 2), atol=1e-6)
assert np.allclose(vals[:, 2], 2 * np.pi * eps_dots ** 2, atol=1e-6)
assert np.allclose(vals[:, 3], vals[:, 1], atol=1e-9) and abs(energies(1.0)[1]) < 1e-9
print("E_D, A, E_C at eps = 0.5:", np.round(energies(0.5)[:3], 4))

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
e = np.linspace(0, 1.5, 200)
for y, c, lab in ((np.pi * (1 + e ** 2), col["tangent"], r"$E_D$"), (np.pi * (1 - e ** 2), col["third"], r"$\mathcal{A}$"),
                  (2 * np.pi * e ** 2, col["normal"], r"$E_C = E_D - \mathcal{A}$")):
    ax.plot(e, y, color=c, lw=1.6, label=lab)
ax.plot(eps_dots, vals[:, 0], "o", color=col["tangent"], ms=3)
ax.plot(eps_dots, vals[:, 1], "o", color=col["third"], ms=3)
ax.plot(eps_dots, vals[:, 2], "o", color=col["normal"], ms=3)
ax.axhline(0, color=col["aux"], lw=0.7)
ax.axvline(1, color=col["aux"], lw=0.7, ls="--")
ax.set_xlabel(r"$\varepsilon$")
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(r"(a) energies of $w + \varepsilon\bar w$", fontsize=11)

ax = axes[1]
tb = np.linspace(0, 2 * np.pi, 300)
for eps, c in ((0.3, col["tangent"]), (0.7, col["third"]), (1.0, col["aux"]), (1.3, col["accent"])):
    zb = np.exp(1j * tb) + eps * np.exp(-1j * tb)
    ax.plot(zb.real, zb.imag, color=c, lw=1.3, label=r"$\varepsilon = %.1f$" % eps)
    k = 40
    if eps != 1.0:
        ax.annotate("", xy=(zb[k + 4].real, zb[k + 4].imag), xytext=(zb[k].real, zb[k].imag),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3))
ax.set_aspect("equal")
ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-1.6, 1.6)
ax.set_axis_off()
ax.legend(fontsize=8, frameon=False, loc="lower center", ncol=4, handlelength=1.2, columnspacing=0.7, bbox_to_anchor=(0.5, -0.12))
ax.set_title(r"(b) images of $|w| = 1$", fontsize=11)
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.17, wspace=0.12)
dgfig.save(fig, __file__)
