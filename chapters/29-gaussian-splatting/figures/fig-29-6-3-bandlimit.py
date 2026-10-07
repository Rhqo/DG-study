"""Figure 29.6.3: degree 3 is a low-pass filter on the sphere (Section 29.6).

Zonal lobes f(d) = exp(kappa (<d, a> - 1)) around a direction a, for kappa = 5, 20, 100 (half width at half maximum
30.5, 15.1, 6.7 degrees). Their L2-best approximations by spherical harmonics of degree <= L are computed with Legendre
polynomials and 400-point Gauss-Legendre quadrature.
(a) Profiles as functions of the angle from a: true lobe (dashed) and degree <= 3 approximation (solid).
(b) Relative L2 error of the degree <= L approximation versus L (log scale); gray line: 10%; dotted line: L = 3.
Self-checks: degree-3 peaks 0.80, 0.33, 0.08 and errors 0.19, 0.66, 0.92; degrees for 10% error 4, 9, 21.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-6-3-bandlimit.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
tg, wg = np.polynomial.legendre.leggauss(400)


def coeffs(kappa, L):
    f = np.exp(kappa * (tg - 1))
    return [(2 * l + 1) / 2 * np.sum(wg * f * np.polynomial.legendre.Legendre.basis(l)(tg)) for l in range(L + 1)]


def proj(kappa, L, t):
    return sum(a * np.polynomial.legendre.Legendre.basis(l)(t) for l, a in enumerate(coeffs(kappa, L)))


def rel_err(kappa, L):
    f = np.exp(kappa * (tg - 1))
    return np.sqrt(np.sum(wg * (proj(kappa, L, tg) - f) ** 2) / np.sum(wg * f ** 2))


kap = [(5, C["tangent"]), (20, C["accent"]), (100, C["normal"])]
assert [round(proj(k, 3, np.array([1.0]))[0], 2) for k, _ in kap] == [0.80, 0.33, 0.08]
assert [round(rel_err(k, 3), 2) for k, _ in kap] == [0.19, 0.66, 0.92]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw=dict(wspace=0.3))
ang = np.linspace(0, 180, 721)
t = np.cos(np.radians(ang))
for k, col in kap:
    axa.plot(ang, np.exp(k * (t - 1)), color=col, lw=1.4, ls=(0, (4, 2.5)))
    axa.plot(ang, proj(k, 3, t), color=col, lw=1.9, label=r"$\kappa = %d$" % k)
axa.axhline(0, color="#BBBBBB", lw=0.6)
axa.set_xlim(0, 180)
axa.set_ylim(-0.12, 1.05)
axa.set_xlabel(r"angle from the lobe axis $a$ (degrees)")
axa.set_ylabel(r"$f$")
axa.legend(fontsize=9.5, frameon=False, loc="upper right", title="dashed: lobe\nsolid: degree $\\leq 3$", title_fontsize=9)
axa.set_title(r"(a) lobes and their degree-3 approximations", fontsize=11)
axa.tick_params(labelsize=9)

Ls = np.arange(0, 26)
need = []
for k, col in kap:
    e = np.array([rel_err(k, L) for L in Ls])
    axb.semilogy(Ls, e, "o-", color=col, ms=3, lw=1.5, label=r"$\kappa = %d$" % k)
    need.append(int(Ls[np.argmax(e < 0.1)]))
assert need == [4, 9, 21]
axb.axhline(0.1, color=C["aux"], lw=0.8)
axb.axvline(3, color=C["main"], lw=0.8, ls=(0, (2, 2)))
axb.text(3.4, 1.15, "3DGS: degree 3", fontsize=9.5)
axb.text(14.5, 0.115, "10%", fontsize=9.5, color="#555555")
axb.set_xlabel(r"maximum degree $L$")
axb.set_ylabel(r"relative $L^2$ error")
axb.set_ylim(1e-3, 2)
axb.legend(fontsize=9.5, frameon=False, loc="lower right")
axb.set_title("(b) error versus degree", fontsize=11)
axb.tick_params(labelsize=9)

dgfig.save(fig, __file__)
