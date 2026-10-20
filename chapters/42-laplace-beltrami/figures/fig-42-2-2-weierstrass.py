"""Figure 42.2.2: an energy whose infimum is not attained (Section 42.2, History box).

J(phi) = int_{-1}^{1} x^2 phi'(x)^2 dx with phi(-1) = -1, phi(1) = 1, and the family
phi_eps(x) = arctan(x / eps) / arctan(1 / eps).
(a) phi_eps for eps = 0.5, 0.1, 0.02 (the steeper the smaller eps);
(b) J(phi_eps) = (eps arctan(1/eps) - eps^2/(1 + eps^2)) / arctan(1/eps)^2 against eps (log-log); the dotted line is
    2 eps / pi, the small-eps asymptote. Self-check: the closed form agrees with a 20000-point midpoint rule.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-2-2-weierstrass.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS


def phi(x, e):
    return np.arctan(x / e) / np.arctan(1 / e)


def J_closed(e):
    return (e * np.arctan(1 / e) - e ** 2 / (1 + e ** 2)) / np.arctan(1 / e) ** 2


xm = -1 + (np.arange(20000) + 0.5) / 10000
for e in (0.5, 0.1, 0.02):
    dphi = (1 / np.arctan(1 / e)) * e / (e ** 2 + xm ** 2)
    assert abs(np.sum(xm ** 2 * dphi ** 2) / 10000 - J_closed(e)) < 1e-6
    assert abs(phi(-1, e) + 1) < 1e-14 and abs(phi(1, e) - 1) < 1e-14
es = np.geomspace(1e-3, 1.0, 50)
assert np.all(np.diff(J_closed(es)) > 0) and abs(J_closed(1e-3) / (2e-3 / np.pi) - 1) < 2e-3
print("J(phi_eps) for eps = 0.5, 0.1, 0.02, 0.001:", [f"{J_closed(e):.4f}" for e in (0.5, 0.1, 0.02, 1e-3)])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))
xs = np.linspace(-1, 1, 801)
for e, c in ((0.5, col["accent"]), (0.1, col["third"]), (0.02, col["tangent"])):
    axa.plot(xs, phi(xs, e), color=c, lw=1.6, label=rf"$\varepsilon = {e}$")
axa.plot([-1, 1], [-1, 1], "o", color=col["main"], ms=4)
axa.set_xlabel(r"$x$")
axa.set_ylabel(r"$\varphi_\varepsilon(x)$")
axa.legend(fontsize=9, frameon=False, loc="upper left")
axa.set_title(r"(a) $\varphi_\varepsilon = \arctan(x/\varepsilon)/\arctan(1/\varepsilon)$", fontsize=10.5)
axa.spines[["top", "right"]].set_visible(False)
axb.loglog(es, J_closed(es), color=col["tangent"], lw=1.6)
axb.loglog(es, 2 * es / np.pi, ":", color=col["aux"], lw=1.1)
axb.text(3e-2, 6e-3, r"$2\varepsilon/\pi$", fontsize=10, color=col["aux"])
axb.set_xlabel(r"$\varepsilon$")
axb.set_ylabel(r"$J(\varphi_\varepsilon)$")
axb.set_title(r"(b) $J(\varphi_\varepsilon) \to 0$", fontsize=10.5)
axb.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.18, wspace=0.38)
dgfig.save(fig, __file__)
