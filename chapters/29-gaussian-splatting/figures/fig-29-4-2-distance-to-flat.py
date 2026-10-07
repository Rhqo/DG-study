"""Figure 29.4.2: how far is a flat Gaussian? Distance from I to diag(1, eps) under the four metrics (Section 29.4).

Euclidean: 1 - eps. log-Euclidean and affine-invariant: |log eps| (the two coincide here). Bures-Wasserstein:
1 - sqrt(eps). eps = s_2^2 is the squared smaller scale, from 1 down to 1e-6 (log axis).
Self-check: closed forms agree with the distance functions used in the verify script.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-4-2-distance-to-flat.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
eps = np.geomspace(1e-6, 1, 300)


def fun(S, f):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(f(w)) @ V.T


for e in (1e-1, 1e-4):
    D = np.diag([1.0, e])
    assert abs(np.linalg.norm(D - np.eye(2)) - (1 - e)) < 1e-12
    assert abs(np.linalg.norm(fun(D, np.log)) - abs(np.log(e))) < 1e-12
    assert abs(np.sqrt(2 + (1 + e) - 2 * np.trace(fun(D, np.sqrt))) - (1 - np.sqrt(e))) < 1e-9

fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.semilogx(eps, 1 - eps, color=C["main"], lw=1.9, label="Euclidean: $1 - \\varepsilon$")
ax.semilogx(eps, np.abs(np.log(eps)), color=C["tangent"], lw=1.9, label="log-Euclidean: $|\\log\\varepsilon|$")
ax.semilogx(eps, np.abs(np.log(eps)), color=C["accent"], lw=1.9, ls=(0, (4, 3)),
            label="affine-invariant: $|\\log\\varepsilon|$")
ax.semilogx(eps, 1 - np.sqrt(eps), color=C["third"], lw=1.9, label="Bures–Wasserstein: $1 - \\sqrt{\\varepsilon}$")
ax.set_xlim(1, 1e-6)
ax.set_ylim(0, 15)
ax.set_xlabel(r"$\varepsilon = s_2^2$ (flatter $\rightarrow$)")
ax.set_ylabel(r"distance from $I$ to $\mathrm{diag}(1, \varepsilon)$")
ax.annotate("grows without bound:\nflat Gaussians are at infinity", xy=(1e-4, 9.21), xytext=(0.5, 11.5), fontsize=10,
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8), va="center")
ax.annotate("stays below 1:\nthe boundary is reachable", xy=(1e-5, 1.0), xytext=(1e-3, 4.0), fontsize=10,
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8), va="center")
ax.legend(fontsize=9.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.tick_params(labelsize=9)

dgfig.save(fig, __file__)
