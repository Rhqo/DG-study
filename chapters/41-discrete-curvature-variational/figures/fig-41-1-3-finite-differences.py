"""Figure 41.1.3: checking a gradient with finite differences.

The bumpy closed mesh of Implementation 41.1 (icosphere(1), vertex k scaled by 1 + 0.2 u_k, u_k uniform in [-1, 1],
seed 4) and its total area A. For a fixed random unit direction W (seed 5) in R^{3 n0} the directional derivative
dA(W) = <grad A, W> from Proposition 41.1.4 is compared with the forward difference (A(x + eps W) - A(x)) / eps and the
central difference (A(x + eps W) - A(x - eps W)) / (2 eps) for eps = 10^-1, ..., 10^-12. Log-log axes; gray guide lines
of slope 1 and 2.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-1-3-finite-differences.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

C = dgfig.COLORS
rng = np.random.default_rng(4)
V, F = dm.icosphere(1)
V = V * (1 + 0.2 * rng.uniform(-1, 1, (len(V), 1)))
area = lambda X: dm.face_areas(X, F).sum()  # noqa: E731
W = np.random.default_rng(5).normal(size=V.shape)
W /= np.linalg.norm(W)
exact = np.sum(dd.area_gradient(V, F) * W)
eps = 10.0 ** -np.arange(1, 12.01, 0.25)
fwd = np.array([abs((area(V + e * W) - area(V)) / e - exact) for e in eps])
cen = np.array([abs((area(V + e * W) - area(V - e * W)) / (2 * e) - exact) for e in eps])
k = int(np.argmin(cen))
assert 1e-6 < eps[k] < 1e-3 and cen[k] < 1e-9
assert fwd[np.argmin(np.abs(eps - 1e-3))] / fwd[np.argmin(np.abs(eps - 1e-4))] > 8
assert cen[np.argmin(np.abs(eps - 1e-2))] / cen[np.argmin(np.abs(eps - 1e-3))] > 80

fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.loglog(eps, fwd, "s--", color=C["normal"], ms=3.5, lw=1.2, label="forward difference")
ax.loglog(eps, cen, "o-", color=C["tangent"], ms=3.5, lw=1.4, label="central difference")
e0 = np.array([1e-1, 1e-4])
ax.loglog(e0, 0.05 * e0, color=C["aux"], lw=0.8)
ax.loglog(e0, 2e-3 * e0 ** 2, color=C["aux"], lw=0.8)
ax.text(4e-3, 6e-5, "slope 1", color=C["aux"], fontsize=9.5)
ax.text(1.2e-3, 3e-10, "slope 2", color=C["aux"], fontsize=9.5)
ax.annotate("rounding error", (1e-10, cen[np.argmin(np.abs(eps - 1e-10))]), (3e-8, 3e-2),
            fontsize=9.5, arrowprops=dict(arrowstyle="->", color=C["main"], lw=0.8))
ax.set_xlabel(r"step $\varepsilon$")
ax.set_ylabel(r"$|$difference quotient $-\, dA(W)|$")
ax.invert_xaxis()
ax.legend(loc="lower right", fontsize=9.5, frameon=False)
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.14, top=0.97)
dgfig.save(fig, __file__)
print("exact dA(W) =", exact, "best central eps", eps[k], cen[k], "fwd at 1e-8", fwd[np.argmin(np.abs(eps - 1e-8))])
