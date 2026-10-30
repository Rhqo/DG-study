"""Figure 49.5.3: non-uniform Gaussian centres distort the manifold estimator unless the generator is normalized
(Section 49.5).

60000 points on the unit sphere with density proportional to p(z) = 1 + 0.9 z (dense at the north pole, sparse at the
south pole), t = 0.005, 15 query points on a meridian. Three versions of the generator L f(q) = sum_j a_j (f_j - f(q)),
k_j = exp(-|x_j - q|^2 / t):
  paper (45) with a constant C:  a_j = (4/t) k_j / (pbar pi t), pbar = N/(4 pi) the mean density (uniform assumption);
  the same, corrected:            tau / (tr K / 2)^2;
  random walk:                    a_j = (4/t) k_j / sum_l k_l.
Plotted: the mean (tau_1 + tau_2)/2 of the two principal curvatures (outward normal; true value -1) against z,
and the prediction -(p(z)/pbar)^2 = -(1 + 0.9 z)^2 for the uncorrected version.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-5-3-density.py
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import geo49  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
N, t = 60000, 0.005
X = geo49.sphere_points_density(N, a=0.9, seed=0)
zs = np.linspace(-0.85, 0.95, 15)
Qp = np.stack([np.sqrt(1 - zs ** 2), 0 * zs, zs], axis=1)
X2 = np.vstack([Qp, X])
pbar = len(X2) / (4 * np.pi)
raw, cor, rw, trk = [], [], [], []
for k in range(len(zs)):
    rp = geo49.manifold_estimate(X2, X2[k], t, mode="paper", density=pbar, n_ref=Qp[k])
    rr = geo49.manifold_estimate(X2, X2[k], t, mode="rw", n_ref=Qp[k])
    s = np.trace(rp["K"]) / 2
    raw.append(rp["tau"].mean())
    cor.append(rp["tau"].mean() / s ** 2)
    rw.append(rr["tau"].mean())
    trk.append(s)
raw, cor, rw, trk = map(np.array, (raw, cor, rw, trk))
pred = -(1 + 0.9 * zs) ** 2
print("raw", np.round(raw, 2))
print("pred", np.round(pred, 2))
print("cor", np.round(cor, 2), "rw", np.round(rw, 2))
print("trK/2", np.round(trk, 2), "p/pbar", np.round(1 + 0.9 * zs, 2))
assert np.all(np.abs(raw - pred) < 0.25 * np.abs(pred) + 0.03)
assert np.all(np.abs(cor + 1) < 0.2) and np.all(np.abs(rw + 1) < 0.2)
assert np.all(np.abs(trk - (1 + 0.9 * zs)) < 0.15 * (1 + 0.9 * zs) + 0.02)

fig, ax = plt.subplots(figsize=(5.4, 3.6))
zz = np.linspace(-0.9, 1.0, 200)
ax.plot(zz, -(1 + 0.9 * zz) ** 2, color=C["aux"], lw=1.0, ls=":", label=r"prediction $-(p/\bar p)^2$")
ax.plot(zs, raw, "o-", color=C["normal"], ms=4, lw=1.3, label="paper (45), constant C")
ax.plot(zs, cor, "s--", color=C["main"], ms=4, lw=1.3, label=r"same, divided by $(\mathrm{tr}\,K/2)^2$")
ax.plot(zs, rw, "^-.", color=C["third"], ms=4, lw=1.3, label="random-walk normalization")
ax.axhline(-1, color=C["tangent"], lw=0.8)
ax.text(0.62, -0.8, "true value -1", color=C["tangent"], fontsize=8.5)
ax.set_xlabel(r"height $z$ of the query point (density $\propto 1 + 0.9z$)")
ax.set_ylabel(r"mean estimated curvature")
ax.set_ylim(-3.9, 0.2)
ax.legend(loc="lower left", fontsize=8, frameon=False)
fig.tight_layout()
dgfig.save(fig, __file__)
