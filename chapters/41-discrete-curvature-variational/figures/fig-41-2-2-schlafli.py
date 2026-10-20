"""Figure 41.2.2: the Schlaefli formula along a deformation.

icosphere(2) deformed by x(t) = x0 (1 + 0.35 t g(x0)), g(x) = x y + 0.6 z^2 - 0.3 x, t in [0, 1] (closed, faces stay
nondegenerate). T(t) = (1/2) sum l theta. With central differences in t (step 1e-5):
  dT/dt (black), (1/2) sum theta dl/dt (blue dashed), (1/2) sum l dtheta/dt (orange).
(a) the three on a linear scale; (b) |(1/2) sum l dtheta/dt| and |dT/dt| on a log scale.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-2-2-schlafli.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402

C = dgfig.COLORS
V0, F = dm.icosphere(2)
g = V0[:, 0] * V0[:, 1] + 0.6 * V0[:, 2] ** 2 - 0.3 * V0[:, 0]
X = lambda t: V0 * (1 + 0.35 * t * g)[:, None]  # noqa: E731
keys = sorted(dm.edge_bending_angles(V0, F))


def lt(t):
    b = dm.edge_bending_angles(X(t), F)
    return np.array([b[k][1] for k in keys]), np.array([b[k][0] for k in keys])


ts = np.linspace(0, 1, 41)
h = 1e-5
dT, A, B = [], [], []
for t in ts:
    l0, th0 = lt(t)
    lp, thp = lt(t + h)
    lm, thm = lt(t - h)
    dl, dth = (lp - lm) / (2 * h), (thp - thm) / (2 * h)
    dT.append((0.5 * np.dot(lp, thp) - 0.5 * np.dot(lm, thm)) / (2 * h))
    A.append(0.5 * np.dot(th0, dl))
    B.append(0.5 * np.dot(l0, dth))
dT, A, B = map(np.array, (dT, A, B))
assert np.abs(B).max() < 1e-8 and np.abs(dT - A).max() < 1e-8 and np.abs(dT).min() > 0.1
assert min(np.min(dm.face_areas(X(t), F)) for t in ts) > 1e-3
assert all(dm.signed_volume(X(t), F) > 0 for t in ts)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax = axes[0]
ax.plot(ts, dT, color=C["main"], lw=2.2, label=r"$dT/dt$")
ax.plot(ts, A, color=C["tangent"], lw=1.4, ls="--", label=r"$\frac{1}{2}\sum\theta\,\dot\ell$")
ax.plot(ts, B, color=C["accent"], lw=1.6, label=r"$\frac{1}{2}\sum\ell\,\dot\theta$")
ax.axhline(0, color=C["aux"], lw=0.6)
ax.set_xlabel(r"$t$")
ax.legend(fontsize=10, frameon=False, loc="center right")
ax.set_title("(a)", loc="left", fontsize=12)
ax = axes[1]
ax.semilogy(ts, np.abs(dT), color=C["main"], lw=2.0, label=r"$|dT/dt|$")
ax.semilogy(ts, np.abs(B), "o", color=C["accent"], ms=3, label=r"$|\frac{1}{2}\sum\ell\,\dot\theta|$")
ax.set_ylim(1e-13, 1e2)
ax.set_xlabel(r"$t$")
ax.legend(fontsize=10, frameon=False, loc="center right")
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.9, wspace=0.28)
dgfig.save(fig, __file__)
print("dT/dt range", dT.min(), dT.max(), "max |B|", np.abs(B).max())
