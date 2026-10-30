"""Figure 49.4.2: the varifold estimator (12)/(38) of [GeoSplat25] on point clouds with exact normals (Section 49.4).

Kernels as in [BLM22, §7.1]: Upsilon = rho(s) = exp(-1/(1 - s^2)), chi = xi with the natural pair 3 xi(s) = -s rho'(s);
masses m_j = 1, all neighbours within distance eps (the paper uses KNN and does not specify kernels or eps).
(a) unit sphere, 40000 uniform points, outward normals (true principal curvatures -1, -1). Mean of the two eigenvalues
    over 10 points against eps, for three versions: second term with "-" (derived) and the factor 1/eps;
    second term with "+" (printed) and 1/eps; "-" without 1/eps (as printed in (31)).
(b) torus T_{2,0.8}, 150000 uniform points, outward normals, eps = 0.2: the two eigenvalues at 25 points along a meridian
    (tube angle u from the outer to the inner equator) against the exact principal curvatures -1.25 and -cos u/(2 + 0.8 cos u).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-4-2-varifold-accuracy.py
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
X = geo49.sphere_points(40000, seed=1)
m = np.ones(len(X))
Q = range(0, 40000, 4000)
eps_list = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
versions = [("derived: minus, with 1/eps", -1.0, True), ("printed (37): plus, with 1/eps", 1.0, True),
            ("printed (31): minus, no 1/eps", -1.0, False)]
curves = {}
for name, sg, ie in versions:
    curves[name] = np.array([np.mean([np.linalg.eigvalsh(geo49.geosplat_B_fast(X, X, m, i, e, sign=sg, with_inv_eps=ie)).mean()
                                      for i in Q]) for e in eps_list])
d, p, n = (curves[v[0]] for v in versions)
print("sphere:", np.round(d, 3), np.round(p, 3), np.round(n, 3))
assert np.all(np.abs(d + 1) < 0.07) and np.all(np.abs(n + eps_list) < 0.04 * 1 + 0.03)
assert abs(d[2] + 0.984) < 0.01 and abs(p[2] + 0.964) < 0.01 and abs(p[-1] + 0.86) < 0.02

R, r = 2.0, 0.8
T = geo49.torus_points(150000, seed=2)
rho = np.hypot(T[:, 0], T[:, 1])
NT = (T - np.stack([R * T[:, 0] / rho, R * T[:, 1] / rho, 0 * rho], axis=1)) / r
us = np.linspace(0, np.pi, 25)
Qp = np.stack([R + r * np.cos(us), 0 * us, r * np.sin(us)], axis=1)
T2 = np.vstack([Qp, T])
N2 = np.vstack([(Qp - np.array([R, 0, 0])) / r, NT])
m2 = np.ones(len(T2))
est = np.array([sorted(np.linalg.eigvalsh(geo49.geosplat_B_fast(T2, N2, m2, k, 0.2)), key=lambda x: -abs(x)) for k in range(len(us))])
k_par = -np.cos(us) / (R + r * np.cos(us))
err_mer = np.abs(est[:, 0] + 1.25)
k_est_par = np.where(np.abs(est[:, 0] + 1.25) < np.abs(est[:, 1] + 1.25), est[:, 1], est[:, 0])
k_est_mer = np.where(np.abs(est[:, 0] + 1.25) < np.abs(est[:, 1] + 1.25), est[:, 0], est[:, 1])
print("torus max errors: meridian", np.abs(k_est_mer + 1.25).max(), "parallel", np.abs(k_est_par - k_par).max())
assert np.abs(k_est_mer + 1.25).max() < 0.25 and np.abs(k_est_par - k_par).max() < 0.1

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.5))
styles = [(C["main"], "-", "o"), (C["normal"], "--", "s"), (C["aux"], ":", "^")]
for (name, _, _), (col, ls, mk) in zip(versions, styles):
    ax.plot(eps_list, curves[name], color=col, ls=ls, marker=mk, ms=4, lw=1.5, label=name)
ax.axhline(-1, color=C["tangent"], lw=0.8)
ax.text(0.105, -1.06, "true value -1", color=C["tangent"], fontsize=8.5)
ax.set_xlabel(r"scale $\varepsilon$")
ax.set_ylabel("mean of the two eigenvalues")
ax.set_ylim(-1.15, 0.0)
ax.legend(loc="center right", fontsize=7.5, frameon=False)
ax.set_title("(a) unit sphere", fontsize=10.5)

ug = np.linspace(0, np.pi, 200)
ax2.plot(np.degrees(ug), -1.25 + 0 * ug, color=C["tangent"], lw=1.2, label="exact")
ax2.plot(np.degrees(ug), -np.cos(ug) / (R + r * np.cos(ug)), color=C["tangent"], lw=1.2)
ax2.plot(np.degrees(us), k_est_mer, "o", color=C["main"], ms=4, label="estimated")
ax2.plot(np.degrees(us), k_est_par, "o", color=C["main"], ms=4)
ax2.text(60, -0.9, "around the tube", fontsize=8.5)
ax2.text(100, -0.25, "along the parallel", fontsize=8.5)
ax2.set_xlabel(r"tube angle $u$ (degrees; 0 = outer equator)")
ax2.set_ylabel("principal curvature")
ax2.set_xticks([0, 45, 90, 135, 180])
ax2.set_ylim(-1.45, 1.25)
ax2.legend(loc="upper left", fontsize=8, frameon=False)
ax2.set_title(r"(b) torus, $\varepsilon = 0.2$", fontsize=10.5)
fig.tight_layout()
dgfig.save(fig, __file__)
