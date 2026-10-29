"""Figure 46.7.3: the bounded Lipschitz distance.

(a) On the real line, Delta(delta_0, delta_h) = min(h, 2). The optimal test function phi(x) = clip(h/2 - x, -1, 1)
(slope -1, cut at +-1) for h = 0.8 (blue) and h = 3 (vermilion); the dots mark phi(0) and phi(h).
(b) The unit circle C against the point cloud W_N of N equally spaced atoms (cell centres, exact tangents, masses 2 pi/N):
Delta(v(C, 1), W_N) = N [8 (1 - cos(pi/2N)) + 2 (1 - cos(pi/N))] (Example 46.7.10), log-log, with 2 pi^2 / N (dashed).
The self-check confirms that the test function min(1, min_i(|x - x_i| + ||P - P_i||)) attains the value.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-7-3-bounded-lipschitz.py
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


def phi_opt(x, h):
    return np.clip(h / 2 - x, -1.0, 1.0)


for h in (0.3, 0.8, 2.0, 3.0):
    assert abs(phi_opt(0.0, h) - phi_opt(h, h) - v.bl_diracs(h)) < 1e-12

Ns = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128])
exact = np.array([v.circle_cell_bound_exact(int(N)) for N in Ns])
for N in (8, 32):
    t, X, P, m = v.circle_atoms(N)

    def phi_star(x, Pm, X=X, P=P):
        d = np.linalg.norm(x[:, None, :] - X[None, :, :], axis=2)
        op = np.abs(np.linalg.eigvalsh(Pm[:, None, :, :] - P[None, :, :, :])).max(axis=2)
        return np.minimum(1.0, (d + op).min(axis=1))

    gap = v.circle_varifold_integral(phi_star, nq=16, pieces=N * 8) - v.atoms_integral(phi_star, X, P, m)
    assert abs(gap - v.circle_cell_bound_exact(N)) < 1e-9
assert np.allclose(exact * Ns / (2 * math.pi ** 2), 1, atol=0.01)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.35})
ax = axes[0]
x = np.linspace(-1.5, 4.5, 600)
for h, col in ((0.8, C["tangent"]), (3.0, C["normal"])):
    ax.plot(x, phi_opt(x, h), color=col, lw=1.5, label=f"$h = {h}$")
    ax.plot([0, h], [phi_opt(0.0, h), phi_opt(h, h)], "o", color=col, ms=4)
ax.axhline(0, color=C["aux"], lw=0.6)
ax.plot([0, 0], [-1.25, 1.25], color=C["aux"], lw=0.6, ls=":")
ax.text(0.05, -1.2, r"$0$", fontsize=10, ha="left", va="bottom")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\varphi(x)$")
ax.set_ylim(-1.3, 1.45)
ax.set_yticks([-1, 0, 1])
ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.3)
ax.set_title(r"(a) $\delta_0$ against $\delta_h$", fontsize=11)

ax = axes[1]
ax.loglog(Ns, exact, "o-", color=C["main"], ms=4, lw=1.2, label=r"$\Delta^{1,1}$")
ax.loglog(Ns, 2 * math.pi ** 2 / Ns, "--", color=C["accent"], lw=1.2, label=r"$2\pi^2/N$")
ax.set_xlabel(r"$N$")
ax.set_xticks([8, 16, 32, 64, 128])
ax.set_xticklabels(["8", "16", "32", "64", "128"])
ax.minorticks_off()
ax.set_yticks([0.2, 0.5, 1, 2])
ax.set_yticklabels(["0.2", "0.5", "1", "2"])
ax.legend(loc="upper right", frameon=False, fontsize=9, handlelength=1.6, handletextpad=0.4)
ax.set_title("(b) circle against N atoms", fontsize=11)

dgfig.save(fig, __file__)
print("exact", np.round(exact, 4))
