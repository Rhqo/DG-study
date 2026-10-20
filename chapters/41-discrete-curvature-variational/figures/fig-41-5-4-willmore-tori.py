"""Figure 41.5.4: the Willmore energy of the tori of revolution.

W(T_{R,r}) = int H^2 dA = pi^2 a^2 / sqrt(a^2 - 1), a = R / r (Example 41.5.14), minimum 2 pi^2 at a = sqrt 2.
Dots: the discrete energy W_h = sum_i |(L x)_i|^2 / (4 A_i) (mixed area) of torus_mesh(a, 1, 48, 96).
Gray dashed line: 4 pi, the energy of every round sphere. The running example (R, r) = (2, 0.8) is a = 2.5.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-5-4-willmore-tori.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402

C = dgfig.COLORS
W = lambda a: np.pi ** 2 * a * a / np.sqrt(a * a - 1)  # noqa: E731
avals = np.array([1.15, 1.25, 1.414, 1.6, 2.0, 2.5, 3.0])
Wh = np.array([ddg41.willmore_energy(*dm.torus_mesh(a, 1.0, 48, 96)) for a in avals])
assert np.all(Wh < W(avals)) and np.all(W(avals) - Wh < 0.2)
assert avals[np.argmin(Wh)] == 1.414
assert abs(ddg41.willmore_energy(*dm.torus_mesh(2.0, 0.8, 48, 96)) - ddg41.willmore_energy(*dm.torus_mesh(2.5, 1.0, 48, 96))) < 1e-9

fig, ax = plt.subplots(figsize=(5.6, 3.6))
aa = np.linspace(1.08, 3.2, 300)
ax.plot(aa, W(aa), color=C["main"], lw=1.6, label=r"$\pi^2a^2/\sqrt{a^2-1}$")
ax.plot(avals, Wh, "o", color=C["tangent"], ms=5, label=r"$W_h$, torus_mesh$(a, 1, 48, 96)$")
ax.axhline(4 * np.pi, color=C["aux"], lw=1.0, ls="--")
ax.text(2.55, 4 * np.pi + 0.5, r"$4\pi$ (sphere)", color=C["aux"], fontsize=10)
ax.axvline(np.sqrt(2), color=C["aux"], lw=0.7, ls=":")
ax.annotate(r"$a=\sqrt{2}$, $W=2\pi^2$", (np.sqrt(2), 2 * np.pi ** 2), (1.75, 16.0), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=C["main"], lw=0.8))
ax.annotate("(2, 0.8)", (2.5, W(2.5)), (2.65, 24.0), fontsize=10, arrowprops=dict(arrowstyle="->", color=C["main"], lw=0.8))
ax.set_xlabel(r"$a = R/r$")
ax.set_ylabel(r"Willmore energy")
ax.set_ylim(10, 36)
ax.legend(fontsize=9.5, frameon=False, loc="upper center")
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.96)
dgfig.save(fig, __file__)
print(np.round(Wh, 3).tolist(), np.round(W(avals), 3).tolist())
