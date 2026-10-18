"""Figure 37.6.2: explicit versus implicit Euler on the model problem u' = -lambda u (lambda = 1, u(0) = 1).

(a) forward Euler u_{k+1} = (1 - h lambda) u_k for h lambda = 0.5, 1.5, 2.2; (b) backward Euler
u_{k+1} = u_k / (1 + h lambda) for the same steps. The grey curve is the exact solution e^{-t}.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-6-2-model-problem.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
T_END = 9.0
hs = [(0.5, C["tangent"], "o"), (1.5, C["accent"], "s"), (2.2, C["normal"], "D")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
tt = np.linspace(0, T_END, 300)
for ax, kind in zip(axes, ("explicit", "implicit")):
    ax.plot(tt, np.exp(-tt), color=C["aux"], lw=1.2, label=r"$e^{-t}$")
    for h, col, mk in hs:
        k = np.arange(int(T_END / h) + 1)
        g = (1 - h) if kind == "explicit" else 1 / (1 + h)
        u = g ** k
        if kind == "explicit" and h > 2:
            assert abs(u[-1]) > 1                   # |1 - h| > 1: grows
        if kind == "implicit":
            assert np.all(np.diff(u) < 0) and np.all(u > 0)
        ax.plot(k * h, u, marker=mk, ms=3.5, color=col, lw=1.2, label=r"$h\lambda = %.1f$" % h)
    ax.axhline(0, color=C["aux"], lw=0.5)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlim(0, T_END)
    ax.set_xlabel(r"$t$")
    ax.set_title("(a)" if kind == "explicit" else "(b)", fontsize=12, loc="left")
    for sp2 in ("top", "right"):
        ax.spines[sp2].set_visible(False)
axes[0].set_ylabel(r"$u_k$")
axes[0].legend(fontsize=9, frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
dgfig.save(fig, __file__)
