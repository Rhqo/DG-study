"""Figure 37.6.1: the inner product decides the flow.

The flower curve r = 1 + 0.3 cos 5t sampled with 64 vertices at equal parameter steps, moved by forward Euler with
small steps until the same time t = 0.2:
- Euclidean inner product (M = I): x_i <- x_i - h grad_i L, h = dt / mean(lbar) (the integrated B flow of Chapter 32);
- lbar-weighted inner product (M = diag(lbar)): x_i <- x_i - dt grad_i L / lbar_i.
(a) the two curves at t = 0.2 with their vertices; (b) the smallest edge length divided by its initial value.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-6-1-inner-product.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402

C = dgfig.COLORS
X0, _ = ddg37.sample("flower", 64)
ell0 = dd.curve_edge_lengths(X0, True)
dt = 1e-4
steps = 2000
hE = dt / ddg37.dual_lengths(X0).mean()
XE, XW = X0.copy(), X0.copy()
ts, rE, rW = [0.0], [1.0], [1.0]
for k in range(1, steps + 1):
    XE = ddg37.explicit_step(XE, hE, weighted=False)
    XW = ddg37.explicit_step(XW, dt, weighted=True)
    if k % 20 == 0:
        ts.append(k * dt)
        rE.append((dd.curve_edge_lengths(XE, True) / ell0).min())
        rW.append((dd.curve_edge_lengths(XW, True) / ell0).min())
rE, rW = np.array(rE), np.array(rW)
assert rE[-1] < 0.01 and rW[-1] > 0.15             # Euclidean: an edge collapses; weighted: no collapse
# the Euclidean flow is the integrated B flow of Chapter 32 (kappa^B n = -grad L)
assert np.allclose(dd.length_gradient(X0, True), -dd.curvature_length_variation(X0, True)[:, None]
                   * dd.curve_vertex_normals(X0, True))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
s = np.linspace(0, 2 * np.pi, 400)
F = ddg37.point("flower", s)
ax.plot(F[:, 0], F[:, 1], color=C["aux"], lw=0.8, ls="--")
for X, col, lab in ((XE, C["normal"], r"$M = I$"), (XW, C["tangent"], r"$M = \mathrm{diag}(\bar\ell_i)$")):
    Xc = np.vstack([X, X[:1]])
    ax.plot(Xc[:, 0], Xc[:, 1], color=col, lw=1.4, label=lab)
    ax.plot(X[:, 0], X[:, 1], "o", color=col, ms=2.2)
ax.set_aspect("equal")
ax.set_axis_off()
ax.legend(fontsize=9.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
ax.set_title("(a)", fontsize=12, loc="left")
bx.semilogy(ts, rE, color=C["normal"], lw=1.6, label=r"$M = I$ (integrated)")
bx.semilogy(ts, rW, color=C["tangent"], lw=1.6, label=r"$M = \mathrm{diag}(\bar\ell_i)$ (pointwise)")
bx.set_xlabel(r"time $t$")
bx.set_ylabel(r"$\min_i \ell_i(t)/\ell_i(0)$")
bx.legend(fontsize=9.5, frameon=False, loc="lower left")
bx.set_title("(b)", fontsize=12, loc="left")
for sp2 in ("top", "right"):
    bx.spines[sp2].set_visible(False)
fig.tight_layout()
dgfig.save(fig, __file__)
print(f"min edge ratio at t = 0.2: Euclidean {rE[-1]:.4f}, weighted {rW[-1]:.4f}")
