"""Figure 37.6.1: the inner product decides the flow.

The flower curve r = 1 + 0.3 cos 5t sampled with 64 vertices at equal parameter steps, moved by forward Euler with
small steps until the same time t = 0.2:
- Euclidean inner product (M = I): x_i <- x_i - h grad_i L, h = dt / mean(lbar) (the integrated B flow of Chapter 32);
- lbar-weighted inner product (M = diag(lbar)): x_i <- x_i - dt grad_i L / lbar_i.
(a) every edge length at t = 0.2 divided by its initial value, against the edge index (dotted: the five petal tips);
(b) the smallest of these ratios against time.

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
n = 64
X0, _ = ddg37.sample("flower", n)
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
qE = dd.curve_edge_lengths(XE, True) / ell0        # per-edge ratios at t = 0.2
qW = dd.curve_edge_lengths(XW, True) / ell0
assert rE[-1] < 0.01 and rW[-1] > 0.15             # Euclidean: an edge collapses; weighted: no collapse
assert np.isclose(qE.min(), rE[-1]) and np.isclose(qW.min(), rW[-1])
# the Euclidean flow is the integrated B flow of Chapter 32 (kappa^B n = -grad L)
assert np.allclose(dd.length_gradient(X0, True), -dd.curvature_length_variation(X0, True)[:, None]
                   * dd.curve_vertex_normals(X0, True))
# petal tips at t = 2 pi k / 5, i.e. at vertex position 64 k / 5; edge i is centred at i + 1/2
tips = n * np.arange(6) / 5
mid = np.arange(n) + 0.5
near_tip = np.min(np.abs(mid[:, None] - tips[None, :]), axis=1) <= 0.5   # the edges whose segment holds a tip
assert np.all(near_tip[np.argsort(qE)[:6]])        # the six smallest M = I ratios flank the five tips
assert qE[near_tip].max() < 0.035 and qE[~near_tip].min() > 0.08
assert np.all(near_tip[np.argsort(qW)[:6]]) and qW.min() > 0.17

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
for x in tips:
    ax.axvline(x, color=C["aux"], lw=0.8, ls=":")
ax.semilogy(mid, qE, "o-", color=C["normal"], lw=1.0, ms=2.6, label=r"$M = I$")
ax.semilogy(mid, qW, "s-", color=C["tangent"], lw=1.0, ms=2.4, label=r"$M = \mathrm{diag}(\bar\ell_i)$")
ax.set_xlim(0, n)
ax.set_ylim(1.5e-3, 12.0)
ax.set_xticks([0, 16, 32, 48, 64])
ax.set_xlabel(r"edge $i$")
ax.set_ylabel(r"$\ell_i(0.2)/\ell_i(0)$")
ax.annotate("petal tips", xy=(tips[2], 2.0e-3), xytext=(tips[2] + 4, 4.0e-3), fontsize=9.5,
            color=C["aux"], arrowprops=dict(arrowstyle="->", color=C["aux"], lw=0.8))
ax.legend(fontsize=9.0, frameon=False, loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.0)
ax.set_title("(a)", fontsize=12, loc="left")
bx.semilogy(ts, rE, color=C["normal"], lw=1.6, label=r"$M = I$ (integrated)")
bx.semilogy(ts, rW, color=C["tangent"], lw=1.6, label=r"$M = \mathrm{diag}(\bar\ell_i)$ (pointwise)")
bx.set_xlabel(r"time $t$")
bx.set_ylabel(r"$\min_i \ell_i(t)/\ell_i(0)$")
bx.legend(fontsize=9.5, frameon=False, loc="lower left")
bx.set_title("(b)", fontsize=12, loc="left")
for a in (ax, bx):
    for sp2 in ("top", "right"):
        a.spines[sp2].set_visible(False)
fig.tight_layout()
dgfig.save(fig, __file__)
print(f"min edge ratio at t = 0.2: Euclidean {rE[-1]:.4f}, weighted {rW[-1]:.4f}; "
      f"M = I at the tips {qE[near_tip].min():.4f}..{qE[near_tip].max():.4f}, between {qE[~near_tip].min():.3f}")
