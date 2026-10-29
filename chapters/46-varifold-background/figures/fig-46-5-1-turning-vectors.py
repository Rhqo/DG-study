"""Figure 46.5.1: the first variation of a polygon sits at its vertices as turning vectors.

Ellipse (2 cos t, sin t) and its inscribed 12-gon (vertices t_k = 2 pi k / 12, counter-clockwise).
(a) at every vertex the turning vector t_k - t_{k-1} (blue, drawn at half length); delta V = -sum (t_k - t_{k-1})
delta_{gamma_k}. For this polygon it equals exactly the smooth integrated curvature vector over the dual arc,
int kappa n ds = T(t_{k+1/2}) - T(t_{k-1/2}) (a chord of an ellipse is parallel to the tangent at the middle parameter).
(b) the curvature kappa(t) of the ellipse (black) and the ratio |delta V(phi_k)| / ||V||(phi_k) =
|t_k - t_{k-1}| / ((l_{k-1} + l_k) / 2) for the hat function phi_k of each vertex, for N = 12 (blue) and N = 48 (orange).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-5-1-turning-vectors.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
N = 12
tk = 2 * np.pi * np.arange(N) / N
G = v.ellipse(tk)
W = v.polyline_first_variation(G, closed=True)        # w_k = t_{k-1} - t_k
turn = -W                                              # turning vectors t_k - t_{k-1}
half = tk + np.pi / N
smooth = v.ellipse_tangent(half) - v.ellipse_tangent(half - 2 * np.pi / N)

# self-checks: sum of turning vectors is 0 (closed polygon); |t_k - t_{k-1}| = 2 sin(theta_k/2);
# T(b) - T(a) equals the integral of kappa n ds (fundamental theorem of calculus, checked by quadrature)
assert np.allclose(turn.sum(axis=0), 0)
E = np.roll(G, -1, axis=0) - G
ang = np.arctan2(E[:, 1], E[:, 0])
theta = (ang - np.roll(ang, 1) + np.pi) % (2 * np.pi) - np.pi
assert np.allclose(np.linalg.norm(turn, axis=1), 2 * np.abs(np.sin(theta / 2)))
for k in (0, 3):
    a, b = half[k] - 2 * np.pi / N, half[k]
    q = v.integrate_1d(lambda t: (v.ellipse_kappa(t) * v.ellipse_speed(t))[:, None] * v.ellipse_normal(t), a, b, n=64)
    assert np.allclose(q, smooth[k], atol=1e-10)

def hat_ratio(n):
    t = 2 * np.pi * np.arange(n) / n
    P = v.ellipse(t)
    w = v.polyline_first_variation(P, closed=True)
    ell = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    mass = 0.5 * (ell + np.roll(ell, 1))
    return t, np.linalg.norm(w, axis=1) / mass


fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.3})
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 400)
ax.plot(*v.ellipse(tt).T, color=C["aux"], lw=0.8, ls=":")
ax.plot(*np.vstack([G, G[:1]]).T, color=C["main"], lw=1.3)
ax.plot(G[:, 0], G[:, 1], "o", color=C["main"], ms=3)
s = 0.5
for g, w1 in zip(G, turn):
    ax.annotate("", xy=g + s * w1, xytext=g, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4,
                                                             shrinkA=0, shrinkB=0, mutation_scale=9))
ax.set_aspect("equal")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.2, 1.2)
ax.set_axis_off()
ax.set_title("(a) turning vectors", fontsize=12)

ax = axes[1]
ax.plot(np.degrees(tt), v.ellipse_kappa(tt), color=C["main"], lw=1.2, label=r"$\kappa$")
for n, col, mk in ((12, C["tangent"], "o"), (48, C["accent"], ".")):
    t, r = hat_ratio(n)
    ax.plot(np.degrees(t), r, mk, color=col, ms=5 if n == 12 else 4, label=f"$N = {n}$")
ax.set_xlabel(r"$t$ (degrees)")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_ylim(0, 2.65)
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend(loc="upper center", frameon=False, fontsize=8, ncol=3, handlelength=1.0, handletextpad=0.3, columnspacing=0.6)
ax.set_title(r"(b) $|\delta V(\varphi_k)|\,/\,\|V\|(\varphi_k)$", fontsize=12)

dgfig.save(fig, __file__)
t12, r12 = hat_ratio(12)
print("turning lengths:", np.round(np.linalg.norm(turn, axis=1)[:4], 4), " ratio N=12 at t=0, 90:", np.round(r12[[0, 3]], 4))
