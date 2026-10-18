"""Figure 37.1.2: recovering a plane curve from its signed curvature (Theorem 4.4.10, Derivation in Section 37.1).

(a) kappa_s(s) of the ellipse (2 cos t, sin t) as a function of arc length, with the tangent angle phi(s) = phi_0 +
int kappa_s; (b) curves integrated from curvature data: the ellipse's kappa_s with two choices of (p_0, phi_0) (they
differ by a rigid motion), and kappa_s(s) = 1 + 0.5 cos s on [0, 2 pi], whose total curvature is 2 pi but which does
not close (gap 2 pi J_1(0.5) = 1.5222).

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-1-2-recovery.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg37  # noqa: E402

C = dgfig.COLORS


def integrate(kap, L, phi0, p0):
    """trapezoid rule for phi = phi0 + int kappa, gamma = p0 + int (cos phi, sin phi) (the Implementation box)."""
    m = len(kap) - 1
    ds = L / m
    phi = phi0 + np.concatenate([[0.0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * ds)])
    T = np.stack([np.cos(phi), np.sin(phi)], 1)
    return phi, np.asarray(p0, float) + np.vstack([[0.0, 0.0], np.cumsum(0.5 * (T[1:] + T[:-1]) * ds, 0)])


L = ddg37.smooth_length("ellipse")
m = 2000
s = np.linspace(0, L, m + 1)
t = ddg37.t_of_s("ellipse", s, 200000)
kap = ddg37.kappa("ellipse", t)
phi, G1 = integrate(kap, L, np.pi / 2, (2.0, 0.0))         # starts where the ellipse starts, heading up
_, G2 = integrate(kap, L, np.pi / 2 + 0.6, (5.6, 0.2))       # another choice: rotated by 0.6, translated
assert abs(phi[-1] - phi[0] - 2 * np.pi) < 1e-6                # total curvature 2 pi (rotation index 1)
assert ddg37.rigid_align_error(G1, ddg37.point("ellipse", t)) < 1e-5
assert np.linalg.norm(G1[-1] - G1[0]) < 1e-6
sb = np.linspace(0, 2 * np.pi, 4001)
_, G3 = integrate(1 + 0.5 * np.cos(sb), 2 * np.pi, 0.0, (0.2, -3.6))
gap = G3[-1] - G3[0]
assert abs(gap[0] + 1.52221761) < 1e-6 and abs(gap[1]) < 1e-9

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1, 1.45]})

# (a) curvature along the ellipse and the tangent angle it integrates to
ax.fill_between(s, 0, kap, color=C["region"], alpha=0.25, lw=0)
ax.plot(s, kap, color=C["main"], lw=1.6, label=r"$\kappa_s(s)$")
ax.plot(s, (phi - phi[0]) / (2 * np.pi), color=C["tangent"], lw=1.4, ls="--",
        label=r"$(\varphi(s) - \varphi_0)/2\pi$")
ax.text(2.4, 0.1, r"area $= 2\pi$", ha="center", fontsize=10)
ax.set_xlabel(r"arc length $s$")
ax.set_xlim(0, L)
ax.set_ylim(0, 2.15)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False, fontsize=10.5, handlelength=1.6)
ax.set_title("(a)", fontsize=12, loc="left")
for sp2 in ("top", "right"):
    ax.spines[sp2].set_visible(False)

# (b) curves integrated from curvature data
bx.plot(G1[:, 0], G1[:, 1], color=C["main"], lw=1.8)
bx.plot(G2[:, 0], G2[:, 1], color=C["aux"], lw=1.4)
bx.plot(G3[:, 0], G3[:, 1], color=C["accent"], lw=1.6, ls="--")
for G, col, ds in ((G1, C["main"], L / m), (G2, C["aux"], L / m), (G3, C["accent"], 2 * np.pi / 4000)):
    bx.plot([G[0, 0]], [G[0, 1]], "o", color=col, ms=4, zorder=5)
    k = int(0.6 / ds)                                          # arrow along the first 0.6 of arc length
    bx.annotate("", xy=G[k], xytext=G[0], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=9))
bx.plot([G3[-1, 0]], [G3[-1, 1]], "s", color=C["accent"], ms=4, zorder=5)
bx.annotate("", xy=G3[0], xytext=G3[-1], arrowprops=dict(arrowstyle="<->", color=C["normal"], lw=1.0,
                                                          shrinkA=3, shrinkB=3))
bx.text(0.5 * (G3[0, 0] + G3[-1, 0]), G3[0, 1] - 0.42, "gap", color=C["normal"], ha="center", fontsize=10)
bx.text(0, -0.05, r"ellipse $\kappa_s$", fontsize=10.5, ha="center")
bx.text(1.6, -2.9, r"$\kappa_s = 1 + 0.5\cos s$", color=C["accent"], fontsize=10.5)
bx.set_aspect("equal")
bx.set_axis_off()
bx.set_title("(b)", fontsize=12, loc="left")

fig.tight_layout()
dgfig.save(fig, __file__)
