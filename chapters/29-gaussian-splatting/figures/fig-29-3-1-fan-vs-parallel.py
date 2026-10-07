"""Figure 29.3.1: perspective projection vs. its linearization (EWA), seen in the x-z plane (Section 29.3).

Camera center O at the origin, depth t_z upward, image line t_z = f = 1 (u = t_x / t_z).
A 2D Gaussian (the slice t_y = 0) with center t0 = (z0 tan 30deg, z0), z0 = 2.5, isotropic std s = 0.5 (s/t_z = 0.2).
(a) Orange: the two exact rays from O tangent to the 2-sigma circle; they cut the image line in the exact interval.
    Blue dashed: the two lines parallel to the central ray O->t0 tangent to the same circle (the fibers of the
    linearized map); they meet the depth line t_z = z0 at P-, P+, and the thin blue rays O->P+- cut the image line in
    the EWA interval u0 +- 2 sd, sd = s sqrt(1 + tan^2 30deg) / z0.
(b) Image-line densities: exact pushforward density p(u) = int N((u z, z); t0, s^2 I) z dz (orange) and the EWA Gaussian
    N(u0, sd^2) (blue).
Self-checks: EWA interval endpoints equal the construction with P+-; exact interval = tan(30deg +- asin(2s/|t0|));
the exact density integrates to 1.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-3-1-fan-vs-parallel.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
z0, th, s, k = 2.5, np.radians(30), 0.5, 2.0
t0 = np.array([z0 * np.tan(th), z0])
rho = k * s
u0 = t0[0] / t0[1]
sd = s * np.sqrt(1 + np.tan(th) ** 2) / z0

# exact tangent rays from O
beta = np.arcsin(rho / np.linalg.norm(t0))
angs = [th - beta, th + beta]
u_exact = np.tan(angs)
# EWA: lines parallel to t0 tangent to the circle; their feet on t_z = z0
dirn = t0 / np.linalg.norm(t0)
perp = np.array([dirn[1], -dirn[0]])            # points toward +x side
tang = [t0 - rho * perp, t0 + rho * perp]       # tangent points of the parallel lines
feet = []
for p in tang:
    lam = (z0 - p[1]) / dirn[1]
    feet.append(p + lam * dirn)
u_ewa = [q[0] / q[1] for q in feet]
assert np.allclose(sorted(u_ewa), [u0 - k * sd, u0 + k * sd])
print("exact interval", u_exact, "EWA interval", u_ewa)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.8, 4.2), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.28))

# ---------------- (a) geometry
xs = np.linspace(-0.6, 3.4, 300)
zs = np.linspace(0.0, 4.2, 300)
XX, ZZ = np.meshgrid(xs, zs)
G = np.exp(-((XX - t0[0]) ** 2 + (ZZ - t0[1]) ** 2) / (2 * s * s))
axa.imshow(G, extent=(xs[0], xs[-1], zs[0], zs[-1]), origin="lower", cmap="Greys", vmin=0, vmax=2.6, zorder=0,
           interpolation="bilinear")
tt = np.linspace(0, 2 * np.pi, 200)
axa.plot(t0[0] + rho * np.cos(tt), t0[1] + rho * np.sin(tt), color=C["main"], lw=1.2, zorder=3)
axa.plot([-0.6, 3.4], [1, 1], color=C["main"], lw=1.4, zorder=2)
axa.plot([-0.6, 3.4], [z0, z0], color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=2)
axa.text(3.35, 0.95, r"image line $t_z = f$", fontsize=10, va="top", ha="right", zorder=7,
          bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
axa.text(-0.57, z0 + 0.06, "depth line", fontsize=9.5, color=C["aux"], va="bottom", zorder=7,
          bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
# central ray
axa.plot([0, 1.5 * t0[0]], [0, 1.5 * t0[1]], color=C["aux"], lw=0.8, zorder=2)
# exact rays
for a in angs:
    L = 4.2 / np.cos(a)
    axa.plot([0, L * np.sin(a)], [0, L * np.cos(a)], color=C["accent"], lw=1.6, zorder=4)
# parallel (linearized) lines and feet
for p, q in zip(tang, feet):
    a0 = p - 1.6 * dirn
    a1 = p + 1.4 * dirn
    axa.plot([a0[0], a1[0]], [a0[1], a1[1]], color=C["tangent"], lw=1.4, ls=(0, (5, 3)), zorder=4)
    axa.plot([0, q[0]], [0, q[1]], color=C["tangent"], lw=0.8, zorder=4)
    axa.plot(*q, "o", color=C["tangent"], ms=4, zorder=6)
axa.text(feet[0][0] - 0.06, feet[0][1] - 0.08, r"$P_-$", color=C["tangent"], fontsize=11, ha="right", va="top", zorder=7,
          bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
axa.text(feet[1][0] + 0.08, feet[1][1] - 0.08, r"$P_+$", color=C["tangent"], fontsize=11, va="top", zorder=7,
          bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
# intervals on the image line
for (lo, hi), yb, col, lab in ((u_exact, 1.12, C["accent"], "exact"), (u_ewa, 0.88, C["tangent"], "EWA")):
    axa.plot([lo, hi], [yb, yb], color=col, lw=2.2, zorder=5)
    for xe in (lo, hi):
        axa.plot([xe, xe], [yb - 0.07, yb + 0.07], color=col, lw=2.0, zorder=5)
    axa.text(-0.55, yb, lab, color=col, fontsize=10, va="center", ha="left", zorder=6,
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
axa.plot(0, 0, "o", color=C["main"], ms=5, zorder=6)
axa.text(0.08, 0.02, r"$O$", fontsize=12)
axa.plot(*t0, "o", color=C["main"], ms=4, zorder=6)
axa.text(t0[0] + 0.07, t0[1] + 0.05, r"$t_0$", fontsize=12)
axa.text(2.35, 3.95, "exact rays", color=C["accent"], fontsize=10.5)
axa.text(2.2, 0.06, "EWA lines:\nparallel to " + r"$Ot_0$", color=C["tangent"], fontsize=10.5, zorder=7,
          bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
axa.text(1.5 * t0[0] - 0.05, 1.5 * t0[1] + 0.05, r"central ray $Ot_0$", color=C["aux"], fontsize=9.5, ha="right",
          va="bottom", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
axa.set_xlim(-0.6, 3.4)
axa.set_ylim(0, 4.2)
axa.set_aspect("equal")
axa.set_xlabel(r"$t_x$")
axa.set_ylabel(r"$t_z$ (depth)")
axa.set_title("(a) rays in the $t_x t_z$-plane", fontsize=11)
axa.tick_params(labelsize=9)

# ---------------- (b) densities on the image line
uu = np.linspace(-0.4, 1.9, 1151)
zz = np.linspace(0.02, 6.0, 4000)
UU, ZZ2 = np.meshgrid(uu, zz, indexing="ij")
dens = (np.exp(-((UU * ZZ2 - t0[0]) ** 2 + (ZZ2 - t0[1]) ** 2) / (2 * s * s)) / (2 * np.pi * s * s) * ZZ2).sum(1) \
    * (zz[1] - zz[0])
assert abs(dens.sum() * (uu[1] - uu[0]) - 1.0) < 3e-3
ewa = np.exp(-(uu - u0) ** 2 / (2 * sd * sd)) / np.sqrt(2 * np.pi * sd * sd)
axb.fill_between(uu, dens, color=C["accent"], alpha=0.18, lw=0)
axb.plot(uu, dens, color=C["accent"], lw=1.8, label=r"exact $p(u)$")
axb.plot(uu, ewa, color=C["tangent"], lw=1.8, label="EWA Gaussian")
axb.axvline(u0, color=C["tangent"], lw=0.8, ls=(0, (3, 3)))
mean_ex = (uu * dens).sum() / dens.sum()
axb.axvline(mean_ex, color=C["accent"], lw=0.8, ls=(0, (3, 3)))
axb.text(u0 - 0.03, 1.95, r"$u_0 = \varphi(t_0)$", color=C["tangent"], fontsize=10, ha="right", va="top")
axb.text(mean_ex + 0.03, 1.95, "exact mean", color="#B07700", fontsize=10, ha="left", va="top")
axb.annotate("longer tail\n(near side)", xy=(1.35, 0.25), xytext=(1.3, 0.85), fontsize=10, color="#B07700",
             arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=0.8))
axb.set_xlim(uu[0], uu[-1])
axb.set_ylim(0, 2.3)
axb.set_xlabel(r"image coordinate $u = t_x/t_z$")
axb.set_ylabel("density")
axb.legend(fontsize=9.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55))
axb.set_title("(b) footprint on the image line", fontsize=11)
axb.tick_params(labelsize=9)

dgfig.save(fig, __file__)
