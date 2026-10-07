"""Figure 29.5.2: the same loss, three metrics, three paths (Section 29.5).

Toy problem: a 1D footprint G(x) = exp(-(x - mu)^2 / (2 s^2)) (peak 1, like a 3DGS splat) is fitted to the target
mu* = 1, s* = 1 with the L2 loss L(mu, s) = int (G - G*)^2 dx (closed form; gray contours).
Three updates with step 0.05 from three starts (-0.6, 0.35), (-0.4, 2.6), (2.6, 0.5):
black = gradient descent in (mu, s); blue = gradient descent in (mu, log s);
orange = natural gradient with the Fisher metric of N(mu, s^2), diag(1/s^2, 2/s^2).
Dots every 10 iterations. Self-checks: all nine runs reach the minimum; iteration counts match the verify script.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-5-2-trajectories.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
SP = np.sqrt(np.pi)


def Ltoy(m, sd):
    a = sd * sd + 1.0
    return SP * sd + SP - 2 * np.sqrt(2 * np.pi) * sd / np.sqrt(a) * np.exp(-(m - 1.0) ** 2 / (2 * a))


def gtoy(m, sd, h=1e-6):
    return np.array([(Ltoy(m + h, sd) - Ltoy(m - h, sd)) / (2 * h), (Ltoy(m, sd + h) - Ltoy(m, sd - h)) / (2 * h)])


def run(kind, m, sd, eta=0.05, n=3000):
    P = [(m, sd)]
    for _ in range(n):
        gm, gsd = gtoy(m, sd)
        if kind == "s":
            m, sd = m - eta * gm, sd - eta * gsd
        elif kind == "log":
            m, sd = m - eta * gm, sd * np.exp(-eta * sd * gsd)
        else:
            m, sd = m - eta * sd * sd * gm, sd - eta * sd * sd / 2 * gsd
        P.append((m, sd))
    return np.array(P)


kinds = [("s", r"GD in $(\mu, s)$", C["main"]), ("log", r"GD in $(\mu, \log s)$", C["tangent"]),
         ("nat", "natural gradient (Fisher)", C["accent"])]
expected = {((-0.6, 0.35), "s"): 63, ((-0.6, 0.35), "log"): 66, ((-0.6, 0.35), "nat"): 160,
            ((-0.4, 2.6), "s"): 69, ((-0.4, 2.6), "log"): 62, ((-0.4, 2.6), "nat"): 60,
            ((2.6, 0.5), "s"): 61, ((2.6, 0.5), "log"): 61, ((2.6, 0.5), "nat"): 99}
fig, ax = plt.subplots(figsize=(6.4, 4.6))
M, S = np.meshgrid(np.linspace(-1.2, 3.2, 300), np.linspace(0.05, 3.0, 300))
cs = ax.contour(M, S, Ltoy(M, S), levels=np.linspace(0.1, 3.4, 16), colors="#BBBBBB", linewidths=0.6, zorder=0)
for start in ((-0.6, 0.35), (-0.4, 2.6), (2.6, 0.5)):
    ax.plot(*start, "s", color=C["main"], ms=6, zorder=6)
    for kind, lab, col in kinds:
        P = run(kind, *start)
        d = np.hypot(P[:, 0] - 1, P[:, 1] - 1)
        assert int(np.argmax(d < 0.01)) == expected[(start, kind)]
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.6, zorder=3, label=lab if start == (-0.6, 0.35) else None)
        ax.plot(P[:200:10, 0], P[:200:10, 1], "o", color=col, ms=2.6, zorder=4)
ax.plot(1, 1, "*", color=C["normal"], ms=13, zorder=7)
ax.text(1.06, 1.06, r"minimum $(\mu^*, s^*) = (1, 1)$", fontsize=10, color=C["normal"])
for start, off in (((-0.6, 0.35), (-0.05, -0.22)), ((-0.4, 2.6), (0.08, 0.06)), ((2.6, 0.5), (-0.1, -0.24))):
    ax.text(start[0] + off[0], start[1] + off[1], "start", fontsize=9.5)
ax.set_xlim(-1.2, 3.2)
ax.set_ylim(0.05, 3.0)
ax.set_xlabel(r"center $\mu$")
ax.set_ylabel(r"scale $s$")
ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, loc="upper right")
ax.tick_params(labelsize=9)

dgfig.save(fig, __file__)
