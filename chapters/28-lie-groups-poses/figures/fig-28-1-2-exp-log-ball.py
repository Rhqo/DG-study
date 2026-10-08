"""Figure 28.1.2: exp is not one-to-one on R^3, and log jumps at theta = pi (Section 28.1).

(a) The ball model of SO(3) (Tour Figure 0.6.4), cut by the plane w^2 = 0. The ray t -> t*n (n at 35 degrees from
    the w^1 axis, 0 <= t <= 2 pi) is mapped by Exp and pulled back by the robust Log: for t <= pi the point
    Log(Exp(t n)) = t n runs out to the boundary point pi*n (blue); for t > pi it reappears at -pi*n and runs back
    to 0 (orange), because Exp(t n) = Exp((t - 2 pi) n).
(b) The component <Log(Exp(t n)), n> (black) and the rotation angle |Log(Exp(t n))| (gray dashed) as functions of t.
    The component jumps from +pi to -pi at t = pi; the angle is the tent function min(t, 2 pi - t).
Self-check: Log(Exp(t n)) = t n for t < pi and (t - 2 pi) n for t > pi to 1e-12; Exp(pi n) = Exp(-pi n).

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-1-2-exp-log-ball.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dglie as L  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
ang = np.radians(35)
n = np.array([np.cos(ang), 0.0, np.sin(ang)])       # unit vector in the plane w^2 = 0
ts = np.linspace(0, 2 * np.pi, 721)
logs = np.array([L.logSO3(L.expSO3(t * n)) for t in ts])
comp = logs @ n
angle = np.linalg.norm(logs, axis=1)

# self-checks
for t, w in zip(ts, logs):
    if t < np.pi - 1e-6:
        assert np.allclose(w, t * n, atol=1e-12)
    elif t > np.pi + 1e-6:
        assert np.allclose(w, (t - 2 * np.pi) * n, atol=1e-12)
assert np.allclose(L.expSO3(np.pi * n), L.expSO3(-np.pi * n), atol=1e-14)
assert np.allclose(angle, np.minimum(ts, 2 * np.pi - ts), atol=1e-12)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.32))

# (a) ball cross-section
th = np.linspace(0, 2 * np.pi, 400)
ax.fill(np.pi * np.cos(th), np.pi * np.sin(th), color=C["region"], alpha=0.25, lw=0)
ax.plot(np.pi * np.cos(th), np.pi * np.sin(th), color=C["main"], lw=1.2)
ax.plot(np.pi / 2 * np.cos(th), np.pi / 2 * np.sin(th), color=C["aux"], lw=0.8, ls=(0, (3, 3)))
first = ts <= np.pi
P = logs[:, [0, 2]]
ax.plot(P[first, 0], P[first, 1], color=C["tangent"], lw=2.2, zorder=4)
ax.plot(P[~first, 0], P[~first, 1], color=C["accent"], lw=2.2, ls=(0, (5, 2)), zorder=4)
pn = np.pi * n[[0, 2]]
ax.plot(*pn, "s", color=C["accent"], ms=7, zorder=6)
ax.plot(*(-pn), "s", color=C["accent"], ms=7, zorder=6)
ax.plot(0, 0, "o", color=C["main"], ms=5, zorder=6)
ax.annotate(r"$\pi n$", pn, xytext=(pn[0] + 0.15, pn[1] + 0.45), fontsize=12, color=C["main"])
ax.annotate(r"$-\pi n$", -pn, xytext=(-pn[0] - 0.55, -pn[1] - 0.75), fontsize=12, color=C["main"])
ax.annotate("same rotation", xy=(-pn[0] + 0.05, -pn[1] + 0.12), xytext=(-3.45, 2.75), fontsize=11,
            color=C["accent"], arrowprops=dict(arrowstyle="->", color=C["accent"], lw=0.9))
ax.annotate("", xy=(pn[0] - 0.05, pn[1] + 0.1), xytext=(-1.05, 2.85),
            arrowprops=dict(arrowstyle="->", color=C["accent"], lw=0.9))
q1 = 0.5 * np.pi * n[[0, 2]]
ax.text(0.05, 1.4, r"$0 \leq t \leq \pi$", color=C["tangent"], fontsize=11,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
q2 = -0.55 * np.pi * n[[0, 2]]
ax.text(-3.05, -0.55, r"$\pi < t \leq 2\pi$", color=C["accent"], fontsize=11,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
ax.text(0.2, -0.5, r"$0 \mapsto I_3$", fontsize=11)
ax.text(-1.0, -np.pi - 0.62, r"$|\omega| = \pi$ (boundary)", fontsize=10.5, color=C["main"])
ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-3.9, 3.6)
ax.set_aspect("equal")
ax.set_xlabel(r"$\omega^1$")
ax.set_ylabel(r"$\omega^3$")
ax.set_title(r"(a) $\mathrm{Log}(\mathrm{Exp}(t\,n))$ in the ball", fontsize=12)
ax.tick_params(labelsize=10)

# (b) component along n and angle
bx.axhline(0, color=C["aux"], lw=0.6)
bx.plot(ts[first], comp[first], color=C["tangent"], lw=2.0, label=r"$\langle \mathrm{Log}, n\rangle$, $t \leq \pi$")
bx.plot(ts[~first], comp[~first], color=C["accent"], lw=2.0, ls=(0, (5, 2)),
        label=r"$\langle \mathrm{Log}, n\rangle$, $t > \pi$")
bx.plot(ts, angle, color=C["aux"], lw=1.4, ls=(0, (2, 2)), label=r"angle $|\mathrm{Log}|$")
bx.plot([np.pi, np.pi], [np.pi, -np.pi], color=C["main"], lw=0.8, ls=":")
bx.plot(np.pi, np.pi, "s", color=C["accent"], ms=6)
bx.plot(np.pi, -np.pi, "s", color=C["accent"], ms=6)
bx.annotate(r"jump: $+\pi \to -\pi$", xy=(np.pi, 0), xytext=(3.55, -1.3), fontsize=11,
            arrowprops=dict(arrowstyle="->", lw=0.9))
bx.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
bx.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
bx.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
bx.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
bx.set_xlabel(r"$t$ (parameter of the ray $t\,n$)")
bx.set_ylabel("value")
bx.legend(loc="lower left", fontsize=10.5, frameon=False)
bx.set_title(r"(b) along the ray: $\mathrm{Log}$ is discontinuous at $t = \pi$", fontsize=12)
bx.set_xlim(0, 2 * np.pi)
bx.set_ylim(-3.6, 3.9)

dgfig.save(fig, __file__)
