"""Figure 44.1.3: a geodesic that is not a minimum of the energy (Section 44.1, Example 44.1.8).

The equator arc gamma(t) = x(pi/2, t), 0 <= t <= ell, of the unit sphere (x(theta, phi) of GUIDELINES 7) and its
variation gamma_eps(t) = x(pi/2 - eps sin(pi t/ell), t), which keeps the endpoints.
(a) ell = 1.4 pi seen from elev 25: the arc (black) and gamma_eps for eps = +-0.35 (orange, blue).
(b) E(eps) - E(0) for ell = 0.6 pi, pi, 1.4 pi (E = (1/2) int |gamma'|^2 dt, trapezoid rule with 4001 points);
    E''(0) = (ell/2)(pi^2/ell^2 - 1) (Example 44.1.8).
Self-checks: E''(0) by finite differences matches the closed form; the variation fixes the endpoints.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-1-3-energy-saddle.py
"""

import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS


def X(th, ph):
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th) + 0 * ph], -1)


def energy(eps, ell, n=4001):
    t = np.linspace(0, ell, n)
    w = np.pi / ell
    th = np.pi / 2 - eps * np.sin(w * t)
    dth = -eps * w * np.cos(w * t)
    integrand = 0.5 * (dth ** 2 + np.sin(th) ** 2)
    return float(np.trapezoid(integrand, t))


for ell in (0.6 * np.pi, np.pi, 1.4 * np.pi):
    h = 1e-3
    E2 = (energy(h, ell) - 2 * energy(0, ell) + energy(-h, ell)) / h ** 2
    exact = ell / 2 * (np.pi ** 2 / ell ** 2 - 1)
    assert abs(E2 - exact) < 1e-5, (E2, exact)
    print(f"ell = {ell / np.pi:.1f} pi: E''(0) = {exact:+.4f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.0, 1.15]})
cam = dm.Ortho(elev=25, azim=0)


def draw3(ax, P, **kw):
    P = np.asarray(P)
    vis = P @ cam.view > -1e-9
    Q = cam(P)
    for mask, ls in ((vis, kw.pop("ls", "-")), (~vis, ":")):
        segm = np.where(mask, 1.0, np.nan)
        ax.plot(Q[:, 0] * segm, Q[:, 1] * segm, ls=ls, **kw)


tt = np.linspace(0, 2 * np.pi, 400)
axa.plot(np.cos(tt), np.sin(tt), color=col["main"], lw=0.9)
for th in (np.pi / 4, np.pi / 2, 3 * np.pi / 4):
    draw3(axa, X(th, tt), color=col["aux"], lw=0.4)
ell = 1.4 * np.pi
t = np.linspace(0, ell, 300)
ph = t - ell / 2           # centre the arc on the front meridian phi = 0
draw3(axa, X(np.pi / 2 + 0 * t, ph), color=col["main"], lw=2.0)
for eps, c in ((0.35, col["accent"]), (-0.35, col["tangent"])):
    draw3(axa, X(np.pi / 2 - eps * np.sin(np.pi * t / ell), ph), color=c, lw=1.5)
for P_ in (X(np.pi / 2, -ell / 2), X(np.pi / 2, ell / 2)):
    c2 = cam(P_[None])[0]
    axa.plot(*c2, "o", color=col["main"], ms=4)
axa.text(0.55, -0.62, r"$\varepsilon = -0.35$", fontsize=10, color=col["tangent"])
axa.text(0.55, 0.5, r"$\varepsilon = 0.35$", fontsize=10, color=col["accent"])
axa.set_aspect("equal")
axa.set_xlim(-1.1, 1.1)
axa.set_ylim(-1.1, 1.1)
axa.set_axis_off()
axa.set_title(r"(a) equator arc of length $1.4\pi$", fontsize=11)

eps = np.linspace(-0.6, 0.6, 121)
for ell, c, lab in ((0.6 * np.pi, col["third"], r"$\ell = 0.6\pi$"), (np.pi, col["aux"], r"$\ell = \pi$"),
                    (1.4 * np.pi, col["normal"], r"$\ell = 1.4\pi$")):
    E0 = energy(0, ell)
    axb.plot(eps, [energy(e, ell) - E0 for e in eps], color=c, lw=1.8, label=lab)
axb.axhline(0, color=col["aux"], lw=0.5)
axb.set_xlabel(r"$\varepsilon$")
axb.set_ylabel(r"$E(\varepsilon) - E(0)$")
axb.legend(frameon=False, loc="upper center")
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("(b) energy along the variation", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.88, bottom=0.17, wspace=0.28)
dgfig.save(fig, __file__)
