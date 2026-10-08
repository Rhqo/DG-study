"""Figure 30.4.4: the inverse depth of a plane is affine in the pixel coordinates; the depth is not (Section 30.4).

Flatland, normalized image coordinate u = x/z (focal length 1, u in [-0.5, 0.5], about 53 deg field of view).
Planes (lines) through (0, 3) whose normal makes the angle theta = 0, 30, 45, 60 deg with the optical axis:
d(u) = 3 cos(theta)/(cos(theta) - u sin(theta)).
(a) depth d(u): curved (convex) for theta != 0;
(b) inverse depth 1/d(u) = (1 - u tan(theta))/3: straight lines.
Self-checks: second differences of 1/d vanish to machine precision, those of d do not.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-4-4-inverse-depth.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
u = np.linspace(-0.5, 0.5, 401)
tilts = [(0, C["main"]), (30, C["tangent"]), (45, C["third"]), (60, C["accent"])]

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 7.4), sharex=True, gridspec_kw=dict(hspace=0.18))
for deg, col in tilts:
    th = np.radians(deg)
    d = 3 * np.cos(th) / (np.cos(th) - u * np.sin(th))
    rho = 1 / d
    assert np.max(np.abs(np.diff(rho, 2))) < 1e-15
    if deg:
        assert np.max(np.abs(np.diff(d, 2))) > 1e-6
    axa.plot(u, d, color=col, lw=2.2, label=rf"$\theta = {deg}^\circ$")
    axb.plot(u, rho, color=col, lw=2.2)
    chord = d[0] + (d[-1] - d[0]) * (u - u[0]) / (u[-1] - u[0])
    if deg == 45:
        axa.plot(u, chord, color=col, lw=1.2, ls=(0, (4, 3)))
        k = np.argmin(np.abs(u))
        assert abs(chord[k] - 4.0) < 1e-9 and abs(d[k] - 3.0) < 1e-9
        axa.annotate("", xy=(0.0, 3.0), xytext=(0.0, 4.0), arrowprops=dict(arrowstyle="<->", lw=1.0, color=C["main"]))
        axa.annotate(r"chord of $\theta = 45^\circ$ at $u = 0$: 4.0" + "\nvs depth 3.0: not affine",
                     xy=(0.0, 3.6), xytext=(-0.47, 8.6), fontsize=11.5, arrowprops=dict(arrowstyle="->", lw=0.9))
axa.set_ylim(0, 12.5)
axa.set_ylabel(r"depth $d(u)$")
axa.set_title("(a) depth of tilted planes: curved", fontsize=13)
axb.set_ylabel(r"inverse depth $1/d(u)$")
axb.set_xlabel(r"normalized image coordinate $u = x/z$")
axb.text(0.30, 0.86, r"$1/d = (1 - u\tan\theta)/3$", fontsize=12.5, transform=axb.transAxes)
axb.set_title("(b) inverse depth of the same planes: straight lines", fontsize=13)
axb.set_xlim(-0.5, 0.5)
hh, ll = axa.get_legend_handles_labels()
axb.legend(hh, ll, loc="lower left", frameon=False, fontsize=11, ncol=2, title="same color = same plane",
           title_fontsize=10.5)
dgfig.save(fig, __file__)
