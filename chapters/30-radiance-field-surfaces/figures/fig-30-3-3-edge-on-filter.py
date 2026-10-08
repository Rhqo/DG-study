"""Figure 30.3.3: a splat seen edge-on collapses to a line; the screen-space low-pass filter of 2DGS (Section 30.3).

Flatland, focal length f = 1000 px, splat with scale s = 0.004 centered on the optical axis at depth 2
(so its 1-sigma half-width is s f/z = 2 px when it faces the camera), tilted by theta (angle between t_w and the view ray).
(a) 1-sigma half-width of the footprint on the image (exact, from the ray-splat intersection) versus theta; it follows
    2 cos(theta) px. Dashed: the filter radius sigma = sqrt(2)/2 px (FilterSize in diff-surfel-rasterization).
(b) theta = 88 deg, pixel centres at x = k + 0.4 (the projected center sits 0.4 px from the nearest pixel centre):
    thin orange = G(u(x)) (a spike of half-width 0.07 px), gray dashed = the filter Gaussian exp(-|x - c|^2/(2 sigma^2)),
    black = G_hat = max of the two (2DGS, eq. (11) of [Huang24]); stems = values at pixel centres
    (orange: raw, black: filtered).
Self-checks: half-width formula; raw values at pixel centres < 1e-6; filtered value at the nearest pixel exp(-0.16).

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-3-3-edge-on-filter.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
FPX, S, Z = 1000.0, 0.004, 2.0
SIG = np.sqrt(2) / 2


def splat(phi_deg):
    cc = np.array([0.0, Z])
    ph = np.radians(phi_deg)
    n = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]]) @ np.array([0.0, -1.0])
    return cc, np.array([n[1], -n[0]])


def xi_of_u(u, cc, t):
    p = cc + S * u * t
    return p[0] / p[1]


def u_of_x(xpix, cc, t):
    xi = xpix / FPX
    return -(cc[0] - cc[1] * xi) / (S * (t[0] - t[1] * xi))


phis = np.linspace(0, 89.5, 400)
hw = np.array([abs(xi_of_u(1, *splat(p)) - xi_of_u(-1, *splat(p))) * FPX / 2 for p in phis])
assert np.allclose(hw, 2 * np.cos(np.radians(phis)), atol=2e-3)
phi_cross = np.degrees(np.arccos(SIG / 2))
print(f"half-width = sigma at phi = {phi_cross:.1f} deg")

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 7.6), gridspec_kw=dict(hspace=0.42))
axa.plot(phis, hw, color=C["accent"], lw=2.4, label=r"footprint half-width $\approx 2\cos\theta$")
axa.axhline(SIG, color=C["aux"], lw=1.6, ls=(0, (5, 3)), label=r"filter $\sigma = \sqrt{2}/2$ px")
axa.fill_between(phis, 0, SIG, where=hw < SIG, color=C["covector"], alpha=0.15, lw=0)
axa.axvline(88, color=C["main"], lw=0.9, ls=":")
axa.text(87.0, 1.55, r"$\theta = 88^\circ$" + "\n(panel b)", ha="right", fontsize=11)
axa.text(55, 0.22, "narrower than the filter:\nmay fall between pixels", fontsize=11, ha="center",
         bbox=dict(fc="white", ec="none", pad=1))
axa.set_xlim(0, 90)
axa.set_ylim(0, 2.2)
axa.set_xlabel(r"tilt $\theta$ (angle between $t_w$ and the view ray, deg)")
axa.set_ylabel("half-width (px)")
axa.legend(loc="upper right", frameon=False, fontsize=11)
axa.set_title("(a) the footprint vanishes as the splat turns edge-on", fontsize=13)

cc, t = splat(88.0)
xs = np.linspace(-3.2, 3.2, 20001)
Graw = np.exp(-0.5 * u_of_x(xs, cc, t) ** 2)
Gfil = np.exp(-0.5 * (xs - 0.0) ** 2 / SIG ** 2)
Ghat = np.maximum(Graw, Gfil)
pix = np.arange(-3, 3) + 0.4
gr = np.exp(-0.5 * u_of_x(pix, cc, t) ** 2)
gh = np.maximum(gr, np.exp(-0.5 * pix ** 2 / SIG ** 2))
assert gr.max() < 1e-6 and abs(gh.max() - np.exp(-0.16)) < 1e-12
axb.plot(xs, Gfil, color=C["aux"], lw=1.4, ls=(0, (5, 3)), label="filter Gaussian")
axb.plot(xs, Graw, color=C["accent"], lw=1.6, label=r"$G(u(x))$")
axb.plot(xs, Ghat, color=C["main"], lw=2.0, label=r"$\hat G = \max$")
mk, st, bl = axb.stem(pix, gh, linefmt="k-", markerfmt="ko", basefmt=" ")
plt.setp(st, lw=1.2)
plt.setp(mk, ms=6, mfc="white")
axb.plot(pix, gr, "o", color=C["accent"], ms=6, zorder=6)
axb.annotate("raw value at every\npixel centre < " + r"$10^{-6}$", xy=(pix[3], 0.0), xytext=(1.9, 0.45), fontsize=11,
             ha="center", arrowprops=dict(arrowstyle="->", lw=0.8, color=C["accent"]))
for p in pix:
    axb.axvline(p, color="#DDDDDD", lw=0.6, zorder=0)
axb.set_xlim(-3.2, 3.2)
axb.set_ylim(-0.04, 1.12)
axb.set_xlabel("image x (px), pixel centres at k + 0.4")
axb.set_ylabel("value")
axb.legend(loc="upper left", frameon=False, fontsize=10.5)
axb.set_title(r"(b) $\theta = 88^\circ$: the filter keeps the splat visible", fontsize=13)
dgfig.save(fig, __file__)
