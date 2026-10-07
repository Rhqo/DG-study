"""Figure 29.3.4: adding covariance is a low-pass filter; dilation with and without normalization (Section 29.3).

A thin, long 2D Gaussian on the screen, Sigma = R(20deg) diag(100, 0.05) R(20deg)^T px^2 (std 10 px along the line,
0.22 px across), peak value 1, centered in an 8 x 5 pixel patch.
(a) The values G(pixel center) that a rasterizer writes (each square = one pixel, white 0 ... black 1), with the 2-sigma
    ellipse of Sigma (blue). Title: the range of the per-column sums of the pixel values.
(b) The same for the dilated Sigma + 0.3 I (the screen-space dilation of the official 3DGS implementation), with its
    2-sigma ellipse (orange dashed).
(c) Profile across the thin direction (1D): original G (std 0.224, peak 1, blue), plain dilation (std 0.592, peak 1,
    orange dashed), and the normalized convolution with N(0, 0.3) (std 0.592, peak sqrt(0.05/0.35) = 0.378, green).
    Numbers: area under each curve = sqrt(2 pi) * std * peak.
Self-checks: the normalized curve equals the numerical convolution of the blue curve with N(0, 0.3); the per-column
sums are 0.23-0.90 (original) and 1.45-1.58 (dilated).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-3-4-dilation.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
a = np.radians(20)
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
Sig = R @ np.diag([100.0, 0.05]) @ R.T
SigD = Sig + 0.3 * np.eye(2)
center = np.array([4.0, 2.5])


def ell(Sm, k=2.0, n=400):
    w, V = np.linalg.eigh(Sm)
    t = np.linspace(0, 2 * np.pi, n)
    return center + (np.stack([np.cos(t), np.sin(t)], 1) * k * np.sqrt(w)) @ V.T


px, py = np.meshgrid(np.arange(8) + 0.5, np.arange(5) + 0.5)
P = np.stack([px, py], -1) - center
G0 = np.exp(-0.5 * np.einsum("...i,ij,...j->...", P, np.linalg.inv(Sig), P))
G1 = np.exp(-0.5 * np.einsum("...i,ij,...j->...", P, np.linalg.inv(SigD), P))
cs0, cs1 = G0.sum(0), G1.sum(0)
print("column sums original", np.round(cs0, 2), "dilated", np.round(cs1, 2))
assert (round(cs0.min(), 2), round(cs0.max(), 2)) == (0.23, 0.90)
assert (round(cs1.min(), 2), round(cs1.max(), 2)) == (1.45, 1.58)

fig = plt.figure(figsize=(7.8, 4.1))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.55, wspace=0.28)
axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
for ax, Gv, Sm, col, ls, title in (
        (axa, G0, Sig, C["tangent"], "-",
         r"(a) $\Sigma$: column sums %.2f–%.2f" % (cs0.min(), cs0.max())),
        (axb, G1, SigD, C["accent"], (0, (4, 2.5)),
         r"(b) $\Sigma + 0.3\,I$: column sums %.2f–%.2f" % (cs1.min(), cs1.max()))):
    im = ax.imshow(Gv, extent=(0, 8, 5, 0), cmap="Greys", vmin=0, vmax=1, interpolation="nearest", zorder=0)
    for x in range(9):
        ax.plot([x, x], [0, 5], color="#BBBBBB", lw=0.5, zorder=1)
    for y in range(6):
        ax.plot([0, 8], [y, y], color="#BBBBBB", lw=0.5, zorder=1)
    ax.plot(*ell(Sm).T, color=col, lw=1.5, ls=ls, zorder=3)
    ax.plot(px.ravel(), py.ravel(), ".", color="#888888", ms=2.0, zorder=2)
    ax.set_xlim(0, 8)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11.5)
    ax.tick_params(labelsize=9.5)
    ax.set_ylabel("pixel $v$", fontsize=11.5)
axb.set_xlabel("pixel $u$", fontsize=11.5)
axa.tick_params(labelbottom=False)

# (c) 1D profile across the thin axis
x = np.linspace(-2.5, 2.5, 2001)
s0, sD = np.sqrt(0.05), np.sqrt(0.35)
g0 = np.exp(-x ** 2 / (2 * s0 ** 2))
gD = np.exp(-x ** 2 / (2 * sD ** 2))
gN = np.sqrt(0.05 / 0.35) * gD
ker = np.exp(-x ** 2 / (2 * 0.3)) / np.sqrt(2 * np.pi * 0.3)
conv = np.convolve(g0, ker, mode="same") * (x[1] - x[0])
assert np.max(np.abs(conv - gN)) < 1e-4
area = lambda sd, pk: np.sqrt(2 * np.pi) * sd * pk
axc.plot(x, g0, color=C["tangent"], lw=1.8, label="original: area %.2f" % area(s0, 1.0))
axc.plot(x, gD, color=C["accent"], lw=1.8, ls=(0, (4, 2.5)), label="dilated: area %.2f" % area(sD, 1.0))
axc.plot(x, gN, color=C["third"], lw=1.8, label="normalized: area %.2f" % area(sD, np.sqrt(0.05 / 0.35)))
for xx in np.arange(-2, 3):
    axc.axvline(xx, color="#DDDDDD", lw=0.6, zorder=0)
axc.set_xlim(-2.5, 2.5)
axc.set_ylim(0, 1.45)
axc.set_xlabel("offset across the line (px)", fontsize=12)
axc.set_ylabel("value", fontsize=12)
axc.legend(fontsize=10.5, frameon=False, loc="upper center", ncol=1)
axc.set_title("(c) cross-section", fontsize=12)
axc.tick_params(labelsize=10)

dgfig.save(fig, __file__)
