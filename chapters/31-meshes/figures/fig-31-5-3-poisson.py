"""Figure 31.5.3: Poisson reconstruction in 1D and 2D (Section 31.5).

(a) 1D: the indicator chi of the interval [-0.5, 0.6] smoothed by a Gaussian of width 0.04 (black), its derivative
    (grey), which is the smoothed inward normal field V: +delta at the left end (inward = +x), -delta at the right end
    (inward = -x). Integrating V recovers chi up to a constant (orange dashed).
(b), (c) 2D: the closed curve r(theta) = 1 + 0.15 cos 3 theta + 0.1 sin 2 theta (dashed black), sampled at 160 equally
    spaced theta, with inward unit normals scaled by the arc length per sample. V = sum_s |P_s| G_sigma(q - p_s) n_s
    (sigma = 2.5 grid cells, grid 256 x 256 on [-1.6, 1.6)^2, periodic), and Delta chi = div V solved with the FFT.
    The surface is the level set chi = mean of chi at the samples [Kazhdan06].
    (b) samples with theta in (0.4, 1.2) removed (a gap); (c) all samples, but the normals with theta in (2.4, 3.3)
    flipped (pointing outward).
    Colour: chi; arrows: the inward normals (every second sample); orange: the reconstructed curve.
Self-check: chi jumps by 1 across the curve; intersection over union with the true region: 0.995 (gap),
0.853 (flipped); 0.999 with all samples and correct normals.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-5-3-poisson.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
N, LB = 256, 1.6
xs = (np.arange(N) / N) * 2 * LB - LB
dx = xs[1] - xs[0]
X, Y = np.meshgrid(xs, xs, indexing="ij")
TH = np.linspace(0, 2 * np.pi, 160, endpoint=False)


def curve(th):
    r = 1 + 0.15 * np.cos(3 * th) + 0.1 * np.sin(2 * th)
    dr = -0.45 * np.sin(3 * th) + 0.2 * np.cos(2 * th)
    p = np.stack([r * np.cos(th), r * np.sin(th)], -1)
    t = np.stack([dr * np.cos(th) - r * np.sin(th), dr * np.sin(th) + r * np.cos(th)], -1)
    return p, t


def poisson(th, flip=None, sigma=2.5 * dx):
    p, t = curve(th)
    ds = np.linalg.norm(t, axis=1) * (2 * np.pi / len(TH))
    tn = t / np.linalg.norm(t, axis=1, keepdims=True)
    n_in = np.stack([-tn[:, 1], tn[:, 0]], -1)          # counterclockwise curve: the left normal points inward
    if flip is not None:
        n_in[flip] *= -1
    Vx, Vy = np.zeros((N, N)), np.zeros((N, N))
    for q, n, w in zip(p, n_in, ds):
        g = np.exp(-((X - q[0]) ** 2 + (Y - q[1]) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        Vx += w * n[0] * g
        Vy += w * n[1] * g
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    chi_hat = -(1j * KX * np.fft.fft2(Vx) + 1j * KY * np.fft.fft2(Vy)) / K2   # Delta chi = div V
    chi_hat[0, 0] = 0.0
    chi = np.real(np.fft.ifft2(chi_hat))
    ii = np.clip(np.round((p[:, 0] + LB) / dx).astype(int), 0, N - 1)
    jj = np.clip(np.round((p[:, 1] + LB) / dx).astype(int), 0, N - 1)
    return chi, chi[ii, jj].mean(), p, n_in


ang = np.arctan2(Y, X)
true_in = np.hypot(X, Y) < 1 + 0.15 * np.cos(3 * ang) + 0.1 * np.sin(2 * ang)


def iou(chi, iso):
    inside = chi > iso
    return (inside & true_in).sum() / (inside | true_in).sum()


chi_f, iso_f, _, _ = poisson(TH)
assert round(iou(chi_f, iso_f), 3) == 0.999 and abs((chi_f.max() - chi_f.min()) - 1) < 0.05
gap = (TH < 0.4) | (TH > 1.2)
chi_g, iso_g, p_g, n_g = poisson(TH[gap])
flip = (TH > 2.4) & (TH < 3.3)
chi_x, iso_x, p_x, n_x = poisson(TH, flip)
assert round(iou(chi_g, iso_g), 3) == 0.995 and round(iou(chi_x, iso_x), 3) == 0.853

fig = plt.figure(figsize=(7.6, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 1.0], hspace=0.32, wspace=0.08)

# ---------------------------------------------------------------- (a) 1D
ax = fig.add_subplot(gs[0, :])
x = np.linspace(-1, 1, 2001)
h1 = x[1] - x[0]
s1 = 0.04
chi1 = 0.5 * (np.tanh((x + 0.5) / (s1 * 0.8)) - np.tanh((x - 0.6) / (s1 * 0.8)))
V1 = np.gradient(chi1, h1)
rec = np.cumsum(V1) * h1
ax.plot(x, chi1, color=C["main"], lw=2.0, label=r"smoothed indicator $\chi$")
ax.plot(x, V1 / np.abs(V1).max() * 0.9, color=C["aux"], lw=1.2, label=r"$V = \chi'$ (scaled)")
ax.plot(x, rec, color=C["accent"], lw=1.6, ls=(0, (5, 3)), label=r"$\int\, V\,dx$")
ax.annotate("inward normal +x", xy=(-0.5, 0.9), xytext=(-0.98, 0.55), fontsize=10, color=C["tangent"],
            arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=0.9))
ax.annotate("inward normal -x", xy=(0.6, -0.9), xytext=(0.65, -0.55), fontsize=10, color=C["normal"],
            arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=0.9))
assert np.abs(rec - chi1).max() < 1e-2
ax.axhline(0, color=C["aux"], lw=0.5)
ax.set_xlim(-1, 1)
ax.set_ylim(-1.1, 1.25)
ax.set_xlabel(r"$x$")
ax.legend(fontsize=9.5, frameon=False, loc="lower left", ncol=3, bbox_to_anchor=(0.0, 0.02))
ax.set_title(r"(a) 1D: the derivative of the indicator is the inward normal on the boundary", fontsize=11)

# ---------------------------------------------------------------- (b), (c) 2D
pt, _ = curve(np.linspace(0, 2 * np.pi, 400))
for col, (title, chi, iso, p, n, mark) in enumerate(
        (("(b) a gap in the samples", chi_g, iso_g, p_g, n_g, None),
         ("(c) some normals flipped", chi_x, iso_x, curve(TH)[0], n_x, flip))):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(chi.T, origin="lower", extent=(-LB, LB, -LB, LB), cmap="Greys", vmin=chi.min(), vmax=chi.max(),
              alpha=0.55)
    ax.plot(*np.vstack([pt, pt[:1]]).T, color=C["main"], lw=1.0, ls=(0, (4, 3)))
    ax.contour(xs, xs, chi.T, levels=[iso], colors=[C["accent"]], linewidths=2.0)
    sel = np.arange(0, len(p), 2)
    colors = np.array([C["tangent"]] * len(p), dtype=object)
    if mark is not None:
        colors[mark] = C["normal"]
    for k in sel:
        ax.annotate("", xy=p[k] + 0.18 * n[k], xytext=p[k],
                    arrowprops=dict(arrowstyle="-|>", color=colors[k], lw=0.9, mutation_scale=6))
    ax.plot(p[:, 0], p[:, 1], "o", ms=1.8, color=C["main"])
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
fig.text(0.5, 0.02, "dashed: true curve,  orange: reconstruction,  arrows: normals (vermillion = flipped)",
         ha="center", fontsize=10)
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
dgfig.save(fig, __file__)
