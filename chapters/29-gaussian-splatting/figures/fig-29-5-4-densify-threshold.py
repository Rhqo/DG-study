"""Figure 29.5.4: what the NDC metric of the densification rule does (Section 29.5).

The official implementation compares tau = 2e-4 with the norm of dL/du_ndc = diag(W/2, H/2) dL/du_px.
(a) Resolution. Toy experiment: one 2D Gaussian splat (opacity 0.8, center (0.1, -0.05) and covariance
    diag(0.03^2, 0.05^2) in NDC units) over a black background, target = the same splat shifted by (0.012, 0.004) NDC,
    loss = mean over pixels of |I - I*| (the L1 term of 3DGS). The same scene is rendered at 16:9 resolutions
    W = 240, 480, 960, 1920. Black: the norm of the pixel-unit gradient dL/du_px (falls like 1/W).
    Orange: the norm of the NDC gradient (constant to 0.2%). Both are divided by their value at W = 1920.
(b) Aspect ratio. At 1920 x 1080 the set ||diag(W/2, H/2) g|| <= tau in the pixel-gradient plane g (units 1e-7 per
    pixel) is the ellipse with semi-axes 2 tau/W = 2.08 and 2 tau/H = 3.70 (blue). The two black points have the
    same pixel length 3: (3, 0) has NDC norm 2.88e-4 > tau (densified), (0, 3) has 1.62e-4 < tau (not densified).
    Gray dashed: the circle of pixel length 3.
Self-checks: 1/W scaling of the pixel gradient and constancy of the NDC gradient; semi-axes; the two NDC norms.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-5-4-densify-threshold.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
tau = 2e-4


def toy_grads(W, H, u_ndc=(0.1, -0.05), S_ndc=np.diag([0.03 ** 2, 0.05 ** 2]), shift=(0.012, 0.004), o=0.8):
    """Gradient of mean |I - I*| w.r.t. the splat center, in pixel units and in NDC units."""
    to_px = lambda v, S: ((v + 1) * S - 1) / 2                     # ndc2Pix of the official rasterizer
    u = np.array([to_px(u_ndc[0], W), to_px(u_ndc[1], H)])
    us = np.array([to_px(u_ndc[0] + shift[0], W), to_px(u_ndc[1] + shift[1], H)])
    A = np.diag([W / 2, H / 2])                                     # d pixel / d ndc
    Si = np.linalg.inv(A @ S_ndc @ A)
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    def img(c):
        dx, dy = X - c[0], Y - c[1]
        return o * np.exp(-0.5 * (Si[0, 0] * dx * dx + 2 * Si[0, 1] * dx * dy + Si[1, 1] * dy * dy)), dx, dy

    I, dx, dy = img(u)
    T, _, _ = img(us)
    w = np.sign(I - T) * I / (W * H)                                # d mean|I - T| / dI  times  I
    g_px = np.array([np.sum(w * (Si[0, 0] * dx + Si[0, 1] * dy)), np.sum(w * (Si[1, 0] * dx + Si[1, 1] * dy))])
    return g_px, A @ g_px


Ws = np.array([240, 360, 480, 720, 960, 1440, 1920])
G = [toy_grads(W, W * 9 // 16) for W in Ws]
npx = np.array([np.linalg.norm(g[0]) for g in G])
nndc = np.array([np.linalg.norm(g[1]) for g in G])
assert np.allclose(npx * Ws / (npx[-1] * Ws[-1]), 1, atol=0.01)      # pixel gradient ~ 1/W
assert np.allclose(nndc / nndc[-1], 1, atol=0.002)                    # NDC gradient ~ constant

fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.38))

# (a)
axa.loglog(Ws, npx / npx[-1], "o-", color=C["main"], lw=1.8, ms=4, label=r"pixel units $\|\partial L/\partial u_{\mathrm{px}}\|$")
axa.loglog(Ws, nndc / nndc[-1], "s-", color=C["accent"], lw=1.8, ms=4, label=r"NDC $\|\partial L/\partial u_{\mathrm{ndc}}\|$")
axa.set_xticks([240, 480, 960, 1920])
axa.set_xticklabels(["240", "480", "960", "1920"])
axa.minorticks_off()
axa.set_yticks([1, 2, 4, 8])
axa.set_yticklabels(["1", "2", "4", "8"])
axa.set_ylim(0.7, 11)
axa.text(560, 5.6, r"$\propto 1/W$", fontsize=11, color=C["main"])
axa.text(600, 1.15, "constant", fontsize=11, color=C["accent"])
axa.set_xlabel(r"image width $W$ (16:9, same scene)")
axa.set_ylabel("gradient norm (relative to $W = 1920$)")
axa.legend(fontsize=9, frameon=False, loc="upper right")
axa.set_title("(a) NDC cancels the resolution", fontsize=11)
axa.tick_params(labelsize=9)

# (b)
W, H = 1920, 1080
a, b = 2 * tau / W * 1e7, 2 * tau / H * 1e7
assert abs(a - 2.08) < 0.005 and abs(b - 3.70) < 0.005
t = np.linspace(0, 2 * np.pi, 300)
axb.fill(a * np.cos(t), b * np.sin(t), color=C["tangent"], alpha=0.10, lw=0)
axb.plot(a * np.cos(t), b * np.sin(t), color=C["tangent"], lw=1.9, label=r"$\|\partial L/\partial u_{\mathrm{ndc}}\| = \tau$")
axb.plot(3 * np.cos(t), 3 * np.sin(t), color=C["aux"], lw=0.9, ls=(0, (3, 3)), label="pixel length 3")
n_h = np.linalg.norm(np.array([W / 2, H / 2]) * np.array([3e-7, 0]))
n_v = np.linalg.norm(np.array([W / 2, H / 2]) * np.array([0, 3e-7]))
assert abs(n_h - 2.88e-4) < 5e-7 and abs(n_v - 1.62e-4) < 5e-7
for p_, txt, xy in (((3, 0), "NDC $2.88\\times10^{-4} > \\tau$\ndensified", (3.3, 0.35)),
                    ((0, 3), "NDC $1.62\\times10^{-4} < \\tau$\nnot densified", (0.35, 4.3))):
    axb.plot(*p_, "o", color=C["main"], ms=6, zorder=5)
    axb.text(*xy, txt, fontsize=9.5, va="bottom" if p_[1] == 0 else "center")
axb.text(-5.3, 5.6, "inside the blue\nellipse: not\ndensified", fontsize=9.5, va="top")
axb.axhline(0, color="#DDDDDD", lw=0.6, zorder=0)
axb.axvline(0, color="#DDDDDD", lw=0.6, zorder=0)
axb.set_xlim(-5.5, 8.6)
axb.set_ylim(-6, 6)
axb.set_aspect("equal")
axb.set_xlabel(r"$\partial L/\partial u_{\mathrm{px}}$  ($10^{-7}$/pixel)")
axb.set_ylabel(r"$\partial L/\partial v_{\mathrm{px}}$  ($10^{-7}$/pixel)")
axb.legend(fontsize=9, frameon=False, loc="lower right")
axb.set_title(r"(b) $1920 \times 1080$: the directions differ", fontsize=11)
axb.tick_params(labelsize=9)

dgfig.save(fig, __file__)
