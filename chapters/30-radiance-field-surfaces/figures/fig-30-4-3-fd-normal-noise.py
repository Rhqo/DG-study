"""Figure 30.4.3: error of finite-difference normals from a depth map versus the tilt, for two kinds of depth error.

3D depth map, f = 500 px, a plane through (0, 0, 3) tilted about the image x-axis so that its normal makes the angle
theta with the viewing ray at the image center (depth changes along the image v-direction only). The normal at the
center pixel is estimated as in depth_to_normal of the 2DGS code: central differences of the back-projected points of
the four neighbours (u, v) = (0, +-1) and (+-1, 0), N = (X(0,1) - X(0,-1)) x (X(1,0) - X(-1,0)).
Monte Carlo with 40000 trials per theta (seed 0), rms angle error (deg), split into two components:
"along the tilt" = rotation of the normal inside the plane that contains the tilt (what a 2D cross-section sees),
"across the tilt" = rotation about the tilt direction (only visible in 3D).
(a) constant depth noise: d + N(0, sigma^2) at each pixel, sigma = 0.02 % of the depth. First order, with
    k = sigma f/(sqrt 2 d) = 4.05 deg: along k cos^2(theta), across k cos(theta), total k sqrt(cos^4 + cos^2).
(b) depth read at a random sub-pixel position: each pixel gets d(u + a, v + b), a, b ~ U(-1/2, 1/2), but is
    back-projected through its pixel center (aliasing / misregistration). First order: along sin cos/sqrt(24),
    across sin/sqrt(24), total (sin/sqrt(24)) sqrt(1 + cos^2) (rad), independent of f and d.
Self-checks: Monte Carlo matches the first-order formulas (within 3 % in (a), 10 % of the maximum in (b); the
remaining gap in (b) is second order, Monte Carlo slightly above); (a) is worst facing the camera, (b) near grazing.

Run from the project root: ``OMP_NUM_THREADS=4 PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-4-3-fd-normal-noise.py``
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
F, D0, SIG = 500.0, 3.0, 2e-4
NB_U = np.array([0.0, 0.0, -1.0, 1.0])               # neighbours: (0,-1), (0,+1), (-1,0), (+1,0)
NB_V = np.array([-1.0, 1.0, 0.0, 0.0])


def plane_depth(u, v, th):
    """z-depth of the plane through (0, 0, D0) with normal n = (0, sin th, -cos th) (facing the camera)."""
    return D0 * np.cos(th) / (np.cos(th) - v / F * np.sin(th))


def mc_errors(th, model, trials=40000, seed=0):
    rg = np.random.default_rng(seed)
    if model == "const":
        dd = plane_depth(NB_U, NB_V, th)[None] + SIG * D0 * rg.normal(size=(trials, 4))
    else:
        dd = plane_depth(NB_U[None] + rg.uniform(-0.5, 0.5, size=(trials, 4)),
                         NB_V[None] + rg.uniform(-0.5, 0.5, size=(trials, 4)), th)
    r = np.stack([NB_U / F, NB_V / F, np.ones(4)], -1)
    X = dd[..., None] * r[None]
    N = np.cross(X[:, 1] - X[:, 0], X[:, 3] - X[:, 2])
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    n = np.array([0.0, np.sin(th), -np.cos(th)])
    across = np.arcsin(np.clip(N[:, 0], -1, 1))                        # rotation about the tilt direction
    t_in = np.array([0.0, np.cos(th), np.sin(th)])                      # in-plane direction of the tilt
    along = np.arctan2(N @ t_in, N @ n)                                 # rotation inside the tilt plane
    total = np.arccos(np.clip(np.abs(N @ n), -1, 1))
    rms = lambda a: np.degrees(np.sqrt(np.mean(a ** 2)))
    return rms(total), rms(along), rms(across)


thd = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 88])
th_line = np.linspace(0, 89.5, 300)
c_line, s_line = np.cos(np.radians(th_line)), np.sin(np.radians(th_line))
K = np.degrees(SIG * F / np.sqrt(2))                                    # sigma f/(sqrt2 d) with sigma = SIG d
mc_c = np.array([mc_errors(np.radians(t), "const") for t in thd])
mc_j = np.array([mc_errors(np.radians(t), "jitter") for t in thd])
ct, st = np.cos(np.radians(thd)), np.sin(np.radians(thd))
pred_c = np.stack([K * np.sqrt(ct ** 4 + ct ** 2), K * ct ** 2, K * ct], 1)
pred_j = np.degrees(np.stack([st / np.sqrt(24) * np.sqrt(1 + ct ** 2), st * ct / np.sqrt(24), st / np.sqrt(24)], 1))
assert np.all(np.abs(mc_c - pred_c) <= 0.03 * pred_c + 0.01)
assert np.all(np.abs(mc_j - pred_j) <= 0.1 * pred_j.max(axis=0) + 0.01)     # second-order terms ~ 8 %
assert np.argmax(mc_c[:, 0]) == 0 and np.argmax(mc_j[:, 0]) >= len(thd) - 3   # (a) worst facing, (b) worst grazing
print("const (total, along, across):\n", np.round(mc_c, 3), "\njitter:\n", np.round(mc_j, 3))

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 8.0), sharex=True, gridspec_kw=dict(hspace=0.25))
axa.plot(th_line, K * np.sqrt(c_line ** 4 + c_line ** 2), color=C["tangent"], lw=2.4,
         label=r"total $k\sqrt{\cos^4\theta + \cos^2\theta}$")
axa.plot(th_line, K * c_line ** 2, color=C["tangent"], lw=1.3, ls=(0, (5, 3)),
         label=r"along the tilt (2D section) $k\cos^2\theta$")
axa.plot(th_line, K * c_line, color=C["tangent"], lw=1.3, ls=":", label=r"across the tilt $k\cos\theta$")
axa.plot(thd, mc_c[:, 0], "o", color=C["main"], mfc="white", ms=6, label="Monte Carlo (total)")
axa.text(2, 0.35, r"$k = \sigma f/(\sqrt{2}\,d) = 4.05^\circ$", fontsize=11.5)
axa.set_ylabel("rms normal error (deg)")
axa.set_ylim(0, 7.6)
axa.legend(loc="upper right", frameon=False, fontsize=10.5)
axa.set_title(r"(a) constant depth noise ($\sigma = 0.02\%$ of $d$)", fontsize=13)

axb.plot(th_line, np.degrees(s_line / np.sqrt(24) * np.sqrt(1 + c_line ** 2)), color=C["accent"], lw=2.4,
         label=r"total $\frac{\sin\theta}{\sqrt{24}}\sqrt{1 + \cos^2\theta}$")
axb.plot(th_line, np.degrees(s_line * c_line / np.sqrt(24)), color=C["accent"], lw=1.3, ls=(0, (5, 3)),
         label=r"along the tilt (2D section)")
axb.plot(th_line, np.degrees(s_line / np.sqrt(24)), color=C["accent"], lw=1.3, ls=":", label=r"across the tilt")
axb.plot(thd, mc_j[:, 0], "o", color=C["main"], mfc="white", ms=6, label="Monte Carlo (total)")
axb.set_ylabel("rms normal error (deg)")
axb.set_xlabel(r"tilt $\theta$ (deg)")
axb.set_ylim(0, 17.5)
axb.set_xlim(0, 90)
axb.legend(loc="upper left", frameon=False, fontsize=10.5)
axb.set_title(r"(b) depth read $\pm\frac{1}{2}$ px off the pixel center", fontsize=13)
dgfig.save(fig, __file__)
