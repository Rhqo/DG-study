"""Figure 29.6.2: rotating a Gaussian requires rotating its SH coefficients with D^l(R) (Section 29.6).

Color function f = L2 projection onto degree <= 3 of the lobe exp(8 (<d, r0> - 1)), r0 = (cos 60, sin 60, 0.3)/|.|
(the "bright side" of the Gaussian). The object is rotated by R = 100 degrees about the z axis.
Maps: equirectangular (longitude phi horizontal 0..360, colatitude theta vertical 0..180, north up).
(a) f before the rotation; the white x marks r0.
(b) after rotating the Gaussian but keeping the coefficients: the map is unchanged, while the object's bright side is
    now at R r0 (white x): the highlight did not move with the object.
(c) after replacing c_l by D^l(R) c_l (D^l found by least squares on 400 random directions): f o R^{-1}; the maximum
    is at R r0.
Self-checks: (c) equals f o R^{-1} to 1e-10; argmax of (a) and (c) within 3 degrees of r0 and R r0.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-6-2-rotation.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
pi = np.pi
rng = np.random.default_rng(63)
FS = [
    lambda x, y, z: 0 * x + 1 / (2 * np.sqrt(pi)),
    lambda x, y, z: np.sqrt(3 / (4 * pi)) * y, lambda x, y, z: np.sqrt(3 / (4 * pi)) * z,
    lambda x, y, z: np.sqrt(3 / (4 * pi)) * x,
    lambda x, y, z: np.sqrt(15 / pi) / 2 * x * y, lambda x, y, z: np.sqrt(15 / pi) / 2 * y * z,
    lambda x, y, z: np.sqrt(5 / pi) / 4 * (2 * z * z - x * x - y * y),
    lambda x, y, z: np.sqrt(15 / pi) / 2 * x * z, lambda x, y, z: np.sqrt(15 / pi) / 4 * (x * x - y * y),
    lambda x, y, z: np.sqrt(35 / (2 * pi)) / 4 * y * (3 * x * x - y * y),
    lambda x, y, z: np.sqrt(105 / pi) / 2 * x * y * z,
    lambda x, y, z: np.sqrt(21 / (2 * pi)) / 4 * y * (4 * z * z - x * x - y * y),
    lambda x, y, z: np.sqrt(7 / pi) / 4 * z * (2 * z * z - 3 * x * x - 3 * y * y),
    lambda x, y, z: np.sqrt(21 / (2 * pi)) / 4 * x * (4 * z * z - x * x - y * y),
    lambda x, y, z: np.sqrt(105 / pi) / 4 * z * (x * x - y * y),
    lambda x, y, z: np.sqrt(35 / (2 * pi)) / 4 * x * (x * x - 3 * y * y),
]
DEG = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]


def Ymat(D):
    return np.stack([np.broadcast_to(f(D[..., 0], D[..., 1], D[..., 2]), D.shape[:-1]) for f in FS], -1)


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


tq, wq = np.polynomial.legendre.leggauss(40)
phq = np.linspace(0, 2 * pi, 80, endpoint=False)
T, PH = np.meshgrid(tq, phq, indexing="ij")
Wq = (wq[:, None] * np.ones_like(PH) * (2 * pi / 80)).ravel()
Dq = np.stack([np.sqrt(1 - T ** 2) * np.cos(PH), np.sqrt(1 - T ** 2) * np.sin(PH), T], -1).reshape(-1, 3)
r0 = np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0.3]); r0 /= np.linalg.norm(r0)
c = Ymat(Dq).T @ (np.exp(8 * (Dq @ r0 - 1)) * Wq)
R = rotz(100)
Dfull = np.zeros((16, 16))
d = rng.normal(size=(400, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
for l in range(4):
    idx = [i for i in range(16) if DEG[i] == l]
    M, *_ = np.linalg.lstsq(Ymat(d)[:, idx], Ymat(d @ R)[:, idx], rcond=None)
    Dfull[np.ix_(idx, idx)] = M
c_rot = Dfull @ c
dd = rng.normal(size=(3000, 3)); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
assert np.allclose(Ymat(dd) @ c_rot, Ymat(dd @ R) @ c, atol=1e-10)

th, ph = np.meshgrid(np.linspace(0, pi, 181), np.linspace(0, 2 * pi, 361), indexing="ij")
G = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
fa = Ymat(G) @ c
fc = Ymat(G) @ c_rot


def thph(v):
    return np.degrees(np.arccos(v[2])), np.degrees(np.arctan2(v[1], v[0])) % 360


Rr0 = R @ r0
ia, ic = np.unravel_index(np.argmax(fa), fa.shape), np.unravel_index(np.argmax(fc), fc.shape)
assert np.degrees(np.arccos(G[ia] @ r0)) < 3 and np.degrees(np.arccos(G[ic] @ Rr0)) < 3
vmin, vmax = min(fa.min(), fc.min()), max(fa.max(), fc.max())

fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.6), sharex=True, gridspec_kw=dict(hspace=0.32))
panels = [(fa, r0, "(a) before rotation: bright side at $r_0$"),
          (fa, Rr0, "(b) rotated, coefficients kept: highlight stays"),
          (fc, Rr0, r"(c) rotated, $c_\ell \leftarrow D^\ell(R)\,c_\ell$: highlight follows")]
for ax, (F, mark, title) in zip(axes, panels):
    im = ax.imshow(F, extent=(0, 360, 180, 0), cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto")
    tm, pm = thph(mark)
    ax.plot(pm, tm, "x", color="white", ms=11, mew=2.4)
    ax.plot(pm, tm, "x", color=C["main"], ms=11, mew=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(r"colatitude $\theta$ (deg)")
    ax.set_yticks([0, 90, 180])
    ax.tick_params(labelsize=9)
axes[-1].set_xlabel(r"longitude $\varphi$ (deg)")
axes[-1].set_xticks([0, 90, 180, 270, 360])
axes[0].annotate(r"$r_0$", xy=thph(r0)[::-1], xytext=(thph(r0)[1] + 25, thph(r0)[0] - 35), color="white", fontsize=11,
                 arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
for ax in axes[1:]:
    ax.annotate(r"$Rr_0$", xy=thph(Rr0)[::-1], xytext=(thph(Rr0)[1] + 25, thph(Rr0)[0] - 35), color="white", fontsize=11,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.set_label("color value $f(d)$", fontsize=10)
cb.ax.tick_params(labelsize=9)

dgfig.save(fig, __file__)
