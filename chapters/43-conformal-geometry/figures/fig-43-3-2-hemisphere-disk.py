"""Figure 43.3.2: two different conformal maps of the southern hemisphere onto the unit disk (Example 43.3.12).

H = {x in S^2 : x3 <= 0} with outward normal. zeta(x) = conj(sigma(x)) = (x1 - i x2)/(1 - x3) maps H onto the closed
unit disk and is holomorphic (Example 43.3.7(b)); psi = phi_a o zeta with phi_a(w) = (w - a)/(1 - conj(a) w),
a = 0.45 (Example 43.2.12) is another one. Each panel shows H seen from below with the preimage of the polar grid
of the disk (circles |w| = 0.2, 0.4, 0.6, 0.8 and 12 radii) under (a) zeta, (b) psi; the black dot is the point that
goes to 0 (the south pole in (a), zeta^{-1}(a) in (b)).
Self-check: |zeta| <= 1 on H with equality exactly on the equator; zeta^{-1}(w) lies on S^2; the grid curves of (b)
are circles on the sphere (they lie in planes: smallest singular value of the centred points ~ 0); the point
zeta^{-1}(a) has x3 = -(1 - a^2)/(1 + a^2).

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-3-2-hemisphere-disk.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
A = 0.45


def zeta_inv(w):
    """inverse of zeta = conj(sigma) on the southern hemisphere: sigma^{-1}(conj w)."""
    s = np.conj(w)
    d = 1 + np.abs(s) ** 2
    return np.stack([2 * s.real / d, 2 * s.imag / d, (np.abs(s) ** 2 - 1) / d], -1)


phi_inv = lambda w: (w + A) / (1 + np.conj(A) * w)  # noqa: E731   phi_a^{-1} = phi_{-a}
zeta = lambda P: (P[..., 0] - 1j * P[..., 1]) / (1 - P[..., 2])  # noqa: E731

rng = np.random.default_rng(0)
Q = rng.standard_normal((500, 3))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
Q[:, 2] = -np.abs(Q[:, 2])
assert np.all(np.abs(zeta(Q)) <= 1 + 1e-12)
eq = np.c_[np.cos(np.arange(8)), np.sin(np.arange(8)), np.zeros(8)]
assert np.allclose(np.abs(zeta(eq)), 1)
ww = 0.7 * np.exp(1j * np.linspace(0, 6, 20))
assert np.allclose(np.linalg.norm(zeta_inv(ww), axis=1), 1) and np.allclose(zeta(zeta_inv(ww)), ww)
t = np.linspace(0, 2 * np.pi, 241)
planar = []
for rr in (0.2, 0.4, 0.6, 0.8):
    C = zeta_inv(phi_inv(rr * np.exp(1j * t)))
    planar.append(np.linalg.svd(C - C.mean(0), compute_uv=False)[-1])
assert max(planar) < 1e-10
c_b = zeta_inv(np.array([A + 0j]))[0]
assert abs(c_b[2] + (1 - A * A) / (1 + A * A)) < 1e-12
print("centre of (b):", np.round(c_b, 4), "planarity of the circles", max(planar))

fig = plt.figure(figsize=(6.6, 3.3))
for k, (g, title) in enumerate(((lambda w: w, r"(a) $\zeta = \bar\sigma$"), (phi_inv, r"(b) $\varphi_a\circ\zeta$, $a = 0.45$"))):
    ax = dgfig.axes3d(fig, 121 + k, elev=-58, azim=-70)
    th = np.linspace(np.pi / 2, np.pi, 30)
    ph = np.linspace(0, 2 * np.pi, 70)
    TH, PH = np.meshgrid(th, ph)
    dgfig.surface(ax, np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH), alpha=0.35, grid=False)
    vv = dgfig.view_vector(ax)
    for rr in (0.2, 0.4, 0.6, 0.8, 1.0):
        C = zeta_inv(g(rr * np.exp(1j * t)))
        dgfig.curve3d(ax, C, role="main" if rr == 1.0 else "tangent", lw=1.3 if rr == 1.0 else 0.9, visible=C @ vv > 0)
    rad = np.linspace(0, 0.999, 120)
    for j in range(12):
        C = zeta_inv(g(rad * np.exp(1j * j * np.pi / 6)))
        dgfig.curve3d(ax, C, role="third", lw=0.7, visible=C @ vv > 0)
    c0 = zeta_inv(g(np.array([0j])))[0]
    dgfig.point3d(ax, c0)
    dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 0]]), zoom=1.3)
    ax.set_title(title, fontsize=11)
fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0, wspace=0.0)
dgfig.save(fig, __file__)
