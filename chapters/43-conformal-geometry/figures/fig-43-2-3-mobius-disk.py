"""Figure 43.2.3: a Möbius transformation of the unit disk onto itself maps circles to circles (Example 43.2.12,
Proposition 43.2.11).

(a) the polar grid of the unit disk: circles |z| = 0.2, 0.4, 0.6, 0.8 (grey), the unit circle (black), 12 radii
    (light grey); the point a = 0.5 (orange) and the centre 0 (vermilion).
(b) the images under phi_a(z) = (z - a) / (1 - conj(a) z), a = 0.5: the unit circle goes to itself, the circles to
    circles that are no longer concentric, the radii to arcs of circles through phi_a(0) = -a and phi_a(inf) = 1/conj(a)
    (outside the picture); both families still meet at right angles. phi_a(a) = 0.
Self-check: every image curve is a circle (or a line) to 1e-12 (least-squares circle fit), |phi_a| = 1 on the unit
circle, the image of |z| = r is the circle with centre -a(1 - r^2)/(1 - a^2 r^2) and radius
r(1 - a^2)/(1 - a^2 r^2) (r = 0.4: -0.4375, 0.3125), and the images of a radius
and a circle cross at 90 deg.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-2-3-mobius-disk.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
A = 0.5
phi = lambda z: (z - A) / (1 - np.conj(A) * z)  # noqa: E731


def circle_fit(w):
    """least-squares circle |w - c|^2 = R^2: returns (c, R, max residual)."""
    M = np.c_[2 * w.real, 2 * w.imag, np.ones(len(w))]
    rhs = np.abs(w) ** 2
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    c = sol[0] + 1j * sol[1]
    R = np.sqrt(sol[2] + abs(c) ** 2)
    return c, R, np.abs(np.abs(w - c) - R).max()


t = np.linspace(0, 2 * np.pi, 721)
res, centres = [], {}
for r in (0.2, 0.4, 0.6, 0.8, 1.0):
    c, R, e = circle_fit(phi(r * np.exp(1j * t)))
    res.append(e)
    centres[r] = (c, R)
rr = np.linspace(0, 0.999, 400)
for k in range(12):
    w = phi(rr * np.exp(1j * k * np.pi / 6))
    if k % 6 == 0:                                   # the real diameter maps into the real axis (a line)
        res.append(np.abs(w.imag).max())
    else:
        res.append(circle_fit(w)[2])
assert max(res) < 1e-10
for r, (c, R) in centres.items():
    assert abs(c + A * (1 - r * r) / (1 - A * A * r * r)) < 1e-12 and abs(R - r * (1 - A * A) / (1 - A * A * r * r)) < 1e-12
assert np.allclose(np.abs(phi(np.exp(1j * t))), 1) and abs(phi(A)) < 1e-15 and abs(phi(0) + A) < 1e-15
# right angle between the image of the radius at 60 deg and the image of |z| = 0.6 at their crossing
z0 = 0.6 * np.exp(1j * np.pi / 3)
d_rad, d_circ = np.exp(1j * np.pi / 3), 1j * np.exp(1j * np.pi / 3)
dphi = (1 - abs(A) ** 2) / (1 - np.conj(A) * z0) ** 2
assert abs(abs(np.angle((dphi * d_circ) / (dphi * d_rad))) - np.pi / 2) < 1e-12
print("image circles (centre, radius):", {r: (np.round(c, 4), round(float(R), 4)) for r, (c, R) in centres.items()})
print("max circle-fit residual %.1e" % max(res))

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
for ax, g, title in ((axes[0], lambda z: z, r"(a) unit disk, $a = 0.5$"), (axes[1], phi, r"(b) $\varphi_a(z) = \dfrac{z - a}{1 - \bar a z}$")):
    for k in range(12):
        w = g(rr * np.exp(1j * k * np.pi / 6))
        ax.plot(w.real, w.imag, color="#BBBBBB", lw=0.6)
    for r in (0.2, 0.4, 0.6, 0.8):
        w = g(r * np.exp(1j * t))
        ax.plot(w.real, w.imag, color=col["aux"], lw=0.9)
    w = g(np.exp(1j * t))
    ax.plot(w.real, w.imag, color=col["main"], lw=1.4)
    pa, p0 = g(np.array([A + 0j]))[0], g(np.array([0j]))[0]
    ax.plot([pa.real], [pa.imag], "o", color=col["accent"], ms=5)
    ax.plot([p0.real], [p0.imag], "o", color=col["normal"], ms=5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
axes[0].text(A + 0.04, -0.13, r"$a$", fontsize=11, color=col["accent"])
axes[0].text(-0.12, -0.13, r"$0$", fontsize=11, color=col["normal"])
axes[1].text(0.04, -0.13, r"$\varphi_a(a) = 0$", fontsize=10, color=col["accent"])
axes[1].annotate(r"$\varphi_a(0) = -a$", xy=(-A, -0.02), xytext=(-1.2, -1.15), fontsize=10, color=col["normal"],
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=col["normal"]))
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.07, wspace=0.08)
dgfig.save(fig, __file__)
