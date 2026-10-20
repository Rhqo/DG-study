"""Figure 43.1.1: a conformal map keeps small circles round, an area-preserving map keeps their areas
(Section 43.1, Non-example 43.1.3).

Small circles of radius eps = 0.045 centred at the points of a polar grid in the unit disk (rho = 0.2, 0.4, 0.6,
0.8, 12 angles, and the centre) and their images under
  (a) the holomorphic map f(z) = z + 0.3 z^2 (|f'| = |1 + 0.6 z| lies in (0.4, 1.6); injective because
      Re f' > 0 on the disk),
  (b) the twist T(rho, theta) = (rho, theta + 1.2 rho^2) (area-preserving, a shear rho*tau'(rho) = 2.4 rho^2 in the
      polar frame).
The images of the grid lines rho = const and theta = const are drawn thin. Self-check: in (a) the image of every
circle is a circle up to O(eps^2) (axis ratio of the moment ellipse < 1.01) and its radius is |f'| eps; in (b) every image has area
pi eps^2 (to 1 %) and the axis ratio at rho = 0.8 is ((s + sqrt(s^2 + 4)) / 2)^2 with s = 2.4 * 0.64.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-1-1-circles.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
EPS = 0.045
t = np.linspace(0, 2 * np.pi, 241)


def f(z):
    return z + 0.3 * z ** 2


def twist(z):
    rho, th = np.abs(z), np.angle(z)
    return rho * np.exp(1j * (th + 1.2 * rho ** 2))


def shoelace(w):
    return 0.5 * np.sum(w.real * np.roll(w.imag, -1) - w.imag * np.roll(w.real, -1))


def axis_ratio(w):
    """ratio of the axes of the ellipse with the same second moments as the closed curve w (samples)"""
    d = np.c_[w.real - w.real.mean(), w.imag - w.imag.mean()]
    ev = np.linalg.eigvalsh(d.T @ d)
    return np.sqrt(ev[1] / ev[0])


centres = [0j] + [r * np.exp(1j * a) for r in (0.2, 0.4, 0.6, 0.8) for a in 2 * np.pi * np.arange(12) / 12]
stats = {"a": [], "b": []}
for c in centres:
    circ = c + EPS * np.exp(1j * t)
    wa, wb = f(circ), twist(circ)
    stats["a"].append((abs(c), axis_ratio(wa), np.abs(wa - f(c)).mean() / (EPS * abs(1 + 0.6 * c))))
    stats["b"].append((abs(c), axis_ratio(wb), shoelace(wb) / (np.pi * EPS ** 2)))
ra = np.array(stats["a"])
rb = np.array(stats["b"])
assert ra[:, 1].max() < 1.01 and np.allclose(ra[:, 2], 1, atol=0.01)
assert np.allclose(rb[:, 2], 1, atol=0.01)
s = 2.4 * 0.64
pred = ((s + np.sqrt(s * s + 4)) / 2) ** 2
assert abs(rb[rb[:, 0] > 0.79, 1].mean() - pred) / pred < 0.03
print("(a) max axis ratio %.4f, |f'| range %.2f..%.2f" % (ra[:, 1].max(), 0.4, 1.6))
print("(b) area ratio %.4f..%.4f, axis ratio at rho=0.8: %.2f (predicted %.2f), at rho=0.4: %.2f"
      % (rb[:, 2].min(), rb[:, 2].max(), rb[rb[:, 0] > 0.79, 1].mean(), pred, rb[np.abs(rb[:, 0] - 0.4) < 1e-9, 1].mean()))

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
for ax, g, title in ((axes[0], f, r"(a) conformal: $z + 0.3z^2$"), (axes[1], twist, r"(b) area-preserving twist")):
    for r in (0.2, 0.4, 0.6, 0.8, 1.0):
        w = g(r * np.exp(1j * t))
        ax.plot(w.real, w.imag, color=col["aux"], lw=0.5 if r < 1 else 1.0)
    rr = np.linspace(0, 1, 80)
    for a in 2 * np.pi * np.arange(12) / 12:
        w = g(rr * np.exp(1j * a))
        ax.plot(w.real, w.imag, color=col["aux"], lw=0.5)
    for c in centres:
        w = g(c + EPS * np.exp(1j * t))
        ax.fill(w.real, w.imag, color=col["region"], alpha=0.55, lw=0)
        ax.plot(w.real, w.imag, color=col["tangent"], lw=0.9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
axes[0].annotate("small", xy=(f(-0.8).real, f(-0.8).imag), xytext=(-0.85, -0.75), fontsize=10,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
axes[0].annotate("large", xy=(f(0.8).real + 0.06, 0.09), xytext=(1.0, 0.75), fontsize=10,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
p = twist(0.8 * np.exp(-1j * np.pi / 2))
axes[1].annotate("stretched", xy=(p.real, p.imag - 0.03), xytext=(0.55, -1.2), fontsize=10,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.08, wspace=0.08)
dgfig.save(fig, __file__)
