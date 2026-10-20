"""Figure 43.2.2: f(z) = z^2 preserves right angles wherever f' != 0 and doubles the angle at 0 (Non-example 43.2.8).

(a) the grid lines x = k/5, y = k/5 (k = 0..6) of the square [0, 1.2]^2, the point p = 0.8 + 0.6 i with the two grid
    lines through it (blue: y = 0.6, green: x = 0.8) and the corner 0, where the two sides meet at 90 deg.
(b) their images under f(z) = z^2: u = x^2 - y^2, v = 2 x y. The lines through p become parabolas that still meet at
    90 deg at f(p) = 0.28 + 0.96 i; the two sides through 0 become the positive and the negative real axis (180 deg).
Self-check: the angle between f_x and f_y at p is 90 deg to 1e-12 and |f'(p)| = 2|p|; the images of the two sides
of the square near 0 point in the directions 0 deg and 180 deg.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-2-2-square-map.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
f = lambda z: z ** 2  # noqa: E731
p = 0.8 + 0.6j
fx, fy = 2 * p, 2 * p * 1j                       # partial derivatives of z^2 at p
ang_p = np.degrees(np.angle(fy / fx))
assert abs(ang_p - 90) < 1e-12 and abs(abs(fx) - 2 * abs(p)) < 1e-12
e = 1e-3
assert abs(np.angle(f(e))) < 1e-12 and abs(abs(np.angle(f(1j * e))) - np.pi) < 1e-12
print("angle at f(p) = %.6f deg, |f'(p)| = %.4f, f(p) = %s" % (ang_p, abs(fx), f(p)))

s = np.linspace(0, 1.2, 200)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw=dict(width_ratios=[1, 1.45]))
for ax, g, title in ((axes[0], lambda z: z, r"(a) $z$"), (axes[1], f, r"(b) $f(z) = z^2$")):
    for k in range(7):
        for line in (k / 5 + 1j * s, s + 1j * k / 5):
            w = g(line)
            ax.plot(w.real, w.imag, color=col["aux"], lw=0.5 if k else 1.2)
    wb, wg = g(s + 0.6j), g(0.8 + 1j * s)
    ax.plot(wb.real, wb.imag, color=col["tangent"], lw=1.6)
    ax.plot(wg.real, wg.imag, color=col["third"], lw=1.6)
    q = g(p)
    ax.plot([q.real], [q.imag], "o", color=col["main"], ms=4)
    ax.plot([0], [0], "o", color=col["normal"], ms=4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
# right-angle markers
r0 = 0.09
axes[0].plot([0.8 + r0, 0.8 + r0, 0.8], [0.6, 0.6 + r0, 0.6 + r0], color=col["main"], lw=0.8)
u1, u2 = fx / abs(fx), fy / abs(fy)
q = f(p)
corner = np.array([q + 0.15 * u1, q + 0.15 * (u1 + u2), q + 0.15 * u2])
axes[1].plot(corner.real, corner.imag, color=col["main"], lw=0.8)
t = np.linspace(0, np.pi / 2, 30)
axes[0].plot(0.16 * np.cos(t), 0.16 * np.sin(t), color=col["normal"], lw=1.0)
axes[0].text(0.15, 0.17, r"$90^\circ$", color=col["normal"], fontsize=10)
t = np.linspace(0, np.pi, 40)
axes[1].plot(0.13 * np.cos(t), 0.13 * np.sin(t), color=col["normal"], lw=1.0)
axes[1].text(-0.55, 0.12, r"$180^\circ$", color=col["normal"], fontsize=10)
axes[0].text(0.83, 0.5, r"$p$", fontsize=11)
axes[1].text(q.real + 0.07, q.imag - 0.2, r"$f(p)$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.08)
dgfig.save(fig, __file__)
