"""Figure 43.2.1: multiplying by a complex number is a rotation and a scaling; only C-linear maps keep squares
square and oriented (Lemma 43.2.1, Proposition 43.2.7).

(a) the unit square spanned by w = 1 and i w = i (light blue, blue and green arrows) and its image under
    multiplication by c = 1.25 e^{i 40 deg} (orange outline, arrows c w and c (i w) = i (c w)).
(b) the images of the same unit square under three real-linear maps:
    C-linear     T1(w) = c w                    (square, counterclockwise order kept),
    anti-linear  T2(w) = c conj(w)              (square, order reversed: the image of i is -i turned),
    real-linear  T3 = [[1.2, 0.7], [0, 0.9]]    (parallelogram: angle 90 deg -> 52.1 deg).
Self-check: T1 has the matrix [[a, -b], [b, a]] with a + i b = c and T1(i w) = i T1(w); T2 has determinant -|c|^2;
the angle of T3 e1, T3 e2 is arccos(0.7 / sqrt(1.3)) = 52.1 deg.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-2-1-complex-linear.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
c = 1.25 * np.exp(1j * np.radians(40))
a, b = c.real, c.imag
M1 = np.array([[a, -b], [b, a]])
M2 = np.array([[a, b], [b, -a]])          # w -> c conj(w)
M3 = np.array([[1.2, 0.7], [0.0, 0.9]])
w = np.array([1 + 0j, 1j])
assert np.allclose(M1 @ [0, 1], [(1j * c).real, (1j * c).imag])
assert np.isclose(np.linalg.det(M2), -abs(c) ** 2)
v1, v2 = M3[:, 0], M3[:, 1]
ang3 = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
assert abs(ang3 - 52.1) < 0.05
print("c =", np.round(c, 4), "matrix", np.round(M1, 4), "angle of T3 image %.1f deg" % ang3)

sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T


def arrow(ax, p, d, color, label=None, off=(0, 0)):
    ax.annotate("", xy=(p[0] + d[0], p[1] + d[1]), xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=0, shrinkB=0))
    if label:
        ax.text(p[0] + d[0] + off[0], p[1] + d[1] + off[1], label, color=color, fontsize=11)


fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
ax = axes[0]
ax.fill(sq[0], sq[1], color=col["region"], alpha=0.25, lw=0)
im = M1 @ sq
ax.fill(im[0], im[1], color=col["accent"], alpha=0.25, lw=0)
ax.plot(im[0], im[1], color=col["accent"], lw=1.2)
arrow(ax, (0, 0), (1, 0), col["tangent"], r"$w$", (0.03, -0.12))
arrow(ax, (0, 0), (0, 1), col["third"], r"$\mathrm{i}w$", (-0.32, 0.0))
arrow(ax, (0, 0), (a, b), col["tangent"], r"$cw$", (0.03, -0.05))
arrow(ax, (0, 0), (-b, a), col["third"], r"$\mathrm{i}cw$", (-0.25, 0.06))
t = np.linspace(0, np.radians(40), 30)
ax.plot(0.45 * np.cos(t), 0.45 * np.sin(t), color=col["aux"], lw=0.8)
ax.text(0.5, 0.12, r"$40^\circ$", fontsize=10, color=col["aux"])
ax.set_aspect("equal")
ax.set_xlim(-1.0, 1.4)
ax.set_ylim(-0.25, 1.95)
ax.set_axis_off()
ax.set_title(r"(a) $w \mapsto cw$, $c = 1.25e^{\mathrm{i}40^\circ}$", fontsize=11)

ax = axes[1]
for M, shift, color, name in ((M1, (0.0, 0.0), col["accent"], r"$cw$"), (M2, (2.1, 0.0), col["covector"], r"$c\bar w$"),
                              (M3, (0.6, -1.9), col["aux"], "real-linear")):
    im = M @ sq + np.array(shift)[:, None]
    ax.fill(im[0], im[1], color=color, alpha=0.25, lw=0)
    ax.plot(im[0], im[1], color=color, lw=1.2)
    arrow(ax, shift, M[:, 0], col["tangent"])
    arrow(ax, shift, M[:, 1], col["third"])
    cen = im[:, :4].mean(1)
    if name == "real-linear":
        ax.text(cen[0], im[1].min() - 0.22, name, ha="center", va="center", fontsize=10)
    else:
        ax.text(cen[0], cen[1], name, ha="center", va="center", fontsize=10)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(b) three images of the unit square", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)
