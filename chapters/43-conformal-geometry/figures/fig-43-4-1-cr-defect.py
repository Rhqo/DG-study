"""Figure 43.4.1: the Cauchy-Riemann defect (star dz + i dz)(X) of a linear map has the same length for every unit X
(Definition 43.4.4, Lemma 43.2.1(c)).

The real-linear map dz(X) = alpha X + beta conj(X) with alpha = 1 + 0.2 i, beta = 0.35 (sigma1 = |alpha| + |beta|,
sigma2 = |alpha| - |beta|). (a) the unit circle of tangent vectors with six unit vectors X_k = e^{i k pi / 3} (blue) and
J X_1 = i X_1 (green) for k = 1. (b) the image ellipse dz(unit circle) (grey) and the same ellipse turned by 90 deg (dotted); for X = X_1: dz(X)
(blue), dz(JX) (green), i dz(X) (dashed blue) and the defect dz(JX) - i dz(X) = -(star dz + i dz)(X) (vermilion). Self-check: (star dz)(X) = -dz(J X) by Proposition 43.4.1, the defect equals
-2 i beta conj(X) and has length 2|beta| = 0.7 for every unit X; the pointwise conformal energy density
|star dz + i dz|^2 / 4 (|.|^2 of a 1-form = sum over an orthonormal basis) equals (sigma1 - sigma2)^2 / 2 = 2|beta|^2.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-4-1-cr-defect.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
al, be = 1 + 0.2j, 0.35
dz = lambda X: al * X + be * np.conj(X)  # noqa: E731
star_dz = lambda X: -dz(1j * X)  # noqa: E731
Xs = np.exp(1j * np.pi * np.arange(6) / 3)
defect = np.array([dz(1j * X) - 1j * dz(X) for X in Xs])
assert np.allclose(defect, -(np.array([star_dz(X) + 1j * dz(X) for X in Xs])))
assert np.allclose(defect, -2j * be * np.conj(Xs)) and np.allclose(np.abs(defect), 2 * abs(be))
s1, s2 = abs(al) + abs(be), abs(al) - abs(be)
dens = 0.25 * sum(abs(star_dz(e) + 1j * dz(e)) ** 2 for e in (1, 1j))
assert np.isclose(dens, 0.5 * (s1 - s2) ** 2) and np.isclose(dens, 2 * be ** 2)
print("sigma1 = %.4f, sigma2 = %.4f, |defect| = %.3f, density = %.4f" % (s1, s2, abs(defect[0]), dens))

t = np.linspace(0, 2 * np.pi, 300)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
ax = axes[0]
ax.plot(np.cos(t), np.sin(t), color=col["aux"], lw=1.0)
for k, X in enumerate(Xs):
    ax.annotate("", xy=(X.real, X.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col["tangent"], lw=1.5 if k == 1 else 0.8))
X1 = Xs[1]
ax.annotate("", xy=((1j * X1).real, (1j * X1).imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col["third"], lw=1.5))
ax.text(X1.real + 0.05, X1.imag + 0.02, r"$X$", color=col["tangent"], fontsize=11)
ax.text((1j * X1).real - 0.32, (1j * X1).imag + 0.05, r"$JX$", color=col["third"], fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.3, 1.3)
ax.set_axis_off()
ax.set_title("(a) unit tangent vectors", fontsize=11)

ax = axes[1]
E = dz(np.exp(1j * t))
ax.plot(E.real, E.imag, color=col["aux"], lw=1.0)
Ei = 1j * E
ax.plot(Ei.real, Ei.imag, color=col["aux"], lw=0.8, ls=":")
a1, a2, a3 = dz(X1), dz(1j * X1), 1j * dz(X1)
ax.annotate("", xy=(a1.real, a1.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col["tangent"], lw=1.5))
ax.annotate("", xy=(a2.real, a2.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col["third"], lw=1.5))
ax.plot([0, a3.real], [0, a3.imag], ls="--", color=col["tangent"], lw=1.2)
ax.annotate("", xy=(a2.real, a2.imag), xytext=(a3.real, a3.imag), arrowprops=dict(arrowstyle="-|>", color=col["normal"], lw=1.6))
ax.text(a1.real + 0.05, a1.imag, r"$dz(X)$", color=col["tangent"], fontsize=10)
ax.text(a2.real - 0.25, a2.imag - 0.3, r"$dz(JX)$", color=col["third"], fontsize=10)
ax.text(a3.real - 0.45, a3.imag + 0.14, r"$\mathrm{i}\,dz(X)$", color=col["tangent"], fontsize=10)
mid = 0.5 * (a2 + a3)
ax.text(mid.real - 0.7, mid.imag - 0.03, "defect", color=col["normal"], fontsize=9)
ax.set_aspect("equal")
ax.set_xlim(-1.75, 1.75)
ax.set_ylim(-1.45, 1.45)
ax.set_axis_off()
ax.set_title(r"(b) image: defect $= -(\star dz + \mathrm{i}\,dz)(X)$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)
