"""Figure 42.1.1: the Laplacian measures the deviation from the local average (Section 42.1, Proposition 42.1.2).

(a) 1D: u(x) = sin(2x) + 0.3 x^2 on [0, 3.2]. At x0 = 2.3 (u'' > 0) and x1 = 0.75 (u'' < 0) the average of the two
    neighbours x -+ eps (eps = 0.45) is drawn as the midpoint of the chord (orange); the vertical gap
    avg - u(x) has the sign of u''. Self-check: (2 / eps^2)(avg - u) = (u(x+e) - 2u(x) + u(x-e)) / e^2 -> u''.
(b) 2D: the quantity Q(eps) = (4 / eps^2) (mean of u over the circle of radius eps about p - u(p)), p = (0.3, 0.5),
    for u = x^2 - y^2 (Q = 0 for every eps), u = x^2 + y^2 (Q = 4 for every eps) and u = e^x sin 2y
    (Q -> Delta u(p) = -3 e^0.3 sin 1 = -3.4076, error O(eps^2)). Circle means by the trapezoidal rule with 720 points
    (exact up to rounding for trigonometric polynomials, spectrally accurate here).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-1-1-mean-value.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS


def circle_mean(u, p, eps, n=720):
    t = 2 * np.pi * np.arange(n) / n
    return np.mean(u(p[0] + eps * np.cos(t), p[1] + eps * np.sin(t)))


u1 = lambda x: np.sin(2 * x) + 0.3 * x ** 2  # noqa: E731
d2u1 = lambda x: -4 * np.sin(2 * x) + 0.6  # noqa: E731
eps1 = 0.45
x_cvx, x_ccv = 2.3, 0.75
assert d2u1(x_cvx) > 0 and d2u1(x_ccv) < 0
for x0 in (x_cvx, x_ccv):
    avg = 0.5 * (u1(x0 - eps1) + u1(x0 + eps1))
    assert np.sign(avg - u1(x0)) == np.sign(d2u1(x0))
# second difference -> u''
for x0 in (x_cvx, x_ccv):
    q = [(u1(x0 + e) - 2 * u1(x0) + u1(x0 - e)) / e ** 2 for e in (0.1, 0.01)]
    assert abs(q[1] - d2u1(x0)) < 1e-3

p = np.array([0.3, 0.5])
funcs = {
    r"$x^2 - y^2$": (lambda x, y: x ** 2 - y ** 2, 0.0),
    r"$x^2 + y^2$": (lambda x, y: x ** 2 + y ** 2, 4.0),
    r"$e^x \sin 2y$": (lambda x, y: np.exp(x) * np.sin(2 * y), -3 * np.exp(0.3) * np.sin(1.0)),
}
eps = np.geomspace(0.02, 1.0, 40)
Q = {k: np.array([4 / e ** 2 * (circle_mean(f, p, e) - f(*p)) for e in eps]) for k, (f, _) in funcs.items()}
for k, (f, lap) in funcs.items():
    assert abs(Q[k][0] - lap) < 2e-3
assert np.abs(Q[r"$x^2 - y^2$"]).max() < 1e-9 and np.abs(Q[r"$x^2 + y^2$"] - 4).max() < 1e-9
lap3 = funcs[r"$e^x \sin 2y$"][1]
err = np.abs(Q[r"$e^x \sin 2y$"] - lap3)
slope = np.polyfit(np.log(eps[:15]), np.log(err[:15]), 1)[0]
assert abs(slope - 2) < 0.05
K3 = r"$e^x \sin 2y$"
f3 = funcs[K3][0]
q01 = 4 / 0.01 * (circle_mean(f3, p, 0.1) - f3(*p))
print(f"Delta u(p) = {lap3:.4f}, Q(1) = {Q[K3][-1]:.4f}, Q(0.1) = {q01:.4f}, slope {slope:.3f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
xs = np.linspace(0, 3.2, 300)
axa.plot(xs, u1(xs), color=col["main"], lw=1.6)
for x0, lab in ((x_cvx, r"$u'' > 0$"), (x_ccv, r"$u'' < 0$")):
    xl, xr = x0 - eps1, x0 + eps1
    avg = 0.5 * (u1(xl) + u1(xr))
    axa.plot([xl, xr], [u1(xl), u1(xr)], color=col["accent"], lw=1.2)
    axa.plot([xl, xr], [u1(xl), u1(xr)], "o", color=col["accent"], ms=4)
    axa.plot([x0], [avg], "s", color=col["accent"], ms=4)
    axa.plot([x0], [u1(x0)], "o", color=col["main"], ms=4)
    axa.annotate("", xy=(x0, avg), xytext=(x0, u1(x0)),
                 arrowprops=dict(arrowstyle="->", color=col["tangent"], lw=1.3))
    dx, dy = (0.1, -0.32) if x0 == x_cvx else (-0.2, -0.62)
    axa.text(x0 + dx, 0.5 * (avg + u1(x0)) + dy, lab, fontsize=10, color=col["tangent"])
axa.set_xlabel(r"$x$")
axa.set_ylabel(r"$u(x)$")
axa.set_title("(a) average minus value", fontsize=11)
axa.spines[["top", "right"]].set_visible(False)

styles = {r"$x^2 - y^2$": (col["third"], "-"), r"$x^2 + y^2$": (col["accent"], "-"),
          r"$e^x \sin 2y$": (col["tangent"], "-")}
for k in funcs:
    c, ls = styles[k]
    axb.semilogx(eps, Q[k], ls, color=c, lw=1.6)
axb.axhline(lap3, color=col["aux"], ls=":", lw=0.9)
axb.text(0.025, lap3 - 0.75, r"$\Delta u(p) = -3.408$", fontsize=9, color=col["aux"])
axb.set_xlabel(r"radius $\varepsilon$")
axb.set_ylabel(r"$Q(\varepsilon)$")
axb.set_ylim(-5.2, 5.2)
axb.text(0.025, 4.3, r"$u = x^2 + y^2$", fontsize=9.5, color=col["accent"])
axb.text(0.025, 0.3, r"$u = x^2 - y^2$", fontsize=9.5, color=col["third"])
axb.text(0.12, -2.4, r"$u = e^x \sin 2y$", fontsize=9.5, color=col["tangent"])
axb.set_title(r"(b) circle means about $p = (0.3, 0.5)$", fontsize=11)
axb.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.17, wspace=0.42)
dgfig.save(fig, __file__)
