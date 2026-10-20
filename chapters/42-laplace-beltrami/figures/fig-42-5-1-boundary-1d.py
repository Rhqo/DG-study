"""Figure 42.5.1: Dirichlet and Neumann conditions for u'' = 1 on [0, 1] (Section 42.5, Example 42.5.3).

(a) Dirichlet: u(0) = a, u(1) = b has exactly one solution u = x^2/2 + (b - a - 1/2) x + a; drawn for
    (a, b) = (0, 0), (0, 0.4), (0.3, -0.1); dots mark the prescribed values.
(b) Neumann: the outward derivatives -u'(0) = g_0 and u'(1) = g_1 can be prescribed only if g_0 + g_1 = int_0^1 1 dx = 1.
    For (g_0, g_1) = (1/2, 1/2) the solutions are u = (x^2 - x)/2 + c (drawn for c = -0.15, 0, 0.15); the short arrows
    show the prescribed slopes. For (g_0, g_1) = (1, 1) there is no solution (u'(1) - u'(0) would have to be 2).
Self-check: u'' = 1, boundary values and slopes; the compatibility int f = g_0 + g_1.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-5-1-boundary-1d.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
xs = np.linspace(0, 1, 201)
dir_cases = [((0.0, 0.0), col["tangent"]), ((0.0, 0.4), col["third"]), ((0.3, -0.1), col["accent"])]


def u_dir(x, a, b):
    return 0.5 * x ** 2 + (b - a - 0.5) * x + a


for (a, b), _ in dir_cases:
    assert abs(u_dir(0, a, b) - a) < 1e-15 and abs(u_dir(1, a, b) - b) < 1e-15
u_neu = lambda x, c: 0.5 * (x ** 2 - x) + c  # noqa: E731
du_neu = lambda x: x - 0.5  # noqa: E731
assert abs(-du_neu(0) - 0.5) < 1e-15 and abs(du_neu(1) - 0.5) < 1e-15 and abs((0.5 + 0.5) - 1.0) < 1e-15

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for (a, b), c in dir_cases:
    axa.plot(xs, u_dir(xs, a, b), color=c, lw=1.6)
    axa.plot([0, 1], [a, b], "o", color=c, ms=5)
axa.set_title(r"(a) Dirichlet: $u(0) = a$, $u(1) = b$", fontsize=10.5)
for c, colr in ((-0.15, col["accent"]), (0.0, col["tangent"]), (0.15, col["third"])):
    axb.plot(xs, u_neu(xs, c), color=colr, lw=1.6)
    for x0 in (0.0, 1.0):
        s = du_neu(x0)
        axb.annotate("", xy=(x0 + (0.12 if x0 == 0 else -0.12) * (1 if x0 == 0 else 1),
                             u_neu(x0, c) + s * (0.12 if x0 == 0 else -0.12)),
                     xytext=(x0, u_neu(x0, c)), arrowprops=dict(arrowstyle="-", color=col["main"], lw=2.2))
axb.text(0.5, 0.33, r"$-u'(0) = u'(1) = \frac{1}{2}$", ha="center", fontsize=10)
axb.text(0.5, 0.22, r"$u + c$: not unique", ha="center", fontsize=9.5, color=col["aux"])
axb.set_title(r"(b) Neumann: $-u'(0) = g_0$, $u'(1) = g_1$", fontsize=10.5)
for ax in (axa, axb):
    ax.set_xlabel(r"$x$")
    ax.axhline(0, color="#BBBBBB", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axa.set_ylabel(r"$u(x)$")
axa.set_ylim(-0.32, 0.45)
fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.17, wspace=0.12)
dgfig.save(fig, __file__)
