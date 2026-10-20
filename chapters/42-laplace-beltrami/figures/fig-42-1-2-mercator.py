"""Figure 42.1.2: the Laplace-Beltrami operator in conformal coordinates (Section 42.1, Proposition 42.1.10,
Example 42.1.11).

Unit sphere in Mercator coordinates (Example 38.2.10): x(phi, t) = (sech t cos phi, sech t sin phi, tanh t),
g = e^{2 mu}(dphi^2 + dt^2) with mu = -log cosh t, so e^{-2 mu} = cosh^2 t. The height is z = tanh t.
(a) as functions of t: z (black), the flat second derivative d^2 z / dt^2 = -2 sech^2 t tanh t (gray dashed), and
    Delta z = e^{-2 mu} d^2 z / dt^2 = -2 tanh t = -2 z (blue). Self-check: sympy identity and the values at t = 1.
(b) the sphere with the Mercator grid t = -2.4, -1.6, ..., 2.4 and phi = 0, 30, ..., 330 degrees: the parameter
    squares of side 0.8 become smaller toward the poles; their area is about e^{2 mu} 0.64 = sech^2(t) 0.64.
    Faces of the sampled grid are shaded by sech^2 t (light = small factor).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-1-2-mercator.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

col = dgfig.COLORS
t_ = sp.symbols("t", real=True)
z_ = sp.tanh(t_)
flat = sp.diff(z_, t_, 2)
lap = sp.cosh(t_) ** 2 * flat
assert sp.simplify(lap + 2 * z_) == 0
assert sp.simplify(flat + 2 * sp.sech(t_) ** 2 * sp.tanh(t_)) == 0
# the parametrization is the unit sphere and is conformal with e^{2 mu} = sech^2 t
ph_ = sp.symbols("phi", real=True)
X = sp.Matrix([sp.sech(t_) * sp.cos(ph_), sp.sech(t_) * sp.sin(ph_), sp.tanh(t_)])
assert sp.simplify(X.dot(X) - 1) == 0
Xp, Xt = X.diff(ph_), X.diff(t_)
E, Fc, G = [sp.simplify(a) for a in (Xp.dot(Xp), Xp.dot(Xt), Xt.dot(Xt))]
assert sp.simplify(E - sp.sech(t_) ** 2) == 0 and Fc == 0 and sp.simplify(G - sp.sech(t_) ** 2) == 0
f_flat = sp.lambdify(t_, flat, modules="numpy")
print(f"t = 1: z = {np.tanh(1):.4f}, flat = {f_flat(1.0):.4f}, Delta z = {-2 * np.tanh(1):.4f}, factor cosh^2 = {np.cosh(1)**2:.4f}")

fig = plt.figure(figsize=(6.8, 3.0))
axa = fig.add_axes([0.08, 0.17, 0.46, 0.73])
ts = np.linspace(-2.6, 2.6, 400)
axa.plot(ts, np.tanh(ts), color=col["main"], lw=1.6)
axa.plot(ts, f_flat(ts), "--", color=col["aux"], lw=1.3)
axa.plot(ts, -2 * np.tanh(ts), color=col["tangent"], lw=1.6)
axa.axhline(0, color="#BBBBBB", lw=0.6)
axa.text(1.15, 1.08, r"$z = \tanh t$", fontsize=10, color=col["main"])
axa.text(-2.55, 0.42, r"$\partial_t^2 z$", fontsize=10, color=col["aux"])
axa.text(-2.55, -1.65, r"$e^{-2\mu}\partial_t^2 z = -2z$", fontsize=10, color=col["tangent"])
axa.set_xlabel(r"$t$")
axa.set_ylim(-2.3, 2.3)
axa.set_title(r"(a) $\Delta z$ in Mercator coordinates", fontsize=11)
axa.spines[["top", "right"]].set_visible(False)

axb = fig.add_axes([0.58, 0.04, 0.40, 0.86])
import dgmesh as dm  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

cam = dm.Ortho(elev=12, azim=-60)
tg = np.arange(-2.4, 2.41, 0.8)
pg = np.radians(np.arange(0, 361, 30))
xm = lambda T, P: np.stack([np.cos(P) / np.cosh(T), np.sin(P) / np.cosh(T), np.tanh(T)], -1)  # noqa: E731
polys, shades, depths = [], [], []
for i in range(len(tg) - 1):
    for j in range(len(pg) - 1):
        tt = np.linspace(tg[i], tg[i + 1], 5)
        pp = np.linspace(pg[j], pg[j + 1], 5)
        ring = np.concatenate([xm(tt, np.full(5, pp[0])), xm(np.full(5, tt[-1]), pp)[1:],
                               xm(tt[::-1], np.full(5, pp[-1]))[1:], xm(np.full(5, tt[0]), pp[::-1])[1:]])
        c = xm(0.5 * (tg[i] + tg[i + 1]), 0.5 * (pg[j] + pg[j + 1]))
        if c @ cam.view <= 0:
            continue
        polys.append(cam(ring))
        shades.append(1.0 / np.cosh(0.5 * (tg[i] + tg[i + 1])) ** 2)
        depths.append(c @ cam.view)
order = np.argsort(depths)
cmap = plt.get_cmap("Blues")
pc = PolyCollection([polys[k] for k in order], facecolors=[cmap(0.15 + 0.55 * shades[k]) for k in order],
                    edgecolors=col["accent"], linewidths=0.8)
axb.add_collection(pc)
caps = [xm(np.full(200, tc), np.linspace(0, 2 * np.pi, 200)) for tc in (-2.4, 2.4)]
for cap in caps:
    vis = cap @ cam.view > 0
    q = cam(cap)
    axb.plot(np.where(vis, q[:, 0], np.nan), np.where(vis, q[:, 1], np.nan), color=col["accent"], lw=0.8)
th = np.linspace(0, 2 * np.pi, 300)
axb.plot(np.cos(th), np.sin(th), color=col["main"], lw=1.0)
axb.set_aspect("equal")
axb.set_xlim(-1.05, 1.05)
axb.set_ylim(-1.05, 1.05)
axb.set_axis_off()
axb.set_title(r"(b) Mercator grid, $\Delta t = 0.8$", fontsize=11)
dgfig.save(fig, __file__)
