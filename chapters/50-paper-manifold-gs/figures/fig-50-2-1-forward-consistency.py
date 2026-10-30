"""Figure 50.2.1: the forward consistency bound (7) of [ManifoldGS26] on the unit sphere, planes in the Frobenius norm.

(a) the latitude-longitude tiling with n_theta = 8 (128 cells, the tiling of Example 46.8.6): one cell C_i, its atom mu_i
(the parameter midpoint) and the plane P_i (the tangent plane at mu_i); the masses are the exact cell areas.
(b) against the cell size h (n_theta = 4, 8, 16, 32): the bound A(M)(h + eps_T) + eps_q of (7), the cell-integral form
(46.7.6) which equals Delta^{1,1} up to 0.8 percent (the lower bound by the test function min(1, distance to the atoms) is
computed for n_theta <= 16), and the largest |V_G(phi) - V_M(phi)| over 20 smooth 1-Lipschitz test functions.
Dashed: the same with masses q_i = A(C_i)(1 + 0.05 z_i), a 5 percent error that follows the height (eps_q = 0.1 pi).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-2-1-forward-consistency.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
import mgs50 as g  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
NTS = (4, 8, 16, 32)
res = {nt: g.forward_consistency(nt, lower=(nt <= 16)) for nt in NTS}
noisy = {nt: g.forward_consistency(nt, qbias=0.05, lower=False) for nt in NTS}
H = np.array([res[nt]["h"] for nt in NTS])
B = np.array([res[nt]["bound"] for nt in NTS])
U = np.array([res[nt]["upper"] for nt in NTS])
S = np.array([res[nt]["smooth"] for nt in NTS])
Bn = np.array([noisy[nt]["bound"] for nt in NTS])
Sn = np.array([noisy[nt]["smooth"] for nt in NTS])
# self-checks: the bound holds; the lower and upper forms bracket Delta^{1,1} tightly; bound / Delta is about 2;
# the smooth error falls like h^2 without noise and stops at a floor with noise
for nt in NTS[:3]:
    r = res[nt]
    assert r["lower"] <= r["upper"] <= r["bound"] and (r["upper"] - r["lower"]) / r["upper"] < 0.009
assert np.all(S <= U) and np.all(Sn <= Bn)
assert np.all(np.abs(B / U - 2.0) < 0.05)
slope_s = np.polyfit(np.log(H), np.log(S), 1)[0]
slope_u = np.polyfit(np.log(H), np.log(U), 1)[0]
assert 1.9 < slope_s < 2.3 and 0.95 < slope_u < 1.05
assert Sn[-1] > 5 * S[-1]

fig = plt.figure(figsize=(6.6, 3.4))
ax = dgfig.axes3d(fig, pos=121, elev=18, azim=-60)
uu, vv = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60), indexing="ij")
X = g.sph(uu, vv)
dgfig.surface(ax, X[..., 0], X[..., 1], X[..., 2], alpha=0.25, grid=False)
view = dgfig.view_vector(ax)
for th in np.linspace(0, np.pi, 9)[1:-1]:
    p = np.linspace(0, 2 * np.pi, 200)
    pts = g.sph(np.full_like(p, th), p)
    dgfig.curve3d(ax, pts, role="aux", lw=0.5, visible=pts @ view > 0, hidden=None)
for ph in np.linspace(0, 2 * np.pi, 17)[:-1]:
    t = np.linspace(0, np.pi, 100)
    pts = g.sph(t, np.full_like(t, ph))
    dgfig.curve3d(ax, pts, role="aux", lw=0.5, visible=pts @ view > 0, hidden=None)
cells, mus, areas = g.tiling(8)
vis = mus @ view > 0.05
ax.scatter(mus[vis, 0], mus[vis, 1], mus[vis, 2], s=5, color=C["main"], depthshade=False, zorder=8)
k = [i for i, c in enumerate(cells) if abs(c[0] - 3 * np.pi / 8) < 1e-9 and abs(c[2] - 15 * np.pi / 8) < 1e-9][0]
t0, t1, p0, p1 = cells[k]
TT, PP = np.meshgrid(np.linspace(t0, t1, 8), np.linspace(p0, p1, 8), indexing="ij")
Y = g.sph(TT, PP) * 1.003
ax.plot_surface(Y[..., 0], Y[..., 1], Y[..., 2], color=C["region"], alpha=0.6, linewidth=0, zorder=6)
mu = mus[k]
e1 = g.unit(np.cross([0, 0, 1.0], mu))
e2 = np.cross(mu, e1)
dgfig.tangent_plane(ax, mu, e1, e2, size=0.55, zorder=9)
dgfig.point3d(ax, mu, label=r"$\mu_i$", label_offset=(0.05, 0.05, 0.12), zorder=12)
ax.text(*(g.sph(t1 + 0.12, p0 - 0.05) * 1.02), r"$C_i$", fontsize=12, zorder=12)
ax.text(*(mu + 0.36 * e1 - 0.25 * e2), r"$P_i$", fontsize=12, color=C["tangent"], zorder=12)
dgfig.equal_aspect(ax, X[..., 0], X[..., 1], X[..., 2], zoom=1.25)
ax.set_title(r"(a) $n_\theta = 8$: atoms and one cell", fontsize=11)

bx = fig.add_subplot(122)
bx.loglog(H, B, "-o", color=C["main"], ms=4, label="bound (7)")
bx.loglog(H, U, "-s", color=C["tangent"], ms=4, label=r"$\Delta^{1,1}$ (cell form)")
bx.loglog(H, S, "-^", color=C["normal"], ms=4, label="smooth tests")
bx.loglog(H, Bn, "--", color=C["main"], lw=1.0)
bx.loglog(H, Sn, "--^", color=C["normal"], ms=3, lw=1.0, label=r"mass error $\varepsilon_q = 0.31$")
bx.text(H[2] * 0.93, S[2] * 2.4, "slope 2", fontsize=9, color=C["normal"], ha="right")
bx.text(H[1] * 1.05, U[1] * 0.5, "slope 1", fontsize=9, color=C["tangent"], ha="left")
bx.set_xlabel(r"cell size $h$")
bx.set_ylabel("error / bound")
bx.set_ylim(1.5e-5, 60)
bx.set_xticks(H)
bx.set_xticklabels([f"{x:.2f}" for x in H])
bx.xaxis.set_minor_formatter(plt.NullFormatter())
bx.legend(loc="lower right", fontsize=8, frameon=False, handlelength=1.6)
bx.set_title(r"(b) against $h$", fontsize=11)
fig.tight_layout()
dgfig.save(fig, __file__)
for nt in NTS:
    r, q = res[nt], noisy[nt]
    print(f"n_theta={nt:2d} atoms={r['atoms']:4d} h={r['h']:.4f} eT={r['eT']:.4f} bound={r['bound']:.3f} "
          f"upper={r['upper']:.4f} lower={r['lower']:.4f} smooth={r['smooth']:.5f} | noisy eq={q['eq']:.3f} "
          f"bound={q['bound']:.3f} smooth={q['smooth']:.5f}")
print("slopes: smooth %.2f, cell form %.2f" % (slope_s, slope_u))
