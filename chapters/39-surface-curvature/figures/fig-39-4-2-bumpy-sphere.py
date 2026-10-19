"""Figure 39.4.2: total curvature does not see the bumps.

Bumpy sphere x(t, v) = rho(t) (sin t cos v, sin t sin v, cos t), rho = 1 + 0.2 cos 4t (a surface of revolution,
N = x_t x x_v / |...| outward). (a) the surface colored by K (RdBu_r centered at 0; K from -5.01 to 9.34 (3025/324 at the poles)).
(b) the integral of K dA over the cap t <= t0 against t0 for the bumpy sphere (solid black) and the round unit sphere
(gray), and 2 pi - (boundary integral of kappa_g) for the bumpy sphere at 13 values of t0 (vermilion dots,
Proposition 39.4.6). Both totals end at 4 pi. Also writes an interactive version.

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-4-2-bumpy-sphere.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg39  # noqa: E402

C = dgfig.COLORS
prof = ddg39.bumpy_profile
tt = np.linspace(0.02, np.pi - 0.02, 400)
Kt = ddg39.revolution_K(prof, tt)
# checks: K from the profile = K from the 2D numeric Weingarten map; sign change; GB balance
for t0 in (0.4, 1.1, 2.2):
    Kn, Hn = ddg39.KH(ddg39.bumpy_sphere, t0, 0.3)
    assert abs(Kn - ddg39.revolution_K(prof, t0)) < 1e-4
    assert ddg39.unit_normal(ddg39.bumpy_sphere, t0, 0.3) @ ddg39.bumpy_sphere(t0, 0.3) > 0     # outward
assert Kt.min() < -4.9 and Kt.max() > 9.2
cum = np.array([ddg39.cap_terms(prof, x, n=2000)[0] for x in tt])
bdy = np.array([ddg39.cap_terms(prof, x, n=2000)[1] for x in tt])
assert np.max(np.abs(cum + bdy - 2 * np.pi)) < 1e-4
total = ddg39.cap_terms(prof, np.pi, n=8000)[0]
assert abs(total - 4 * np.pi) < 1e-4

fig = plt.figure(figsize=(7.2, 3.6))
ax = dgfig.axes3d(fig, 121, elev=12, azim=-60)
T, V = np.meshgrid(np.linspace(1e-3, np.pi - 1e-3, 90), np.linspace(0, 2 * np.pi, 90), indexing="ij")
X = ddg39.bumpy_sphere(T, V)
KK = ddg39.revolution_K(prof, T)
norm = TwoSlopeNorm(vmin=-5.1, vcenter=0.0, vmax=9.4)
cmap = plt.get_cmap("RdBu_r")
ax.plot_surface(*X, facecolors=cmap(norm(KK)), linewidth=0, antialiased=True, shade=False, rstride=1, cstride=1,
                rasterized=True)
dgfig.equal_aspect(ax, X.reshape(3, -1).T, zoom=1.3)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.0, shrink=0.75)
cb.set_label(r"$K$", fontsize=11)
cb.set_ticks([-5, 0, 5, 9])
cb.ax.tick_params(labelsize=9)
ax.set_title("(a)", loc="left", fontsize=12, y=0.96)

ax = fig.add_subplot(122)
ax.plot(tt, cum, color=C["main"], lw=1.8, label="bumpy sphere")
ax.plot(tt, 2 * np.pi * (1 - np.cos(tt)), color=C["aux"], lw=1.4, label="unit sphere")
sel = np.linspace(0, len(tt) - 1, 13).astype(int)
ax.plot(tt[sel], 2 * np.pi - bdy[sel], "o", color=C["normal"], ms=4,
        label=r"$2\pi - \int_{\partial S}\kappa_g\,ds$")
ax.axhline(4 * np.pi, color=C["aux"], lw=0.7, ls=(0, (3, 2)))
ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
ax.set_yticks([0, 2 * np.pi, 4 * np.pi])
ax.set_yticklabels([r"$0$", r"$2\pi$", r"$4\pi$"])
ax.set_xlabel(r"$t_0$")
ax.set_xlim(0, np.pi)
ax.legend(fontsize=9.5, frameon=False, loc="center left")
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.0, right=0.98, bottom=0.14, top=0.92, wspace=0.15)
dgfig.save(fig, __file__)

# interactive version: a triangulated bumpy sphere colored by K
Vm, Fm = dm.icosphere(4)
tm = np.arccos(np.clip(Vm[:, 2], -1, 1))
Vb = Vm * ddg39.bumpy_rho(tm)[:, None]
Kv = ddg39.revolution_K(prof, np.clip(tm, 1e-3, np.pi - 1e-3))
tr = [dm.plotly_mesh(Vb, Fm, intensity=Kv, colorscale="RdBu_r", cmin=-9.4, cmax=9.4, name="K", showscale=True,
                     colorbar_title="K")]
dm.plotly_save(tr, __file__, "Bumpy sphere rho = 1 + 0.2 cos 4t colored by K (total curvature 4 pi)", "fig-39-4-2")
print(f"K in [{Kt.min():.3f}, {Kt.max():.3f}], total {total:.6f}")
