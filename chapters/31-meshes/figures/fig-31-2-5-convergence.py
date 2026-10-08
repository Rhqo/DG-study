"""Figure 31.2.5: pointwise curvature does not converge on irregular meshes, integrated curvature does (Section 31.2).

Torus T_{R,r}, R = 2, r = 0.8. "regular": the (u, v) grid mesh. "jittered": the same connectivity, but every vertex
is moved inside the surface by changing (u, v) by up to 20% of the grid step (uniform noise, seed 0); all vertices
still lie exactly on the torus.
(a) The pointwise error delta_i / A_i - K(x_i) against u for the jittered 24 x 48 (orange) and 96 x 192 (blue)
    meshes. The band does not get thinner (95th percentile of |error|: about the same).
(b) Errors against the mean edge length h, log-log. Pointwise: max_i |delta_i / A_i - K(x_i)|.
    Integrated: |sum_i delta_i cos u_i - int K cos u dA| (int K cos u dA = 2 pi^2).
Self-check: on the jittered meshes the pointwise error does not decrease, while the integrated error decreases
by a factor of about 4 per refinement on both kinds of meshes.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-2-5-convergence.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8
K_exact = lambda u: np.cos(u) / (r * (R + r * np.cos(u)))  # noqa: E731
WEAK_EXACT = 2 * np.pi ** 2


def torus(nu, nv, jit):
    V, F = dm.torus_mesh(R, r, nu, nv)
    rng = np.random.default_rng(0)
    u = np.repeat(2 * np.pi * np.arange(nu) / nu, nv)
    v = np.tile(2 * np.pi * np.arange(nv) / nv, nu)
    u = u + jit * (2 * np.pi / nu) * rng.uniform(-1, 1, u.shape)
    v = v + jit * (2 * np.pi / nv) * rng.uniform(-1, 1, v.shape)
    V = np.stack([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)], 1)
    return V, F, u


def errors(nu, nv, jit):
    V, F, u = torus(nu, nv, jit)
    d = dm.angle_defect(V, F)
    A = dm.mass_barycentric(V, F)
    h = np.mean([np.linalg.norm(V[a] - V[b]) for a, b in dm.edges(F)])
    return h, np.abs(d / A - K_exact(u)).max(), abs((d * np.cos(u)).sum() - WEAK_EXACT), u, d / A


fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.3))

# ---------------------------------------------------------------- (a)
widths = {}
for (nu, nv), col, ms, lab, z in (((96, 192), C["tangent"], 1.2, r"jittered $96 \times 192$", 2),
                                  ((24, 48), C["accent"], 3.2, r"jittered $24 \times 48$", 3)):
    _, _, _, u, Ki = errors(nu, nv, 0.2)
    res_ = Ki - K_exact(u)
    widths[nu] = np.percentile(np.abs(res_), 95)
    ax.plot(u, res_, "o", ms=ms, color=col, alpha=0.8, mew=0, label=lab, zorder=z, rasterized=True)
assert widths[96] > 0.9 * widths[24]   # the band does not get thinner
ax.axhline(0, color=C["main"], lw=1.0, zorder=4)
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$\delta_i / A_i - K(x_i)$")
ax.legend(fontsize=10.5, frameon=False, loc="lower left", markerscale=2.5)
ax.set_ylim(-0.45, 0.45)
ax.set_title("(a) pointwise error, jittered meshes", fontsize=11.5)

# ---------------------------------------------------------------- (b)
grids = [(12, 24), (24, 48), (48, 96), (96, 192)]
res = {}
for jit in (0.0, 0.2):
    res[jit] = np.array([errors(nu, nv, jit)[:3] for nu, nv in grids])
reg, jt = res[0.0], res[0.2]
axb.loglog(reg[:, 0], reg[:, 1], "o-", color=C["third"], lw=1.5, ms=5, label="pointwise, regular")
axb.loglog(jt[:, 0], jt[:, 1], "s--", color=C["normal"], lw=1.5, ms=5, label="pointwise, jittered")
axb.loglog(reg[:, 0], reg[:, 2], "o-", color=C["tangent"], lw=1.5, ms=5, label="integrated, regular")
axb.loglog(jt[:, 0], jt[:, 2], "s--", color=C["accent"], lw=1.5, ms=5, label="integrated, jittered")
assert jt[-1, 1] > 0.9 * jt[0, 1]                      # pointwise error does not decrease
for arr in (reg, jt):
    rates = np.log(arr[:-1, 2] / arr[1:, 2]) / np.log(arr[:-1, 0] / arr[1:, 0])
    assert (rates > 1.8).all()                         # integrated error ~ h^2
rates = np.log(reg[:-1, 1] / reg[1:, 1]) / np.log(reg[:-1, 0] / reg[1:, 0])
assert (rates > 1.8).all()
axb.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5]))
axb.xaxis.set_major_formatter(ScalarFormatter())
axb.xaxis.set_minor_formatter(NullFormatter())
axb.set_xlabel(r"mean edge length $h$")
axb.set_ylabel("error")
axb.legend(fontsize=10.5, frameon=False, loc="lower right")
axb.grid(True, which="major", lw=0.3, alpha=0.5)
axb.set_title(r"(b) errors of $K$ from angle defects", fontsize=11.5)

fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.15)
dgfig.save(fig, __file__)
