"""Figure 48.2.1: principal and Gaussian curvature of the torus of revolution T_{2,0.8} read from the plane field alone.

The torus x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u), R = 2, r = 0.8 (GUIDELINES §7) is sampled on
the (u, v) grid of 48 x 96 points. Only the positions x and the tangent projections P = I - N N^T at the grid points
are used: x_u, x_v, P_u, P_v by central differences on the grid, grad^M P_jk = sum g^{ab} (d_a P_jk) x_b,
A_ijk = <grad^M P_jk, e_i> ((1.2) of [BLM22]), B^k_ij = (A_ijk + A_jik - A_kij)/2 ((1.4)), and the 2 x 2 matrix
Q^T (sum_k B^k_ij N_k) Q with the outward normal N (Proposition 48.2.1). Its eigenvalues approximate the principal
curvatures (-1/r, -cos u/(R + r cos u)) for the outward normal.
(a) the torus coloured by the Gaussian curvature det(Q^T B_N Q) computed this way (RdBu_r, 0 in the middle).
(b) kappa_1, kappa_2 against u: exact curves and the grid values (dots, v = 0).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/figures/fig-48-2-1-torus-from-planes.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import var48 as v  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8
NU, NV = 48, 96
hu, hv = 2 * np.pi / NU, 2 * np.pi / NV
U, Vg = np.meshgrid(np.arange(NU) * hu - np.pi, np.arange(NV) * hv, indexing="ij")
X = v.torus_point(U, Vg, R, r)
Nout = v.torus_outward(U, Vg)
P = v.proj_from_normal(Nout)


def cdiff(F, axis, h):
    return (np.roll(F, -1, axis=axis) - np.roll(F, 1, axis=axis)) / (2 * h)


Xu, Xv = cdiff(X, 0, hu), cdiff(X, 1, hv)
Pu, Pv = cdiff(P, 0, hu), cdiff(P, 1, hv)
g = np.stack([np.stack([np.einsum("...a,...a", Xu, Xu), np.einsum("...a,...a", Xu, Xv)], -1),
              np.stack([np.einsum("...a,...a", Xv, Xu), np.einsum("...a,...a", Xv, Xv)], -1)], -2)
gi = np.linalg.inv(g)
# grad^M P_jk = g^{uu} P_u x_u + g^{uv} (P_u x_v + P_v x_u) + g^{vv} P_v x_v ;  A[..., i, j, k]
A = (np.einsum("...,...jk,...i->...ijk", gi[..., 0, 0], Pu, Xu)
     + np.einsum("...,...jk,...i->...ijk", gi[..., 0, 1], Pu, Xv)
     + np.einsum("...,...jk,...i->...ijk", gi[..., 1, 0], Pv, Xu)
     + np.einsum("...,...jk,...i->...ijk", gi[..., 1, 1], Pv, Xv))
B = v.B_from_A(A)                                                          # (1.4), batched over the grid
BN = np.einsum("...ijk,...k->...ij", B, Nout)
e_v = np.stack([-np.sin(Vg), np.cos(Vg), np.zeros_like(Vg)], -1)          # unit tangent along v
e_u = np.stack([-np.sin(U) * np.cos(Vg), -np.sin(U) * np.sin(Vg), np.cos(U)], -1)
Q = np.stack([e_u, e_v], -1)
Wbar = np.einsum("...ai,...ab,...bj->...ij", Q, BN, Q)
kap = np.linalg.eigvalsh(0.5 * (Wbar + np.swapaxes(Wbar, -1, -2)))
Kfd = kap[..., 0] * kap[..., 1]
k_u, k_v = v.torus_kappa_outward(U, R, r)
ex = np.sort(np.stack([k_u, k_v], -1), axis=-1)
Kex = k_u * k_v

# self-checks
err = np.abs(kap - ex).max()
assert err < 0.02, err
assert np.abs(Kfd - Kex).max() < 0.02
assert np.allclose(Kex, np.cos(U) / (r * (R + r * np.cos(U))))               # GUIDELINES §7
assert abs(Kex[NU // 2, 0] - 1 / (r * (R + r))) < 1e-12                       # outer equator u = 0: 0.4464
assert abs(Kex[0, 0] + 1 / (r * (R - r))) < 1e-12                             # inner equator u = -pi: -1.0417
assert np.allclose(np.einsum("...qiq->...i", A), -(1 / r + np.cos(U) / (R + r * np.cos(U)))[..., None] * Nout,
                   atol=0.03)                                                 # sum_q A_qiq = H (sum convention)

fig = plt.figure(figsize=(6.6, 3.0))
ax = dgfig.axes3d(fig, elev=38, azim=-60)
ax.set_position([-0.05, 0.0, 0.48, 1.0])
UU = np.vstack([U, U[:1] + 2 * np.pi])
VV = np.vstack([Vg, Vg[:1]])
UU = np.hstack([UU, UU[:, :1]])
VV = np.hstack([VV, VV[:, :1] + 2 * np.pi])
S = v.torus_point(UU, VV, R, r)
Kcl = np.vstack([Kfd, Kfd[:1]])
Kcl = np.hstack([Kcl, Kcl[:, :1]])
Kc = 0.25 * (Kcl[:-1, :-1] + Kcl[1:, :-1] + Kcl[:-1, 1:] + Kcl[1:, 1:])
norm = TwoSlopeNorm(vmin=-1.0417, vcenter=0.0, vmax=0.4464)
cmap = plt.get_cmap("RdBu_r")
ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], facecolors=cmap(norm(Kc)), rstride=1, cstride=1, linewidth=0,
                antialiased=False, shade=False, rasterized=True, zorder=1)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.3)
ax.set_title(r"(a) $K$ from $P$ only", fontsize=11, y=0.92)
cax = fig.add_axes([0.435, 0.25, 0.013, 0.5])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([-1.0417, 0, 0.4464])
cb.set_ticklabels([r"$-1.04$", r"$0$", r"$0.45$"])

ax2 = fig.add_axes([0.63, 0.18, 0.35, 0.68])
uu = np.linspace(-np.pi, np.pi, 400)
a1, a2 = v.torus_kappa_outward(uu, R, r)
ax2.plot(uu, a1, color=C["tangent"], lw=1.4, label=r"$-1/r$")
ax2.plot(uu, a2, color=C["normal"], lw=1.4, label=r"$-\cos u/(R + r\cos u)$")
ax2.plot(U[:, 0], kap[:, 0, 0], "o", ms=2.6, color=C["main"], label="grid")
ax2.plot(U[:, 0], kap[:, 0, 1], "o", ms=2.6, color=C["main"])
ax2.axhline(0, color=C["aux"], lw=0.6, ls=":")
ax2.set_xlabel(r"$u$")
ax2.set_ylabel(r"$\kappa_1,\ \kappa_2$")
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax2.set_title(r"(b) along $v = 0$", fontsize=11)
ax2.set_ylim(-1.45, 0.95)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.40), fontsize=8.5, frameon=False, ncol=1)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

print(f"max |kappa_grid - kappa_exact| = {err:.4f}, max |K_grid - K| = {np.abs(Kfd - Kex).max():.4f}")
dgfig.save(fig, __file__)
