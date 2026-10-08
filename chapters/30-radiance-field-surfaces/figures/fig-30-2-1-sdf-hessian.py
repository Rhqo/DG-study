"""Figure 30.2.1: the eigenvalues of the Hessian of an SDF are minus the principal curvatures of its level sets.

Torus of revolution T_{R,r}, R = 2, r = 0.8, exact SDF f = sqrt((rho - R)^2 + z^2) - r (outward N = grad f).
Eigenvalues of Hess f computed by central finite differences (step 1e-4) of the exact SDF, compared with formulas.
(a) along the outward normal line through the outer equator (u = 0), as a function of the signed distance t:
    1/(r + t) (meridian direction), 1/(R + r + t) (parallel direction), 0 (normal direction). Blow-up at t = -r:
    the core circle of the tube = the medial axis.
(b) on the surface (t = 0) as a function of the meridian angle u: 1/r and cos u/(R + r cos u), compared with
    -kappa_1, -kappa_2 computed from the parametrization of GUIDELINES §7 (dgsym.shape_operator, sign flipped so that
    the principal curvatures refer to the outward normal).
Self-checks: numerical eigenvalues match the formulas to 1e-5.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-2-1-sdf-hessian.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
R0, r0 = 2.0, 0.8


def sdf(P):
    rho = np.hypot(P[..., 0], P[..., 1])
    return np.hypot(rho - R0, P[..., 2]) - r0


def num_hess(P, h=1e-4):
    Hm = np.zeros((3, 3))
    E = np.eye(3)
    for i in range(3):
        for j in range(3):
            Hm[i, j] = (sdf(P + h * E[i] + h * E[j]) - sdf(P + h * E[i] - h * E[j]) - sdf(P - h * E[i] + h * E[j])
                        + sdf(P - h * E[i] - h * E[j])) / (4 * h * h)
    return np.sort(np.linalg.eigvalsh(0.5 * (Hm + Hm.T)))


def torus_point(u, v, t):
    rr = r0 + t
    return np.array([(R0 + rr * np.cos(u)) * np.cos(v), (R0 + rr * np.cos(u)) * np.sin(v), rr * np.sin(u)])


# (a) along the normal line at u = 0
ts = np.linspace(-0.62, 1.5, 400)
td = np.linspace(-0.6, 1.5, 15)
ev_num = np.array([num_hess(torus_point(0.0, 0.3, t)) for t in td])
ev_ref = np.sort(np.stack([np.zeros_like(td), 1 / (R0 + r0 + td), 1 / (r0 + td)], 1), axis=1)
assert np.max(np.abs(ev_num - ev_ref)) < 1e-5

# (b) on the surface vs u, with the parametric principal curvatures (dgsym, GUIDELINES §7 torus)
ex = dgsym.EXAMPLES["torus"]
Rs, rs = ex["params"]
us, vs = ex["coords"]
Wpar = sp.lambdify(us, dgsym.shape_operator(ex["expr"], us, vs, positive=ex["positive"]).subs({Rs: R0, rs: r0}),
                   modules="numpy")
uu = np.linspace(0, 2 * np.pi, 400)
ud = np.linspace(0, 2 * np.pi, 13)
kin = np.array([np.sort(np.linalg.eigvals(np.array(Wpar(u), dtype=float)).real) for u in ud])   # inward normal
neg_kappa_out = np.sort(kin, axis=1)                                                             # -kappa_out = kappa_in
ev_s = np.array([num_hess(torus_point(u, 1.0, 0.0)) for u in ud])
nz = np.array([e[np.argsort(np.abs(e))][1:] for e in ev_s])
assert np.max(np.abs(np.sort(nz, axis=1) - neg_kappa_out)) < 1e-5
assert np.max(np.abs(np.array([e[np.argmin(np.abs(e))] for e in ev_s]))) < 1e-5

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 7.6), gridspec_kw=dict(hspace=0.45))

axa.axvspan(-0.9, 0.0, color=C["region"], alpha=0.15, lw=0)
axa.plot(ts, 1 / (r0 + ts), color=C["tangent"], lw=2.2, label=r"$1/(r + t)$  meridian")
axa.plot(ts, 1 / (R0 + r0 + ts), color=C["third"], lw=2.2, label=r"$1/(R + r + t)$  parallel")
axa.plot(ts, 0 * ts, color=C["normal"], lw=2.2, label=r"$0$  normal $\mathbf{N}$")
axa.plot(td, ev_num[:, 2], "o", color=C["main"], ms=4.5, mfc="white", zorder=5)
axa.plot(td, ev_num[:, 1], "o", color=C["main"], ms=4.5, mfc="white", zorder=5)
axa.plot(td, ev_num[:, 0], "o", color=C["main"], ms=4.5, mfc="white", zorder=5, label="finite differences")
axa.axvline(0, color=C["main"], lw=1.0)
axa.axvline(-r0, color=C["accent"], lw=1.4, ls=(0, (4, 3)))
axa.text(0.03, 4.3, r"$S$ ($t = 0$)", fontsize=12)
axa.text(-0.76, 4.3, "core circle\n(medial axis)", fontsize=11.5, color=C["main"], va="top")
axa.text(-0.4, -0.75, "inside", fontsize=11.5, ha="center", color=C["tangent"])
axa.set_xlim(-0.9, 1.5)
axa.set_ylim(-1.0, 5.0)
axa.set_xlabel(r"signed distance $t$ along $\mathbf{N}$ at $u = 0$")
axa.set_ylabel(r"eigenvalues of $\mathrm{Hess}\,f$")
axa.legend(loc="upper right", frameon=False, fontsize=11)
axa.set_title(r"(a) along the normal line (offset level sets)", fontsize=13.5)

axb.axhline(0, color=C["normal"], lw=2.2, label=r"$0$ (normal $\mathbf{N}$)")
axb.plot(uu, 0 * uu + 1 / r0, color=C["tangent"], lw=2.2, label=r"$1/r$")
axb.plot(uu, np.cos(uu) / (R0 + r0 * np.cos(uu)), color=C["third"], lw=2.2, label=r"$\cos u/(R + r\cos u)$")
axb.plot(ud, neg_kappa_out[:, 0], "s", color=C["main"], ms=5, mfc="none", zorder=5)
axb.plot(ud, neg_kappa_out[:, 1], "s", color=C["main"], ms=5, mfc="none", zorder=5,
         label=r"$-\kappa_1, -\kappa_2$ (parametrization)")
axb.axvspan(np.pi / 2, 3 * np.pi / 2, color=C["covector"], alpha=0.10, lw=0)
axb.text(np.pi, 0.62, "inner half:\nopposite signs\n($K < 0$)", ha="center", fontsize=11, va="center")
axb.set_xlim(0, 2 * np.pi)
axb.set_ylim(-1.0, 2.3)
axb.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
axb.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
axb.set_xlabel(r"meridian angle $u$ ($u = 0$: outer equator)")
axb.set_ylabel(r"eigenvalues on $S$")
axb.legend(loc="upper center", frameon=False, fontsize=11, ncol=2, columnspacing=1.0, handlelength=1.8)
axb.set_title(r"(b) on the surface: $\mathrm{eig}\,\mathrm{Hess}\,f = (0, -\kappa_1, -\kappa_2)$", fontsize=13.5)

dgfig.save(fig, __file__)
