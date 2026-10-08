"""Figure 30.2.3: what an L1 mean-curvature penalty sees on a rounded fold (Section 30.2).

Surface: the boundary of the quarter-space {x <= 0, y <= 0} with its 90-degree edge rounded by a cylinder of radius rho,
extruded along z (so kappa_z = 0 and |H| = kappa/2). Exact SDF: offset of the shifted quarter-space.
|H| along the cross-section is measured as |Delta f|/2 with the 6-tap finite-difference Laplacian (step 1e-4) of the SDF
at points of the zero set.
(a) |H| as a function of arc length s (s = 0 at the middle of the fillet) for rho = 0.1, 0.3, 0.6: boxes of height
    1/(2 rho) and width pi rho/2. Inset: the three cross-sections.
(b) per unit length of the edge: int |H| dA (= pi/4 for every rho) and int H^2 dA (= pi/(8 rho)), log-log, rho in
    [0.02, 1]; markers = numerical integration, lines = formulas.
Self-checks: numerical integrals match the formulas to 0.2 %.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-2-3-l1-vs-l2.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS


def sdf(P, rho):
    qx, qy = P[..., 0] + rho, P[..., 1] + rho
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rho


def curve(rho, n_flat=400, n_arc=1200, L=1.2):
    """zero set, parametrized by arc length s with s = 0 at the middle of the arc."""
    a = np.linspace(0, np.pi / 2, n_arc)
    arc = np.stack([-rho + rho * np.cos(a), -rho + rho * np.sin(a)], 1)
    s_arc = rho * (a - np.pi / 4)
    yv = np.linspace(-rho - L, -rho, n_flat, endpoint=False)
    right = np.stack([0 * yv, yv], 1)                          # x = 0 face, below the arc
    s_r = -(np.pi / 4) * rho - (-rho - yv)
    xv = np.linspace(-rho, -rho - L, n_flat + 1)[1:]
    top = np.stack([xv, 0 * xv], 1)                            # y = 0 face, left of the arc
    s_t = (np.pi / 4) * rho + (-rho - xv)
    P = np.concatenate([right, arc, top])
    s = np.concatenate([s_r, s_arc, s_t])
    return P, s


def absH(P, rho, h=1e-4):
    E = np.eye(2)
    lap = sum(sdf(P + h * e, rho) + sdf(P - h * e, rho) - 2 * sdf(P, rho) for e in E) / h ** 2
    return np.abs(lap) / 2


rhos = [0.1, 0.3, 0.6]
cols = [C["normal"], C["accent"], C["tangent"]]

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 7.8), gridspec_kw=dict(hspace=0.42))
for rho, col in zip(rhos, cols):
    P, s = curve(rho)
    assert np.max(np.abs(sdf(P, rho))) < 1e-12
    hval = absH(P, rho)
    inside = np.abs(s) < np.pi * rho / 4 - 1e-3
    assert np.allclose(hval[inside], 1 / (2 * rho), rtol=1e-3)
    assert np.max(hval[np.abs(s) > np.pi * rho / 4 + 2e-3]) < 1e-3
    order = np.argsort(s)
    axa.plot(s[order], hval[order], color=col, lw=2.2, label=rf"$r_{{\mathrm{{f}}}} = {rho}$")
    axa.fill_between(s[order], 0, hval[order], color=col, alpha=0.12, lw=0)
    if rho == 0.1:
        axa.text(0.0, 1 / (2 * rho) + 0.12, rf"$1/(2 r_{{\mathrm{{f}}}}) = {1 / (2 * rho):.2f}$", color=col, ha="center", fontsize=11.5)
    else:
        axa.text(np.pi * rho / 4 + 0.02, 1 / (2 * rho) + 0.08, rf"${1 / (2 * rho):.2f}$", color=col, ha="left",
                 fontsize=11.5)
axa.set_xlim(-0.75, 0.75)
axa.set_ylim(0, 6.3)
axa.set_xlabel(r"arc length $s$ across the edge")
axa.set_ylabel(r"$|H| = |\Delta f|/2$")
axa.set_title(r"(a) same area $\int |H|\,ds = \pi/4$, different heights", fontsize=13.5)
axa.legend(loc="upper right", frameon=False)
ins = axa.inset_axes([0.03, 0.34, 0.31, 0.5])
for rho, col in zip(rhos, cols):
    P, s = curve(rho, L=0.6)
    ins.plot(P[np.argsort(s), 0], P[np.argsort(s), 1], color=col, lw=1.6)
ins.fill_between([-1.0, 0.0], -1.0, 0.0, color=C["surface"], alpha=0.5, lw=0)
ins.set_xlim(-0.8, 0.15)
ins.set_ylim(-0.8, 0.15)
ins.set_aspect("equal")
ins.set_xticks([])
ins.set_yticks([])
ins.set_title("cross-sections", fontsize=10.5, pad=2)

# (b) integrals vs rho
rr = np.geomspace(0.02, 1.0, 100)
rd = np.geomspace(0.02, 1.0, 9)
L1n, L2n = [], []
for rho in rd:
    P, s = curve(rho, n_arc=4001)
    hv = absH(P, rho)
    assert np.max(hv[np.abs(s) > np.pi * rho / 4 + 1e-3]) < 1e-3            # flat faces contribute nothing
    arc = np.abs(s) <= np.pi * rho / 4 + 1e-12                               # integrate over the fillet
    order = np.argsort(s[arc])
    L1n.append(np.trapezoid(hv[arc][order], s[arc][order]))
    L2n.append(np.trapezoid(hv[arc][order] ** 2, s[arc][order]))
L1n, L2n = np.array(L1n), np.array(L2n)
assert np.allclose(L1n, np.pi / 4, rtol=2e-3), L1n
assert np.allclose(L2n, np.pi / (8 * rd), rtol=2e-3)
axb.loglog(rr, np.pi / 4 + 0 * rr, color=C["main"], lw=2.2, label=r"$\int |H|\,dA = \pi/4$ (L1)")
axb.loglog(rr, np.pi / (8 * rr), color=C["normal"], lw=2.2, label=r"$\int H^2\,dA = \pi/(8 r_{\mathrm{f}})$ (L2)")
axb.loglog(rd, L1n, "o", color=C["main"], mfc="white", ms=6)
axb.loglog(rd, L2n, "o", color=C["normal"], mfc="white", ms=6)
axb.set_xlabel(r"fillet radius $r_{\mathrm{f}}$ (sharper $\leftarrow$)")
axb.set_ylabel("per unit edge length")
axb.set_title(r"(b) L1 is blind to sharpness, L2 is not", fontsize=13.5)
axb.legend(loc="upper right", frameon=False)
axb.set_ylim(0.2, 40)

dgfig.save(fig, __file__)
