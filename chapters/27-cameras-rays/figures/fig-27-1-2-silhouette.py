"""Figure 27.1.2: the projection of a surface loses rank exactly on the occluding contour (Section 27.1).

Sphere of radius 1 centered at (0, 0, 3) in camera coordinates, normalized camera (f = 1).
(a) Cross-section t_y = 0. The meridian point at angle psi (psi = 0: point nearest the camera) is
    t(psi) = (sin psi, 0, 3 - cos psi). Visible arc solid, hidden arc dashed. Orange: the two rays tangent to the
    sphere (contour points at psi = +-70.53 deg). Blue segments: the tangent line of the meridian at
    psi = 25, 50, 70.53 deg; gray: the ray through each point, extended a little beyond it. The number next to each
    point is the angle between the ray and the tangent line, asin |<N, t_hat>| = 53.6, 22.0, 0 deg.
(b) The two singular values of d(phi|_S) at t(psi): sigma_out = 1/t_z (direction out of the plane) and
    sigma_in = |t| |<N, t/|t|>| / t_z^2 (direction along the meridian), as functions of psi.
    The dots on the sigma_in curve mark psi = 25 and 50 deg (sigma_in = 0.392, 0.167), the same points as in (a).
Self-checks: the formulas equal the SVD of J restricted to T_pS; sigma_in = 0 exactly at psi = acos(1/3) = 70.53 deg;
the ray-tangent angles and the marked sigma_in values.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-1-2-silhouette.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
Z0, RHO = 3.0, 1.0
psi_c = np.arccos(RHO / Z0)


def point(psi):
    n = np.array([np.sin(psi), 0.0, -np.cos(psi)])
    return np.array([0, 0, Z0]) + RHO * n, n


def J(t):
    return np.array([[1 / t[2], 0, -t[0] / t[2] ** 2], [0, 1 / t[2], -t[1] / t[2] ** 2]])


def sing(psi):
    t, n = point(psi)
    e_th = np.array([np.cos(psi), 0, np.sin(psi)])
    sv = np.linalg.svd(J(t) @ np.column_stack([e_th, [0, 1.0, 0]]), compute_uv=False)
    s_out = 1 / t[2]
    s_in = np.linalg.norm(t) * abs(n @ t / np.linalg.norm(t)) / t[2] ** 2
    assert np.allclose(np.sort(sv), np.sort([s_out, s_in]), atol=1e-12)
    return s_out, s_in


tc, nc = point(psi_c)
assert abs(nc @ tc) < 1e-12 and sing(psi_c)[1] < 1e-12

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw=dict(width_ratios=[1.05, 1.2]))
# ---------------------------------------------------------------- (a)
ps = np.linspace(-np.pi, np.pi, 721)
pts = np.array([point(p)[0] for p in ps])
vis = np.abs(ps) <= psi_c
ax.plot(pts[vis, 0], pts[vis, 2], color="k", lw=1.8)
for side in (ps > psi_c, ps < -psi_c):
    ax.plot(pts[side, 0], pts[side, 2], color=C["aux"], lw=1.2, ls="--")
for sgn in (1, -1):
    tcs = np.array([sgn * tc[0], 0, tc[2]])
    ax.plot([0, 1.25 * tcs[0]], [0, 1.25 * tcs[2]], color=C["accent"], lw=1.8)
    ax.plot(tcs[0], tcs[2], "o", color=C["normal"], ms=5, zorder=6)
def ray_tangent_angle(psi):
    t, n = point(psi)
    return np.degrees(np.arcsin(abs(n @ t) / np.linalg.norm(t)))


assert abs(ray_tangent_angle(np.radians(25)) - 53.6) < 0.05 and abs(ray_tangent_angle(np.radians(50)) - 22.0) < 0.05
assert abs(sing(np.radians(25))[1] - 0.392) < 5e-4 and abs(sing(np.radians(50))[1] - 0.167) < 5e-4
for deg, lab_xy in ((25, (0.03, 2.27)), (50, (0.22, 2.57))):
    t, n = point(np.radians(deg))
    ax.plot([0, 1.18 * t[0]], [0, 1.18 * t[2]], color=C["aux"], lw=0.9)
    e = np.array([np.cos(np.radians(deg)), 0, np.sin(np.radians(deg))])
    seg = np.array([t - 0.38 * e, t + 0.38 * e])
    ax.plot(seg[:, 0], seg[:, 2], color=C["tangent"], lw=2.0, zorder=5)
    ax.plot(t[0], t[2], "ko", ms=4, zorder=6)
    ax.text(*lab_xy, rf"$\psi={deg}^\circ$", fontsize=10)
e = np.array([np.cos(psi_c), 0, np.sin(psi_c)])
seg = np.array([tc - 0.45 * e, tc + 0.45 * e])
ax.plot(seg[:, 0], seg[:, 2], color=C["tangent"], lw=2.0, zorder=5)
ax.annotate("contour:\nray lies in $T_pS$", xy=(tc[0], tc[2]), xytext=(1.2, 1.55), fontsize=11, color=C["normal"],
            arrowprops=dict(arrowstyle="->", color=C["normal"], lw=0.9))
ax.text(-0.42, 3.45, "hidden", fontsize=11, color=C["aux"])
ax.text(-0.62, 1.62, "visible", fontsize=11)
rows = ["angle between ray", "and tangent line:"] + [
    rf"$\psi={deg}^\circ$:  ${ray_tangent_angle(np.radians(deg)):.1f}^\circ$" for deg in (25, 50)] + [
    r"contour:  $0^\circ$"]
ax.text(0.40, 0.06, "\n".join(rows), fontsize=9.5, va="bottom", linespacing=1.3,
        bbox=dict(facecolor="white", edgecolor=C["aux"], lw=0.6, pad=3))
ax.plot(0, 0, "ko", ms=4)
ax.text(0.1, -0.08, r"$O$", fontsize=12)
ax.set_xlim(-1.6, 2.3)
ax.set_ylim(-0.25, 4.2)
ax.set_aspect("equal")
ax.set_xlabel(r"$t_x$")
ax.set_ylabel(r"$t_z$ (depth)")
ax.set_title("(a) sphere seen from $O$ (section $t_y=0$)", fontsize=12)

# ---------------------------------------------------------------- (b)
pp = np.linspace(0, np.pi, 721)
so, si = np.array([sing(p) for p in pp]).T
bx.plot(np.degrees(pp), so, color=C["tangent"], lw=1.8, label=r"$\sigma_{\rm out}=1/t_z$")
bx.plot(np.degrees(pp), si, color=C["normal"], lw=1.8, label=r"$\sigma_{\rm in}=|t|\,|\langle \mathbf{N},\hat t\rangle|/t_z^2$")
for deg in (25, 50):
    sv_in = sing(np.radians(deg))[1]
    bx.plot(deg, sv_in, "ko", ms=4, zorder=6)
    bx.text(deg + 2.5, sv_in + 0.012, rf"$\psi={deg}^\circ$", fontsize=10)
bx.axvline(np.degrees(psi_c), color=C["accent"], lw=1.2, ls="--")
bx.axvspan(np.degrees(psi_c), 180, color=C["surface"], alpha=0.4, lw=0)
bx.text(np.degrees(psi_c) - 2, 0.495, r"contour $\psi_c=70.5^\circ$", fontsize=11, color=C["accent"], ha="right")
bx.text(np.degrees(psi_c) + 4, 0.40, "back\n(hidden)", fontsize=11, color=C["aux"])
bx.annotate(r"rank 1", xy=(np.degrees(psi_c), 0.0), xytext=(84, 0.12), fontsize=11, color=C["normal"],
            arrowprops=dict(arrowstyle="->", color=C["normal"], lw=0.9))
bx.set_xlim(0, 180)
bx.set_ylim(0, 0.55)
bx.set_xlabel(r"position on the meridian $\psi$ (deg)")
bx.set_ylabel(r"singular values of $d(\varphi|_S)$")
bx.legend(loc="upper right", fontsize=10, frameon=False, bbox_to_anchor=(1.0, 1.0))
bx.set_title(r"(b) $\sigma_{\rm in}\to 0$ at the contour", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)
