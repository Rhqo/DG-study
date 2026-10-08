"""Figure 30.3.4: normal consistency = "every splat is tangent to the surface rendered from all splats" (Section 30.3).

Flatland 2DGS renderer (x horizontal, depth z vertical, camera at the origin below the plotted window, image coordinate
xi = x/z, 101 pixels in xi in [-0.25, 0.25]). True surface: the line z = 2 + 0.8 x (gray dashed).
16 splats with centers on that line, scale s = 0.06, opacity 0.95, Gaussian falloff exp(-u^2/2):
(a) "shingles": every splat is parallel to the image line (normal (0, -1), facing the camera);
(b) every splat is tangent to the line.
Per pixel: ray-splat intersections sorted by depth, alpha blending w_i = a_i prod_{j<i}(1 - a_j), expected depth
sum w_i z_i / sum w_i (the default depth_ratio = 0 of the official code); back-projected points p(xi) = z (xi, 1);
depth normal N from central differences of p (rotated by 90 deg and oriented toward the camera);
loss L_n = sum_i w_i (1 - <n_i, N>) with n_i the splat normal oriented toward the camera ([Huang24] eq. (14)).
Drawn: splats (light blue segments, +-1.5 s), splat normals n_i (vermillion), rendered depth curve (black), depth normals
N at every 10th pixel (blue), true surface (gray dashed). Mean L_n over pixels is printed in each panel.
Self-checks: tangent splats give L_n < 1e-6; shingles give mean L_n ~ 0.2 (1 - cos 38.7 deg = 0.219 times coverage).

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-3-4-normal-consistency.py``
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
SLOPE, S, ALPHA = 0.8, 0.06, 0.95
xi = np.linspace(-0.25, 0.25, 101)
centers = np.stack([np.linspace(-0.75, 0.75, 16), 2 + SLOPE * np.linspace(-0.75, 0.75, 16)], 1)
tan = np.array([1, SLOPE]) / np.hypot(1, SLOPE)
nrm = np.array([-SLOPE, 1]) / np.hypot(1, SLOPE)


def render(splats):
    ed = np.full(len(xi), np.nan)
    per_pixel = []
    for k, x in enumerate(xi):
        hits = []
        for (cc, t, n) in splats:
            den = S * (t[0] - t[1] * x)
            if abs(den) < 1e-12:
                continue
            uu = -(cc[0] - cc[1] * x) / den
            p = cc + S * uu * t
            a = ALPHA * np.exp(-0.5 * uu * uu)
            if p[1] <= 0 or a < 1 / 255:
                continue
            nn = n if n @ (-p) > 0 else -n                  # rasterizer: normal faces the camera
            hits.append((p[1], a, nn))
        hits.sort(key=lambda h: h[0])
        T, A, zsum, wl = 1.0, 0.0, 0.0, []
        for z, a, nn in hits:
            w = a * T
            zsum += w * z
            A += w
            wl.append((w, nn))
            T *= 1 - a
        ed[k] = zsum / A
        per_pixel.append(wl)
    return ed, per_pixel


def depth_normals(z):
    p = np.stack([xi * z, z], 1)
    d = p[2:] - p[:-2]
    N = np.stack([d[:, 1], -d[:, 0]], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    flip = np.einsum("ij,ij->i", N, -p[1:-1]) < 0
    N[flip] *= -1
    return p, N


configs = {"shingles": [(c0, np.array([1.0, 0.0]), np.array([0.0, 1.0])) for c0 in centers],
           "tangent": [(c0, tan, nrm) for c0 in centers]}
results = {}
for name, sp_ in configs.items():
    z, wl = render(sp_)
    p, N = depth_normals(z)
    L = np.array([sum(w * (1 - nv @ N[k - 1]) for w, nv in wl[k]) for k in range(1, len(xi) - 1)])
    results[name] = (p, N, L)
    print(f"{name}: mean L_n = {L.mean():.4f}, max {L.max():.4f}")
assert results["tangent"][2].max() < 1e-6
assert 0.15 < results["shingles"][2].mean() < 0.219

fig, axs = plt.subplots(2, 1, figsize=(5.8, 8.0), sharex=True, gridspec_kw=dict(hspace=0.22))
titles = {"shingles": "(a) splats parallel to the image (\"shingles\")", "tangent": "(b) splats tangent to the surface"}
xl = np.array([-0.62, 0.62])
for ax, name in zip(axs, ("shingles", "tangent")):
    p, N, L = results[name]
    ax.plot(xl, 2 + SLOPE * xl, color=C["aux"], lw=1.0, ls=(0, (5, 3)), zorder=1)
    for (c0, t, n) in configs[name]:
        if abs(c0[0]) > 0.62:
            continue
        seg = np.stack([c0 - 1.5 * S * t, c0 + 1.5 * S * t])
        ax.plot(*seg.T, color=C["region"], lw=5, solid_capstyle="round", zorder=2)
        nn = n if n @ (-c0) > 0 else -n
        ax.annotate("", xy=c0 + 0.11 * nn, xytext=c0,
                    arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.4, mutation_scale=10), zorder=4)
    ax.plot(p[:, 0], p[:, 1], color=C["main"], lw=1.8, zorder=3)
    for k in range(5, len(p), 10):
        ax.annotate("", xy=p[k] + 0.11 * N[k], xytext=p[k],
                    arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4, mutation_scale=10), zorder=5)
    ax.text(0.03, 0.93, rf"mean $\mathcal{{L}}_n = {L.mean():.3f}$", transform=ax.transAxes, fontsize=12.5, va="top",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_aspect("equal")
    ax.set_ylim(1.35, 2.62)
    ax.set_ylabel(r"depth $z$")
    ax.set_title(titles[name], fontsize=13)
axs[1].set_xlim(*xl)
axs[1].set_xlabel(r"$x$ (camera at the origin, below)")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=C["region"], lw=5, label="splat"),
           Line2D([], [], color=C["normal"], lw=1.6, label=r"splat normal $n_i$"),
           Line2D([], [], color=C["main"], lw=1.8, label="rendered depth"),
           Line2D([], [], color=C["tangent"], lw=1.6, label=r"depth normal $\mathbf{N}$"),
           Line2D([], [], color=C["aux"], lw=1.0, ls=(0, (5, 3)), label="true surface")]
axs[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=11)
dgfig.save(fig, __file__)
