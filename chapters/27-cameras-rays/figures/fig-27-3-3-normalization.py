"""Figure 27.3.3: Hartley's normalization makes the 8-point algorithm better conditioned (Section 27.3).

Two cameras K = [[1500, 0, 960], [0, 1500, 540], [0, 0, 1]] (a 1920 x 1080 image), relative pose
R = exp([(0.02, -0.25, 0.03)]_x), T = (-1, 0.05, 0.1). 50 random points in [-3, 3] x [-2, 2] x [5, 12], Gaussian pixel
noise of standard deviation sigma in both images, 200 trials per sigma (seed 0). The 8-point algorithm (SVD of the 50 x 9
design matrix, then rank 2 by zeroing the smallest singular value) is run on raw pixel coordinates and on Hartley-normalized
coordinates (centroid at 0, mean distance sqrt(2)).
Plotted: the median over trials of the median symmetric epipolar distance (px) of the noise-free correspondences.
The same experiment (same code and seed) is in verify/v-27-3-essential-manifold.py.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-3-3-normalization.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expso3(w):
    th = np.linalg.norm(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def design(P1, P2):
    return np.array([np.kron(b, a) for a, b in zip(P1, P2)])


def eight_point(x1, x2, normalize):
    def nmat(x):
        c = x[:, :2].mean(0)
        d = np.mean(np.linalg.norm(x[:, :2] - c, axis=1))
        sc = np.sqrt(2) / d
        return np.array([[sc, 0, -sc * c[0]], [0, sc, -sc * c[1]], [0, 0, 1]])

    if normalize:
        N1, N2 = nmat(x1), nmat(x2)
    else:
        N1 = N2 = np.eye(3)
    y1, y2 = (N1 @ x1.T).T, (N2 @ x2.T).T
    A = design(y1, y2)
    _, sA, Vt_ = np.linalg.svd(A)
    Fh = Vt_[-1].reshape(3, 3)
    U_, s_, V_ = np.linalg.svd(Fh)
    Fh = U_ @ np.diag([s_[0], s_[1], 0]) @ V_
    Fh = N2.T @ Fh @ N1
    return Fh / np.linalg.norm(Fh), sA[0] / sA[-2]


def epi_err(Fh, x1, x2):
    l2_ = (Fh @ x1.T).T
    l1_ = (Fh.T @ x2.T).T
    r = np.abs(np.sum(x2 * l2_, axis=1))
    return 0.5 * (r / np.hypot(l2_[:, 0], l2_[:, 1]) + r / np.hypot(l1_[:, 0], l1_[:, 1]))


def experiment(seed=0, sigmas=(0.1, 0.3, 1.0, 3.0), trials=200, npts=50):
    rr = np.random.default_rng(seed)
    Kc = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    Rc = expso3(np.array([0.02, -0.25, 0.03]))
    Tc = np.array([-1.0, 0.05, 0.1])
    res = {"raw": [], "norm": []}
    cond = {"raw": [], "norm": []}
    for sg in sigmas:
        er = {"raw": [], "norm": []}
        for _ in range(trials):
            X = np.column_stack([rr.uniform(-3, 3, npts), rr.uniform(-2, 2, npts), rr.uniform(5, 12, npts)])
            x1 = (Kc @ X.T).T
            x1 /= x1[:, 2:]
            q = (Rc @ X.T).T + Tc
            x2 = (Kc @ q.T).T
            x2 /= x2[:, 2:]
            n1 = x1.copy()
            n2 = x2.copy()
            n1[:, :2] += rr.normal(scale=sg, size=(npts, 2))
            n2[:, :2] += rr.normal(scale=sg, size=(npts, 2))
            for key, nz in (("raw", False), ("norm", True)):
                Fh, cn = eight_point(n1, n2, nz)
                er[key].append(np.median(epi_err(Fh, x1, x2)))
                cond[key].append(cn)
        for key in er:
            res[key].append(np.median(er[key]))
    return np.array(sigmas), {k: np.array(v) for k, v in res.items()}, {k: np.median(v) for k, v in cond.items()}


sig, res, cond = experiment()
assert np.all(res["norm"] < res["raw"])
assert abs(res["raw"][2] - 0.749) < 5e-3 and abs(res["norm"][2] - 0.343) < 5e-3
fig, ax = plt.subplots(figsize=(6.2, 4.3))
ax.loglog(sig, res["raw"], "o-", color=C["normal"], lw=1.8, ms=6,
          label=rf"raw pixels ($\sigma_1/\sigma_8 \approx {cond['raw']:.0e}$)".replace("e+0", r"\times 10^{").replace("$)", "}$)"))
ax.loglog(sig, res["norm"], "s-", color=C["tangent"], lw=1.8, ms=6,
          label=rf"normalized ($\sigma_1/\sigma_8 \approx {cond['norm']:.0f}$)")
ax.loglog(sig, sig, color=C["aux"], lw=0.8, ls=":")
ax.text(1.25, 1.75, r"error $=\sigma$", fontsize=10, color=C["aux"], rotation=33, ha="center", va="bottom")
for x, yr, yn in zip(sig, res["raw"], res["norm"]):
    ax.text(x * 0.93, yr * 1.22, f"{yr:.2f}", fontsize=10.5, color=C["normal"], ha="right", va="bottom")
    ax.text(x * 1.08, yn * 0.82, f"{yn:.2f}", fontsize=10.5, color=C["tangent"], ha="left", va="top")
    ax.annotate("", xy=(x, yn * 1.08), xytext=(x, yr * 0.92), arrowprops=dict(arrowstyle="<->", color=C["aux"], lw=0.7))
ax.set_xlabel(r"pixel noise $\sigma$ (px)")
ax.set_ylabel("median epipolar distance (px)")
ax.set_xlim(0.07, 4.5)
ax.set_xticks([0.1, 0.3, 1, 3])
ax.set_xticklabels(["0.1", "0.3", "1", "3"])
ax.set_ylim(0.02, 6)
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.text(2.2, 0.035, r"gap $\approx\times 2$ at every $\sigma$", fontsize=10.5, color=C["aux"], ha="center")
ax.set_title("8-point algorithm: raw vs normalized coordinates", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)
print(res, cond)
