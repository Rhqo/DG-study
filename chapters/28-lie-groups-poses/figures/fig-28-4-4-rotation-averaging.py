"""Figure 28.4.4: chordal L2, geodesic L2 (Karcher) and geodesic L1 means of rotations with outliers (Section 28.4).

Data (seed = trial index): a random true rotation R*; 20 inliers R* Exp(e_i), e_i ~ N(0, (10 deg)^2 I_3); k outliers
R* Exp(theta n), n uniform on S^2, theta uniform in [90, 170] deg.
Means (lib/dglie): chordal L2 = SVD projection of sum R_i (= the quaternion eigenvector method); geodesic L2 = Karcher
mean by the iteration of Tour 0.8 started at the chordal mean; geodesic L1 = Weiszfeld iteration in the tangent space
[Hartley13] started at the Karcher mean.
(a) one data set with k = 6 (trial 0), in degrees, projected to the "pull plane" spanned by the Karcher offset
    Log(R*^T R_Karcher) and the part of the chordal offset orthogonal to it (so these two errors are drawn to scale):
    gray dots = inliers (projected), red arrows on the outer ring = directions towards the 6 outliers (which are
    90-170 degrees away, outside the window; drawn on a 23.5-degree ring), dashed red arrow = the outliers' share (1/n) sum_out Log(R*^T R_i) of a
    Karcher step from R*, black + = R*, markers = the three means, each labelled with its true error angle.
(b) the error angle d(R*, mean) in degrees, averaged over 30 trials, as a function of k (0..8); gray dotted = the
    Karcher mean of the inliers only.
Self-check: the Karcher offset points along the outliers' net pull (within 15 deg); chordal SVD mean = quaternion eigenvector mean (1e-12); all means satisfy their optimality conditions
(sum Log = 0 for Karcher, Weiszfeld fixed point for L1); with k = 0 the three means agree within 1.5 deg on average.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-4-4-rotation-averaging.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dglie as L  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
NIN, TRIALS, KS = 20, 30, list(range(9))


def make_data(k, seed):
    r = np.random.default_rng(seed)
    Rs = L.random_rotation(r)
    ins = [Rs @ L.expSO3(np.radians(10) * r.normal(size=3)) for _ in range(NIN)]
    outs = []
    for _ in range(k):
        n = r.normal(size=3)
        n /= np.linalg.norm(n)
        outs.append(Rs @ L.expSO3(np.radians(r.uniform(90, 170)) * n))
    return Rs, ins, outs


def means(Rlist):
    Rc = L.mean_chordal(Rlist)
    Rk, _ = L.mean_geodesic_L2(Rlist, R0=Rc)
    R1, _ = L.mean_geodesic_L1(Rlist, R0=Rk)
    return Rc, Rk, R1


err = np.zeros((len(KS), TRIALS, 3))
base = np.zeros((len(KS), TRIALS))
for a, k in enumerate(KS):
    for t in range(TRIALS):
        Rs, ins, outs = make_data(k, t)
        Rl = ins + outs
        Ms = means(Rl)
        err[a, t] = [np.degrees(L.rot_angle(Rs.T @ M)) for M in Ms]
        base[a, t] = np.degrees(L.rot_angle(Rs.T @ L.mean_geodesic_L2(ins)[0]))
        if t == 0:
            assert np.allclose(Ms[0], L.mean_quat_eig(Rl), atol=1e-12)
            assert np.linalg.norm(np.sum([L.logSO3(Ms[1].T @ R) for R in Rl], axis=0)) < 1e-9
assert np.mean(np.abs(err[0, :, 0] - err[0, :, 2])) < 1.5 and np.mean(np.abs(err[0, :, 1] - err[0, :, 2])) < 1.5
me = err.mean(axis=1)
print("mean error (deg), rows k = 0..8, columns chordal, Karcher, L1:")
print(np.round(me, 2))
print("inlier-only Karcher:", np.round(base.mean(axis=1)[0], 2))

Rs, ins, outs = make_data(6, 0)
Ms = means(ins + outs)
Wi = np.degrees(np.array([L.logSO3(Rs.T @ R) for R in ins]))
Wo = np.degrees(np.array([L.logSO3(Rs.T @ R) for R in outs]))
Wm = [np.degrees(L.logSO3(Rs.T @ M)) for M in Ms]          # chordal, Karcher, L1 (deg)
errs_a = [float(np.linalg.norm(w)) for w in Wm]
# the "pull plane": e1 along the Karcher offset, e2 along the part of the chordal offset orthogonal to it.
# Both offsets lie in this plane, so their distances to the truth are drawn without distortion.
e1 = Wm[1] / np.linalg.norm(Wm[1])
e2 = Wm[0] - (Wm[0] @ e1) * e1
e2 /= np.linalg.norm(e2)
P = np.vstack([e1, e2])
pull = Wo.sum(axis=0) / (len(ins) + len(outs))             # outliers' share of (1/n) sum Log(R*^T R_i)
assert abs(np.linalg.norm(P @ Wm[1]) - errs_a[1]) < 1e-9 and abs(np.linalg.norm(P @ Wm[0]) - errs_a[0]) < 1e-9
assert np.degrees(np.arccos(pull @ e1 / np.linalg.norm(pull))) < 15     # the Karcher mean moves along the net pull
assert errs_a[1] > 3 * errs_a[0] and errs_a[2] < errs_a[0]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 4.3), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.05]))
RAD = 23.5
th = np.linspace(0, 2 * np.pi, 300)
ax.plot(10 * np.cos(th), 10 * np.sin(th), color="#CCCCCC", lw=0.7, ls=":")
ax.text(10 * np.cos(2.3), 10 * np.sin(2.3) + 0.6, r"$10^\circ$", fontsize=9, color=C["aux"], ha="center")
pi_ = Wi @ P.T
ax.plot(pi_[:, 0], pi_[:, 1], "o", color="#9A9A9A", ms=3.5, label="inliers (projected)", zorder=2)
for k_, w in enumerate(Wo):
    u = P @ w
    u = u / np.linalg.norm(u)
    ax.annotate("", xy=(RAD + 1.6) * u, xytext=(RAD - 1.8) * u,
                arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.4, mutation_scale=11), zorder=3)
ax.plot([], [], color=C["normal"], lw=1.4, label=r"towards an outlier ($90^\circ$-$170^\circ$ away)")
pp = P @ pull
ax.annotate("", xy=pp, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.2, ls=(0, (3, 2)),
                                                       mutation_scale=10), zorder=4)
ax.text(pp[0] * 0.5, pp[1] * 0.5 - 5.2, "net pull of\nthe outliers", fontsize=9.5, color=C["normal"], ha="center")
mk = [("s", C["accent"], "chordal $L_2$"), ("D", C["tangent"], "geodesic $L_2$ (Karcher)"),
      ("*", C["third"], "geodesic $L_1$")]
offs = [(-3.4, 1.4), (1.2, 2.6), (1.4, 3.4)]
for w, (m, col, lab), er, (dx, dy) in zip(Wm, mk, errs_a, offs):
    q = P @ w
    ax.plot(q[0], q[1], m, color=col, ms=9 if m != "*" else 13, mec="black", mew=0.6, label=lab, zorder=6)
    ax.text(q[0] + dx, q[1] + dy, r"$%.1f^\circ$" % er, fontsize=10.5, color=col if m != "s" else "#B07400",
            ha="center", zorder=7)
ax.plot(0, 0, "+", color=C["main"], ms=13, mew=1.8, label=r"truth $R^*$", zorder=7)
ax.set_aspect("equal")
ax.set_xlim(-RAD - 2.5, RAD + 2.5)
ax.set_ylim(-RAD - 2.5, RAD + 2.5)
ax.set_xlabel("deg, along the Karcher offset")
ax.set_ylabel("deg, perpendicular")
ax.set_title(r"(a) one data set ($k = 6$), pull plane", fontsize=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.8, frameon=False)

for c_, (m, col, lab) in enumerate(mk):
    bx.plot(KS, me[:, c_], marker=m, color=col, lw=1.9, ms=6 if m != "*" else 10, label=lab)
bx.plot(KS, base.mean(axis=1), color=C["aux"], lw=1.2, ls=":", label="Karcher of the inliers only")
bx.set_xlabel("number of outliers $k$ (with 20 inliers)")
bx.set_ylabel(r"error $d(R^*, \bar R)$ (deg), mean of 30")
bx.set_title("(b) error vs. outliers", fontsize=12)
bx.set_ylim(0, 19)
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9, frameon=False)
bx.grid(True, color="#E5E5E5", lw=0.5)

dgfig.save(fig, __file__)
