"""Figure 27.3.2: the four decompositions of an essential matrix and cheirality (Section 27.3).

True relative pose (camera-1 frame = world): R = exp([(0.05, -0.3, 0.02)]_x), T = 2 (-1, 0.1, 0.2)/|.|, so the baseline has
length 2. Scene point X = (1.2, 0.2, 5), observed at pbar1 = X/X_z and pbar2 = (R X + T)/(.)_z.
From E = [T]_x R the SVD gives four candidates (R_a, T), (R_a, -T), (R_b, T), (R_b, -T) (|T| = 2 fixed) with R_a = U W V^T,
R_b = U W^T V^T (W = R_z(-90 deg), as in COLMAP). For each candidate we draw, in the top view (x, z):
camera 1 at the origin with its viewing direction (+z), camera 2's center c2 = -R_i^T T_i with its viewing direction
R_i^T e_z, both rays (solid in front of the camera, dashed behind) and the triangulated point (s1 pbar1).
Self-checks: each candidate reproduces E up to sign; exactly one gives positive depths in both cameras and it equals the
true pose; the twisted pair differs by the rotation R_T(pi) about the baseline.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-3-2-four-decompositions.py``
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


R = expso3(np.array([0.05, -0.3, 0.02]))
T = np.array([-1.0, 0.1, 0.2])
T *= 2.0 / np.linalg.norm(T)
E = hat(T) @ R
X = np.array([1.2, 0.2, 5.0])
p1 = X / X[2]
q = R @ X + T
p2 = q / q[2]
U, s, Vt = np.linalg.svd(E)
if np.linalg.det(U) < 0:
    U = -U
if np.linalg.det(Vt) < 0:
    Vt = -Vt
W = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
Ra, Rb, t = U @ W @ Vt, U @ W.T @ Vt, 2.0 * U[:, 2]
cands = [(Ra, t), (Ra, -t), (Rb, t), (Rb, -t)]
results = []
for Rm, Tm in cands:
    assert min(np.abs(hat(Tm) @ Rm - E).max(), np.abs(hat(Tm) @ Rm + E).max()) < 1e-12
    A = np.column_stack([Rm @ p1, -p2])
    (s1, s2), *_ = np.linalg.lstsq(A, -Tm, rcond=None)
    results.append((s1, s2))
good = [i for i, (a, b) in enumerate(results) if a > 0 and b > 0]
assert len(good) == 1
gi = good[0]
assert np.allclose(cands[gi][0], R, atol=1e-10) and np.allclose(cands[gi][1], T, atol=1e-10)
Tu = T / np.linalg.norm(T)
RT = 2 * np.outer(Tu, Tu) - np.eye(3)
other_R = Rb if np.allclose(Ra, R, atol=1e-10) else Ra
assert np.allclose(other_R, RT @ R, atol=1e-10)

# order panels: true (R,T), (R,-T), (R',T), (R',-T)
Rtrue = cands[gi][0]
order = [(Rtrue, T), (Rtrue, -T), (other_R, T), (other_R, -T)]
names = [r"(a) $(R,\ T)$", r"(b) $(R,\ -T)$", r"(c) $(R',\ T)$, twisted", r"(d) $(R',\ -T)$, twisted"]
fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
for axx, (Rm, Tm), nm in zip(axes.ravel(), order, names):
    A = np.column_stack([Rm @ p1, -p2])
    (s1, s2), *_ = np.linalg.lstsq(A, -Tm, rcond=None)
    c2 = -Rm.T @ Tm
    d2 = Rm.T @ p2                      # direction of ray 2 in frame 1
    Xi = s1 * p1
    for cc, dd, ss in ((np.zeros(3), p1, s1), (c2, d2, s2)):
        span = np.linspace(0, 9, 2)
        fw = np.array([cc + u * dd / np.linalg.norm(dd) * 1.0 for u in span])
        bw = np.array([cc - u * dd / np.linalg.norm(dd) * 1.0 for u in span])
        axx.plot(fw[:, 0], fw[:, 2], color=C["accent"], lw=1.4)
        axx.plot(bw[:, 0], bw[:, 2], color=C["accent"], lw=1.0, ls="--")
    for cc, Rc, lab in ((np.zeros(3), np.eye(3), r"$C_1$"), (c2, Rm, r"$C_2$")):
        vd = Rc.T @ np.array([0, 0, 1.0])
        axx.annotate("", xy=(cc[0] + 1.3 * vd[0], cc[2] + 1.3 * vd[2]), xytext=(cc[0], cc[2]),
                     arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.6, mutation_scale=13))
        axx.plot(cc[0], cc[2], "ks", ms=5)
        axx.text(cc[0] + (-0.75 if lab == r"$C_1$" else 0.25), cc[2] - 0.75, lab, fontsize=12)
    ok = s1 > 0 and s2 > 0
    axx.plot(Xi[0], Xi[2], "o", color=C["third"] if ok else C["normal"], ms=9, zorder=6)
    sg = lambda v: "+" if v > 0 else "-"
    dy = 0.35 if Xi[2] > 0 else -0.95
    axx.text(np.clip(Xi[0], -2.2, 2.2), Xi[2] + dy, rf"depths $({sg(s1)},\,{sg(s2)})$", fontsize=11.5, ha="center",
             color=C["third"] if ok else C["normal"], bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    axx.set_title(nm + ("  [chosen]" if ok else ""), fontsize=12.5)
    axx.set_xlim(-4.0, 4.0)
    axx.set_ylim(-6.5, 6.5)
    axx.set_aspect("equal")
    axx.axhline(0, color=C["aux"], lw=0.5, ls=":")
    axx.set_xlabel(r"$x$")
    axx.set_ylabel(r"$z$")
    axx.tick_params(labelsize=10)
fig.tight_layout()
dgfig.save(fig, __file__)
print([(np.round(a, 2), np.round(b, 2)) for a, b in results])
