"""Figure 29.5.3: gradient descent through q/|q| pushes |q| outward and shrinks the effective step (Section 29.5).

(a) 2D analogue (q in R^2, q/|q| on the unit circle). From q_0 = (1, 0) two gradient steps of the same Euclidean
    length 0.6, each tangent to the circle through the current q (because grad_q L is orthogonal to q).
    |q_1| = sqrt(1 + 0.36) = 1.166, |q_2| = sqrt(1.36 + 0.36) = 1.311; the angles moved are atan(0.6) = 31.0 deg and
    atan(0.6/1.166) = 27.2 deg. Gray dashed rays show the normalized points q_k/|q_k| on the circle.
(b) |q_k| over 400 iterations for the toy loss L = |Sigma(q) - Sigma*|_F^2 / 2 (s = (1, 0.6, 0.3), random target,
    same as the verify script) starting from q = (1, 0, 0, 0): plain gradient descent with step 0.05, 0.2, 0.5
    (monotone increasing) and Adam with learning rate 0.01, 0.05 (not monotone).
Self-checks: |q_k|^2 = |q_{k-1}|^2 + |delta|^2 in (a); monotonicity of GD and the Adam values in (b).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-5-3-quaternion-norm.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw=dict(width_ratios=[1, 1.2], wspace=0.42))

# (a)
t = np.linspace(0, 2 * np.pi, 300)
axa.plot(np.cos(t), np.sin(t), color=C["main"], lw=1.4)
q = np.array([1.0, 0.0])
pts = [q]
for k in range(2):
    perp = np.array([-q[1], q[0]]) / np.linalg.norm(q)
    qn = q + 0.6 * perp
    assert abs(qn @ qn - (q @ q + 0.36)) < 1e-12
    axa.annotate("", xy=qn, xytext=q, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.6, mutation_scale=12,
                                                       shrinkA=0, shrinkB=0))
    q = qn
    pts.append(q)
norms = [np.linalg.norm(p) for p in pts]
assert np.allclose(norms, [1, np.sqrt(1.36), np.sqrt(1.72)])
angles = [np.degrees(np.arctan2(p[1], p[0])) for p in pts]
assert abs(angles[1] - 30.96) < 0.01 and abs(angles[2] - angles[1] - 27.23) < 0.01
for p in pts:
    u = p / np.linalg.norm(p)
    axa.plot([0, p[0]], [0, p[1]], color=C["aux"], lw=0.8, ls=(0, (3, 3)))
    axa.plot(*u, "o", color=C["accent"], ms=5, zorder=5)
    axa.plot(*p, "o", color=C["main"], ms=4, zorder=6)
axa.text(1.06, -0.14, r"$q_0$", fontsize=11)
axa.text(pts[1][0] + 0.08, pts[1][1] - 0.06, r"$q_1$" + "\n" + r"$|q_1| = 1.17$", fontsize=10, va="center")
axa.text(pts[2][0] + 0.08, pts[2][1] + 0.1, r"$q_2$" + "\n" + r"$|q_2| = 1.31$", fontsize=10, va="center")
for (r_, a_), lab in (((0.55, 15.5), r"$31.0^\circ$"), ((0.8, 44.6), r"$27.2^\circ$")):  # wedge bisectors
    axa.text(r_ * np.cos(np.radians(a_)), r_ * np.sin(np.radians(a_)), lab, fontsize=9.5, color=C["accent"],
             ha="center", va="center")
axa.annotate(r"$|q| = 1$", xy=(np.cos(np.radians(220)), np.sin(np.radians(220))), xytext=(-1.15, -1.3), fontsize=10,
             arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7))
axa.text(-1.15, 1.62, "blue: two steps, each of length 0.6\norange: normalized " + r"$q_k/|q_k|$", fontsize=9.5)
dgfig.schematic_axes(axa, (-1.2, 1.75), (-1.35, 2.0))
axa.set_title("(a) tangent steps leave the circle", fontsize=11)


# (b)
def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Rq(q):
    q = q / np.linalg.norm(q)
    return np.eye(3) + 2 * q[0] * hat(q[1:]) + 2 * hat(q[1:]) @ hat(q[1:])


rng = np.random.default_rng(295)
# reproduce the random draws of the verify script up to the target quaternion
_ = rng.normal(size=(2, 2)); _ = rng.normal(size=2); _ = rng.normal()
sc = np.array([1.0, 0.6, 0.3])
qs = rng.normal(size=4); qs /= np.linalg.norm(qs)
St = Rq(qs) @ np.diag(sc ** 2) @ Rq(qs).T


def Lq(q):
    Rm = Rq(q)
    return 0.5 * np.sum((Rm @ np.diag(sc ** 2) @ Rm.T - St) ** 2)


def gq(q, h=1e-7):
    return np.array([(Lq(q + h * e) - Lq(q - h * e)) / (2 * h) for e in np.eye(4)])


def run_gd(eta, n=400):
    q = np.array([1.0, 0, 0, 0]); out = [1.0]
    for _ in range(n):
        q = q - eta * gq(q); out.append(np.linalg.norm(q))
    return np.array(out)


def run_adam(lr, n=400, b1=0.9, b2=0.999, eps=1e-15):
    q = np.array([1.0, 0, 0, 0]); m = np.zeros(4); v = np.zeros(4); out = [1.0]
    for k in range(1, n + 1):
        g_ = gq(q); m = b1 * m + (1 - b1) * g_; v = b2 * v + (1 - b2) * g_ * g_
        q = q - lr * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + eps); out.append(np.linalg.norm(q))
    return np.array(out)


for eta, col in ((0.05, C["third"]), (0.2, C["tangent"]), (0.5, C["main"])):
    nr = run_gd(eta)
    assert np.all(np.diff(nr) >= -1e-12)
    axb.plot(nr, color=col, lw=1.8, label=r"GD, step $%g$" % eta)
for lr, col in ((0.01, C["accent"]), (0.05, C["normal"])):
    na = run_adam(lr)
    axb.plot(na, color=col, lw=1.6, ls=(0, (4, 2)), label=r"Adam, lr $%g$" % lr)
assert abs(run_adam(0.01).min() - 0.748) < 1e-3 and abs(run_gd(0.5)[-1] - 1.3845) < 1e-3
axb.axhline(1.0, color="#BBBBBB", lw=0.6, zorder=0)
axb.set_xlabel("iteration")
axb.set_ylabel(r"$|q_k|$", labelpad=2)
axb.set_xlim(0, 400)
axb.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
axb.set_title(r"(b) $|q_k|$ during training (toy loss)", fontsize=11)
axb.tick_params(labelsize=9)

dgfig.save(fig, __file__)
