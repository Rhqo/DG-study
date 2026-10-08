"""Figure 27.2.3: the hierarchy Euclidean < similarity < affine < projective, acting on the same picture (Section 27.2).

Original picture (light gray in every panel): the unit square [0, 1]^2 with a 5 x 5 grid and its inscribed circle
(center (0.5, 0.5), radius 0.5). Each panel applies one 3 x 3 matrix of the given class (in the chart w = 1):
(a) Euclidean: rotation 25 deg + translation (0.3, 0.1);
(b) similarity: the same rotation and scale 1.5, translation (0.2, -0.1);
(c) affine: A = [[1.2, 0.6], [-0.2, 0.8]], translation (0.2, 0.0);
(d) projective: H = [[1.0, 0.2, 0.1], [0.0, 1.1, 0.0], [0.6, 0.3, 1.0]].
Self-checks: lengths kept in (a); angles kept in (b); parallel grid lines stay parallel in (c); in (d) parallel lines meet
(their images have a finite intersection) and the circle goes to a conic that is not a circle; the cross-ratio of four
collinear grid points is kept in (d).

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-2-3-hierarchy.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def Rz(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def hom(A, b):
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = b
    return M


a25 = np.radians(25)
mats = [
    ("(a) Euclidean (3 dof)", "lengths, angles, areas", hom(Rz(a25), [0.3, 0.1])),
    ("(b) similarity (4 dof)", "angles, length ratios", hom(1.5 * Rz(a25), [0.2, -0.1])),
    ("(c) affine (6 dof)", "parallelism, area ratios", hom(np.array([[1.2, 0.6], [-0.2, 0.8]]), [0.2, 0.0])),
    ("(d) projective (8 dof)", "lines, incidence, cross-ratio", np.array([[1.0, 0.2, 0.1], [0.0, 1.1, 0.0], [0.6, 0.3, 1.0]])),
]


def apply(M, xy):
    h = M @ np.vstack([xy, np.ones(xy.shape[1])])
    return h[:2] / h[2]


s = np.linspace(0, 1, 200)
grid = [np.vstack([np.full_like(s, g), s]) for g in np.linspace(0, 1, 5)] + \
       [np.vstack([s, np.full_like(s, g)]) for g in np.linspace(0, 1, 5)]
ang = np.linspace(0, 2 * np.pi, 400)
circ = np.vstack([0.5 + 0.5 * np.cos(ang), 0.5 + 0.5 * np.sin(ang)])

# self-checks
E = mats[0][2]
e1 = apply(E, np.array([[0, 1.0], [0, 0]]))
assert abs(np.linalg.norm(e1[:, 1] - e1[:, 0]) - 1) < 1e-12
S = mats[1][2]
v1 = apply(S, np.array([[0, 1, 0], [0, 0, 1.0]]))
assert abs((v1[:, 1] - v1[:, 0]) @ (v1[:, 2] - v1[:, 0])) < 1e-12
A = mats[2][2]
la, lb = apply(A, grid[0][:, [0, -1]]), apply(A, grid[1][:, [0, -1]])
da, db = la[:, 1] - la[:, 0], lb[:, 1] - lb[:, 0]
assert abs(da[0] * db[1] - da[1] * db[0]) < 1e-12
Hp = mats[3][2]
la, lb = apply(Hp, grid[0][:, [0, -1]]), apply(Hp, grid[1][:, [0, -1]])
da, db = la[:, 1] - la[:, 0], lb[:, 1] - lb[:, 0]
assert abs(da[0] * db[1] - da[1] * db[0]) > 1e-3
c4 = apply(Hp, circ)
rr = np.hypot(*(c4 - c4.mean(1, keepdims=True)))
assert rr.max() / rr.min() > 1.2
xs = np.array([0.0, 0.25, 0.5, 1.0])
img = apply(Hp, np.vstack([xs, np.full(4, 0.25)]))
tt = np.hypot(*(img - img[:, :1]))
cr = lambda a, b, c_, d: ((c_ - a) * (d - b)) / ((c_ - b) * (d - a))
assert abs(cr(*tt) - cr(*xs)) < 1e-10

fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0))
for axx, (ttl, inv, M) in zip(axes.ravel(), mats):
    for g in grid:
        axx.plot(*g, color="#BBBBBB", lw=0.6)
    axx.plot(*circ, color="#BBBBBB", lw=0.8)
    for g in grid:
        axx.plot(*apply(M, g), color=C["tangent"], lw=1.0)
    axx.plot(*apply(M, circ), color=C["normal"], lw=1.6)
    axx.set_xlim(-0.6, 2.1)
    axx.set_ylim(-0.5, 2.2)
    axx.set_aspect("equal")
    axx.set_title(ttl, fontsize=13)
    axx.text(-0.55, 2.05, "keeps: " + inv, fontsize=11.5, va="top")
    axx.tick_params(labelsize=10)
fig.tight_layout()
dgfig.save(fig, __file__)
