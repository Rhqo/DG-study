"""Figure 43.1.2: one triangle determines an angle-preserving piecewise linear map (Proposition 43.1.4).

A planar mesh with 9 vertices and 9 faces (coordinates below). Panel (a): the mesh, the start face T_0 = (0, 1, 2)
(orange) and the order in which a breadth-first search over shared edges reaches the other faces. Panel (b): the image
of T_0 is prescribed (z_0, z_1, z_2 = a p + b with a = 0.75 e^{i 35 deg}, b = 3.2 + 0.4 i); every other face is then
forced: its two vertices on the shared edge are already placed and the orientation-preserving similarity of the face is
fixed by them, so the third vertex is z_k = z_i + (p_k - p_i) / (p_j - p_i) * (z_j - z_i). Self-check: the vertex
reached last agrees with a p + b to 1e-12 for every vertex, i.e. the whole map is the similarity of T_0.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-1-2-rigidity.py``
"""

from collections import deque

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
P = np.array([0 + 0j, 1, 0.45 + 0.85j, -0.6 + 0.75j, -0.9 - 0.15j, -0.25 - 0.85j, 0.75 - 0.7j, 1.5 + 0.7j, 1.6 - 0.3j])
F = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 1), (1, 7, 2), (1, 8, 7), (1, 6, 8)]
for tri in F:
    a, b, c = P[list(tri)]
    assert ((np.conj(b - a) * (c - a)).imag) > 0          # counterclockwise faces
A_SIM = 0.75 * np.exp(1j * np.radians(35))
B_SIM = 3.2 + 0.4j

z = {i: A_SIM * P[i] + B_SIM for i in F[0]}
order = {0: 0}
queue = deque([0])
while queue:
    f = queue.popleft()
    for g, tri in enumerate(F):
        if g in order:
            continue
        shared = set(tri) & set(F[f])
        if len(shared) == 2:
            # oriented shared edge (i, j) as traversed by g, third vertex k
            for s in range(3):
                i, j, k = tri[s], tri[(s + 1) % 3], tri[(s + 2) % 3]
                if {i, j} == shared:
                    break
            zk = z[i] + (P[k] - P[i]) / (P[j] - P[i]) * (z[j] - z[i])
            if k in z:
                assert abs(z[k] - zk) < 1e-12
            z[k] = zk
            order[g] = len(order)
            queue.append(g)
assert len(order) == len(F) and len(z) == len(P)
err = max(abs(z[i] - (A_SIM * P[i] + B_SIM)) for i in z)
assert err < 1e-12
print("BFS order", order, "max deviation from a p + b: %.1e" % err)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, pts, title in ((axes[0], P, r"(a) mesh, start face $T_0$"), (axes[1], np.array([z[i] for i in range(len(P))]),
                                                               r"(b) forced image: $a\,p + b$")):
    for g, tri in enumerate(F):
        w = pts[list(tri)]
        ax.fill(w.real, w.imag, color=col["accent"] if g == 0 else col["region"], alpha=0.65 if g == 0 else 0.3, lw=0)
        ax.plot(np.r_[w, w[0]].real, np.r_[w, w[0]].imag, color=col["main"], lw=0.9)
        c = w.mean()
        ax.text(c.real, c.imag, r"$T_0$" if g == 0 else str(order[g]), ha="center", va="center", fontsize=10)
    ax.plot(pts.real, pts.imag, "o", color=col["main"], ms=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
for i in (0, 1, 2):
    off = {0: (-0.16, 0.06), 1: (0.05, -0.17), 2: (0.0, 0.1)}[i]
    off_a = (0.07, 0.04) if i == 2 else off                  # keep p_2 clear of the panel title
    axes[0].text(P[i].real + off_a[0], P[i].imag + off_a[1], r"$p_%d$" % i, fontsize=10)
    q = z[i]
    axes[1].text(q.real + 0.75 * off[0], q.imag + 0.75 * off[1], r"$z_%d$" % i, fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.1)
dgfig.save(fig, __file__)
