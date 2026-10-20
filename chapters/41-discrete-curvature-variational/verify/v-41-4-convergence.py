"""Verification for Section 41.4 (Numerical Tests and Convergence).

Run from the project root: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/verify/v-41-4-convergence.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "41-4-convergence.html"
R, r = 2.0, 0.8

# ---------------------------------------------------------------- Implementation 41.4 and Exercises 41.4.4, 41.4.5
check("Implementation 41.4: at most 30 lines", dgcode.code_lines(PAGE, "code-41-4-1") <= 30)
ns, out = ddg41.run_printed(PAGE, "code-41-4-1")
check("Implementation 41.4: printed output matches the page", out.strip() == ddg41.printed_block(PAGE, "out-41-4-1").strip(), out)
Vs, Fs = ns["subdivided"](2)
Vl, Fl, cls = ddg41.subdivided_torus(2)
close("Implementation 41.4: subdivided() = ddg41.subdivided_torus", Vs, Vl, 1e-12)
check("Implementation 41.4: same faces", np.array_equal(Fs, Fl))
ns2, out2 = ddg41.run_printed(PAGE, "code-ex-41-4-4", dict(ns))
check("Exercise 41.4.4: printed output matches the page", out2.strip() == ddg41.printed_block(PAGE, "out-ex-41-4-4").strip(), out2)
ns3, out3 = ddg41.run_printed(PAGE, "code-ex-41-4-5", dict(ns2))
check("Exercise 41.4.5: printed output matches the page", out3.strip() == ddg41.printed_block(PAGE, "out-ex-41-4-5").strip(), out3)

# ---------------------------------------------------------------- Example 41.4.1
cases = {"icosphere(5)": (dm.icosphere(5), 10242, 2.4e-12, 9e-15, 8e-15), "torus 96 x 192": (dm.torus_mesh(R, r, 96, 192), 18432, 4.5e-12, 3e-13, 9e-15),
         "subdivided torus L4": (ddg41.subdivided_torus(4)[:2], 32768, 8.2e-12, 7e-14, 1.2e-14),
         "jittered sphere L4": (ddg41.jittered_sphere(4, 0.15, 7), 2562, 6.2e-13, 1e-15, 1e-15), "cube 8 x 8": (ddg41.cube_mesh(8), 386, 2.9e-13, 3e-17, 0.0)}
for name, ((V, F), n0, gb, lx, kn) in cases.items():
    chi = dm.euler_characteristic(F)
    res = [abs(dm.angle_defect(V, F).sum() - 2 * np.pi * chi), np.abs(dm.cotan_apply(V, F, V).sum(0)).max(),
           np.abs(ddg41.gauss_curvature_normal(V, F).sum(0)).max()]
    check(f"Example 41.4.1: {name}: n0 = {n0}, all residuals < 1e-11", len(V) == n0 and max(res) < 1e-11)
    check(f"Example 41.4.1: {name}: table magnitudes within a factor 3 ({res[0]:.1e}, {res[1]:.0e}, {res[2]:.0e})",
          all((b / 3 <= a <= 3 * b) if b > 0 else a < 1e-16 for a, b in zip(res, (gb, lx, kn))))
    others = [np.abs(ddg41.vector_area(V, F).sum(0)).max(), np.abs(np.cross(V, dm.cotan_apply(V, F, V)).sum(0)).max()]
    check(f"Example 41.4.1: {name}: vector areas and torques sum to 0", max(others) < 1e-11)
close("Example 41.4.1: cube total mean curvature = -3 pi", ddg41.total_mean_curvature(*ddg41.cube_mesh(8)), -3 * np.pi, 1e-12)


# ---------------------------------------------------------------- Examples 41.4.3, 41.4.4
def measures(V, F):
    u, K, H, k1, k2 = ddg41.torus_exact(V, R, r)
    A = dm.mass_voronoi(V, F)
    d = dm.angle_defect(V, F)
    Hd = dd.mean_curvature(V, F)
    Hs = dm.mean_curvature_dihedral(V, F, mass=A)
    a, b, cl = ddg41.principal_curvatures(Hd, d / A)
    rho2 = np.sum(V[:, :2] ** 2, 1)
    L2 = lambda e: np.sqrt(np.sum(A * e ** 2) / A.sum())  # noqa: E731
    return dict(h=ddg41.mean_edge_length(V, F), K=np.abs(d / A - K).max(), K2=L2(d / A - K), H=np.abs(Hd - H).max(),
                Hs=np.abs(Hs - H).max(), k1=np.abs(a - k1).max(), wK=abs(np.sum(d * rho2) - 4 * np.pi ** 2 * R * r))


reg = [measures(*dm.torus_mesh(R, r, n, 2 * n)) for n in (12, 24, 48, 96)]
close("Example 41.4.3: h = 0.536, 0.269, 0.135, 0.067", [m["h"] for m in reg], [0.536, 0.269, 0.135, 0.067], 6e-4)
close("Example 41.4.3: E_inf(K) = 2.9e-2, 7.5e-3, 1.9e-3, 4.8e-4", np.array([m["K"] for m in reg]) / np.array([2.9e-2, 7.5e-3, 1.9e-3, 4.8e-4]), np.ones(4), 0.02)
orders = np.log2(np.array([m["K"] for m in reg[:-1]]) / np.array([m["K"] for m in reg[1:]]))
close("Example 41.4.3: observed orders 1.95, 1.99, 2.00", orders, [1.95, 1.99, 2.00], 6e-3)
close("Example 41.4.3: n = 96: cotan H 1.8e-4, Steiner H 2.6e-4, kappa_1 3.8e-4", [reg[-1]["H"] / 1.8e-4, reg[-1]["Hs"] / 2.6e-4, reg[-1]["k1"] / 3.8e-4], [1, 1, 1], 0.03)
for key in ("H", "Hs", "k1", "K2", "wK"):
    o = np.log2(reg[-2][key] / reg[-1][key])
    check(f"Example 41.4.3: order of {key} in the last step is about 2 ({o:.3f})", 1.9 < o < 2.1)
sub = [measures(*ddg41.subdivided_torus(k)[:2]) for k in (1, 2, 3, 4)]
close("Example 41.4.4: h = 0.403, 0.202, 0.101, 0.051", [m["h"] for m in sub], [0.403, 0.202, 0.101, 0.051], 6e-4)
close("Example 41.4.4: E_inf(K) = 0.038, 0.029, 0.027, 0.027", [m["K"] for m in sub], [0.038, 0.029, 0.027, 0.027], 6e-4)
close("Example 41.4.4: E_inf(H) = 0.017, 0.024, 0.029, 0.031", [m["H"] for m in sub], [0.017, 0.024, 0.029, 0.031], 6e-4)
close("Example 41.4.4: E_2(K) = 8.2e-3, 3.9e-3, 2.0e-3, 9.7e-4", np.array([m["K2"] for m in sub]) / np.array([8.2e-3, 3.9e-3, 2.0e-3, 9.7e-4]), np.ones(4), 0.03)
close("Example 41.4.4: weak error 0.45, 0.11, 0.029, 0.0072", np.array([m["wK"] for m in sub]) / np.array([0.45, 0.11, 0.029, 0.0072]), np.ones(4), 0.05)
close("Example 41.4.4: int K rho^2 dA = 4 pi^2 R r = 63.165", 4 * np.pi ** 2 * R * r, 63.165, 5e-4)
V, F, cls = ddg41.subdivided_torus(4)
u, K, *_ = ddg41.torus_exact(V)
e = np.abs(dm.angle_defect(V, F) / dm.mass_voronoi(V, F) - K)
close("Example 41.4.4 / Figure 41.4.2(a): class maxima 0.027, 0.0069, 0.0028", [e[cls == c].max() for c in (1, 2, 3)], [0.027, 0.0069, 0.0028], 4e-4)
check("Example 41.4.4: 128 original vertices", np.sum(cls == 1) == 128)
check("Example 41.4.4: all vertices of the subdivided meshes have valence 6", all(np.all(np.array([len(s) for s in dm.neighbors(ddg41.subdivided_torus(k)[1], len(ddg41.subdivided_torus(k)[0]))]) == 6) for k in (1, 2)))
check("Example 41.4.4: all vertices lie on the torus", np.abs(np.hypot(np.hypot(V[:, 0], V[:, 1]) - R, V[:, 2]) - r).max() < 1e-12)

# ---------------------------------------------------------------- Example 41.4.5 (table from Exercise 41.4.5) and lattice deviation
sc = ns3["star_curvature"]
table = {name: [sc(P, 1e-3, *abc) for abc in ns3["surfaces"].values()] for name, P in ns3["stars"].items()}
want = {"regular hexagon": [1, -1, -0.95], "lattice hexagon": [1, -1, -0.95], "irregular hexagon": [1, -1.0952, -1.0536],
        "regular pentagon": [1, -0.8090, -0.7514], "square": [1, -1, -0.91]}
for name in want:
    close(f"Example 41.4.5: {name}", table[name], want[name], 6e-5)
    close(f"Example 41.4.5: {name}: h = 1e-4 gives the same four digits", [sc(ns3["stars"][name], 1e-4, *abc) for abc in ns3["surfaces"].values()], want[name], 6e-5)
L = np.array(ns3["stars"]["lattice hexagon"], float)
check("Example 41.4.5: lattice hexagon: P_{k+3} = -P_k and P_2 = P_1 + P_3", np.allclose(L[3:], -L[:3]) and np.allclose(L[1], L[0] + L[2]))
close("Example 41.4.5: tilted K = ac - b^2 = -0.95, square gives ac = -0.91", [1.3 * -0.7 - 0.2 ** 2, 1.3 * -0.7], [-0.95, -0.91], 1e-12)


def lattice_dev(V, F, ids):
    N = dm.vertex_normals(V, F, kind="angle")
    worst = 0.0
    for i in ids:
        nxt = {}
        for f in F[np.any(F == i, axis=1)]:
            a = list(f).index(i)
            nxt[f[(a + 1) % 3]] = f[(a + 2) % 3]
        ring = [next(iter(nxt))]
        while len(ring) < len(nxt):
            ring.append(nxt[ring[-1]])
        P = V[ring] - V[i]
        P -= np.outer(P @ N[i], N[i])
        dev = max(np.linalg.norm(P[k] + P[k + 3]) for k in range(3)) + max(np.linalg.norm(P[(k + 1) % 6] - P[k] - P[(k + 2) % 6]) for k in range(6))
        worst = max(worst, dev / np.linalg.norm(P, axis=1).mean())
    return worst


rng = np.random.default_rng(0)
devs = []
for n in (24, 48, 96):
    V, F = dm.torus_mesh(R, r, n, 2 * n)
    devs.append(lattice_dev(V, F, rng.choice(len(V), 200, replace=False)))
close("Example 41.4.5 (text): regular grid lattice deviation 0.41, 0.20, 0.10", devs, [0.41, 0.20, 0.10], 0.011)
for lev in (2, 3, 4):
    V, F, cls = ddg41.subdivided_torus(lev)
    d = [lattice_dev(V, F, rng.choice(np.where(cls == c)[0], min(200, np.sum(cls == c)), replace=False)) for c in (1, 2, 3)]
    check(f"Example 41.4.5 (text): subdivided level {lev}: deviations about 1.20, 0.96, 0.11-0.12 ({np.round(d, 3)})",
          abs(d[0] - 1.20) < 0.02 and abs(d[1] - 0.96) < 0.02 and 0.10 < d[2] < 0.13)
summary()
