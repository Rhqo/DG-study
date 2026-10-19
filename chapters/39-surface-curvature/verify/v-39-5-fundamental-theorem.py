"""Verification for Section 39.5 (Fundamental Forms and the Fundamental Theorem of Surfaces).

Run from the project root: PYTHONPATH=tools python3 chapters/39-surface-curvature/verify/v-39-5-fundamental-theorem.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
sys.path.insert(0, str(HERE.parents[2] / "38-surfaces-smooth-discrete" / "lib"))
import dgcode  # noqa: E402
import dgmesh as dm  # noqa: E402
import dgsym  # noqa: E402
import ddg38  # noqa: E402
import ddg39  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "39-5-fundamental-theorem.html"
ex = dgsym.EXAMPLES
rng = np.random.default_rng(395)

# ---------------------------------------------------------------- Implementation 39.5 and Exercise 39.5.6
for blk in ("code-39-5-1", "code-39-5-2", "code-ex-39-5-6"):
    check(f"{blk}: at most 30 lines", dgcode.code_lines(PAGE, blk) <= 30)
ns, out1 = ddg39.run_printed(PAGE, "code-39-5-1")
check("Implementation 39.5 (block 1) prints nothing", out1.strip() == "")
ns, out2 = ddg39.run_printed(PAGE, "code-39-5-2", ns)
check("Implementation 39.5: printed output matches the page", out2.strip() == ddg39.printed_block(PAGE, "out-39-5-2").strip(),
      out2)
rec = ns["reconstruct"]
V0, F0 = dm.icosahedron()
_, W0, _ = ddg38.dimpled_icosahedron()
check("Implementation 39.5: the inline dimple equals ddg38.dimpled_icosahedron", np.allclose(ns["W0"], W0, atol=1e-14))
for name, (V, F) in [("icosphere(1)", dm.icosphere(1)), ("torus_mesh(2,0.8,8,16)", dm.torus_mesh(2, 0.8, 8, 16)),
                     ("dimpled icosahedron", (W0, F0))]:
    ell, th = ddg39.lengths_angles(V, F)
    Y = rec(F, ell, th)
    check(f"Theorem 39.5.10: {name} rebuilt by a proper rigid motion (Kabsch error < 1e-9)", ddg39.rigid_error(V, Y) < 1e-9,
          f"{ddg39.rigid_error(V, Y)}")
ns3, out3 = ddg39.run_printed(PAGE, "code-ex-39-5-6", dict(ns))
check("Exercise 39.5.6: printed output matches the page", out3.strip() == ddg39.printed_block(PAGE, "out-ex-39-5-6").strip(),
      out3)
# same algorithm as the library version used for Figure 39.5.3
V, F = dm.icosphere(1)
ell, th = ddg39.lengths_angles(V, F)
Ylib, seq, tree = ddg39.reconstruct(F, ell, th)
check("Figure 39.5.3: page reconstruct = ddg39.reconstruct", np.allclose(rec(F, ell, th), Ylib, atol=1e-12))
f_new = seq[60]
star_e = [e for e in tree if set(e) <= set(F[f_new].tolist()) and
          any(g in set(seq[:60]) for g, f in enumerate(F.tolist()) if g != f_new and set(e) <= set(f))][0]
th2 = dict(th)
th2[star_e] += 0.3
Y2p = rec(F, ell, th2)
Y2l, _, _ = ddg39.reconstruct(F, ell, th2)
bad = [e for e, l in ell.items() if abs(np.linalg.norm(Y2l[e[1]] - Y2l[e[0]]) - l) > 1e-6]
check("Figure 39.5.3: perturbed data give 14 wrong edges, the changed edge keeps its length, page = library",
      len(bad) == 14 and star_e not in bad and np.allclose(Y2p, Y2l, atol=1e-12), f"{len(bad)}")
check("Figure 39.5.3: dual spanning tree has n2 - 1 edges", len(tree) == len(F) - 1 and len(seq) == len(F))

# ---------------------------------------------------------------- Lemma 39.5.9 (apex): formula, uniqueness, equivariance
apex = ns["apex"]
for _ in range(5):
    pi, pj = rng.normal(size=3), rng.normal(size=3)
    e = (pj - pi) / np.linalg.norm(pj - pi)
    n = np.cross(e, rng.normal(size=3))
    n /= np.linalg.norm(n)
    pk_true = pi + 0.4 * (pj - pi) + 0.8 * np.cross(n, e)
    lik, ljk = np.linalg.norm(pk_true - pi), np.linalg.norm(pk_true - pj)
    pk = apex(pi, pj, n, lik, ljk)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    c = rng.normal(size=3)
    ok = np.allclose(pk, pk_true) and np.cross(pj - pi, pk - pi) @ n > 0
    ok &= np.allclose(apex(Q @ pi + c, Q @ pj + c, Q @ n, lik, ljk), Q @ pk + c)
    check("Lemma 39.5.9: apex recovers the point, positive normal, proper-rigid-motion equivariant", bool(ok))

# ---------------------------------------------------------------- Theorem 39.5.10 on noisy meshes, with and without boundary
for name, (V, F) in [("noisy icosphere(2)", dm.icosphere(2)), ("saddle grid (boundary)", ddg38.saddle_grid(7))]:
    V = V.copy()
    if name.startswith("noisy"):
        V *= (1 + 0.05 * rng.normal(size=len(V)))[:, None]
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    Vm = V @ Q.T + rng.normal(size=3)
    ell, th = ddg39.lengths_angles(V, F)
    ell2, th2_ = ddg39.lengths_angles(Vm, F)
    same = all(abs(ell[e] - ell2[e]) < 1e-12 for e in ell) and all(abs(th[e] - th2_[e]) < 1e-12 for e in th)
    Y = rec(F, ell, th)
    check(f"Theorem 39.5.10: {name}: (l, theta) invariant under a proper rigid motion; rebuilt mesh congruent",
          same and ddg39.rigid_error(V, Y) < 1e-9, f"{ddg39.rigid_error(V, Y)}")
# reflection: lengths equal, bending angles flip (Prop 39.5.6(c), Exercise 39.5.4)
V, F = dm.icosphere(2)
V = V * (1 + 0.05 * rng.normal(size=len(V)))[:, None]
Vr = V * np.array([1.0, 1.0, -1.0])
ell, th = ddg39.lengths_angles(V, F)
ellr, thr = ddg39.lengths_angles(Vr, F)
check("Prop 39.5.6(c): reflection keeps lengths and flips every bending angle",
      all(abs(ell[e] - ellr[e]) < 1e-12 for e in ell) and all(abs(th[e] + thr[e]) < 1e-12 for e in th))
thf = ddg39.lengths_angles(V, F[:, ::-1])[1]
check("Prop 39.5.6(c): reversing the orientation flips every bending angle", all(abs(th[e] + thf[e]) < 1e-12 for e in th))

# ---------------------------------------------------------------- Definition 39.5.5 and Proposition 39.5.6 (a), (b)
fn = dm.face_normals(V, F)
ok_def, ok_sign, ok_dih = True, True, True
for (i, j), fs in dm.edge_faces(F).items():
    f1, f2 = fs
    a = F[f1].tolist().index(i)
    f, g = (f1, f2) if F[f1][(a + 1) % 3] == j else (f2, f1)
    e = (V[j] - V[i]) / np.linalg.norm(V[j] - V[i])
    t = np.arctan2(np.cross(fn[f], fn[g]) @ e, fn[f] @ fn[g])
    ok_def &= abs(t - th[(int(i), int(j))]) < 1e-12 and np.allclose(ddg39.rotate(fn[f], e, t), fn[g], atol=1e-12)
    l = [v for v in F[g] if v not in (i, j)][0]
    ok_sign &= np.sign(t) == -np.sign(fn[f] @ (V[l] - V[i]))
    k = [v for v in F[f] if v not in (i, j)][0]
    # dihedral angle between the two half-planes at the edge
    pk = V[k] - V[i] - ((V[k] - V[i]) @ e) * e
    pl = V[l] - V[i] - ((V[l] - V[i]) @ e) * e
    dih = np.arccos(np.clip(pk @ pl / np.linalg.norm(pk) / np.linalg.norm(pl), -1, 1))
    ok_dih &= abs(dih - (np.pi - abs(t))) < 1e-9
check("Def 39.5.5: (39.5.3) = dgmesh.edge_bending_angles and N_g = R(e, theta) N_f (noisy icosphere(2))", bool(ok_def))
check("Prop 39.5.6(a): theta > 0 iff <N_f, x_l - x_i> < 0 (all edges)", bool(ok_sign))
check("Prop 39.5.6(b): dihedral angle = pi - |theta| (all edges)", bool(ok_dih))

# ---------------------------------------------------------------- Example 39.5.7 and Non-example 39.5.8
tetV = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
tetF = ddg38.orient_outward(tetV, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
vals = {}
for name, (V_, F_) in [("tetrahedron", (tetV, tetF)), ("octahedron", ddg38.octahedron()), ("icosahedron", dm.icosahedron()),
                       ("cube", ddg38.cube({}))]:
    vals[name] = sorted(set(np.round([t for t, _ in dm.edge_bending_angles(V_, F_).values()], 10)))
close("Example 39.5.7: tetrahedron pi - arccos(1/3) = 1.9106, octahedron 1.2310, icosahedron 0.7297",
      [vals["tetrahedron"][0], vals["octahedron"][0], vals["icosahedron"][0]],
      [np.pi - np.arccos(1 / 3), np.pi - np.arccos(-1 / 3), 0.7297276562], 1e-9)
check("Example 39.5.7: single values for the regular solids; cube has 0 and pi/2",
      len(vals["tetrahedron"]) == len(vals["octahedron"]) == len(vals["icosahedron"]) == 1
      and np.allclose(vals["cube"], [0.0, np.pi / 2]))
Vh, Fh = ddg38.hinge(0.7)
close("Example 39.5.7: hinge folded toward N by psi = 0.7 has theta = -0.7", dm.edge_bending_angles(Vh, Fh)[(0, 1)][0], -0.7, 1e-12)
b1 = {e: t for e, (t, _) in dm.edge_bending_angles(V0, F0).items()}
b2 = {e: t for e, (t, _) in dm.edge_bending_angles(W0, F0).items()}
spokes = [e for e in b1 if 0 in e]
link = [e for e in b1 if 0 not in e and abs(b1[e] - b2[e]) > 1e-9]
check("Example 39.5.7: dimple: 5 spokes -0.7297, 5 link edges 2.0344, 20 others unchanged",
      len(spokes) == 5 and all(abs(b2[e] + b1[e]) < 1e-12 for e in spokes) and len(link) == 5
      and all(abs(b2[e] - 2.0344) < 1e-4 for e in link) and sum(abs(b1[e] - b2[e]) < 1e-12 for e in b1) == 20)
lv = ddg39.lengths_angles(V0, F0)[0]
lw = ddg39.lengths_angles(W0, F0)[0]
check("Example 39.5.7: icosahedron and dimpled icosahedron have the same 30 edge lengths", all(abs(lv[e] - lw[e]) < 1e-12 for e in lv))
Vp, Fp = ddg38.hinge(0.7)
Vm_, _ = ddg38.hinge(-0.7)
tp, tm = dm.edge_bending_angles(Vp, Fp)[(0, 1)][0], dm.edge_bending_angles(Vm_, Fp)[(0, 1)][0]
check("Non-example 39.5.8(c): mirror hinges: same lengths, theta = -psi and +psi, no proper rigid motion",
      np.allclose(ddg39.lengths_angles(Vp, Fp)[0][(0, 3)], ddg39.lengths_angles(Vm_, Fp)[0][(0, 3)])
      and abs(tp + tm) < 1e-12 and ddg39.rigid_error(Vp, Vm_) > 0.1)

# ---------------------------------------------------------------- Proposition 39.5.11 (vertex compatibility)
for name, (V, F) in [("torus_mesh(2,0.8,8,16)", dm.torus_mesh(2, 0.8, 8, 16)), ("noisy icosphere(2)", (V, F))]:
    ell, th = ddg39.lengths_angles(V, F)
    dev = max(np.abs(ddg39.vertex_holonomy(F, ell, th, i) - np.eye(3)).max() for i in range(len(V)))
    check(f"Prop 39.5.11: (39.5.5) holds at every vertex ({name})", dev < 1e-11, f"{dev}")
V, F = dm.icosphere(1)
ell, th = ddg39.lengths_angles(V, F)
e0 = (0, 12)
th[e0] += 0.05
devs = np.array([np.abs(ddg39.vertex_holonomy(F, ell, th, i) - np.eye(3)).max() for i in range(len(V))])
check("Exercise 39.5.6: after changing theta(0,12), only vertices 0 and 12 fail", np.flatnonzero(devs > 1e-9).tolist() == [0, 12])
# all theta = 0: product is Rz(sum of corner angles)
Vf, Ff = dm.grid_mesh(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
ellf, thf = ddg39.lengths_angles(Vf, Ff)
inner = [i for i in range(len(Vf)) if i not in dm.boundary_vertices(Ff)]
check("Section 39.5: flat vertices give Rz(2 pi) = I (theta = 0)",
      all(np.allclose(ddg39.vertex_holonomy(Ff, ellf, thf, i), np.eye(3)) for i in inner) and len(inner) == 4)

# ---------------------------------------------------------------- Proposition 39.5.1, Example 39.5.2, Proposition 39.5.3, Exercises
t_ = sp.symbols("t", real=True)


def forms_sym(X, a, b, pos=()):
    N = dgsym.unit_normal(X, a, b, pos)
    dX = [X.diff(a), X.diff(b)]
    dN = [N.diff(a), N.diff(b)]
    I = sp.Matrix(2, 2, lambda i, j: dX[i].dot(dX[j]))
    II = sp.Matrix(2, 2, lambda i, j: -dN[i].dot(dX[j]))
    III = sp.Matrix(2, 2, lambda i, j: dN[i].dot(dN[j]))
    return N, I, II, III


for key in ("torus", "sphere"):
    e = ex[key]
    X = e["expr"]
    a, b = e["coords"]
    N, I, II, III = forms_sym(X, a, b, e["positive"])
    Xt = X + t_ * N
    It = sp.Matrix(2, 2, lambda i, j: Xt.diff([a, b][i]).dot(Xt.diff([a, b][j])))
    sym_equal(f"Prop 39.5.1: I_t = I - 2 t II + t^2 III ({key})", It, I - 2 * t_ * II + t_ ** 2 * III)
    K, H = dgsym.K_H(X, a, b, e["positive"])
    sym_equal(f"Exercise 39.5.2: III - 2 H II + K I = 0 ({key})", III - 2 * H * II + K * I, sp.zeros(2, 2))
sp_ = ex["sphere"]
_, Is, IIs, IIIs = forms_sym(sp_["expr"], *sp_["coords"], sp_["positive"])
r = sp_["params"][0]
sym_equal("Exercise 39.5.1: sphere II = -I/r, III = I/r^2", [IIs, IIIs], [-Is / r, Is / r ** 2])
u, v = sp.symbols("u v", real=True)
rc = sp.symbols("r", positive=True)
Xc = sp.Matrix([rc * sp.cos(u), rc * sp.sin(u), v])
Yp = sp.Matrix([rc * u, v, 0])
_, Ic, IIc, IIIc = forms_sym(Xc, u, v)
Np, Ip, IIp, _ = forms_sym(Yp, u, v)
sym_equal("Example 39.5.2: plane and cylinder have I = r^2 du^2 + dv^2", [Ic, Ip], [sp.diag(rc ** 2, 1), sp.diag(rc ** 2, 1)])
sym_equal("Example 39.5.2: II_cylinder = -r du^2 (outward), II_plane = 0, N_plane = e3", [IIc, IIp, Np],
          [sp.diag(-rc, 0), sp.zeros(2, 2), sp.Matrix([0, 0, 1])])
close("Figure 39.5.1: (r + t)^2 = 1 - 2 t (-1) + t^2 at r = 1, t = 0.6", 1.6 ** 2, 1 + 1.2 + 0.36, 1e-15)
# Proposition 39.5.3: proper rigid motion keeps II, reflection flips it (numeric on the torus)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
Q *= np.sign(np.linalg.det(Q))
c = rng.normal(size=3)
for A, sgn, name in [(Q, 1, "rotation"), (Q @ np.diag([1, 1, -1.0]), -1, "reflection")]:
    xA = lambda a_, b_, A=A: np.einsum("ij,j...->i...", A, ddg39.torus(a_, b_)) + c.reshape((3,) + (1,) * np.ndim(a_))  # noqa: E731
    g1, h1, *_ = ddg39.forms(ddg39.torus, 0.7, 1.1)
    g2, h2, *_ = ddg39.forms(xA, 0.7, 1.1)
    check(f"Prop 39.5.3: {name}: I equal, II multiplied by det A = {sgn}", np.allclose(g1, g2, atol=1e-6) and np.allclose(h2, sgn * h1, atol=1e-5))
# Exercise 39.5.3
close("Exercise 39.5.3: (1/2) sum theta l = 3 * 1.9106 and 6 * 1.2310 (l = 1)",
      [3 * (np.pi - np.arccos(1 / 3)), 6 * np.arccos(1 / 3)], [5.7319, 7.3858], 1e-4)
# counting remark: closed sphere-like mesh, 2 n1 = 6 n0 - 12
n0, n1, n2 = dm.counts(dm.icosphere(2)[1])
check("Section 39.5: 2 n1 = 6 n0 - 12 on icosphere(2)", 2 * n1 == 6 * n0 - 12)

summary()
