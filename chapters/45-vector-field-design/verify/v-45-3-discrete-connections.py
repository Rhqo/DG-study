"""Verification for Section 45.3 (Discrete Connections and Parallel Transport).

Run from the project root:
``PYTHONPATH=tools python3 chapters/45-vector-field-design/verify/v-45-3-discrete-connections.py``
"""

import pathlib
import sys
import warnings

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch45  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

warnings.simplefilter("ignore")
PAGE = HERE / "45-3-discrete-connections.html"
rng = np.random.default_rng(3)
TWO_PI = 2 * np.pi


def hinge(psi):
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.8, 0], [0.3, -0.9 * np.cos(psi), 0.9 * np.sin(psi)]], float)
    return V, np.array([[0, 1, 2], [1, 0, 3]])


def rot(axis, t):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


# ---------------------------------------------------------------- Definition 45.3.2, Proposition 45.3.3, Example 45.3.4
for deg in (0, 70, 120):
    V, F = hinge(np.radians(deg))
    E, left, right = ch45.left_right_faces(F)
    _, idx = dd.edge_list(F)
    e = idx[(0, 1)]
    check(f"Definition 45.3.2 (hinge {deg} deg): f = (0,1,2) is the left and g = (1,0,3) the right face of [0, 1]",
          left[e] == 0 and right[e] == 1)
    f, g = 0, 1
    N = dm.face_normals(V, F)
    eh = ch45.unit(V[1] - V[0])
    P = np.column_stack([np.outer(eh, eh) @ x + (x @ np.cross(N[g], eh)) * np.cross(N[f], eh) for x in np.eye(3)])
    Bg = np.column_stack([eh, np.cross(N[g], eh)])
    Bf = np.column_stack([eh, np.cross(N[f], eh)])
    check(f"Proposition 45.3.3(a) (hinge {deg} deg): P_e maps the frame (e, N_g x e) to (e, N_f x e)", np.allclose(P @ Bg, Bf))
    w = rng.normal(size=3)
    w -= (w @ N[g]) * N[g]
    check(f"Proposition 45.3.3(a) (hinge {deg} deg): P_e preserves length and lands in the plane of f",
          np.isclose(np.linalg.norm(P @ w), np.linalg.norm(w)) and abs((P @ w) @ N[f]) < 1e-12)
    rho = ch45.transport_angles(V, F)
    close(f"Example 45.3.4(b) (hinge {deg} deg): rho_e = 180 degrees", abs(np.degrees(rho[e])), 180.0, 1e-9)
    r, b, _ = ch45.face_frames3(V, F)
    for psi in (0.3, 1.1, -2.0):
        wg = np.cos(psi) * r[g] + np.sin(psi) * b[g]
        Pw = P @ wg
        check(f"Proposition 45.3.3(b) (hinge {deg} deg, psi = {psi}): the angle becomes psi + rho_e",
              abs(ch45.wrap(np.arctan2(Pw @ b[f], Pw @ r[f]) - psi - rho[e])) < 1e-12)
    if deg > 0:
        th = np.arctan2(np.cross(N[g], N[f]) @ eh, N[g] @ N[f])
        close(f"Exercise 45.3.6 (hinge {deg} deg): |bending angle| = psi", abs(np.degrees(th)), deg, 1e-9)
        Rm = rot(eh, th)
        check(f"Proposition 45.3.6 (hinge {deg} deg): the rotation about e taking N_g to N_f equals P_e on the plane of g",
              np.allclose(Rm @ N[g], N[f]) and np.allclose(Rm @ Bg, P @ Bg))
        Q = np.eye(3) - np.outer(N[f], N[f])
        x = Bg[:, 1]
        close(f"Non-example 45.3.5 / Exercise 45.3.4 (hinge {deg} deg): projection scales N_g x e by cos(bending)",
              np.linalg.norm(Q @ x), abs(np.cos(th)), 1e-12)
# Proposition 45.3.3(c): intrinsic - rho does not change under a rigid motion and only depends on lengths (hinge folds)
V, F = dm.icosphere(2)
Rq = np.linalg.qr(rng.normal(size=(3, 3)))[0]
if np.linalg.det(Rq) < 0:
    Rq[:, 0] *= -1
check("Proposition 45.3.3(c): rho is invariant under a rigid motion of the mesh",
      np.allclose(ch45.wrap(ch45.transport_angles(V @ Rq.T + 0.3, F) - ch45.transport_angles(V, F)), 0, atol=1e-12))
# Example 45.3.4(a): planar mesh, P_e = identity
Vg, Fg = dm.grid_mesh(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
E, left, right = ch45.left_right_faces(Fg)
inner = (left >= 0) & (right >= 0)
th = np.arctan2(*(Vg[Fg[:, 1]] - Vg[Fg[:, 0]])[:, [1, 0]].T)
Vg3 = Vg.copy()
rho_g = ch45.transport_angles(Vg3, Fg)
check("Example 45.3.4(a): on a planar grid rho_e = angle(r_g) - angle(r_f) on interior edges",
      np.allclose(ch45.wrap(rho_g[inner] - (th[right[inner]] - th[left[inner]])), 0, atol=1e-12))
# Non-example 45.3.5: cube edge (bending 90 degrees) kills the perpendicular component
Vc, Fc = ch45.unit_cube()
N = dm.face_normals(Vc, Fc)
E, left, right = ch45.left_right_faces(Fc)
_, idx = dd.edge_list(Fc)
e = idx[(0, 1)]
f, g = left[e], right[e]
eh = ch45.unit(Vc[1] - Vc[0])
x = np.cross(N[g], eh)
check("Non-example 45.3.5: on a cube edge the projection of N_g x e onto the next face is 0",
      abs(N[f] @ N[g]) < 1e-12 and np.linalg.norm(x - (x @ N[f]) * N[f]) < 1e-12)

# ---------------------------------------------------------------- Proposition 45.3.9: holonomy as a pairing
V, F = dm.torus_mesh(2.0, 0.8, 12, 24)
D0, D1 = dd.d0(F, len(V)), dd.d1(F)
rho = ch45.transport_angles(V, F)
xi = rng.normal(0, 0.4, len(rho))
tc, C, Cs = ch45.generators(F)
h1 = ch45.wrap(Cs[0] @ (rho + xi))
# change every reference direction by an angle t_f: rho changes by -d1^T t
t = rng.uniform(-np.pi, np.pi, len(F))
Vtmp = V
r, b, Nn = ch45.face_frames3(V, F)
r2 = np.cos(t)[:, None] * r + np.sin(t)[:, None] * b
E, left, right = ch45.left_right_faces(F)
eh = ch45.unit(V[E[:, 1]] - V[E[:, 0]])
fl, gr = left, right
c1 = np.einsum("ij,ij->i", r2[gr], eh)
c2 = np.einsum("ij,ij->i", r2[gr], np.cross(Nn[gr], eh))
Pv = c1[:, None] * eh + c2[:, None] * np.cross(Nn[fl], eh)
b2 = np.cross(Nn, r2)
rho2 = np.arctan2(np.einsum("ij,ij->i", Pv, b2[fl]), np.einsum("ij,ij->i", Pv, r2[fl]))
check("Proposition 45.3.9: changing the reference directions changes rho by -d1^T t (mod 2 pi)",
      np.allclose(ch45.wrap(rho2 - rho + D1.T @ t), 0, atol=1e-10))
check("Proposition 45.3.9: the holonomy along a dual cycle does not depend on the reference directions",
      abs(ch45.wrap(Cs[0] @ (rho2 + xi) - h1)) < 1e-10)
# transport step by step along the dual loop c*_0 agrees with the pairing
loop_edges = np.where(Cs[0] != 0)[0]
nxt = {}
for e in loop_edges:
    a_, b_ = (right[e], left[e]) if Cs[0][e] > 0 else (left[e], right[e])
    nxt[int(a_)] = (int(b_), int(e), np.sign(Cs[0][e]))
f0 = next(iter(nxt))
p, f = 0.37, f0
while True:
    g, e, sg = nxt[f]
    p += sg * (rho[e] + xi[e])
    f = g
    if f == f0:
        break
check("Proposition 45.3.9: step-by-step transport around c*_0 returns rotated by c*_0 . (rho + xi)",
      abs(ch45.wrap(p - 0.37 - h1)) < 1e-10)

# ---------------------------------------------------------------- Theorem 45.3.10, Example 45.3.11
cases = [("icosphere(3)", *dm.icosphere(3)), ("torus_mesh(2, 0.8, 24, 48)", *dm.torus_mesh(2, 0.8, 24, 48)),
         ("unit cube", *ch45.unit_cube()), ("genus2_mesh(0.2)", *dm.genus2_mesh(0.2))]
for name, V, F in cases:
    D0 = dd.d0(F, len(V))
    hol = -D0.T @ ch45.transport_angles(V, F)
    check(f"Theorem 45.3.10 ({name}): -d0^T rho = delta mod 2 pi at every vertex",
          np.abs(ch45.wrap(hol - dm.angle_defect(V, F))).max() < 1e-10)
Vc, Fc = ch45.unit_cube()
close("Example 45.3.11(a): cube corner holonomy pi/2", ch45.wrap(-(dd.d0(Fc, 8).T @ ch45.transport_angles(Vc, Fc))[7]), np.pi / 2, 1e-12)
hm = dd.HalfedgeMesh(Fc)
check("Example 45.3.11(a): x_0 and x_7 have six faces, the other six corners four faces, every Theta = 3 pi/2",
      [len(hm.vertex_faces(v)) for v in range(8)] == [6, 4, 4, 4, 4, 4, 4, 6]
      and np.allclose(dm.angle_sums(Vc, Fc), 1.5 * np.pi))
Vo = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
Fo = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
check("Example 45.3.11(b): octahedron vertex holonomy 2 pi/3",
      dm.signed_volume(Vo, Fo) > 0 and np.allclose(ch45.wrap(-(dd.d0(Fo, 6).T @ ch45.transport_angles(Vo, Fo))), 2 * np.pi / 3))
Vg, Fg = dm.grid_mesh(np.linspace(0, 1, 6), np.linspace(0, 1, 6))
# interior vertex holonomy on a planar grid via the unfolding: rho telescopes, so the cell sum is 0 mod 2 pi
E, left, right = ch45.left_right_faces(Fg)
inner_v = [i for i in range(len(Vg)) if 0 < Vg[i, 0] < 1 and 0 < Vg[i, 1] < 1]
rg = ch45.transport_angles(Vg, Fg)
cell = -(dd.d0(Fg, len(Vg)).T @ np.where((left >= 0) & (right >= 0), rg, 0.0))
check("Example 45.3.11(c): planar grid interior vertices have holonomy 0 mod 2 pi", np.allclose(ch45.wrap(cell[inner_v]), 0, atol=1e-12))
# Exercise 45.3.2: tetrahedron pi, icosahedron pi/3, two adjacent icosahedron cells 2 pi/3
Vt = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
Ft = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
if dm.signed_volume(Vt, Ft) < 0:
    Ft = Ft[:, ::-1].copy()
check("Exercise 45.3.2: regular tetrahedron vertex holonomy pi",
      np.allclose(np.abs(ch45.wrap(-(dd.d0(Ft, 4).T @ ch45.transport_angles(Vt, Ft)))), np.pi))
Vi, Fi = dm.icosahedron()
Di = dd.d0(Fi, 12)
S = np.zeros(12)
S[[0, 1]] = 1.0                                          # 0 and 1 are adjacent (face [0, 5, 1])
check("Exercise 45.3.2: icosahedron vertex pi/3, two adjacent cells 2 pi/3",
      np.allclose(ch45.wrap(-(Di.T @ ch45.transport_angles(Vi, Fi))), np.pi / 3)
      and np.isclose(ch45.wrap(-(Di @ S) @ ch45.transport_angles(Vi, Fi)), 2 * np.pi / 3))

# ---------------------------------------------------------------- Corollary 45.3.12, Example 45.3.13
for level in (2, 3, 4):
    V, F = dm.icosphere(level)
    D0 = dd.d0(F, len(V))
    rho = ch45.transport_angles(V, F)
    delta = dm.angle_defect(V, F)
    th = np.arccos(np.clip(V[:, 2], -1, 1))
    ok = True
    for t0 in np.linspace(0.1, np.pi - 0.1, 37):
        Sv = (th < t0).astype(float)
        ok = ok and abs(ch45.wrap((-(D0 @ Sv)) @ rho - delta @ Sv)) < 1e-10
    check(f"Corollary 45.3.12 (icosphere({level})): holonomy around every latitude band = sum of delta inside", ok)
    if level == 3:
        Sv = (th < np.pi / 3).astype(float)
        cs = -(D0 @ Sv)
        hol = np.mod(cs @ rho, TWO_PI)
        close("Example 45.3.13: icosphere(3) band at pi/3 has holonomy 3.1594", hol, 3.1594, 6e-5)
        check("Example 45.3.13: the band has 82 faces", int(np.abs(cs).sum()) == 82)
        xi = rng.normal(0, 0.3, len(rho))
        check("Corollary 45.3.12: with a connection xi the region holonomy is sum (delta - d0^T xi) mod 2 pi",
              abs(ch45.wrap(cs @ (rho + xi) - (delta - D0.T @ xi) @ Sv)) < 1e-10)
res = {}
for level in (2, 3, 4):
    V, F = dm.icosphere(level)
    D0 = dd.d0(F, len(V))
    th = np.arccos(np.clip(V[:, 2], -1, 1))
    Sv = (th < np.pi / 3).astype(float)
    res[level] = abs(ch45.wrap((-(D0 @ Sv)) @ ch45.transport_angles(V, F) - np.pi))
check("Example 45.3.13: the band holonomy at pi/3 approaches pi under refinement (icosphere 2, 3, 4)",
      res[2] > res[3] > res[4] and res[4] < 0.005)
print("   |hol - pi| at theta_0 = pi/3:", {k: round(v, 4) for k, v in res.items()})

# ---------------------------------------------------------------- Definition 45.3.14, Proposition 45.3.15
V, F = dm.icosphere(2)
D0 = dd.d0(F, len(V))
for trial in range(3):
    xi = rng.normal(0, 1.0, D0.shape[0])
    kap = dm.angle_defect(V, F) - D0.T @ xi
    check(f"Proposition 45.3.15 (icosphere(2), random xi #{trial}): sum kappa = 4 pi", np.isclose(kap.sum(), 4 * np.pi))
# the example after Definition 45.3.14: pi/3 on the six dual edges around a valence-6 vertex gives kappa = 2 pi
ft = ch45.FlatTorus(8, 8)
i = 0
D0f = dd.d0(ft.F)
xi = np.zeros(D0f.shape[0])
xi[D0f[:, i] != 0] = -D0f[D0f[:, i] != 0, i] * (np.pi / 3)       # counterclockwise around i is -d0 e_i
kap = -(D0f.T @ xi)
check("Definition 45.3.14 example: pi/3 on the six dual edges around a valence-6 vertex gives kappa_i = 2 pi",
      np.isclose(kap[i], 2 * np.pi) and int((D0f[:, i] != 0).sum()) == 6)

# ---------------------------------------------------------------- Implementation box and coding exercise
ns, same, out = ch45.run_printed(PAGE, "code-45-3-1", "out-45-3-1")
check("code-45-3-1 runs and prints exactly out-45-3-1", same)
check("code-45-3-1 has at most 30 lines", dgcode.code_lines(PAGE, "code-45-3-1") <= 30)
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)), ("torus", dm.torus_mesh(2.0, 0.8, 12, 24))):
    check(f"code-45-3-1 transport_angles agrees with lib/ch45.transport_angles ({name})",
          np.allclose(ch45.wrap(ns["transport_angles"](V, F) - ch45.transport_angles(V, F)), 0, atol=1e-12))
ns, same, out = ch45.run_printed(PAGE, "code-ex-45-3-5", "out-ex-45-3-5")
check("code-ex-45-3-5 runs and prints exactly out-ex-45-3-5", same)

# Exercise 45.3.6 / Proposition 45.3.6: with the sign of (39.5.3) the hinge has theta_e = -psi and R(e, -theta_e) N_g = N_f
for deg in (70, 120):
    V, F = hinge(np.radians(deg))
    N = dm.face_normals(V, F)
    eh = ch45.unit(V[1] - V[0])
    th39 = np.arctan2(np.cross(N[0], N[1]) @ eh, N[0] @ N[1])          # (39.5.3): f = (0,1,2), g = (1,0,3)
    check(f"Exercise 45.3.6 (hinge {deg} deg): theta_e = -psi and R(e, -theta_e) maps N_g to N_f",
          np.isclose(th39, -np.radians(deg)) and np.allclose(rot(eh, -th39) @ N[1], N[0]))

summary()
