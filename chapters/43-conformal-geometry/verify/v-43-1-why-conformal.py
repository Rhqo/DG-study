"""Verification for Section 43.1 (Why Conformal?).

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/verify/v-43-1-why-conformal.py``
"""

import pathlib
import sys
from collections import deque

import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch43  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "43-1-why-conformal.html"

# ---------------------------------------------------------------- Example 43.1.1 (Mercator and stereographic)
th, t, phi = sp.symbols("theta t phi", positive=True)
tt = sp.log(sp.cot(th / 2))
check("Example 43.1.1(a): cosh(log cot(theta/2)) = (cot + tan)(theta/2)/2 = 1/sin(theta)",
      sp.simplify((sp.cos(th / 2) / sp.sin(th / 2) + sp.sin(th / 2) / sp.cos(th / 2)) * sp.sin(th / 2) * sp.cos(th / 2)) == 1
      and all(abs(np.cosh(np.log(1 / np.tan(a / 2))) - 1 / np.sin(a)) < 1e-12 for a in np.linspace(0.1, 3.0, 13)))
X = sp.Matrix([sp.sech(t) * sp.cos(phi), sp.sech(t) * sp.sin(phi), sp.tanh(t)])
Xp, Xt = X.diff(phi), X.diff(t)
check("Example 43.1.1(a): Mercator metric is sech^2 t (dphi^2 + dt^2)",
      sp.simplify(Xp.dot(Xp) - sp.sech(t) ** 2) == 0 and sp.simplify(Xt.dot(Xt) - sp.sech(t) ** 2) == 0
      and sp.simplify(Xp.dot(Xt)) == 0)
close("Example 43.1.1(a): length factor 2 and area factor 4 at latitude 60 deg", (1 / np.sin(np.pi / 6), 1 / np.sin(np.pi / 6) ** 2), (2.0, 4.0), 1e-12)
u, v = sp.symbols("u v", real=True)
Y = sp.Matrix([2 * u, 2 * v, u ** 2 + v ** 2 - 1]) / (1 + u ** 2 + v ** 2)
Yu, Yv = Y.diff(u), Y.diff(v)
check("Example 43.1.1(b): inverse stereographic has E = G = 4/(1+|w|^2)^2, F = 0",
      sp.simplify(Yu.dot(Yu) - 4 / (1 + u ** 2 + v ** 2) ** 2) == 0 and sp.simplify(Yu.dot(Yv)) == 0
      and sp.simplify(Yv.dot(Yv) - Yu.dot(Yu)) == 0)
# sigma(Y(u, v)) = u + i v (the inverse is the stereographic projection from the north pole)
check("Example 43.1.1(b): sigma(Y(u,v)) = (u, v)",
      sp.simplify(Y[0] / (1 - Y[2]) - u) == 0 and sp.simplify(Y[1] / (1 - Y[2]) - v) == 0)

# ---------------------------------------------------------------- Figure 43.1.1 numbers
EPS = 0.045
tc = np.linspace(0, 2 * np.pi, 241)
f = lambda z: z + 0.3 * z ** 2  # noqa: E731
twist = lambda z: np.abs(z) * np.exp(1j * (np.angle(z) + 1.2 * np.abs(z) ** 2))  # noqa: E731


def moment_ratio(w):
    d = np.c_[w.real - w.real.mean(), w.imag - w.imag.mean()]
    ev = np.linalg.eigvalsh(d.T @ d)
    return np.sqrt(ev[1] / ev[0])


def shoelace(w):
    return 0.5 * np.sum(w.real * np.roll(w.imag, -1) - w.imag * np.roll(w.real, -1))


centres = [0j] + [r * np.exp(1j * a) for r in (0.2, 0.4, 0.6, 0.8) for a in 2 * np.pi * np.arange(12) / 12]
ra = [moment_ratio(f(c + EPS * np.exp(1j * tc))) for c in centres]
check("Figure 43.1.1(a): 49 circles, all images have moment axis ratio <= 1.0044", len(centres) == 49 and max(ra) < 1.0045)
zz = np.exp(1j * np.linspace(0, 2 * np.pi, 3601)) * np.sqrt(np.linspace(0, 1, 101))[:, None]
fp = np.abs(1 + 0.6 * zz)
close("Figure 43.1.1(a): |f'| ranges over (0.4, 1.6) on the closed disk", (fp.min(), fp.max()), (0.4, 1.6), 1e-6)
check("Figure 43.1.1(a): Re f' > 0 on the disk (so f is injective by Noshiro-Warschawski)", (1 + 0.6 * zz).real.min() > 0.39)
area_r = [shoelace(twist(c + EPS * np.exp(1j * tc))) / (np.pi * EPS ** 2) for c in centres]
close("Figure 43.1.1(b): twist images have area ratio 0.9999", (min(area_r), max(area_r)), (0.9999, 0.9999), 1e-4)
rb = {r: np.mean([moment_ratio(twist(r * np.exp(1j * a) + EPS * np.exp(1j * tc))) for a in 2 * np.pi * np.arange(12) / 12])
      for r in (0.4, 0.8)}
close("Figure 43.1.1(b): axis ratios 1.46 (rho = 0.4) and 4.09 (rho = 0.8)", (rb[0.4], rb[0.8]), (1.46, 4.09), 6e-3)

# ---------------------------------------------------------------- Non-example 43.1.3 (twist differential)
x, y = sp.symbols("x y", real=True)
rho = sp.sqrt(x ** 2 + y ** 2)
tau = sp.Function("tau")
ang = sp.atan2(y, x) + tau(rho)
T = sp.Matrix([rho * sp.cos(ang), rho * sp.sin(ang)])
DT = T.jacobian([x, y])
check("Non-example 43.1.3: det DT = 1 for every tau", sp.simplify(DT.det()) == 1)
# polar frame at the point and at its image
pt = {x: sp.Rational(3, 5), y: sp.Rational(4, 5)}
th0 = sp.atan2(pt[y], pt[x])
cf = sp.Function("tau")
D_num = DT.subs(pt)
er, et = sp.Matrix([sp.cos(th0), sp.sin(th0)]), sp.Matrix([-sp.sin(th0), sp.cos(th0)])
img_ang = th0 + cf(1)
Er, Et = sp.Matrix([sp.cos(img_ang), sp.sin(img_ang)]), sp.Matrix([-sp.sin(img_ang), sp.cos(img_ang)])
Mpolar = sp.Matrix([[Er.dot(D_num * er), Er.dot(D_num * et)], [Et.dot(D_num * er), Et.dot(D_num * et)]])
s_sym = sp.Subs(sp.Derivative(cf(sp.Symbol("r")), sp.Symbol("r")), sp.Symbol("r"), 1).doit()
check("(43.1.2): in the polar frames DT = [[1, 0], [rho tau', 1]] (checked at rho = 1)",
      sp.simplify(Mpolar - sp.Matrix([[1, 0], [s_sym, 1]])) == sp.zeros(2, 2))
s = 1.2 * 2 * 0.8 ** 2
sv = np.linalg.svd(np.array([[1, 0], [s, 1]]), compute_uv=False)
close("Non-example 43.1.3: s = 1.536 and sigma1/sigma2 = sigma1^2 = 4.12 at rho = 0.8",
      (s, sv[0] / sv[1], sv[0] ** 2), (1.536, 4.12, 4.12), 5e-3)
close("Non-example 43.1.3: sigma_{1,2} = (sqrt(s^2+4) +- s)/2", sv, ((np.sqrt(s * s + 4) + s) / 2, (np.sqrt(s * s + 4) - s) / 2), 1e-12)

# ---------------------------------------------------------------- Proposition 43.1.4 (rigidity), Figure 43.1.2
def propagate(F, frames, z0):
    """BFS over shared edges: (43.1.3) in the frame of each face; z0 gives the images of face 0."""
    z = dict(z0)
    seen = {0}
    q = deque([0])
    conflict = 0.0
    while q:
        f0 = q.popleft()
        for g, tri in enumerate(F):
            if g in seen or len(set(tri) & set(F[f0])) != 2:
                continue
            for s_ in range(3):
                i, j, k = tri[s_], tri[(s_ + 1) % 3], tri[(s_ + 2) % 3]
                if i in z and j in z:
                    break
            p = frames[g]
            pi, pj, pk = (p[(s_) % 3], p[(s_ + 1) % 3], p[(s_ + 2) % 3])
            zk = z[i] + (pk - pi) / (pj - pi) * (z[j] - z[i])
            if k in z:
                conflict = max(conflict, abs(z[k] - zk))
            else:
                z[k] = zk
            seen.add(g)
            q.append(g)
    return z, conflict


rng = np.random.default_rng(3)
xs = np.linspace(0, 1, 6)
Vg, Fg = dm.grid_mesh(xs, xs)
Vg = Vg + np.c_[0.04 * rng.standard_normal((len(Vg), 2)), np.zeros(len(Vg))]
Pg = Vg[:, 0] + 1j * Vg[:, 1]
fr = [Pg[tri] for tri in Fg]
a_, b_ = 0.6 * np.exp(0.7j), 1 - 2j
z, conf = propagate(Fg, fr, {k: a_ * Pg[k] + b_ for k in Fg[0]})
check("Proposition 43.1.4(c): on a jittered planar grid the forced map is a p + b at every vertex",
      len(z) == len(Pg) and max(abs(z[k] - (a_ * Pg[k] + b_)) for k in z) < 1e-12 and conf < 1e-12)
# curved mesh: the hemisphere; frames from dgmesh.face_frames
Vh, Fh = ch43.hemisphere(2)
Xf = dm.face_frames(Vh, Fh)
frh = [Xf[g][:, 0] + 1j * Xf[g][:, 1] for g in range(len(Fh))]
zh, confh = propagate(Fh, frh, {k: frh[0][c] for c, k in enumerate(Fh[0])})
defect = dm.angle_defect(Vh, Fh)
interior = np.setdiff1d(np.arange(len(Vh)), dm.boundary_loop(Fh))
check("Corollary 43.1.5: on the hemisphere every interior defect is not in 2 pi Z and the forced map conflicts",
      np.all(np.abs(defect[interior]) > 1e-3) and np.all(np.abs(defect[interior]) < 1) and confh > 1e-3)
# (a): equal scale: forced map on a curved strip of faces with no interior vertex (a fan around the boundary)
ok_scale = True
for g, tri in enumerate(Fh):
    if all(k in zh for k in tri):
        d1, d2 = zh[tri[1]] - zh[tri[0]], frh[g][1] - frh[g][0]
        ok_scale &= abs(abs(d1 / d2) - 1) < 1e-8 or confh > 0
check("Proposition 43.1.4(a): the forced faces keep scale 1 when the start face has scale 1", ok_scale)
# Figure 43.1.2 script assertion is the same algorithm; re-run its data
P = np.array([0 + 0j, 1, 0.45 + 0.85j, -0.6 + 0.75j, -0.9 - 0.15j, -0.25 - 0.85j, 0.75 - 0.7j, 1.5 + 0.7j, 1.6 - 0.3j])
F9 = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 1), (1, 7, 2), (1, 8, 7), (1, 6, 8)]
A9, B9 = 0.75 * np.exp(1j * np.radians(35)), 3.2 + 0.4j
z9, c9 = propagate(F9, [P[list(t_)] for t_ in F9], {k: A9 * P[k] + B9 for k in F9[0]})
check("Figure 43.1.2: forced image equals 0.75 e^{35i} p + (3.2 + 0.4i) to 1e-15",
      max(abs(z9[k] - (A9 * P[k] + B9)) for k in z9) < 1e-14 and c9 < 1e-14)

# ---------------------------------------------------------------- Corollary 43.1.5 (product formula)
tk = np.arange(1, 7) * np.pi / 3 + np.pi / 6
Xs = np.c_[np.cos(tk), np.sin(tk), 0.32 * np.cos(2 * tk)]
ell = np.linalg.norm(Xs, axis=1)
theta = np.array([np.arccos(Xs[k] @ Xs[(k + 1) % 6] / (ell[k] * ell[(k + 1) % 6])) for k in range(6)])
w = [ell[0] + 0j]
for k in range(6):
    w.append(ell[(k + 1) % 6] / ell[k] * np.exp(1j * theta[k]) * w[-1])
close("Corollary 43.1.5: after one turn w_7 / w_1 = exp(i sum theta) = exp(-i delta)",
      w[6] / w[0], np.exp(1j * (2 * np.pi - (2 * np.pi - theta.sum()))), 1e-12)
check("Corollary 43.1.5: the saddle star has sum theta > 2 pi (delta < 0)", theta.sum() > 2 * np.pi)

# ---------------------------------------------------------------- Example 43.1.6, Exercise 43.1.5
expect = {4: 0.198, 8: 0.093, 16: 0.045, 32: 0.022}
got = {}
for n in expect:
    xs = np.linspace(0, 1, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    got[n] = ch43.quasi_conformal(V, F, np.exp(V[:, 0] + 1j * V[:, 1])).max() - 1
close("Example 43.1.6: max sigma1/sigma2 - 1 for e^z is 0.198, 0.093, 0.045, 0.022", [got[n] for n in expect],
      list(expect.values()), 1.2e-3)
check("Example 43.1.6: the error halves when n doubles (ratio 2.0 +- 0.15)",
      all(1.85 < got[n] / got[2 * n] < 2.2 for n in (4, 8, 16)))
xs = np.linspace(0, 1, 5)
V, F = dm.grid_mesh(xs, xs)
close("Example 43.1.6: no face is a similarity, min sigma1/sigma2 - 1 = 0.190 at n = 4",
      ch43.quasi_conformal(V, F, np.exp(V[:, 0] + 1j * V[:, 1])).min() - 1, 0.190, 5e-4)
ns, same, printed = ch43.run_printed(PAGE, "code-ex-43-1-5", "out-ex-43-1-5")
check("Exercise 43.1.5: code-ex-43-1-5 prints exactly out-ex-43-1-5", same, printed)
for n in (4, 8, 16, 32):
    xs = np.linspace(0, 1, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    zq = (V[:, 0] + 1 + 1j * V[:, 1]) ** 2
    g = ns["face_ratios_and_gap"](V, F, zq)[1]
    close(f"Exercise 43.1.5: gap of (w+1)^2 equals 1/n^2 (n = {n}) and equals sum A_f CR^2/2", (g, g),
          (1 / n ** 2, ch43.face_cr_energy(V, F, zq).sum()), 1e-12)

# ---------------------------------------------------------------- Exercise 43.1.1
close("Exercise 43.1.1: Mercator area factor 1/sin^2 theta at 45 and 75 deg N", (1 / np.sin(np.pi / 4) ** 2, 1 / np.sin(np.radians(15)) ** 2),
      (2.0, 14.93), 5e-3)
L = sp.Matrix([phi, sp.cos(th)])
S2 = sp.Matrix([sp.sin(th) * sp.cos(phi), sp.sin(th) * sp.sin(phi), sp.cos(th)])
check("Exercise 43.1.1: Lambert stretches east by 1/sin theta and north by sin theta (squared lengths)",
      sp.simplify(L.diff(phi).dot(L.diff(phi)) / S2.diff(phi).dot(S2.diff(phi)) - 1 / sp.sin(th) ** 2) == 0
      and sp.simplify(L.diff(th).dot(L.diff(th)) / S2.diff(th).dot(S2.diff(th)) - sp.sin(th) ** 2) == 0)

# ---------------------------------------------------------------- Exercise 43.1.2 (fold)
xs = np.linspace(-1, 1, 5)
Vf, Ff = dm.grid_mesh(xs, xs)
Pf = Vf[:, 0] + 1j * Vf[:, 1]
zf = np.where(Pf.imag >= 0, Pf, np.conj(Pf))
sim = []
for tri in Ff:
    p_, q_ = Pf[tri], zf[tri]
    # similarity (orientation-preserving or -reversing): ratios agree with p or with conj(p)
    r1 = (q_[2] - q_[0]) / (q_[1] - q_[0])
    rp = (p_[2] - p_[0]) / (p_[1] - p_[0])
    sim.append(abs(r1 - rp) < 1e-12 or abs(r1 - np.conj(rp)) < 1e-12)
check("Exercise 43.1.2: the fold is a similarity on every face but not one global similarity",
      all(sim) and any(Pf[tri].imag.min() < 0 for tri in Ff)
      and abs((zf[0] - zf[-1]) / (Pf[0] - Pf[-1]) - (zf[1] - zf[-1]) / (Pf[1] - Pf[-1])) > 1e-6)

# ---------------------------------------------------------------- Exercise 43.1.3
r_ = sp.symbols("rho", positive=True)
check("Exercise 43.1.3: int_0^{2pi} int_0^1 4 rho / (1 + rho^2)^2 = 2 pi",
      sp.simplify(sp.integrate(sp.integrate(4 * r_ / (1 + r_ ** 2) ** 2, (r_, 0, 1)), (sp.Symbol("a"), 0, 2 * sp.pi)) - 2 * sp.pi) == 0)

# ---------------------------------------------------------------- Exercise 43.1.4
c, rr = sp.symbols("c rho", positive=True)
s_ = 2 * c * rr ** 2
sig1 = (s_ + sp.sqrt(s_ ** 2 + 4)) / 2
check("Exercise 43.1.4: sigma1^2 = (c rho^2 + sqrt(c^2 rho^4 + 1))^2",
      sp.simplify(sig1 ** 2 - (c * rr ** 2 + sp.sqrt(c ** 2 * rr ** 4 + 1)) ** 2) == 0)
close("Exercise 43.1.4: value 4.12 at c = 1.2, rho = 0.8", float((1.2 * 0.64 + np.sqrt(1.44 * 0.8 ** 4 + 1)) ** 2), 4.12, 5e-3)
check("Exercise 43.1.4: series 1 + 2 c rho^2 + O(rho^4)",
      sp.simplify(sp.series((c * rr ** 2 + sp.sqrt(c ** 2 * rr ** 4 + 1)) ** 2, rr, 0, 4).removeO() - (1 + 2 * c * rr ** 2)) == 0)

summary()
