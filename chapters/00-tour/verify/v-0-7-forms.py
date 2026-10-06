"""Verification for Section 0.7 (Differential Forms and Stokes's Theorem).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-7-forms.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

x, y, z, t = sp.symbols("x y z t", real=True)

# ---------------------------------------------------------------- (0.7.1) df and Figure 0.7.1
f = x ** 2 + y ** 2
df = [sp.diff(f, x), sp.diff(f, y)]
check("Figure 0.7.1: df_p = 2dx + dy at p = (1, 1/2)", [c.subs({x: 1, y: sp.Rational(1, 2)}) for c in df] == [2, 1])
val = 2 * sp.Rational(1, 4) + 1 * sp.Rational(1, 4)
check("Figure 0.7.1: df_p(v) = 3/4 for v = (1/4, 1/4)", val == sp.Rational(3, 4))
check("Figure 0.7.1: v crosses (3/4)/(1/8) = 6 lines", val / sp.Rational(1, 8) == 6)

# Exercise 0.7.1: f = x² y, df at (1, 2) applied to (1, 1)
g = x ** 2 * y
dg = [sp.diff(g, x), sp.diff(g, y)]
dg_p = [c.subs({x: 1, y: 2}) for c in dg]
check("Exercise 0.7.1: df_(1,2) = 4dx + dy", dg_p == [4, 1])
check("Exercise 0.7.1: df_(1,2)(1,1) = 5", dg_p[0] + dg_p[1] == 5)
# the directional derivative agrees
s = sp.symbols("s", real=True)
check("Exercise 0.7.1: equals d/ds f((1,2) + s(1,1)) at s = 0", sp.diff(g.subs({x: 1 + s, y: 2 + s}), s).subs(s, 0) == 5)

# (0.7.2) pullback: d(f o F) = F^*(df) is the row vector df · DF (VJP)
u, v = sp.symbols("u v", real=True)
F = sp.Matrix([u * sp.cos(v), u * sp.sin(v)])  # polar coordinates
fF = f.subs({x: F[0], y: F[1]})
row = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)]]).subs({x: F[0], y: F[1]}) * F.jacobian([u, v])
sym_equal("(0.7.2): d(f o F) = df(F) DF (pullback = VJP)", sp.Matrix([[sp.diff(fF, u), sp.diff(fF, v)]]), row)

# change of coordinates: covector components transform by the transpose Jacobian (row · J), vectors by J
J = F.jacobian([u, v])
w_uv = sp.Matrix([1, 2])                  # tangent vector components in (u, v)
w_xy = J * w_uv
a_xy = sp.Matrix([[3, -1]])               # covector components in (x, y) at the same point
a_uv = a_xy * J
sym_equal("Section covectors: pairing a(w) is coordinate independent", (a_uv * w_uv)[0], (a_xy * w_xy)[0])

# ---------------------------------------------------------------- (0.7.3) line integrals, Example 0.7.2
gam = sp.Matrix([sp.cos(t), sp.sin(t)])
gp = sp.diff(gam, t)
I_xdy = sp.integrate(gam[0] * gp[1], (t, 0, 2 * sp.pi))
check("Example 0.7.2: oint_{S^1} x dy = pi", sp.simplify(I_xdy - sp.pi) == 0)
I_ydx = sp.integrate(-gam[1] * gp[0], (t, 0, 2 * sp.pi))
check("Exercise 0.7.3: oint_{S^1} (-y) dx = pi", sp.simplify(I_ydx - sp.pi) == 0)
# reversing the orientation flips the sign
gr = sp.Matrix([sp.cos(-t), sp.sin(-t)])
check("(0.7.3): reversed orientation gives -pi",
      sp.simplify(sp.integrate(gr[0] * sp.diff(gr[1], t), (t, 0, 2 * sp.pi)) + sp.pi) == 0)
# (0.7.4) fundamental theorem for df along a curve
h = x ** 3 * y + sp.exp(x)
cur = sp.Matrix([t ** 2, 1 + t])
integrand = (sp.diff(h, x).subs({x: cur[0], y: cur[1]}) * sp.diff(cur[0], t)
             + sp.diff(h, y).subs({x: cur[0], y: cur[1]}) * sp.diff(cur[1], t))
lhs = sp.integrate(sp.expand(integrand), (t, 0, 1))
rhs = h.subs({x: 1, y: 2}) - h.subs({x: 0, y: 1})
check("(0.7.4): int_gamma dh = h(end) - h(start)", sp.simplify(lhs - rhs) == 0)

# ---------------------------------------------------------------- wedge product, Figure 0.7.2, Exercise 0.7.2
def wedge(a, b):
    return a[0] * b[1] - a[1] * b[0]


check("Figure 0.7.2: dx^dy(v, w) = 11/4", wedge((2, sp.Rational(1, 2)), (sp.Rational(1, 2), sp.Rational(3, 2))) == sp.Rational(11, 4))
check("Figure 0.7.2: dx^dy(w, v) = -11/4", wedge((sp.Rational(1, 2), sp.Rational(3, 2)), (2, sp.Rational(1, 2))) == -sp.Rational(11, 4))
check("Exercise 0.7.2: dx^dy((1,0),(1,2)) = 2", wedge((1, 0), (1, 2)) == 2)
check("Exercise 0.7.2: dx^dy((1,2),(1,0)) = -2", wedge((1, 2), (1, 0)) == -2)
check("Section wedge: dx^dy(v, v) = 0", wedge((3, 7), (3, 7)) == 0)
check("Section wedge: dx^dy = det",
      sp.Matrix([[2, sp.Rational(1, 2)], [sp.Rational(1, 2), sp.Rational(3, 2)]]).T.det() == sp.Rational(11, 4))

# ---------------------------------------------------------------- (0.7.5) exterior derivative, d∘d = 0
P, Q = sp.Function("P")(x, y), sp.Function("Q")(x, y)
# d(P dx + Q dy) = (Q_x − P_y) dx∧dy; for P dx + Q dy = df this vanishes
ff = sp.Function("f")(x, y)
check("(0.7.5): d(df) = 0 in R^2", sp.simplify(sp.diff(sp.diff(ff, y), x) - sp.diff(sp.diff(ff, x), y)) == 0)
fff = sp.Function("f")(x, y, z)
grad = sp.Matrix([sp.diff(fff, c) for c in (x, y, z)])


def curl(V):
    return sp.Matrix([sp.diff(V[2], y) - sp.diff(V[1], z), sp.diff(V[0], z) - sp.diff(V[2], x),
                      sp.diff(V[1], x) - sp.diff(V[0], y)])


def div(V):
    return sum(sp.diff(V[i], c) for i, c in enumerate((x, y, z)))


Vf = sp.Matrix([sp.Function(n)(x, y, z) for n in ("A", "B", "Cc")])
check("Table (d∘d = 0): curl grad f = 0", sp.simplify(curl(grad)) == sp.zeros(3, 1))
check("Table (d∘d = 0): div curl V = 0", sp.simplify(div(curl(Vf))) == 0)
check("Example 0.7.2 (Stokes): d(x dy) = dx^dy, coefficient 1", sp.diff(x, x) - sp.diff(0, y) == 1)
check("Exercise 0.7.3: d(-y dx) = dx^dy, coefficient 1", sp.diff(0, x) - sp.diff(-y, y) == 1)

# ---------------------------------------------------------------- Theorem 0.7.3 (Stokes) on the unit disk
rr, th = sp.symbols("r theta", positive=True)
area = sp.integrate(sp.integrate(rr, (rr, 0, 1)), (th, 0, 2 * sp.pi))
check("Theorem 0.7.3: int_D dx^dy = pi = oint x dy", sp.simplify(area - I_xdy) == 0)
# Green's theorem for a non-trivial form on the unit square: ω = x²y dx + (x + y³) dy
Pg, Qg = x ** 2 * y, x + y ** 3
inner = sp.integrate(sp.integrate(sp.diff(Qg, x) - sp.diff(Pg, y), (x, 0, 1)), (y, 0, 1))
edges = (sp.integrate(Pg.subs(y, 0), (x, 0, 1)) + sp.integrate(Qg.subs(x, 1), (y, 0, 1))
         - sp.integrate(Pg.subs(y, 1), (x, 0, 1)) - sp.integrate(Qg.subs(x, 0), (y, 0, 1)))
check("Theorem 0.7.3: Green's theorem on [0,1]^2", sp.simplify(inner - edges) == 0)

# ---------------------------------------------------------------- the form dθ on R² ∖ {0}, Figure 0.7.4, Exercise 0.7.4
Pw, Qw = -y / (x ** 2 + y ** 2), x / (x ** 2 + y ** 2)
check("(0.7.7): omega is closed on R^2 minus 0 (Example 0.7.4)", sp.simplify(sp.diff(Qw, x) - sp.diff(Pw, y)) == 0)
Iw = sp.integrate(sp.simplify((Pw * gp[0] + Qw * gp[1]).subs({x: gam[0], y: gam[1]})), (t, 0, 2 * sp.pi))
check("Example 0.7.4: oint_{S^1} omega = 2 pi", sp.simplify(Iw - 2 * sp.pi) == 0)
theta_half = sp.atan(y / x)
check("Exercise 0.7.4: on x > 0, omega = d(arctan(y/x))",
      sp.simplify(sp.diff(theta_half, x) - Pw) == 0 and sp.simplify(sp.diff(theta_half, y) - Qw) == 0)


def loop(c, r, n=40000):
    tt = np.linspace(0, 2 * np.pi, n + 1)
    xx, yy = c[0] + r * np.cos(tt), c[1] + r * np.sin(tt)
    vals = (-yy * (-r * np.sin(tt)) + xx * (r * np.cos(tt))) / (xx ** 2 + yy ** 2)
    return np.sum((vals[1:] + vals[:-1]) / 2 * np.diff(tt))


close("Figure 0.7.4: oint_{gamma_1} omega = 2 pi", loop((0, 0), 1.5), 2 * np.pi, tol=1e-9)
close("Figure 0.7.4: oint_{gamma_2} omega = 0", loop((1.9, 1.1), 0.6), 0.0, tol=1e-9)


# ---------------------------------------------------------------- DEC box: d1 d0 = 0 and discrete cohomology
def complex_from_faces(nv, faces):
    edges = {}
    for fc in faces:
        for a, b in ((fc[0], fc[1]), (fc[1], fc[2]), (fc[2], fc[0])):
            key = (min(a, b), max(a, b))
            edges.setdefault(key, len(edges))
    d0 = np.zeros((len(edges), nv))
    for (a, b), k in edges.items():
        d0[k, a], d0[k, b] = -1, 1                  # (d0 f)(edge a→b) = f(b) − f(a)
    d1 = np.zeros((len(faces), len(edges)))
    for i, fc in enumerate(faces):
        for a, b in ((fc[0], fc[1]), (fc[1], fc[2]), (fc[2], fc[0])):
            k = edges[(min(a, b), max(a, b))]
            d1[i, k] += 1 if a < b else -1          # sum of the edge values around the face boundary
    return d0, d1


def betti(nv, d0, d1):
    r0, r1 = np.linalg.matrix_rank(d0), np.linalg.matrix_rank(d1)
    ne, nf = d0.shape[0], d1.shape[0]
    return nv - r0, (ne - r1) - r0, nf - r1


# octahedron (a sphere)
octa = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
d0, d1 = complex_from_faces(6, octa)
check("CV box (DEC): d1 d0 = 0 on the octahedron", np.allclose(d1 @ d0, 0))
b = betti(6, d0, d1)
check("Section de Rham: sphere mesh has (b0, b1, b2) = (1, 0, 1)", b == (1, 0, 1))
check("Section de Rham: chi(sphere) = 1 - 0 + 1 = 2 = n0 - n1 + n2", 6 - d0.shape[0] + len(octa) == 2 == b[0] - b[1] + b[2])

# 4 × 4 triangulated torus
n = 4
idx = lambda i, j: (i % n) * n + (j % n)  # noqa: E731
tor = []
for i in range(n):
    for j in range(n):
        a, b_, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
        tor += [(a, b_, c), (a, c, d)]
d0, d1 = complex_from_faces(n * n, tor)
check("CV box (DEC): d1 d0 = 0 on the torus mesh", np.allclose(d1 @ d0, 0))
bt = betti(n * n, d0, d1)
check("Section de Rham: torus mesh has (b0, b1, b2) = (1, 2, 1), i.e. dim H^1(T^2) = 2", bt == (1, 2, 1))
check("Section de Rham: chi(torus) = 0", n * n - d0.shape[0] + len(tor) == 0 == bt[0] - bt[1] + bt[2])

summary()
