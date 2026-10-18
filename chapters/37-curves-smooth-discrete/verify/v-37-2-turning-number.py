"""Verification for Section 37.2 (Turning Number and the Whitney-Graustein Theorem).

Run from the project root: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/verify/v-37-2-turning-number.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import ddg37  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "37-2-turning-number.html"
check("Implementation 37.2: at most 30 lines", dgcode.code_lines(PAGE, "code-37-2-1") <= 30)
NS = dgcode.run_block(PAGE, "code-37-2-1")
wn = NS["winding_number"]

# ---------------------------------------------------------------- Example 37.2.1
expected = {"circle": 1, "ellipse": 1, "flower": 1, "figure8": 0, "limacon": 2}
for name, k in expected.items():
    close(f"Example 37.2.1: turning number of {name} = {k}", ddg37.rotation_index_samples(name), k, 1e-9)
t = np.linspace(0, 2 * np.pi, 20001)
cw = np.stack([np.sin(t), np.cos(t)], 1)                         # derivative of (cos t, -sin t)
close("Example 37.2.1: the clockwise circle has turning number -1",
      (ddg37.angle_lift(cw)[-1] - ddg37.angle_lift(cw)[0]) / (2 * np.pi), -1, 1e-9)


# ---------------------------------------------------------------- Prop 37.2.3 (integral formula), Example 37.2.4
def wind_integral(name, p, m=20000, turns=1):
    tt = 2 * np.pi * turns * np.arange(m) / m
    v = ddg37.point(name, tt) - p
    d = ddg37.velocity(name, tt)
    return float(((v[:, 0] * d[:, 1] - v[:, 1] * d[:, 0]) / (v ** 2).sum(1)).sum() * 2 * np.pi * turns / m / (2 * np.pi))


cases = [("circle", (0.0, 0.0), 2, 2), ("circle", (1.5, 0.3), 2, 0), ("figure8", (0.5, 0.0), 1, -1),
         ("figure8", (-0.5, 0.0), 1, 1), ("figure8", (0.0, 0.75), 1, 0), ("limacon", (0.1, 0.0), 1, 2),
         ("limacon", (1.2, 0.0), 1, 1), ("limacon", (2.0, 0.0), 1, 0)]
for name, p, turns, w in cases:
    wi = wind_integral(name, np.array(p), turns=turns)
    P = ddg37.point(name, 2 * np.pi * turns * np.arange(4000) / 4000)
    wp = wn(P, np.array(p))
    close(f"Prop 37.2.3 / Example 37.2.4: wind({name}, {p}) = {w} (integral and polygon)", [wi, wp], [w, w], 1e-8)

# ---------------------------------------------------------------- Prop 37.2.6: rot = wind(gamma', 0)
for name, k in expected.items():
    D = ddg37.velocity(name, 2 * np.pi * np.arange(8000) / 8000)
    close(f"Prop 37.2.6: wind(gamma', 0) = rot = {k} ({name})", wn(D, np.zeros(2)), k, 1e-9)

# ---------------------------------------------------------------- Example 37.2.8, Non-example 37.2.9
tt = np.linspace(0, 1, 2001)
mins = [np.linalg.norm(np.stack([-(2 - lam) * np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt)], 1), axis=1).min()
        for lam in np.linspace(0, 1, 21)]
check("Example 37.2.8(a): ellipse -> circle homotopy is regular (min speed > 0 for all lambda)", min(mins) > 0.99)
a, tsym = sp.symbols("a t", real=True)
lim = sp.Matrix([(a + sp.cos(tsym)) * sp.cos(tsym), (a + sp.cos(tsym)) * sp.sin(tsym)])
sym_equal("Non-example 37.2.9 / Exercise 37.2.4: |gamma_a'|^2 = 1 + a^2 + 2 a cos t",
          lim.diff(tsym).dot(lim.diff(tsym)), 1 + a ** 2 + 2 * a * sp.cos(tsym))
for av, k in ((0.5, 2), (0.8, 2), (1.2, 1), (1.5, 1)):
    tg = np.linspace(0, 2 * np.pi, 20001)
    r = av + np.cos(tg)
    vel = np.stack([-np.sin(tg) * np.cos(tg) - r * np.sin(tg), -np.sin(tg) ** 2 + r * np.cos(tg)], 1)
    rot = (ddg37.angle_lift(vel)[-1] - ddg37.angle_lift(vel)[0]) / (2 * np.pi)
    close(f"Non-example 37.2.9(a): limacon a = {av} has min speed |1 - a| and turning number {k}",
          [np.linalg.norm(vel, axis=1).min(), rot], [abs(1 - av), k], 1e-6)
H = np.stack([-np.sin(2 * np.pi * tt), np.zeros_like(tt)], 1)  # d/dt of (cos 2 pi t, 0) / 2 pi at lambda = 1/2
check("Non-example 37.2.9(b): at lambda = 1/2 the speed vanishes at t = 0 and t = 1/2",
      np.linalg.norm(H[[0, 1000]], axis=1).max() < 1e-12)

# ---------------------------------------------------------------- Theorem 37.2.11 (Proof sketch), Figure 37.2.2(a)
s, phiA, _ = ddg37.unit_length_tangent_angle("limacon")
phiB = phiA[0] + 2 * s
vm, gaps = [], []
for lam in np.linspace(0, 1, 41):
    X, vmin, mean = ddg37.averaged_curve(s, phiA, phiB, lam)
    vm.append(vmin)
    gaps.append(np.linalg.norm(X[-1] - X[0]))
check(f"Theorem 37.2.11 sketch: averaging homotopy limacon -> double circle is regular (min speed {min(vm):.3f} "
      f">= 0.94) and closed", min(vm) >= 0.94 and max(gaps) < 1e-9)
X0, _, m0 = ddg37.averaged_curve(s, phiA, phiB, 0.0)
close("Theorem 37.2.11 sketch: m(0) = 0", m0, [0, 0], 1e-6)
s8, phi8, _ = ddg37.unit_length_tangent_angle("figure8")
phi8m = np.pi - phi8                                              # mirror (x, y) -> (-x, y)
close("Theorem 37.2.11 sketch (k = 0): the mirrored figure-eight has phi_1 = pi - phi_0, so phi_{1/2} = pi/2",
      0.5 * (phi8 + phi8m), np.full_like(phi8, np.pi / 2), 1e-12)
_, vmin8, _ = ddg37.averaged_curve(s8, phi8, phi8m, 0.5)
check("Theorem 37.2.11 sketch (k = 0): the averaged curve at lambda = 1/2 has zero speed", vmin8 < 1e-12)
check("Theorem 37.2.11 sketch (k = 0): the mirror really is the tangent angle of the mirrored curve",
      np.allclose(np.stack([np.cos(phi8m), np.sin(phi8m)], 1),
                  np.stack([-np.cos(phi8), np.sin(phi8)], 1), atol=1e-12))

# ---------------------------------------------------------------- Exercise 37.2.2
tt = np.linspace(0, 2 * np.pi, 200001)
V = np.stack([np.cos(tt), np.cos(2 * tt)], 1)
ang = ddg37.angle_lift(V)
check("Exercise 37.2.2: the lifted angle of gamma' stays in (-3 pi/2, pi/2) starting at pi/4",
      abs(ang[0] - np.pi / 4) < 1e-12 and ang.min() > -1.5 * np.pi and ang.max() < 0.5 * np.pi)

# ---------------------------------------------------------------- Prop 37.2.13 on random polygons
rng = np.random.default_rng(372)
ok = True
for _ in range(30):
    P = rng.normal(size=(9, 2))
    p = rng.normal(size=2) * 0.5
    dense = np.concatenate([P[j] + np.linspace(0, 1, 400, endpoint=False)[:, None] * (P[(j + 1) % 9] - P[j])
                            for j in range(9)])
    dense = np.vstack([dense, dense[:1]]) - p
    lift = ddg37.angle_lift(dense)
    ok &= abs((lift[-1] - lift[0]) / (2 * np.pi) - wn(P, p)) < 1e-9
check("Prop 37.2.13: the angle sum equals the change of a continuous angle along the polygon (30 random cases)", ok)

# ---------------------------------------------------------------- Exercise 37.2.5
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    dgcode.run_block(PAGE, "code-ex-37-2-5", NS)
check("Exercise 37.2.5: printed output matches", buf.getvalue().strip() == dgcode.code_block(PAGE, "out-ex-37-2-5").strip(),
      buf.getvalue())

summary()
