"""Verification for Section 47.6 (Discrete Approximation and the Cotangent Formula).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/verify/v-47-6-discrete-approximation-cotangent.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import importlib.util  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var47 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

# ---------------------------------------------------------------- Example 47.6.4 / Figure 47.6.1: double bubble
db = v.double_bubble_2d(0.6, 1.0)
arcs = db["arcs"]
L = sum(r * (b - a) for c, r, a, b in arcs)
close("Example 47.6.4: double bubble (0.6, 1): length 8.2549, r3 = 1.5, junction (0.0688, 0.5960)",
      [L, db["r3"], *db["p"]], [8.2549, 1.5, 0.0688, 0.5960], 5e-5)
# 120 degree angles at the junction
p = db["p"]
tang = []
for c, r, a0, a1 in arcs:
    d = p - c
    t = np.array([-d[1], d[0]]) / r
    tang.append(t)
ang = [abs(abs(tang[i] @ tang[j]) - 0.5) for i in range(3) for j in range(i + 1, 3)]
check("Example 47.6.4: the three arcs meet at 120 degrees (|cos| = 1/2 between tangent lines)", max(ang) < 1e-12)
spec = importlib.util.spec_from_file_location("f461", HERE.parents[1] / "figures" / "fig-47-6-1-cell-approximation.py")
f461 = importlib.util.module_from_spec(spec)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import dgfig  # noqa: E402

_save = dgfig.save
dgfig.save = lambda *a, **k: None          # run the figure computation without writing the file
spec.loader.exec_module(f461)
dgfig.save = _save
r01 = f461.res[1]
check("Figure 47.6.1(a): 103 cells meet the curve for a = 0.1, 21 of them singular",
      len(r01["cells"]) == 103 and sum(r01["kinds"].values()) == 21)
last = f461.res[-1]
close("Figure 47.6.1(b): a = 0.00625: position 0.0730, regular planes 0.0134, singular planes 0.0065",
      [last["pos"], last["reg"], last["sing"]], [0.0730, 0.0134, 0.0065], 1e-4)
for name, arr in (("position", f461.pos), ("regular planes", f461.reg), ("sum", f461.tot)):
    sl = np.polyfit(np.log(f461.As[1:]), np.log(arr[1:]), 1)[0]
    close(f"Figure 47.6.1(b): {name} term has slope about 1", sl, 1.0, 0.12)
close("Figure 47.6.1(b): regular plane term ~ 2.1 a for small a", f461.reg[-1] / f461.As[-1], 2.1, 0.1)

# min over lines of sum w |sin(theta - alpha)| is attained at a kink: brute force check
rng = np.random.default_rng(6)
for _ in range(20):
    th = rng.uniform(0, np.pi, 7)
    ww = rng.uniform(0.1, 1, 7)
    al = np.linspace(0, np.pi, 200001)
    brute = (np.abs(np.sin(th[None, :] - al[:, None])) @ ww).min()
    kink = (np.abs(np.sin(th[None, :] - th[:, None])) @ ww).min()
    ok_kink = kink <= brute + 1e-12 and brute - kink < 1e-4
    if not ok_kink:
        break
check("lib cell_planes: the minimum over alpha is attained at one of the theta_k (concavity between kinks)", ok_kink)
close("operator norm of the difference of two line projections at angle psi is |sin psi|",
      np.linalg.norm(np.outer([1, 0], [1, 0]) - np.outer([math.cos(0.3), math.sin(0.3)], [math.cos(0.3), math.sin(0.3)]), 2),
      math.sin(0.3), 1e-12)

# ---------------------------------------------------------------- Prop 47.6.7, Remark 47.6.8, Figure 47.6.2
spec2 = importlib.util.spec_from_file_location("f462", HERE.parents[1] / "figures" / "fig-47-6-2-hat-function.py")
f462 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(f462)
x, V = f462.one_ring()
terms, cents, area, Lx = f462.hat_terms(x, V)
dV = terms.sum(0)
close("Prop 47.6.7: delta V_T(phi_x) = -(L x)_x (sphere one-ring)", dV, -Lx, 1e-14)
close("Example 47.6.9: delta V_T(phi_x) = (-6.3e-5, -4.8e-5, 0.2061)", dV, [-6.3e-5, -4.8e-5, 0.2061], 5e-5)
close("Example 47.6.9: ||V_T||(phi_x) = Area/3 = 0.1037", area / 3, 0.1037, 5e-5)
H = -dV / (area / 3)
close("Example 47.6.9: |H_{V_T}(x)| = 1.987 (exact 2)", np.linalg.norm(H), 1.987, 5e-4)


# hat function integral over a triangle is Area/3 and its gradient is (x - y)/h^2 (finite differences)
def hat_check(x, w, vv, n=600):
    # barycentric grid on the triangle, value = barycentric coordinate of x
    s = (np.arange(n) + 1 / 3) / n
    S, T = np.meshgrid(s, s)
    m = S + T < 1
    lam_w, lam_v = S[m], T[m]
    lam_x = 1 - lam_w - lam_v
    A = 0.5 * np.linalg.norm(np.cross(w - x, vv - x))
    return lam_x.mean() * A, A


x0, w0, v0 = np.array([0.1, 0.2, 0.9]), np.array([1.0, 0.0, 0.0]), np.array([0.3, 1.1, -0.2])
I, A0 = hat_check(x0, w0, v0)
close("Exercise 47.6.1: int_F phi_x = Area(F)/3", I, A0 / 3, 2e-3 * A0)
# face term equals -(1/2)(cot(alpha)(v - x) + cot(gamma)(w - x)) with alpha at w, gamma at v
t = (v0 - w0) / np.linalg.norm(v0 - w0)
y = w0 + ((x0 - w0) @ t) * t
h = np.linalg.norm(x0 - y)
face = A0 * (x0 - y) / h ** 2
cot = f462.cot
close("(7.4): Area(F)(x - y)/h^2 = -(|w - v|/(2h))(y - x)", face, -(np.linalg.norm(w0 - v0) / (2 * h)) * (y - x0), 1e-12)
close("(7.4)-(7.5): face term = -(1/2)(cot alpha (v - x) + cot gamma (w - x))", face,
      -0.5 * (cot(w0, x0, v0) * (v0 - x0) + cot(v0, x0, w0) * (w0 - x0)), 1e-12)
close("(7.4): y - x = (|w-x| cos alpha / |w-v|)(v - x) + (|v-x| cos gamma / |w-v|)(w - x)", y - x0,
      (np.linalg.norm(w0 - x0) * ((x0 - w0) @ (v0 - w0)) / (np.linalg.norm(x0 - w0) * np.linalg.norm(v0 - w0))
       / np.linalg.norm(w0 - v0)) * (v0 - x0)
      + (np.linalg.norm(v0 - x0) * ((x0 - v0) @ (w0 - v0)) / (np.linalg.norm(x0 - v0) * np.linalg.norm(w0 - v0))
         / np.linalg.norm(w0 - v0)) * (w0 - x0), 1e-12)

# area gradient: delta V_T(phi_x) is the gradient of the total area with respect to x (finite differences)
def total_area(xx):
    return sum(0.5 * np.linalg.norm(np.cross(V[k] - xx, V[(k + 1) % 6] - xx)) for k in range(6))


g = np.array([(total_area(x + 1e-6 * e) - total_area(x - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
close("Prop 47.6.7: delta V_T(phi_x) = grad_x Area (Theorem 41.2.7)", dV, g, 1e-8)

# hexagonal one-ring of radius 1, apex height 0.4 (Remark 46.5.13)
az = np.radians(np.arange(6) * 60.0)
Vh = np.stack([np.cos(az), np.sin(az), np.zeros(6)], 1)
xh = np.array([0.0, 0.0, 0.4])
th_, _, ah, _ = f462.hat_terms(xh, Vh)
close("Remark 46.5.13 recalled: hexagon apex 0.4: delta V = (0, 0, 1.2579), Area/3 = 0.9539", [*th_.sum(0), ah / 3],
      [0, 0, 1.2579, 0.9539], 5e-5)

summary()
