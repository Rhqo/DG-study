"""Verification for Section 32.1 (What Is Discrete Differential Geometry?).

Run from the project root: PYTHONPATH=tools python3 chapters/32-ddg-overview/verify/v-32-1-what-is-ddg.py
"""
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgdec as dd  # noqa: E402
import ddg32  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "32-1-what-is-ddg.html"


def ngon(n, r=1.0):
    return ddg32.regular_ngon(n, r)


def signed_angle_552(u, v):
    """(5.6.2): angle(u, v) = 2 arctan(det(u, v) / (1 + <u, v>)) for unit u, v (independent of dgdec)."""
    return 2 * np.arctan((u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]) / (1 + (u * v).sum(-1)))


# ---------------------------------------------------------------- Proposition 32.1.1 (a)
n_, r_ = sp.symbols("n r", positive=True)
Ln = 2 * n_ * r_ * sp.sin(sp.pi / n_)
x = sp.symbols("x", positive=True)
# chord length^2 = 2 r^2 (1 - cos 2 pi/n) = 4 r^2 sin^2(pi/n)
sym_equal("Prop 32.1.1(a): |gamma_{i+1} - gamma_i|^2 = 4 r^2 sin^2(pi/n)", 2 * r_ ** 2 * (1 - sp.cos(2 * sp.pi / n_)),
          4 * r_ ** 2 * sp.sin(sp.pi / n_) ** 2, {n_: (3, 50)})
ser = sp.series((2 * sp.pi * r_ - Ln).subs(n_, 1 / x), x, 0, 4).removeO()
sym_equal("Prop 32.1.1(a): 2 pi r - L_n = pi^3 r/(3 n^2) + O(n^-4)", ser, sp.pi ** 3 * r_ * x ** 2 / 3)
ok_len, ok_bound = True, True
for r in (0.5, 1.0, 2.3):
    for n in list(range(3, 60)) + [100, 1000, 5000]:
        X = ngon(n, r)
        L = dd.curve_length(X, True)
        L_hand = np.sum(np.hypot(*(np.roll(X, -1, 0) - X).T))
        ok_len &= abs(L - 2 * n * r * np.sin(np.pi / n)) < 1e-11 and abs(L - L_hand) < 1e-11
        err = 2 * np.pi * r - 2 * n * r * np.sin(np.pi / n)
        ok_bound &= (err > 0) and (err <= np.pi ** 3 * r / (3 * n ** 2) * (1 + 1e-12))
check("Prop 32.1.1(a): L_n = 2 n r sin(pi/n) (dgdec.curve_length and by hand)", ok_len)
check("Prop 32.1.1(a): 0 < 2 pi r - L_n <= pi^3 r / (3 n^2) for n = 3..59, 100, 1000, 5000 and three r", ok_bound)
xs = np.linspace(1e-6, 4, 4001)
check("Prop 32.1.1 proof: x - x^3/6 <= sin x < x on (0, 4]", np.all(xs - xs ** 3 / 6 <= np.sin(xs) + 1e-15)
      and np.all(np.sin(xs) < xs))

# ---------------------------------------------------------------- Proposition 32.1.1 (b)
ok_b, worst = True, 0.0
for n in range(3, 600):
    X = ngon(n, 1.7)
    th = dd.turning_angles(X, True)
    T = dd.curve_tangents(X, True)
    th2 = signed_angle_552(np.roll(T, 1, 0), T)
    ok_b &= np.allclose(th, 2 * np.pi / n, atol=1e-12) and np.allclose(th2, 2 * np.pi / n, atol=1e-12)
    worst = max(worst, abs(th.sum() - 2 * np.pi))
check("Prop 32.1.1(b): every exterior angle = 2 pi/n (dgdec and (5.6.2)), n = 3..599", ok_b)
check(f"Example 32.1.2: sum of exterior angles - 2 pi below 1e-13 (worst {worst:.1e})", worst < 1e-13)

# ---------------------------------------------------------------- Example 32.1.2
e6 = 2 * np.pi - 6.0
close("Example 32.1.2: L_6 = 6", 12 * np.sin(np.pi / 6), 6.0, 1e-12)
close("Example 32.1.2: 2 pi - L_6 = 0.2832", round(e6, 4), 0.2832, 1e-12)
close("Example 32.1.2: pi^3/108 = 0.2871", round(np.pi ** 3 / 108, 4), 0.2871, 1e-12)
L12 = 24 * np.sin(np.pi / 12)
close("Example 32.1.2: L_12 = 6.2117", round(L12, 4), 6.2117, 1e-12)
close("Example 32.1.2: error 0.0715, bound 0.0718", [round(2 * np.pi - L12, 4), round(np.pi ** 3 / 432, 4)],
      [0.0715, 0.0718], 1e-12)
close("Example 32.1.2: (2 pi - L_n) n^2 at n = 6, 12, 96 = 10.19, 10.30, 10.33",
      [round((2 * np.pi - 2 * n * np.sin(np.pi / n)) * n * n, 2) for n in (6, 12, 96)], [10.19, 10.30, 10.33], 1e-12)
close("Example 32.1.2: pi^3/3 = 10.335", round(np.pi ** 3 / 3, 3), 10.335, 1e-12)
ratio = (2 * np.pi - 2 * 24 * np.sin(np.pi / 24)) / (2 * np.pi - L12)
check(f"Example 32.1.2: doubling n divides the error by about 4 (12 -> 24: {1 / ratio:.3f})", abs(1 / ratio - 4) < 0.05)

# ---------------------------------------------------------------- Figure 32.1.1 caption
vals = [round(2 * n * np.sin(np.pi / n) / (2 * np.pi), 3) for n in (3, 6, 12)]
close("Figure 32.1.1: L_n / 2 pi r at n = 3, 6, 12 = 0.827, 0.955, 0.989", vals, [0.827, 0.955, 0.989], 1e-12)

# ---------------------------------------------------------------- Figure 32.1.2: dependencies = GUIDELINES section 9
g = (HERE.parents[3] / "GUIDELINES.md").read_text(encoding="utf-8")
line = next(ln for ln in g.splitlines() if "35 ← 34" in ln)
deps = {}
for m in re.finditer(r"(\d+) ← ([\d·]+)", line):
    deps[int(m.group(1))] = sorted(int(v) for v in m.group(2).split("·"))
want = {35: [34], 36: [33, 35], 37: [32, 36], 38: [35, 36, 37], 39: [38], 40: [33, 39], 41: [40], 42: [36, 41],
        43: [42], 44: [39, 42], 45: [36, 42, 44]}
check("Figure 32.1.2: the drawn dependencies are exactly the GUIDELINES section 9 list", deps == want)
check("Figure 32.1.2: every arrow goes from a smaller to a larger chapter number",
      all(a < b for b, al in deps.items() for a in al))
check("Figure 32.1.2: 37 and 38 use both branches (36 and a curve/surface chapter)",
      36 in deps[37] and 32 in deps[37] and {35, 36, 37} <= set(deps[38]))
check("Figure 32.1.2: 43, 44, 45 all use 42", all(42 in deps[k] for k in (43, 44, 45)))

# ---------------------------------------------------------------- Exercise 32.1.1
close("Exercise 32.1.1: L_8 = 6.1229, error 0.1603, bound 0.1615",
      [round(16 * np.sin(np.pi / 8), 4), round(2 * np.pi - 16 * np.sin(np.pi / 8), 4), round(np.pi ** 3 / 192, 4)],
      [6.1229, 0.1603, 0.1615], 1e-12)

# ---------------------------------------------------------------- Exercise 32.1.2 (circumscribed polygon)
ok = True
for n in range(3, 40):
    R = 1.3 / np.cos(np.pi / n)
    a = 2 * np.pi * np.arange(n) / n + np.pi / n
    Y = R * np.stack([np.cos(a), np.sin(a)], 1)
    E = np.roll(Y, -1, 0) - Y
    mid = Y + 0.5 * E
    ok &= np.allclose(np.linalg.norm(mid, axis=1), 1.3)                     # midpoints on the circle
    ok &= np.allclose((mid * E).sum(1), 0, atol=1e-12)                       # edges tangent at the midpoints
    ok &= abs(dd.curve_length(Y, True) - 2 * n * 1.3 * np.tan(np.pi / n)) < 1e-11
    ok &= abs(dd.turning_angles(Y, True).sum() - 2 * np.pi) < 1e-12
    ok &= 2 * n * 1.3 * np.sin(np.pi / n) < 2 * np.pi * 1.3 < 2 * n * 1.3 * np.tan(np.pi / n)
check("Exercise 32.1.2: circumscribed n-gon, L'_n = 2 n r tan(pi/n), L_n < 2 pi r < L'_n, angle sum 2 pi", ok)
close("Exercise 32.1.2: L'_6 = 4 sqrt 3 = 6.9282", [12 * np.tan(np.pi / 6), round(4 * np.sqrt(3), 4)],
      [4 * np.sqrt(3), 6.9282], 1e-12)

# ---------------------------------------------------------------- Exercise 32.1.3 (area)
ok = True
for n in range(3, 400):
    A = ddg32.polygon_area(ngon(n, 1.0))
    ok &= abs(A - n / 2 * np.sin(2 * np.pi / n)) < 1e-12
    ok &= 0 < np.pi - A <= 2 * np.pi ** 3 / (3 * n * n) * (1 + 1e-12)
check("Exercise 32.1.3: A_n = (n/2) r^2 sin(2 pi/n) (shoelace) and 0 < pi r^2 - A_n <= 2 pi^3 r^2/(3 n^2)", ok)
close("Exercise 32.1.3: A_12 = 3, error 0.1416 <= 0.1435",
      [ddg32.polygon_area(ngon(12)), round(np.pi - 3, 4), round(2 * np.pi ** 3 / 432, 4)], [3.0, 0.1416, 0.1435], 1e-12)

# ---------------------------------------------------------------- Exercise 32.1.5: the printed code and output
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ddg32.run_block(PAGE, "code-ex-32-1-5")
want = ddg32.code_block(PAGE, "out-ex-32-1-5").strip()
check("Exercise 32.1.5: the solution code prints exactly the output shown", buf.getvalue().strip() == want,
      f"got:\n{buf.getvalue()}")

summary()
