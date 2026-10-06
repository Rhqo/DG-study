"""0.1절 What Is Differential Geometry?: 본문과 Quick check의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-1-what-is-dg.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
h = sp.symbols("h", real=True)

# (0.1.1): unit-speed curve의 linear approximation 오차 = (κ h²/2) n + O(h³) -------------------
# helix의 arc length parametrization으로 확인한다 (κ = a/(a²+b²), Example 4.3.10과 같은 매개화).
hel = EX["helix"]
(t,) = hel["coords"]
a, b = hel["params"]
c = sp.sqrt(a ** 2 + b ** 2)
s = sp.symbols("s", real=True)
gam = hel["expr"].subs(t, s / c)                    # arc length parametrization
T, Nn, B = dgsym.frenet_frame(gam, s)
kappa, _ = dgsym.curvature_torsion(gam, s)
sym_equal("(0.1.1): helix κ = a/(a²+b²)", kappa, hel["expected"]["kappa"], hel["domain"])
err = gam.subs(s, s + h) - gam - h * gam.diff(s)
lead = err.applyfunc(lambda e: sp.series(e, h, 0, 3).removeO())
sym_equal("(0.1.1): 오차의 h² 항 = (κ/2) n h²", lead, (kappa / 2) * Nn * h ** 2, {**hel["domain"], s: (0, 5)})

# Figure 0.1.1: ellipse (2cos t, sin t), p = γ(π/3)에서 κ와 상자 B 가장자리의 차이 ---------------------
tt = sp.symbols("t", real=True)
ell = sp.Matrix([2 * sp.cos(tt), sp.sin(tt), 0])
ks = dgsym.signed_curvature(ell, tt)
k0 = float(ks.subs(tt, sp.pi / 3))
close("Figure 0.1.1: p에서 κ ≈ 0.34", k0, 2 / 3.25 ** 1.5, 1e-12)
check("Figure 0.1.1: κ = 0.341…", abs(k0 - 0.3413) < 5e-4)
check("Figure 0.1.1: h = 0.1에서 κh²/2 ≈ 0.0017", abs(k0 * 0.01 / 2 - 0.0017) < 5e-5)

# Exercise 0.1.2: 반지름 r인 원, tangent line까지의 거리 r(1 − cos(h/r)) = h²/(2r) + O(h⁴) ----------
r = sp.symbols("r", positive=True)
circ = sp.Matrix([r * sp.cos(s / r), r * sp.sin(s / r), 0])      # unit speed
kc, _ = dgsym.curvature_torsion(circ, s)
sym_equal("Exercise 0.1.2: circle κ = 1/r", kc, 1 / r)
dist = r * (1 - sp.cos(h / r))
check("Exercise 0.1.2: tangent line(x = r)까지의 거리 = r − r cos(h/r)",
      sp.simplify((sp.Matrix([r, 0, 0]) - circ.subs(s, h))[0] - dist) == 0)
sym_equal("Exercise 0.1.2: r(1 − cos(h/r)) = h²/(2r) + O(h⁴)",
          sp.series(dist, h, 0, 4).removeO(), h ** 2 / (2 * r))
check("Exercise 0.1.2: r = 1, h = 0.1에서 거리 ≈ 0.005",
      abs(float(dist.subs({r: 1, h: sp.Rational(1, 10)})) - 0.005) < 1e-5)

# Exercise 0.1.1(e): 감은 종이에서 x = 0과 x = 3π/2(같은 높이)의 직선거리 √2, 종이 위의 거리 3π/2 ----------
P0 = np.array([np.cos(0.0), np.sin(0.0)])
P1 = np.array([np.cos(1.5 * np.pi), np.sin(1.5 * np.pi)])
close("Exercise 0.1.1(e): chord = √2", np.linalg.norm(P1 - P0), np.sqrt(2), 1e-12)
check("Exercise 0.1.1(e): 3π/2 ≈ 4.71, √2 ≈ 1.41", abs(1.5 * np.pi - 4.71) < 5e-3 and abs(np.sqrt(2) - 1.41) < 5e-3)

summary()
