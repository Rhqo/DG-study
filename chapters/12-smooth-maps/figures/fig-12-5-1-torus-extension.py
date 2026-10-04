"""그림 12.5.1: 원환면 위의 함수 cos u를 ℝ³ 전체의 매끄러운 함수로 늘이기 (12.5절, 예 12.5.6).

R = 2, r = 0.8. T_{R,r} 위의 f = cos u = (ρ − R)/r (ρ = √(x² + y²)).
확장 F = ψ(ρ)·(ρ − R)/r, ψ(ρ) = λ((ρ − a)/(b − a)), a = (R − r)/3 = 0.4, b = 2(R − r)/3 = 0.8, λ(s) = 1 − h(1 + s).
평면 y = 0 (ρ = |x|)에서 F의 값을 색(RdBu_r, 0이 가운데)과 등위선 F = −1, −0.5, 0.5, 1(자주)로 그린다.
검은 원 두 개는 원환면의 단면, 회색 점선은 ρ = a와 ρ = b, 가운데 세로 점선은 z축이다.
z축 근처(ρ ≤ a)에서 F = 0이므로 ρ가 매끄럽지 않은 z축에서도 F는 매끄럽다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-5-1-torus-extension.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

C = dgfig.COLORS
LW = dgfig.LW

R, r = 2.0, 0.8
a, b = (R - r) / 3, 2 * (R - r) / 3


def f(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = np.exp(-1.0 / t[m])
    return out


def h(t):
    p, q = f(2 - np.asarray(t, float)), f(np.asarray(t, float) - 1)
    return p / (p + q)


def lam(s):
    return 1 - h(1 + np.asarray(s, float))


def F(x, y, z):
    rho = np.hypot(x, y)
    return lam((rho - a) / (b - a)) * (rho - R) / r


# 자기검사 (§12.1): 원환면 위에서 F = cos u, ρ ≤ a에서 F = 0
rng = np.random.default_rng(125)
u, v = rng.uniform(-np.pi, np.pi, (2, 3000))
P = np.array([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)])
assert np.allclose(F(*P), np.cos(u))
Q = rng.uniform(-1, 1, (3, 3000)) * np.array([[a], [a], [3]])
Qr = np.hypot(Q[0], Q[1])
assert np.all(F(*Q[:, Qr <= a]) == 0)
assert R - r > b > a > 0

xs = np.linspace(-3.2, 3.2, 641)
zs = np.linspace(-1.4, 1.4, 281)
X, Z = np.meshgrid(xs, zs)
V = F(X, 0 * X, Z)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.set_aspect("equal")
ax.imshow(V, extent=(xs[0], xs[-1], zs[0], zs[-1]), origin="lower", cmap="RdBu_r",
          norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.6, vmax=1.6), alpha=0.55, interpolation="bilinear")
cs = ax.contour(X, Z, V, levels=[-1, -0.5, 0.5, 1], colors=C["covector"], linewidths=0.9, linestyles="solid")
t = np.linspace(0, 2 * np.pi, 400)
for s in (-1, 1):
    ax.plot(s * R + r * np.cos(t), r * np.sin(t), color=C["main"], lw=LW["main"])
for x0 in (-b, -a, a, b):
    ax.axvline(x0, color=C["aux"], lw=0.7, ls=(0, (3, 2)))
ax.axvline(0, color=C["main"], lw=0.7, ls=(0, (5, 3)))
ax.text(a + 0.03, -1.32, r"$a$", fontsize=9, color=C["aux"])
ax.text(b + 0.03, -1.32, r"$b$", fontsize=9, color=C["aux"])
ax.text(0.05, 1.18, r"$z$", fontsize=10)
ax.text(R + r + 0.08, 0.05, r"$\cos u=1$", fontsize=9)
ax.annotate(r"$\cos u=-1$", xy=(R - r, 0), xytext=(1.05, 1.05), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=C["main"], lw=0.6))
ax.set_xlabel(r"$x$", fontsize=10)
ax.set_yticks([-1, 0, 1])
ax.tick_params(labelsize=8.5)
fig.tight_layout()
dgfig.save(fig, __file__)
