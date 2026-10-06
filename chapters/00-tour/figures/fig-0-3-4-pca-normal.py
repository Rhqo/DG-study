"""Figure 0.3.4: PCA로 normal을 추정하기 (0.3절, In computer vision 상자, [Hoppe92]의 방법을 2차원에서 다시 계산).

반지름 1인 원의 호(각 0.15π … 0.85π, 거의 같은 간격) 위의 점 70개에 표준편차 0.02의 Gaussian noise를 더한 point cloud (seed 1).
점 p(각 π/2에 가장 가까운 점)의 k = 15 nearest neighbors(주황)로 covariance matrix C = (1/k) Σ (q − c)(q − c)ᵀ를 만든다.
큰 eigenvalue의 eigenvector(파랑)는 tangent 방향, 작은 eigenvalue의 eigenvector(주홍)는 normal 방향의 추정이다.
eigenvector는 부호가 정해지지 않으므로 normal을 양쪽 화살표(실선과 점선)로 그린다. 회색 점선은 참 normal(반지름 방향).
하늘색 ellipse는 반축이 2√λ인 covariance ellipse.
자기검사: 추정한 normal과 참 normal의 각 오차 < 5°, λ_min/λ_max < 0.05.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-3-4-pca-normal.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

C = dgfig.COLORS
rng = np.random.default_rng(1)
n, K, SIG = 70, 15, 0.02
ang = np.linspace(0.15 * np.pi, 0.85 * np.pi, n) + rng.normal(0, 0.006, n)
P = np.stack([np.cos(ang), np.sin(ang)], axis=1) + SIG * rng.normal(size=(n, 2))

i0 = int(np.argmin(np.abs(ang - np.pi / 2)))
p = P[i0]
d = np.linalg.norm(P - p, axis=1)
nb = np.argsort(d)[:K]
Q = P[nb]
c = Q.mean(axis=0)
Cov = (Q - c).T @ (Q - c) / K
lam, vec = np.linalg.eigh(Cov)              # 오름차순
nrm, tng = vec[:, 0], vec[:, 1]
true_n = np.array([np.cos(ang[i0]), np.sin(ang[i0])])
err = np.degrees(np.arccos(min(1.0, abs(nrm @ true_n))))
assert err < 5, err
assert lam[0] / lam[1] < 0.05, lam
print(f"normal error {err:.2f} deg, eigenvalues {lam}")

fig = plt.figure(figsize=(5.6, 3.3))
ax = fig.add_axes([0, 0, 1, 1])
t = np.linspace(0.12 * np.pi, 0.88 * np.pi, 300)
ax.plot(np.cos(t), np.sin(t), color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.plot(*P.T, "o", color=C["main"], ms=2.6, zorder=3)
ax.plot(*Q.T, "o", color=C["accent"], ms=4.2, zorder=4)
ell = Ellipse(c, 4 * np.sqrt(lam[1]), 4 * np.sqrt(lam[0]), angle=np.degrees(np.arctan2(tng[1], tng[0])),
              fc=C["region"], ec=C["tangent"], alpha=0.35, lw=0.8, zorder=2)
ax.add_patch(ell)
L = 0.33
kw = dict(arrowstyle="-|>", lw=1.6, mutation_scale=12, shrinkA=0, shrinkB=0)
ax.annotate("", xy=c + L * tng, xytext=c, arrowprops=dict(color=C["tangent"], **kw), zorder=6)
ax.annotate("", xy=c + L * nrm, xytext=c, arrowprops=dict(color=C["normal"], **kw), zorder=6)
ax.annotate("", xy=c - L * nrm, xytext=c, arrowprops=dict(color=C["normal"], ls=(0, (3, 2)), **kw), zorder=6)
ax.plot([0.55 * true_n[0], 1.45 * true_n[0]], [0.55 * true_n[1], 1.45 * true_n[1]], color=C["aux"], lw=0.8,
        ls=(0, (2, 2)), zorder=1)
ax.plot(*c, "o", color=C["main"], ms=4, zorder=7)
s_n = np.sign(nrm @ true_n)
ax.text(*(c + 1.12 * L * nrm * s_n + np.array([0.05, 0.0])), r"$\pm\,\mathbf{e}_{\min}$", fontsize=12,
        color=C["normal"])
ax.text(*(c + 1.05 * L * tng + np.array([0.0, 0.04])), r"$\mathbf{e}_{\max}$", fontsize=12, color=C["tangent"])
dgfig.schematic_axes(ax, (-1.05, 1.05), (0.25, 1.5))

dgfig.save(fig, __file__)
