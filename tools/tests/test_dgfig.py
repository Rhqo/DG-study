"""dgfig 표본 그림 테스트. 실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 tools/tests/test_dgfig.py``.

출력: tools/tests/out/sample-sphere-tangent-plane.{svg,png}, sample-chart-schematic.{svg,png}
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import Polygon

from dgcheck import check, close, summary
from dgsym import EXAMPLES, unit_normal

OUT = os.path.join(HERE, "out")
C = dgfig.COLORS

# ---------------------------------------------------------------------------
# 그림 1: 구면 S²(1), 점 p, 접평면, x_θ, x_φ, N (dgsym.EXAMPLES에서 계산)
# ---------------------------------------------------------------------------
sph = EXAMPLES["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
X = sph["expr"].subs(r, 1)
Nsym = unit_normal(X, th, ph, sph["positive"])
f_X = sp.lambdify((th, ph), list(X), "numpy")
f_Xth = sp.lambdify((th, ph), list(X.diff(th)), "numpy")
f_Xph = sp.lambdify((th, ph), list(X.diff(ph)), "numpy")
f_N = sp.lambdify((th, ph), list(Nsym), "numpy")

th0, ph0 = np.deg2rad(42), np.deg2rad(22)
p = np.array(f_X(th0, ph0), float)
Xth = np.array(f_Xth(th0, ph0), float)
Xph = np.array(f_Xph(th0, ph0), float)
Np = np.array(f_N(th0, ph0), float)

close("그림 1: |N| = 1", np.linalg.norm(Np), 1.0, tol=1e-12)
close("그림 1: N ⟂ x_θ", Np @ Xth, 0.0, tol=1e-12)
close("그림 1: N ⟂ x_φ", Np @ Xph, 0.0, tol=1e-12)
close("그림 1: p가 S²(1) 위에 있음", np.linalg.norm(p), 1.0, tol=1e-12)
close("그림 1: 바깥쪽 법벡터 N(p) = p", Np, p, tol=1e-12)

SCALE = 0.6  # 벡터 배율 (캡션에 명시할 값)

fig = plt.figure(figsize=(6, 4.5))
ax = dgfig.axes3d(fig, elev=20, azim=-20)
vv = dgfig.view_vector(ax)

# 곡면 (면만; 격자는 숨은선 처리한 위도·경도선으로 따로)
T, P = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60), indexing="ij")
Xs, Ys, Zs = np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)
dgfig.surface(ax, Xs, Ys, Zs, grid=False, alpha=0.30, shade=False)

s = np.linspace(0, 2 * np.pi, 361)
for lat in np.deg2rad([30, 60, 90, 120, 150]):
    pts = np.stack([np.sin(lat) * np.cos(s), np.sin(lat) * np.sin(s), np.full_like(s, np.cos(lat))], 1)
    dgfig.curve3d(ax, pts, role="aux", lw=0.6, visible=dgfig.visible_mask(ax, pts, pts), zorder=2)
tt = np.linspace(0, np.pi, 181)
for lon in np.deg2rad(np.arange(0, 360, 30)):
    pts = np.stack([np.sin(tt) * np.cos(lon), np.sin(tt) * np.sin(lon), np.cos(tt)], 1)
    dgfig.curve3d(ax, pts, role="aux", lw=0.6, visible=dgfig.visible_mask(ax, pts, pts), zorder=2)

# 윤곽선: 시선에 수직인 대원
e1 = np.cross(vv, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(vv, e1)
rim = np.outer(np.cos(s), e1) + np.outer(np.sin(s), e2)
dgfig.curve3d(ax, rim, role="main", lw=1.2, zorder=3)

# 접평면과 벡터
check("그림 1: p가 보이는 쪽에 있음", bool(Np @ vv > 0))
dgfig.tangent_plane(ax, p, Xth, Xph, size=0.62, zorder=4)
dgfig.arrow3d(ax, p, Xth, "tangent", r"$\mathbf{x}_\theta$", scale=SCALE, label_offset=(0.02, 0.0, -0.06))
dgfig.arrow3d(ax, p, Xph, "tangent", r"$\mathbf{x}_\varphi$", scale=SCALE, label_offset=(0.0, 0.05, 0.04))
dgfig.arrow3d(ax, p, Np, "normal", r"$\mathbf{N}(p)$", scale=SCALE, label_offset=(0.05, 0.0, 0.05))
dgfig.point3d(ax, p, r"$p$", label_offset=(-0.02, -0.12, -0.12))
corner = p - 0.62 * Xth / np.linalg.norm(Xth) + 0.62 * Xph / np.linalg.norm(Xph)
ax.text(*(corner + np.array([0.0, 0.02, 0.06])), r"$T_pS^2$", color=C["tangent"], fontsize=12, zorder=20)

dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), p + SCALE * Np * 1.25, zoom=1.35)
fig.subplots_adjust(0, 0, 1, 1)
paths1 = dgfig.save(fig, __file__, png=True, outdir=OUT, stem="sample-sphere-tangent-plane")

# ---------------------------------------------------------------------------
# 그림 2: 개념도 — 다양체 M의 차트 (U, φ)
# ---------------------------------------------------------------------------
fig, (axM, axR) = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw=dict(wspace=0.55))
dgfig.schematic_axes(axM, (-1.45, 1.45), (-1.3, 1.3))
dgfig.schematic_axes(axR, (-1.25, 1.35), (-1.2, 1.3))

M = dgfig.blob(axM, center=(0, 0), radius=1.05, seed=3, fill="surface", fill_alpha=0.35)
U = dgfig.blob(axM, center=(0.28, 0.12), radius=0.45, seed=11, amp=0.18, fill="region", edge="tangent", lw=1.0)
pM = np.array([0.30, 0.10])
axM.plot(*pM, "o", color=C["main"], ms=4, zorder=5)
axM.text(pM[0] + 0.06, pM[1] - 0.14, r"$p$", fontsize=12)
axM.text(-0.08, 0.62, r"$U$", fontsize=12, color=C["tangent"])
axM.text(-0.95, 0.80, r"$M$", fontsize=14)

# ℝ²: 좌표축, 차트의 치역 Û = φ(U) (좌표격자를 영역으로 잘라 그림)
axR.annotate("", xy=(1.3, -0.9), xytext=(-1.05, -0.9),
             arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=dgfig.LW["aux"]))
axR.annotate("", xy=(-0.95, 1.25), xytext=(-0.95, -1.0),
             arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=dgfig.LW["aux"]))
axR.text(1.3, -1.08, r"$x^1$", fontsize=11, color=C["aux"], ha="right")
axR.text(-1.13, 1.18, r"$x^2$", fontsize=11, color=C["aux"])
axR.text(1.05, -0.72, r"$\mathbb{R}^2$", fontsize=13)

Uhat = dgfig.blob(axR, center=(0.15, 0.1), radius=0.72, seed=5, amp=0.10, fill="region", edge="tangent", lw=1.0)
clip = Polygon(Uhat, closed=True, facecolor="none", edgecolor="none")
axR.add_patch(clip)
for c in np.linspace(-1.0, 1.2, 12):
    for line in (axR.plot([c, c], [-1.0, 1.2], color=C["tangent"], lw=0.4, alpha=0.6)
                 + axR.plot([-1.0, 1.2], [c, c], color=C["tangent"], lw=0.4, alpha=0.6)):
        line.set_clip_path(clip)
pR = np.array([0.18, 0.02])
axR.plot(*pR, "o", color=C["main"], ms=4, zorder=5)
axR.text(pR[0] + 0.07, pR[1] - 0.2, r"$\varphi(p)$", fontsize=12)
axR.text(0.62, 0.95, r"$\hat U = \varphi(U)$", fontsize=12, color=C["tangent"], ha="center")

dgfig.map_arrow(fig, axM, axR, r"$\varphi$", xy_from=(0.70, 0.22), xy_to=(-0.40, 0.30), coords="data", rad=-0.35)

check("그림 2: 점 p가 U 안에 있음",
      bool(Polygon(U, closed=True).get_path().contains_point(pM)))
check("그림 2: φ(p)가 Û 안에 있음",
      bool(Polygon(Uhat, closed=True).get_path().contains_point(pR)))
paths2 = dgfig.save(fig, __file__, png=True, outdir=OUT, stem="sample-chart-schematic")

for pth in paths1 + paths2:
    check(f"출력 파일 존재: {os.path.relpath(pth, HERE)}", os.path.getsize(pth) > 0)

summary()
