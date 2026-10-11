"""Animation 0.1.2: 종이를 말아도 길이와 각은 그대로이고 normal만 변한다 (Section 0.1, Figure 0.1.2, Definition 0.1.1).

Figure 0.1.2와 같은 종이 0 ≤ x ≤ W = 3π/2, 0 ≤ y ≤ 2, 같은 선분 σ(주황), 같은 삼각형 T(파랑),
같은 세 점(y = 1.3, x = W/2 − 1, W/2 + 0.15, W/2 + 1.15)의 unit normal vector N₁, N₂, N₃(주홍, 0.6배).
여기서는 종이를 탁자 위에 눕혀 놓고(처음 normal은 위쪽), x 방향으로 말아 y축에 평행한 cylinder를 만든다.
말기: 단면의 curvature k를 0에서 1로 올리는 isometric bending
    P_k(x, y) = (sin(kξ)/k,  y − 1,  −(1 − cos kξ)/k),   ξ = x − W/2,
k = 0이면 평면 종이, k = 1이면 반지름 1인 cylinder(Figure 0.1.2의 감는 map과 같은 모양, 축만 다르다).
모든 k에서 |∂P/∂x| = |∂P/∂y| = 1, ⟨∂P/∂x, ∂P/∂y⟩ = 0. N = (sin kξ, 0, cos kξ)는 바깥쪽이다.
화면 오른쪽: intrinsic한 양(σ의 길이, T의 세 내각)은 매 순간 다시 계산해도 그대로이고,
extrinsic한 양(N₁과 N₃ 사이의 각)은 0°에서 k·(ξ₃ − ξ₁) = 2.15 rad ≈ 123°로 변한다.
자기검사: k = 0, 0.5, 1에서 σ의 길이와 T의 변 길이·내각이 평면에서의 값과 같다. k = 1에서 점들이 cylinder 위에 있다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from manim import (LEFT, RIGHT, FadeIn, ValueTracker, VGroup, always_redraw, smooth)

import _helpers_a as ha
import dganim as da

C = da.C
DOWN_ = np.array([0.0, -1.0, 0.0])
W, HS = 1.5 * np.pi, 2.0


def sinc_over(k, xi):
    """sin(kξ)/k, (1 − cos kξ)/k (k → 0에서도 매끄럽다)."""
    s1 = xi * np.sinc(k * xi / np.pi)
    s2 = 0.5 * k * xi ** 2 * np.sinc(k * xi / (2 * np.pi)) ** 2
    return s1, s2


def P(Q, k):
    Q = np.asarray(Q, float)
    s1, s2 = sinc_over(k, Q[..., 0] - W / 2)
    return np.stack([s1, Q[..., 1] - 1.0, -s2], axis=-1)


def Nrm(Q, k):
    xi = np.asarray(Q, float)[..., 0] - W / 2
    return np.stack([np.sin(k * xi), 0 * xi, np.cos(k * xi)], axis=-1)


def plen(X):
    return float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1)))


SEG = np.stack([np.linspace(0.25, W - 0.25, 1201), np.linspace(0.2, HS - 0.25, 1201)], axis=1)
TRI = np.array([[1.35, 0.35], [3.05, 0.55], [2.0, 1.55]])
NPTS = [np.array([W / 2 - 1.0, 1.3]), np.array([W / 2 + 0.15, 1.3]), np.array([W / 2 + 1.15, 1.3])]


def tri_angles(k):
    """T의 세 내각(도): 말린 뒤에는 꼭짓점에서 두 변의 처음 접선으로 잰다."""
    out = []
    h = 1e-6
    for i in range(3):
        p, q, s = TRI[i], TRI[(i + 1) % 3], TRI[(i - 1) % 3]
        a = P(p + h * (q - p), k) - P(p, k)
        b = P(p + h * (s - p), k) - P(p, k)
        out.append(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    return np.array(out)


def n_angle(k):
    n1, n3 = Nrm(NPTS[0], k), Nrm(NPTS[2], k)
    return float(np.degrees(np.arccos(np.clip(n1 @ n3, -1, 1))))


L0 = plen(SEG)
ANG0 = tri_angles(0.0)
for kk in (0.0, 0.5, 1.0):
    assert abs(plen(P(SEG, kk)) - L0) < 1e-4 * L0
    assert np.allclose(tri_angles(kk), ANG0, atol=1e-3)
    for i in range(3):                                   # T의 변(곡선)의 길이
        e = TRI[i] + np.linspace(0, 1, 801)[:, None] * (TRI[(i + 1) % 3] - TRI[i])
        assert abs(plen(P(e, kk)) - plen(e)) < 1e-4
    xs = np.linspace(0, W, 7)
    Q = np.stack([xs, 0.5 + 0 * xs], axis=1)
    d = (P(Q + [1e-6, 0], kk) - P(Q, kk)) / 1e-6
    assert np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-5)          # E = 1: isometric bending
    assert np.allclose(np.einsum("ij,ij->i", d, Nrm(Q, kk)), 0, atol=1e-5)
assert np.isclose(ANG0.sum(), 180, atol=1e-6)
Pc = P(SEG, 1.0)
assert np.allclose(np.hypot(Pc[:, 0], Pc[:, 2] + 1.0), 1.0)              # k = 1: 축이 (0, ·, −1)인 반지름 1의 cylinder
assert abs(n_angle(0.0)) < 1e-6 and abs(n_angle(1.0) - np.degrees(2.15)) < 1e-6
assert abs(L0 - 4.49) < 0.01

CAM = ha.Ortho(40, -55, scale=1.2, center=np.array([-3.5, 0.0, 0.0]), origin=(0.0, 0.0, -0.75))
LB = 0.15                                  # 종이 위의 선을 자기 면보다 나중에 그리기 위한 깊이 bias
NSUB = 8
TRI_SUB = []
for i_ in range(NSUB):
    for j_ in range(NSUB - i_):
        bary = [(i_, j_), (i_ + 1, j_), (i_, j_ + 1)] + ([(i_ + 1, j_), (i_ + 1, j_ + 1), (i_, j_ + 1)]
                                                       if i_ + j_ < NSUB - 1 else [])
        for t3 in (bary[:3], bary[3:]):
            if t3:
                TRI_SUB.append(np.array([TRI[0] + (a * (TRI[1] - TRI[0]) + b * (TRI[2] - TRI[0])) / NSUB
                                         for a, b in t3]))
assert len(TRI_SUB) == NSUB ** 2
GQ = np.stack(np.meshgrid(np.linspace(0, W, 37), np.linspace(0, HS, 9), indexing="ij"), axis=-1)


def scene(k):
    pt = ha.Painter(CAM, ambient=0.55)
    pt.grid(P(GQ, k), color=C["surface"], opacity=0.85)
    yy = np.linspace(0, HS, 9)
    for gx in np.linspace(0, W, 13):
        pt.polyline(P(np.stack([gx + 0 * yy, yy], axis=1), k), "#8A8A8A", 1.6, bias=LB)
    xx = np.linspace(0, W, 49)
    for gy in (0.0, 0.5, 1.0, 1.5, 2.0):
        pt.polyline(P(np.stack([xx, gy + 0 * xx], axis=1), k), "#8A8A8A", 1.6, bias=LB)
    pt.polyline(P(SEG[::20], k), C["accent"], 6, bias=LB + 0.01)
    for tri in TRI_SUB:                                   # T의 면: 작은 삼각형으로 나눠 깊이를 맞춘다
        pt.face(P(tri, k), C["tangent"], opacity=0.3, shade=False, stroke=C["tangent"], stroke_width=0,
                bias=LB + 0.02)
    for i in range(3):
        e = TRI[i] + np.linspace(0, 1, 13)[:, None] * (TRI[(i + 1) % 3] - TRI[i])
        pt.polyline(P(e, k), C["tangent"], 4, bias=LB + 0.03)
    for j, q in enumerate(NPTS):
        b, n = P(q, k), Nrm(q, k)
        pt.arrow(b, 0.6 * n, C["normal"], width=6, tip=0.2, bias=LB)
        pt.dot(b, radius=0.05, bias=LB + 0.05)
        if j != 1:
            lab = da.mtex(rf"$\mathbf{{N}}_{j + 1}$", 30, C["normal"]).move_to(CAM(b + 0.88 * n))
            pt.mob(lab, b + 0.88 * n, bias=0.3)
    # 라벨: T는 아래 변 바로 밑, σ는 선분의 끝 아래. 모든 k에서 선분·변·N과 0.35 이상 떨어진 곳(화면 단위)
    qT = np.array([2.5, 0.05])
    pt.mob(da.mtex(r"$T$", 30, C["tangent"]).move_to(CAM(P(qT, k))), P(qT, k), bias=LB + 0.3)
    qS = np.array([4.4, 1.2])
    pt.mob(da.mtex(r"$\sigma$", 38, C["accent"]).move_to(CAM(P(qS, k))), P(qS, k), bias=1.0)   # 앞면 위의 점이라 가릴 것이 없다
    return pt.build()


class Anim(da.Scene2D):
    def construct(self):
        k = ValueTracker(0.0)
        pic = always_redraw(lambda: scene(k.get_value()))
        x0 = 0.75
        h1 = da.label("intrinsic: unchanged", 32).move_to([x0, 2.4, 0], aligned_edge=LEFT)
        l1 = ha.row("length of", r"$\sigma =$", size=30, color=C["accent"])
        l1.next_to(h1, DOWN_, buff=0.35, aligned_edge=LEFT).shift(0.35 * RIGHT)
        v1 = always_redraw(lambda: da.label(f"{plen(P(SEG, k.get_value())):.2f}", 30, C["accent"])
                           .next_to(l1, RIGHT, buff=0.15))
        l2 = ha.row("angles of", r"$T =$", size=30, color=C["tangent"])
        l2.next_to(l1, DOWN_, buff=0.35, aligned_edge=LEFT)
        v2 = always_redraw(lambda: da.label(", ".join(f"{a:.0f}°" for a in tri_angles(k.get_value())), 30,
                                            C["tangent"]).next_to(l2, RIGHT, buff=0.15))
        h2 = da.label("extrinsic: changes", 32).next_to(l2, DOWN_, buff=0.7, aligned_edge=LEFT).shift(0.35 * LEFT)
        l3 = da.mtex(r"$\angle(\mathbf{N}_1, \mathbf{N}_3) =$", 30, C["normal"])
        l3.next_to(h2, DOWN_, buff=0.35, aligned_edge=LEFT).shift(0.35 * RIGHT)
        v3 = always_redraw(lambda: da.label(f"{n_angle(k.get_value()):.0f}°", 30, C["normal"])
                           .next_to(l3, RIGHT, buff=0.15))
        rad = always_redraw(lambda: da.label(
            "flat paper" if k.get_value() < 0.02 else f"cylinder of radius {1 / k.get_value():.2f}", 30, C["aux"])
            .move_to([-3.5, -2.35, 0]))

        self.add(pic, rad)
        self.play(FadeIn(VGroup(h1, l1, v1, l2, v2, h2, l3, v3)), run_time=1.0)
        self.wait(1.5)
        self.play(k.animate.set_value(1.0), run_time=6.0, rate_func=smooth)
        self.wait(2.5)
        self.play(k.animate.set_value(0.0), run_time=4.0, rate_func=smooth)
        self.wait(1.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=9.5)
