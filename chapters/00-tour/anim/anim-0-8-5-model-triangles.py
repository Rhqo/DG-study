"""Animation 0.8.5: 세 model space에서 같은 크기의 geodesic triangle이 커질 때 내각의 합 (0.8절, (0.8.7), Figure 0.8.3).

세 패널에 정삼각형 geodesic triangle을 그린다. 중심에서 세 vertex까지의 거리(geodesic distance)를 모두 ρ로 같게 하고
ρ를 0.15 → 1.2로 키운다.
  (a) unit sphere, K = 1: 중심은 화면 정면을 향하는 점, 변은 great circle의 호.
  (b) Euclidean plane, K = 0: 보통의 정삼각형.
  (c) hyperbolic plane 𝕌², K = −1: 중심 (0, 1). Poincaré disk에서 중심이 0이고 vertex가 반지름 tanh(ρ/2)인 정삼각형을
      Cayley 변환 z ↦ i(1 + z)/(1 − z)로 𝕌²에 옮긴 것이다(등거리사상). 변은 x축 위에 중심을 둔 반원의 호.
내각의 합: (a) π보다 크고, (b) π, (c) π보다 작다. (0.8.7)에 의해 (a)는 π + 넓이, (c)는 π − 넓이다.
자기검사: (c)의 vertex가 (0, 1)에서 hyperbolic distance ρ, 세 변의 끝이 같은 반원 위, 넓이(Stokes ∮ dx/y, Figure 0.8.3과
같은 방법) = π − 내각의 합, (a) 내각의 합 − π = solid angle 공식의 넓이, (b) 60° × 3.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-8-5-model-triangles.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, graticule, great_arc, polyline, sphere_disk  # noqa: E402

C = da.C
FILL = ManimColor("#BCDDF1")
BETA = np.deg2rad([90, 210, 330])

# ---------------------------------------------------------------- (a) sphere
# 아래 90px(멈춤 상태의 control bar)를 비우려고 세 패널을 올렸다(§12.8)
VIEW = View(elev=25, azim=-50, scale=1.8, center=np.array([-4.65, 0.45, 0]))
CEN, U1, U2 = VIEW.d, VIEW.right, VIEW.up


def sph_vertices(rho):
    return [np.cos(rho) * CEN + np.sin(rho) * (np.cos(t) * U1 + np.sin(t) * U2) for t in BETA]


def sph_angles(rho):
    V = sph_vertices(rho)
    out = []
    for k in range(3):
        A, B, Cc = V[k], V[(k + 1) % 3], V[(k + 2) % 3]
        tb, tc = B - (B @ A) * A, Cc - (Cc @ A) * A
        out.append(np.arccos(np.clip(tb @ tc / np.linalg.norm(tb) / np.linalg.norm(tc), -1, 1)))
    return out


def sph_area(rho):
    A, B, Cc = sph_vertices(rho)
    return 2 * np.arctan2(abs(A @ np.cross(B, Cc)), 1 + A @ B + B @ Cc + Cc @ A)


# ---------------------------------------------------------------- (c) hyperbolic plane U²
def cayley(z):
    return 1j * (1 + z) / (1 - z)


def hyp_vertices(rho):
    r = np.tanh(rho / 2)
    # Cayley 변환은 disk의 +1 방향을 𝕌²의 위쪽(∞)으로 보내므로, 각 0°, 120°, 240°의 vertex가 위·왼쪽 아래·오른쪽 아래로 간다
    return [cayley(r * np.exp(1j * t)) for t in np.deg2rad([0, 120, 240])]


def d_hyp(p, q):
    return np.arccosh(1 + abs(p - q) ** 2 / (2 * p.imag * q.imag))


def hyp_arc(P, Q, n=120):
    if abs(P.real - Q.real) < 1e-9:
        y = np.linspace(P.imag, Q.imag, n)
        return P.real + 1j * y
    c = (abs(P) ** 2 - abs(Q) ** 2) / (2 * (P.real - Q.real))
    a0, a1 = np.angle(P - c), np.angle(Q - c)
    t = np.linspace(a0, a1, n)
    return c + abs(P - c) * np.exp(1j * t)


def hyp_tangent(P, Q):
    """P에서 Q로 가는 geodesic의 P에서의 단위 접벡터(복소수). 반원이면 반지름에 수직, 세로선이면 ±i."""
    if abs(P.real - Q.real) < 1e-12:
        return 1j * np.sign(Q.imag - P.imag)
    c = (abs(P) ** 2 - abs(Q) ** 2) / (2 * (P.real - Q.real))
    t = 1j * (P - c) / abs(P - c)
    step = hyp_arc(P, Q, 3)[1] - P                     # 호를 따라 가는 방향으로 부호를 정한다
    return t if (t.conjugate() * step).real > 0 else -t


def hyp_angles(rho):
    """𝕌²의 metric은 conformal이므로 내각은 Euclidean 각과 같다."""
    V = hyp_vertices(rho)
    out = []
    for k in range(3):
        A, B, Cc = V[k], V[(k + 1) % 3], V[(k + 2) % 3]
        out.append(abs(np.angle(hyp_tangent(A, Cc) / hyp_tangent(A, B))))
    return out


def hyp_area(rho):
    V = hyp_vertices(rho)
    tot = 0.0
    for k in range(3):
        pts = hyp_arc(V[k], V[(k + 1) % 3], 20001)
        mid = (pts[1:] + pts[:-1]) / 2
        tot += np.sum(np.diff(pts.real) / mid.imag)
    return abs(tot)


# ---------------------------------------------------------------- 자기검사
for rho in (0.3, 0.7, 1.2):
    for v in hyp_vertices(rho):
        assert np.isclose(d_hyp(1j, v), rho)                                   # 중심 (0, 1)에서 거리 ρ
    sh = sum(hyp_angles(rho))
    assert sh < np.pi and np.isclose(hyp_area(rho), np.pi - sh, atol=1e-5)     # Gauss–Bonnet, K = −1
    ss = sum(sph_angles(rho))
    assert ss > np.pi and np.isclose(ss - np.pi, sph_area(rho))                # Gauss–Bonnet, K = 1
    V = hyp_vertices(rho)
    for k in range(3):
        P, Q = V[k], V[(k + 1) % 3]
        if abs(P.real - Q.real) > 1e-9:
            c = (abs(P) ** 2 - abs(Q) ** 2) / (2 * (P.real - Q.real))
            assert np.isclose(abs(P - c), abs(Q - c))
assert np.allclose(hyp_angles(1.2), hyp_angles(1.2)[0])                      # 정삼각형

PL_C, PL_S = np.array([0.0, 0.45, 0]), 1.55
HY_O, HY_S = np.array([4.75, -1.6, 0]), 1.2


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        rho = ValueTracker(0.15)
        self.add(sphere_disk(v, fill_opacity=0.25), graticule(v, n_lat=5, n_lon=12, color="#C4C4C4", width=1.0))
        bnd = Line(HY_O + LEFT * 2.0, HY_O + RIGHT * 2.0, stroke_color=C["main"], stroke_width=4)
        # y = 0 라벨은 왼쪽 끝에 둔다(오른쪽 아래에는 마지막에 "< 180°"가 나온다)
        self.add(bnd, da.mtex(r"$y = 0$", 30).next_to(HY_O + LEFT * 2.0, DOWN, buff=0.1).shift(RIGHT * 0.4))
        heads = VGroup(da.label("sphere, K = 1", 32).move_to([-4.65, 2.75, 0]),
                       da.label("plane, K = 0", 32).move_to([0.0, 2.75, 0]),
                       da.label("hyperbolic, K = −1", 32).move_to([4.75, 2.75, 0]))
        self.add(heads)

        def tris():
            r = rho.get_value()
            g = VGroup()
            # (a)
            V = sph_vertices(r)
            sides = [great_arc(V[k], V[(k + 1) % 3], 60) for k in range(3)]
            g.add(Polygon(*v.p(np.vstack(sides)), stroke_width=0, fill_color=FILL, fill_opacity=1))
            for s_ in sides:
                g.add(polyline(v.p(s_), C["accent"], 5))
            g.add(*[Dot(v.p(x), radius=0.06) for x in V])
            # (b)
            P = [PL_C + PL_S * r * np.array([np.cos(t), np.sin(t), 0]) for t in BETA]
            g.add(Polygon(*P, stroke_color=C["accent"], stroke_width=5, fill_color=FILL, fill_opacity=1))
            g.add(*[Dot(p, radius=0.06) for p in P])
            # (c)
            H = hyp_vertices(r)
            arcs = [hyp_arc(H[k], H[(k + 1) % 3], 80) for k in range(3)]
            pts = [HY_O + HY_S * np.array([z.real, z.imag, 0]) for a in arcs for z in a]
            g.add(Polygon(*pts, stroke_width=0, fill_color=FILL, fill_opacity=1))
            for a in arcs:
                g.add(polyline([HY_O + HY_S * np.array([z.real, z.imag, 0]) for z in a], C["accent"], 5))
            g.add(*[Dot(HY_O + HY_S * np.array([z.real, z.imag, 0]), radius=0.06) for z in H])
            g.add(Dot(HY_O + HY_S * np.array([0, 1, 0]), radius=0.04, color=C["aux"]))
            return g

        self.add(always_redraw(tris))

        def sums():
            r = rho.get_value()
            vals = (np.rad2deg(sum(sph_angles(r))), 180.0, np.rad2deg(sum(hyp_angles(r))))
            g = VGroup(*[da.label(f"angle sum {val:.1f}°", 32).move_to([x, -2.75, 0])
                         for val, x in zip(vals, (-4.65, 0.0, 4.75))])
            return g

        self.add(always_redraw(sums))
        rl = always_redraw(lambda: da.label(f"distance center → vertex: ρ = {rho.get_value():.2f}", 32)
                           .to_edge(UP, buff=0.2))
        self.add(rl)
        self.wait(1.5)
        self.play(rho.animate.set_value(0.7), run_time=4.0, rate_func=smooth)
        self.wait(1.5)
        self.play(rho.animate.set_value(1.2), run_time=4.0, rate_func=smooth)
        msg = VGroup(da.label("> 180°", 34).move_to([-4.65, -2.2, 0]),
                     da.label("= 180°", 34).move_to([0.0, -2.2, 0]),
                     da.label("< 180°", 34).move_to([4.75, -2.2, 0]))
        self.play(FadeIn(msg), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=12.0)
