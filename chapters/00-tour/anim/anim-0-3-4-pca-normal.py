"""Animation 0.3.4: PCA normal estimation — 이웃의 수 k에 따라 covariance ellipse와 normal 추정이 바뀐다
(Section 0.3, In computer vision 상자, Figure 0.3.4, Exercise 0.3.3, [Hoppe92]의 방법을 2차원에서 다시 계산).

Figure 0.3.4와 같은 point cloud: 반지름 1인 원의 호(각 0.15π … 0.85π) 위의 점 70개 + 표준편차 0.02의 noise (seed 1).
점 p(각 π/2에 가장 가까운 점)의 k nearest neighbors(주황)로 C = (1/k) Σ (q − c)(q − c)ᵀ.
하늘색 ellipse: 반축 2√λ (covariance ellipse). 파랑 e_max, 주홍 ±e_min (실선과 점선: 부호는 정해지지 않는다),
회색 점선: 참 normal(반지름 방향). 파란 가는 선: c를 지나는 맞춘 직선(2차원의 "tangent plane").
k를 3에서 15(Figure 0.3.4)로, 다시 60으로 키운다.
  작은 k: 이웃이 noise 크기 정도라 e_min의 방향이 흔들린다(오차 수 도에서 14°).
  k = 15: 오차 약 1.8° (Figure 0.3.4). 큰 k: 이웃이 호를 따라 휘어 λ_min/λ_max가 다시 커지고
  c가 호 안쪽으로 들어간다(맞춘 직선이 surface에서 떨어진다). curvature가 이웃 크기 안에서 보이기 시작한 것이다.
자기검사: k = 15에서 오차 < 5°, λ_min/λ_max < 0.05 (Figure 0.3.4와 같다). eigh의 e_min은 Σ⟨q − c, n⟩²를 최소로 한다.
k = 60에서 λ_min/λ_max가 k = 20보다 크다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from manim import (LEFT, RIGHT, DashedLine, Dot, Line, ValueTracker, VGroup, always_redraw, linear, Ellipse)

import _helpers_a as ha
import dganim as da

C = da.C
rng = np.random.default_rng(1)
n, SIG = 70, 0.02
ang = np.linspace(0.15 * np.pi, 0.85 * np.pi, n) + rng.normal(0, 0.006, n)
PTS = np.stack([np.cos(ang), np.sin(ang)], axis=1) + SIG * rng.normal(size=(n, 2))
I0 = int(np.argmin(np.abs(ang - np.pi / 2)))
ORDER = np.argsort(np.linalg.norm(PTS - PTS[I0], axis=1))
TRUE_N = np.array([np.cos(ang[I0]), np.sin(ang[I0])])


def pca(k):
    Q = PTS[ORDER[:k]]
    c = Q.mean(axis=0)
    Cov = (Q - c).T @ (Q - c) / k
    lam, vec = np.linalg.eigh(Cov)
    return Q, c, lam, vec[:, 0], vec[:, 1]


def err_deg(nv):
    return float(np.degrees(np.arccos(min(1.0, abs(nv @ TRUE_N)))))


Q15, c15, lam15, n15, t15 = pca(15)
assert err_deg(n15) < 5 and lam15[0] / lam15[1] < 0.05
assert abs(err_deg(n15) - 1.8) < 0.05                                    # Figure 0.3.4: 약 1.8°
th_ = np.linspace(0, np.pi, 3601)                                        # e_min이 Σ⟨q − c, n⟩²의 최소
cost = [np.sum(((Q15 - c15) @ np.array([np.cos(a), np.sin(a)])) ** 2) for a in th_]
amin = th_[int(np.argmin(cost))]
assert abs(abs(np.array([np.cos(amin), np.sin(amin)]) @ n15) - 1) < 1e-5
assert pca(60)[2][0] / pca(60)[2][1] > pca(20)[2][0] / pca(20)[2][1]
assert err_deg(pca(3)[3]) > 10

SC = 5.0
OFF = np.array([-1.2, -4.45, 0.0])


def scr(p):
    return OFF + SC * np.array([p[0], p[1], 0.0])


def frame(kf):
    k = int(round(kf))
    Q, c, lam, nv, tv = pca(k)
    if nv @ TRUE_N < 0:
        nv = -nv
    g = VGroup()
    ang_e = np.degrees(np.arctan2(tv[1], tv[0]))
    e = Ellipse(width=SC * 4 * np.sqrt(lam[1]), height=max(SC * 4 * np.sqrt(lam[0]), 0.01), color=C["tangent"],
                stroke_width=2, fill_color=C["region"], fill_opacity=0.35).rotate(np.radians(ang_e)).move_to(scr(c))
    g.add(e)
    g.add(Line(scr(c - 0.75 * tv), scr(c + 0.75 * tv), color=C["tangent"], stroke_width=2.5))
    g.add(ha.dots_mobject([scr(q) for q in Q], size=0.13, color=C["accent"]))
    L = 0.33
    g.add(ha.arrow2(scr(c), scr(c + L * tv), C["tangent"], width=7, tip=0.26))
    g.add(ha.arrow2(scr(c), scr(c + L * nv), C["normal"], width=7, tip=0.26))
    neg = ha.arrow2(scr(c), scr(c - L * nv), C["normal"], width=6, tip=0.26)
    g.add(neg.set_opacity(0.55))
    g.add(Dot(scr(c), radius=0.06, color=C["main"]))
    g.add(da.mtex(r"$\pm\,\mathbf{e}_{\min}$", 32, C["normal"]).move_to(scr(c + 1.2 * L * nv) + [0.55, 0.0, 0]))
    g.add(da.mtex(r"$\mathbf{e}_{\max}$", 32, C["tangent"]).move_to(scr(c + 1.05 * L * tv) + [0.05, 0.32, 0]))
    x0 = 2.2
    g.add(ha.row(r"$k =$", f"{k}", size=34).move_to([x0, 3.35, 0], aligned_edge=LEFT))
    g.add(ha.row("normal error =", f"{err_deg(nv):.1f}°", size=30, color=C["normal"])
          .move_to([x0, 2.65, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$\lambda_{\min}/\lambda_{\max} =$", f"{lam[0] / lam[1]:.3f}", size=30, color=C["tangent"])
          .move_to([x0, 1.95, 0], aligned_edge=LEFT))
    return g


class Anim(da.Scene2D):
    def construct(self):
        k = ValueTracker(3.0)
        t = np.linspace(0.12 * np.pi, 0.88 * np.pi, 300)
        arc = VGroup(*[DashedLine(scr([np.cos(a), np.sin(a)]), scr([np.cos(b), np.sin(b)]), color=C["aux"],
                                  stroke_width=2.5, dash_length=0.1)
                       for a, b in zip(t[:-1:12], t[12::12])])
        cloud = ha.dots_mobject([scr(q) for q in PTS], size=0.075, color=C["main"])
        tn = DashedLine(scr(0.6 * TRUE_N), scr(1.5 * TRUE_N), color=C["aux"], stroke_width=2.5, dash_length=0.08)
        lab_tn = da.label("true normal", 30, C["aux"]).move_to(scr(1.5 * TRUE_N) + [1.25, -0.05, 0])
        dyn = always_redraw(lambda: frame(k.get_value()))
        self.add(arc, tn, lab_tn, cloud, dyn)
        self.wait(1.5)
        self.play(k.animate.set_value(15), run_time=4.5, rate_func=linear)
        self.wait(2.5)
        self.play(k.animate.set_value(60), run_time=6.0, rate_func=linear)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.5)
