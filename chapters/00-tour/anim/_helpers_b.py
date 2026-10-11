"""Tour 0.5–0.8 애니메이션의 공용 helper (GUIDELINES.md §12.8). 이름이 anim-으로 시작하지 않으므로 렌더 대상이 아니다.

sphere 같은 매끈한 3D 대상은 manim의 ThreeDScene 대신 **2D 장면에서 직접 orthographic projection**으로 그린다.
깊이(앞/뒤)를 우리가 계산하므로 숨은 부분을 점선이나 옅은 선으로 그릴 수 있고(dgfig의 정적 그림과 같은 관례),
렌더도 빠르다. mesh의 face 음영은 tools/dganim.mesh_group을 쓴다.

    from _helpers_b import View, polyline, sphere_curve, arrow, readout
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dganim as da  # noqa: E402
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Circle, DashedVMobject, Dot, Line,  # noqa: E402
                   VGroup, VMobject)

C = da.C


class View:
    """orthographic camera. elev, azim은 degree(dgfig.view_init과 같은 뜻). scale: 단위 길이의 화면 길이.

    d는 원점에서 카메라 쪽을 향하는 단위벡터, right/up은 화면의 오른쪽·위쪽 방향이다.
    """

    def __init__(self, elev=20.0, azim=-60.0, scale=2.5, center=ORIGIN):
        self.set(elev, azim)
        self.scale = float(scale)
        self.center = np.asarray(center, float)

    def set(self, elev, azim):
        e, a = np.deg2rad(elev), np.deg2rad(azim)
        self.elev, self.azim = elev, azim
        self.d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.right = np.array([-np.sin(a), np.cos(a), 0.0])
        self.up = np.cross(self.d, self.right)
        assert np.allclose(np.cross(self.right, self.up), self.d)      # 오른손 좌표계
        return self

    def p(self, X):
        """3D 점(들) → manim 화면 좌표 (…, 3)."""
        X = np.asarray(X, float)
        out = np.stack([X @ self.right, X @ self.up, np.zeros(X.shape[:-1])], axis=-1) * self.scale
        return out + self.center

    def depth(self, X):
        return np.asarray(X, float) @ self.d

    def vec(self, v):
        """3D vector → 화면 vector (평행이동 없음)."""
        v = np.asarray(v, float)
        return np.array([v @ self.right, v @ self.up, 0.0]) * self.scale


def polyline(pts, color=None, width=3.0, opacity=1.0):
    m = VMobject(stroke_color=color if color is not None else C["main"], stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(np.asarray(pts, float))
    return m


def _runs(mask):
    """True가 이어지는 구간 [i0, i1)들."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def sphere_curve(view, P, color=None, width=3.0, hidden="faint", hidden_opacity=0.3, front_eps=0.0):
    """sphere 위(또는 근처)의 3D 곡선 P를 앞쪽은 실선, 뒤쪽은 hidden 방식("faint", "dashed", None)으로 그린다."""
    P = np.asarray(P, float)
    front = view.depth(P) >= front_eps
    g = VGroup()
    for i0, i1 in _runs(front):
        j1 = min(i1 + 1, len(P))
        if j1 - i0 >= 2:
            g.add(polyline(view.p(P[i0:j1]), color, width))
    if hidden is not None:
        for i0, i1 in _runs(~front):
            j1 = min(i1 + 1, len(P))
            if j1 - i0 >= 2:
                seg = polyline(view.p(P[i0:j1]), color, width * 0.8, hidden_opacity if hidden == "faint" else 0.8)
                if hidden == "dashed":
                    n = max(2, int(np.sum(np.linalg.norm(np.diff(view.p(P[i0:j1]), axis=0), axis=1)) / 0.12))
                    seg = DashedVMobject(seg, num_dashes=n, dashed_ratio=0.55)
                g.add(seg)
    return g


def sphere_disk(view, radius=1.0, fill_opacity=0.35, stroke_width=2.5):
    """sphere의 윤곽(원)과 연회색 면. orthographic projection에서 sphere는 원으로 보인다."""
    return Circle(radius=radius * view.scale, stroke_color=C["main"], stroke_width=stroke_width,
                  fill_color=C["surface"], fill_opacity=fill_opacity).move_to(view.center)


def graticule(view, n_lat=5, n_lon=12, color="#A8A8A8", width=1.2, hidden=None):
    """sphere의 경위선(회색). colatitude k·π/(n_lat+1), longitude k·2π/n_lon."""
    g = VGroup()
    s = np.linspace(0, 2 * np.pi, 181)
    for k in range(1, n_lat + 1):
        t = k * np.pi / (n_lat + 1)
        P = np.stack([np.sin(t) * np.cos(s), np.sin(t) * np.sin(s), np.cos(t) * np.ones_like(s)], 1)
        g.add(sphere_curve(view, P, color, width, hidden=hidden))
    tt = np.linspace(0, np.pi, 91)
    for k in range(n_lon):
        a = 2 * np.pi * k / n_lon
        P = np.stack([np.sin(tt) * np.cos(a), np.sin(tt) * np.sin(a), np.cos(tt)], 1)
        g.add(sphere_curve(view, P, color, width, hidden=hidden))
    return g


def arrow(start, end, color=None, width=5.0, tip=0.2, opacity=1.0):
    """끝이 일정한 크기의 화살표(화면 좌표). 길이가 0에 가까우면 점 하나만 돌려준다."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    color = color if color is not None else C["main"]
    L = np.linalg.norm(end - start)
    if L < 1e-3:
        return Dot(start, radius=0.001, fill_opacity=0)
    a = Arrow(start, end, buff=0, stroke_width=width, color=color, tip_length=min(tip, 0.6 * L),
              max_tip_length_to_length_ratio=0.6, max_stroke_width_to_length_ratio=1000)
    a.set_opacity(opacity)
    return a


def sph(theta, phi):
    """§7의 sphere parametrization (r = 1): x(θ, φ) = (sin θ cos φ, sin θ sin φ, cos θ). θ는 colatitude."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def great_arc(a, b, n=120):
    """unit sphere 위 a에서 b까지의 짧은 great circle 호."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.arccos(np.clip(a @ b, -1, 1))
    w = np.linspace(0, 1, n)[:, None]
    return (np.sin((1 - w) * d) * a + np.sin(w * d) * b) / np.sin(d)


def readout(fmt, getter, size=34, color=None, anchor=None, edge=LEFT):
    """always_redraw용 숫자 표시. 값은 fmt로 반올림되므로 같은 글자는 다시 만들지 않는다."""
    from manim import always_redraw
    color = color if color is not None else C["main"]

    def make():
        m = da.label(fmt.format(*np.atleast_1d(getter())), size=size, color=color)
        if anchor is not None:
            m.move_to(anchor, aligned_edge=edge)
        return m

    return always_redraw(make)
