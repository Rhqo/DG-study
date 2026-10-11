"""Chapter 0 (Sections 0.1–0.4) 애니메이션의 공용 도구. render 대상이 아니다(이름이 anim-로 시작하지 않는다).

- Ortho: matplotlib의 view_init(elev, azim)과 같은 정사영 카메라. 3D 점을 manim 화면 좌표로 보낸다.
  ThreeDScene 대신 Scene2D에서 직접 투영하고, 면은 깊이로 정렬해서(painter's algorithm) 그린다.
  면뿐 아니라 곡면 위의 선, 화살표, 점, 라벨까지 한 깊이 순서로 섞어 그리기 위해서다(Painter).
- surface_group: (n, m, 3) 격자 → 음영을 넣은 사각형 면들. 깊이 정렬, 선택적으로 뒷면 제거.
- curve_group: 3D 곡선을 보이는 부분(실선)과 가려진 부분(점선 또는 생략)으로 나눠 그린다.
- arrow2: 화면 좌표의 화살표(굵기와 머리 크기를 고정한다).
- panel_title, readout: 화면에 고정한 짧은 영어 라벨.

스크립트에서:
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _helpers_a as ha
"""

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, DashedVMobject, Line, ManimColor, VGroup,
                   VMobject)

import dganim as da

C = da.C


class Ortho:
    """정사영 카메라. elev, azim은 matplotlib과 같은 뜻(도)이다.

    view: 보는 사람 쪽을 향하는 단위벡터. right, up: 화면의 오른쪽과 위쪽.
    화면 좌표 = center + scale * (<P, right>, <P, up>).
    """

    def __init__(self, elev, azim, scale=1.0, center=ORIGIN, origin=(0.0, 0.0, 0.0)):
        e, a = np.radians(elev), np.radians(azim)
        self.view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.right = np.array([-np.sin(a), np.cos(a), 0.0])
        self.up = np.cross(self.view, self.right)
        self.scale = float(scale)
        self.center = np.asarray(center, float)
        self.origin = np.asarray(origin, float)
        assert np.isclose(self.up @ self.up, 1) and abs(self.up @ self.view) < 1e-12

    def __call__(self, P):
        P = np.asarray(P, float) - self.origin
        x = P @ self.right
        y = P @ self.up
        out = np.stack([x, y, np.zeros_like(x)], axis=-1) * self.scale
        return out + self.center

    def depth(self, P):
        return (np.asarray(P, float) - self.origin) @ self.view


def poly(points, fill, stroke=None, stroke_width=None, opacity=1.0):
    """화면 좌표의 닫힌 다각형 하나 (Polygon보다 가볍다).

    stroke가 None이면 면과 같은 색의 얇은 테두리(0.8)를 둘러 이웃 면 사이의 흰 틈을 감춘다.
    """
    if stroke is None:
        stroke, stroke_width = fill, (0.8 if stroke_width is None else stroke_width)
    m = VMobject(stroke_width=stroke_width or 0.0, stroke_color=stroke, stroke_opacity=opacity,
                 fill_color=fill, fill_opacity=opacity)
    pts = np.asarray(points, float)
    m.set_points_as_corners(np.vstack([pts, pts[:1]]))
    return m


def surface_group(P, cam, color=None, colors=None, opacity=1.0, cull=False, light=(0.35, -0.45, 0.82),
                  ambient=0.38, stroke=None, stroke_width=None, shade=True, two_sided=True):
    """격자 점 P[i, j] (shape (n, m, 3))의 사각형 면을 깊이 순서로 그린 VGroup.

    colors: (n-1, m-1) 배열의 ManimColor(면마다), 없으면 color 하나.
    cull=True면 view와 반대를 보는 면(뒷면)을 그리지 않는다. 면의 방향은 (P[i+1,j]-P[i,j]) x (P[i,j+1]-P[i,j]).
    주의: 앞면은 x_u × x_v 쪽이다. 매개화에 따라 이것이 안쪽을 향할 수 있다(예: 0.3절의 torus 매개화).
    그런 곡면을 cull=True로 그리면 앞쪽 면이 빠지고 안쪽 벽만 보인다. 격자의 한 방향을 뒤집어 바깥을 앞면으로 만든다.
    """
    P = np.asarray(P, float)
    A, B, Cc, D = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    nrm = np.cross(Cc - A, D - B)
    ctr = 0.25 * (A + B + Cc + D)
    dep = cam.depth(ctr)
    facing = nrm @ cam.view
    order = np.argsort(dep, axis=None)
    n, m = dep.shape
    g = VGroup()
    S = {k: cam(v) for k, v in (("A", A), ("B", B), ("C", Cc), ("D", D))}
    base = ManimColor(color) if color is not None else C["surface"]
    for idx in order:
        i, j = divmod(int(idx), m)
        if cull and facing[i, j] <= 0:
            continue
        col = colors[i][j] if colors is not None else base
        if shade:
            nv = nrm[i, j] if (two_sided or facing[i, j] > 0) else -nrm[i, j]
            col = da.shade(col, nv, light, ambient)
        g.add(poly([S["A"][i, j], S["B"][i, j], S["C"][i, j], S["D"][i, j]], col, stroke, stroke_width, opacity))
    return g


def tri_group(V, F, cam, color=None, colors=None, opacity=1.0, cull=True, light=(0.35, -0.45, 0.82),
              ambient=0.38, stroke=None, stroke_width=None):
    """triangle mesh (V, F)를 깊이 순서로 그린 VGroup. F는 바깥쪽을 향하는 방향."""
    V = np.asarray(V, float)
    F = np.asarray(F)
    T = V[F]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    dep = cam.depth(T.mean(axis=1))
    facing = nrm @ cam.view
    S = cam(T.reshape(-1, 3)).reshape(-1, 3, 3)
    base = ManimColor(color) if color is not None else C["surface"]
    g = VGroup()
    for k in np.argsort(dep):
        if cull and facing[k] <= 0:
            continue
        col = colors[k] if colors is not None else base
        g.add(poly(S[k], da.shade(col, nrm[k], light, ambient), stroke, stroke_width, opacity))
    return g


def polyline(points, color, width=4.0):
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners(np.asarray(points, float))
    return m


def runs(mask):
    """bool 배열에서 연속한 True 구간의 (시작, 끝+1) 목록."""
    mask = np.asarray(mask, bool)
    out, start = [], None
    for i, b in enumerate(mask):
        if b and start is None:
            start = i
        if not b and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def curve_group(P, cam, color, width=4.0, visible=None, hidden="dashed", dash=0.08):
    """3D 곡선. visible(bool 배열)이 False인 부분은 hidden='dashed'면 점선, None이면 생략한다."""
    S = cam(P)
    g = VGroup()
    if visible is None:
        visible = np.ones(len(S), bool)
    visible = np.asarray(visible, bool)
    for a, b in runs(visible):
        if b - a >= 2:
            g.add(polyline(S[max(a - 1, 0):b + 1] if a > 0 else S[a:b + 1], color, width))
    if hidden == "dashed":
        for a, b in runs(~visible):
            if b - a >= 2:
                seg = polyline(S[a:b + 1], color, width * 0.7)
                L = np.sum(np.linalg.norm(np.diff(S[a:b + 1], axis=0), axis=1))
                nd = max(2, int(L / dash / 2))
                g.add(DashedVMobject(seg, num_dashes=nd, dashed_ratio=0.55))
    return g


def arrow2(start, end, color, width=5.0, tip=0.22):
    """화면 좌표의 화살표. 길이와 상관없이 머리 크기를 tip(화면 단위)으로 고정한다."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    L = np.linalg.norm(end - start)
    if L < 1e-6:
        return VGroup()
    a = Arrow(start, end, buff=0, color=color, stroke_width=width,
              max_tip_length_to_length_ratio=min(1.0, tip / L), max_stroke_width_to_length_ratio=100)
    return a


def arrow3(cam, base, vec, color, scale=1.0, width=5.0, tip=0.22):
    base = np.asarray(base, float)
    return arrow2(cam(base), cam(base + scale * np.asarray(vec, float)), color, width, tip)


def dashed_line(a, b, color, width=2.5, n=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    n = n or max(3, int(L / 0.16))
    return DashedVMobject(Line(a, b, color=color, stroke_width=width), num_dashes=n, dashed_ratio=0.55)


def fmt(x, nd=2):
    """음수에 진짜 빼기 기호(−)를 쓴 숫자."""
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        if float(s) == 0:
            s = s[1:]
        else:
            s = "−" + s[1:]
    return s


def row(*items, size=30, color=None, buff=0.14):
    """글자와 수식을 가로로 잇는다. '$'로 시작하는 조각은 dganim.mtex, 나머지는 dganim.label.

    글자 조각은 Text(DejaVu Serif)로, 수식 조각은 mathtext(cm)로 그려 글꼴을 구분한다.
    세로는 가운데 정렬이다(글자의 아래끝은 g, y 같은 내림자 때문에 기준선과 다르다).
    """
    from manim import RIGHT as _R
    col = color if color is not None else C["main"]
    parts = VGroup()
    for it in items:
        it = it.strip()
        if not it:
            continue
        m = da.mtex(it, size, col) if it.startswith("$") else da.label(it, size, col)
        parts.add(m)
    parts.arrange(_R, buff=buff)
    return parts


class Painter:
    """면, 선분, 화살표, 점, 라벨을 깊이 순서로 그리는 간단한 painter's algorithm.

    모든 항목은 깊이(카메라 쪽이 큼)로 정렬해 먼 것부터 그린다. 곡면 위의 선은 bias만큼 카메라 쪽으로
    당겨서 자기가 놓인 면보다 나중에 그려지게 한다. 반투명 면 뒤의 선은 흐리게 비친다(숨은선 효과).
    """

    def __init__(self, cam, light=(0.35, -0.45, 0.82), ambient=0.38):
        self.cam = cam
        self.light = light
        self.ambient = ambient
        self.items = []          # (depth, builder)

    def face(self, P3, fill, opacity=1.0, shade=True, stroke=None, stroke_width=None, bias=0.0):
        P3 = np.asarray(P3, float)
        col = ManimColor(fill)
        if shade:
            n = np.cross(P3[1] - P3[0], P3[2] - P3[0])
            col = da.shade(col, n, self.light, self.ambient)
        S = self.cam(P3)
        self.items.append((self.cam.depth(P3.mean(axis=0)) + bias,
                           lambda: poly(S, col, stroke, stroke_width, opacity)))

    def grid(self, P, color=None, colors=None, opacity=1.0, shade=True, cull=False, bias=0.0):
        """격자 P[i, j]의 사각형 면들. cull=True면 (P[i+1,j]-P[i,j]) x (P[i,j+1]-P[i,j])가 뒤를 보는 면을 뺀다.

        앞면은 x_u × x_v 쪽이다. 안쪽을 향하는 매개화(예: torus)는 격자 한 방향을 뒤집어 넘긴다(surface_group 참고).
        """
        P = np.asarray(P, float)
        n, m = P.shape[:2]
        for i in range(n - 1):
            for j in range(m - 1):
                Q = np.array([P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]])
                if cull and np.cross(Q[2] - Q[0], Q[3] - Q[1]) @ self.cam.view <= 0:
                    continue
                self.face(Q, colors[i][j] if colors is not None else (color or C["surface"]), opacity, shade,
                          bias=bias)

    def polyline(self, P3, color, width=4.0, bias=0.02, opacity=1.0, every=1):
        """3D 곡선을 짧은 선분들로 나누어 넣는다. every개의 점마다 한 조각."""
        P3 = np.asarray(P3, float)
        S = self.cam(P3)
        D = self.cam.depth(P3)
        col = ManimColor(color)
        for a in range(0, len(P3) - 1, every):
            b = min(a + every, len(P3) - 1)
            pts = S[a:b + 1]
            self.items.append((float(D[a:b + 1].mean()) + bias,
                               (lambda pts=pts: _seg(pts, col, width, opacity))))

    def arrow(self, base, vec, color, width=5.0, tip=0.22, bias=0.05):
        base, vec = np.asarray(base, float), np.asarray(vec, float)
        a, b = self.cam(base), self.cam(base + vec)
        d = self.cam.depth(base + 0.5 * vec) + bias
        self.items.append((d, lambda: arrow2(a, b, color, width, tip)))

    def dot(self, P, color=None, radius=0.06, bias=0.06):
        from manim import Dot
        P = np.asarray(P, float)
        s = self.cam(P)
        self.items.append((self.cam.depth(P) + bias,
                           lambda: Dot(s, radius=radius, color=color if color is not None else C["main"])))

    def mob(self, mobject, P=None, depth=None, bias=0.1):
        """이미 만든 mobject를 깊이 depth(또는 3D 점 P의 깊이)에 넣는다."""
        d = depth if depth is not None else self.cam.depth(np.asarray(P, float)) + bias
        self.items.append((d, lambda: mobject))

    def build(self):
        g = VGroup()
        for d, f in sorted(self.items, key=lambda t: t[0]):
            m = f()
            if m is not None:
                g.add(m)
        return g


def _seg(pts, color, width, opacity=1.0):
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(pts)
    return m


def dots_mobject(points, size=0.06, color=None):
    """많은 작은 점을 VMobject 하나(작은 정사각형 subpath들)로 만든다. Dot 수천 개보다 훨씬 빠르다."""
    col = color if color is not None else C["main"]
    m = VMobject(stroke_width=0, fill_color=col, fill_opacity=1.0)
    h = size / 2
    first = True
    for p in np.asarray(points, float):
        sq = np.array([[p[0] - h, p[1] - h, 0], [p[0] + h, p[1] - h, 0], [p[0] + h, p[1] + h, 0],
                       [p[0] - h, p[1] + h, 0], [p[0] - h, p[1] - h, 0]])
        if first:
            m.set_points_as_corners(sq)
            first = False
        else:
            m.start_new_path(sq[0])
            m.add_points_as_corners(sq[1:])
    return m
