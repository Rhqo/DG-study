"""Shared helpers for the animations of Chapter 41 (GUIDELINES.md §12.8). Not a render target (the name does not
start with ``anim-``, so render_anims.sh skips it).

Copied from the Chapter 31 helpers (GUIDELINES: an animation does not import another chapter's anim helper):
- Cam, lambert, rgb, cmap_rgb, seg, arrow, dot, polyline, swap and the frame-checking Scene2D come from
  chapters/31-meshes/anim/_helpers_c.py (orthographic camera like dgmesh.Ortho / mplot3d view_init);
- MeshView follows _helpers_c.MeshView (projection, Lambert shading, back-face culling) but keeps a fixed pool of
  slots like _helpers_d.MeshView, so the painter's order and culling stay right during play() (see its docstring);
- Plot, mt (the space work-around for dganim.mtex) and readout come from chapters/31-meshes/anim/_helpers_d.py.
Candidates for tools/dganim.py: Cam, MeshView, Plot, mt, Scene2D (the frame check).

Import from an animation script with::

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _helpers as hp
"""

import os
import re

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy (this machine freezes
    os.environ.setdefault(_k, "4")                                          # under long all-core loads)

import numpy as np  # noqa: E402

import dganim as da  # noqa: E402
from manim import (BLACK, DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, DashedVMobject, Dot, FadeIn, FadeOut, Line,  # noqa: E402
                   ManimColor, Polygon, SVGMobject, Text, VGroup, VMobject, always_redraw, config)

C = da.C
HEX = da.COLORS


# ---------------------------------------------------------------- camera (from _helpers_c.Cam)
class Cam:
    """Orthographic camera looking from (elev, azim) in degrees, like dgmesh.Ortho / mplot3d view_init.

    right = (-sin a, cos a, 0) is well defined even at elev = 90 (top view; then up = (-cos a, -sin a, 0))."""

    def __init__(self, elev=20.0, azim=-60.0, scale=1.0, origin=ORIGIN, pivot=(0.0, 0.0, 0.0)):
        self.scale = float(scale)
        self.origin = np.array(origin, float)[:2] if np.ndim(origin) else np.zeros(2)
        self.pivot = np.asarray(pivot, float)
        self.set(elev, azim)

    def set(self, elev, azim):
        self.elev, self.azim = float(elev), float(azim)
        e, a = np.radians(elev), np.radians(azim)
        self.view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # toward the viewer
        self.right = np.array([-np.sin(a), np.cos(a), 0.0])
        self.up = np.cross(self.view, self.right)
        return self

    def xy(self, P):
        P = np.asarray(P, float) - self.pivot
        return np.stack([P @ self.right, P @ self.up], -1)

    def __call__(self, P):
        """World points (..., 3) -> manim scene points (..., 3) with z = 0."""
        q = self.origin + self.scale * self.xy(P)
        return np.concatenate([q, np.zeros(q.shape[:-1] + (1,))], -1)

    def depth(self, P):
        return (np.asarray(P, float) - self.pivot) @ self.view


def lambert(base_rgb, normals, cam, light=(0.35, 0.45, 0.82), ambient=0.45):
    """Two-sided Lambert shading with a light fixed to the camera (light given in (right, up, view))."""
    l = light[0] * cam.right + light[1] * cam.up + light[2] * cam.view
    l = l / np.linalg.norm(l)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15)
    k = ambient + (1 - ambient) * np.abs(n @ l)
    base = np.asarray(base_rgb, float)
    if base.ndim == 1:
        base = np.broadcast_to(base, (len(n), 3))
    return np.clip(base * k[:, None], 0, 1)


def rgb(color):
    """'#RRGGBB' or ManimColor -> rgb array in [0, 1]."""
    return np.array(ManimColor(color).to_rgb(), float)


def cmap_rgb(values, vmin, vmax, cmap="RdBu_r", center=None):
    """Scalar values -> (n, 3) rgb with a matplotlib colormap; with ``center`` the map is two-slope around it."""
    import matplotlib
    matplotlib.use("Agg")
    v = np.asarray(values, float)
    if center is None:
        t = (v - vmin) / (vmax - vmin)
    else:
        t = np.where(v >= center, 0.5 + 0.5 * (v - center) / (vmax - center), 0.5 - 0.5 * (center - v) / (center - vmin))
    t = np.clip(t, 0, 1)
    return np.asarray(matplotlib.colormaps[cmap](t))[..., :3]


# ---------------------------------------------------------------- projected mesh (after _helpers_c / _helpers_d MeshView)
class MeshView(VGroup):
    """A triangle mesh drawn with an orthographic Cam and the painter's algorithm.

    A fixed pool of VMobject slots (one per face plus ``n_extra``) whose points and colours are reassigned every frame
    in back-to-front order; unused slots are made invisible. The submobject list never changes. This matters in
    manim's Cairo renderer: during play() it renders the family members listed at the start of the animation in that
    order, so a mesh that replaced or reordered its submobjects would show stale faces and keep the first frame's
    depth order (the _helpers_d.MeshView of Chapter 31 uses slots for the same reason).

    colors: None (one base colour ``color``) or (n2, 3) rgb per face (shaded with Lambert unless shade=False).
    cull: hide faces whose normal points away from the viewer (closed, outward oriented meshes only).
    extra_polys: list of (points (m, 3) in scene coordinates, rgb, depth) drawn among the faces by depth (for example
    normal arrows as arrow-shaped polygons, see arrow_poly); set it before set_mesh, at most n_extra of them.
    """

    def __init__(self, V, F, cam, color=HEX["surface"], colors=None, edge_color="#3a3a3a", edge_width=0.8,
                 opacity=1.0, cull=False, sort=True, shade=True, ambient=0.45, light=(0.35, 0.45, 0.82), n_extra=0):
        super().__init__()
        self.F = np.asarray(F, int)
        self.cam = cam
        self.base = rgb(color)
        self.colors = None if colors is None else np.asarray(colors, float)
        self.edge_fill = edge_color == "fill"          # stroke in the face colour: hides the seams between faces
        self.edge_color = ManimColor("#000000" if self.edge_fill else edge_color)
        self.edge_width = edge_width
        self.opacity = opacity
        self.cull, self.sort, self.shade = cull, sort, shade
        self.ambient, self.light = ambient, light
        self.extra_polys = []
        self.slots = [VMobject(fill_opacity=opacity, stroke_width=edge_width, stroke_color=self.edge_color)
                      for _ in range(len(self.F) + n_extra)]
        for m in self.slots:
            m.set_points_as_corners([ORIGIN, ORIGIN, ORIGIN, ORIGIN])
        self.add(*self.slots)
        self.set_mesh(V)

    def set_mesh(self, V=None, colors=None, cam=None):
        if V is not None:
            self.V = np.asarray(V, float)
        if colors is not None:
            self.colors = np.asarray(colors, float)
        if cam is not None:
            self.cam = cam
        V, F, cam = self.V, self.F, self.cam
        P = cam(V)
        tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        base = self.base if self.colors is None else self.colors
        col = lambert(base, n, cam, self.light, self.ambient) if self.shade else np.broadcast_to(base, (len(F), 3))
        depth = cam.depth(tri.mean(1))
        vis = np.ones(len(F), bool)
        if self.cull:
            vis &= (n @ cam.view) > 0
        items = [(depth[k], 0, k) for k in np.nonzero(vis)[0]]
        assert len(self.extra_polys) <= len(self.slots) - len(F), "MeshView: n_extra too small"
        items += [(d, 1, j) for j, (_, _, d) in enumerate(self.extra_polys)]
        if self.sort:
            items.sort(key=lambda t: t[0])
        for slot, (_, kind, k) in zip(self.slots, items):
            if kind == 0:
                a, b, c = P[F[k]]
                slot.set_points_as_corners([a, b, c, a])
                fc = ManimColor(tuple(col[k]))
                slot.set_fill(fc, opacity=self.opacity)
                slot.set_stroke(fc if self.edge_fill else self.edge_color, width=self.edge_width,
                                opacity=min(1.0, 1.5 * self.opacity))
            else:
                pts, c, _ = self.extra_polys[k]
                pts = np.asarray(pts, float)
                slot.set_points_as_corners(np.vstack([pts, pts[:1]]))
                fc = ManimColor(tuple(rgb(c)))
                slot.set_fill(fc, opacity=1.0)
                slot.set_stroke(fc, width=0.5, opacity=1.0)
        c0 = P.mean(0) if len(P) else ORIGIN
        for slot in self.slots[len(items):]:
            slot.set_points_as_corners([c0, c0, c0, c0])
            slot.set_fill(opacity=0.0)
            slot.set_stroke(opacity=0.0)
        return self


def arrow_poly(p, q, width=0.035, head=0.2, head_width=0.09):
    """An arrow from p to q (scene points) as one closed polygon (shaft half-width ``width``), for MeshView slots."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d[:2]) + 1e-12
    u = d / L
    nrm = np.array([-u[1], u[0], 0.0])
    hl = min(head, 0.6 * L)
    r = q - hl * u
    return np.array([p + width * nrm, r + width * nrm, r + head_width * nrm, q, r - head_width * nrm,
                     r - width * nrm, p - width * nrm])


# ---------------------------------------------------------------- small drawing helpers (from _helpers_c)
def seg(p, q, color=BLACK, width=3.0, dashed=False, **kw):
    ln = Line(p, q, color=color, stroke_width=width, **kw)
    return DashedVMobject(ln, num_dashes=max(2, int(np.linalg.norm(np.asarray(q) - np.asarray(p)) / 0.12))) if dashed \
        else ln


def arrow(p, q, color=BLACK, width=5.0, tip=0.22):
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = np.linalg.norm(q - p) + 1e-12
    return Arrow(p, q, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=min(0.5, tip / L),
                 max_stroke_width_to_length_ratio=100)


def dot(p, color=BLACK, r=0.07):
    return Dot(p, radius=r, color=color)


def polyline(points, color=BLACK, width=3.0, **kw):
    m = VMobject(stroke_color=color, stroke_width=width, **kw)
    m.set_points_as_corners(np.asarray(points, float))
    return m


def filled(points, color, opacity=1.0, stroke_width=0.0, stroke_color=None):
    """A filled closed polygon through the scene points (n, 3) (lighter than Polygon for many vertices)."""
    P = np.asarray(points, float)
    m = VMobject(fill_color=color, fill_opacity=opacity, stroke_width=stroke_width,
                 stroke_color=stroke_color if stroke_color is not None else color)
    m.set_points_as_corners(np.vstack([P, P[:1]]))
    return m


def text(s, size=32, color=BLACK, **kw):
    return da.label(s, size=size, color=color, **kw)


def _mt_spaces(expr):
    """matplotlib writes a plain-text space as an SVG glyph without path data, which svgelements cannot parse.
    Replace spaces outside $...$ by the math space '$\\ $' (from _helpers_d.mt)."""
    parts = re.split(r"(\$[^$]*\$)", expr)
    return "".join(p if p.startswith("$") else p.replace(" ", r"$\ $") for p in parts)


def math(expr, size=34, color=BLACK):
    """dganim.mtex with the space work-around (text and math may be mixed)."""
    return da.mtex(_mt_spaces(expr), size, color=color)


def swap(old, new):
    """Cross-fade two labels (morphing Text glyphs garbles them mid-way)."""
    return [FadeOut(old), FadeIn(new)]


def num(v, digits=3):
    """A number with a Unicode minus (dganim.label shows '-' as a hyphen)."""
    s = f"{v:.{digits}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def readout(fn, pos, size=32, color=BLACK, edge=LEFT):
    """A Text label recomputed every frame: fn() -> str, placed with its `edge` at pos (from _helpers_d)."""
    return always_redraw(lambda: da.label(fn(), size, color=color).move_to(pos, aligned_edge=edge))


# ---------------------------------------------------------------- 2D plots (from _helpers_d.Plot)
class Plot:
    """A rectangle [x0, x0 + w] x [y0, y0 + h] in scene units showing data xlim x ylim (optionally log)."""

    def __init__(self, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _t(self, v, lim, log):
        v = np.asarray(v, float)
        if log:
            return (np.log10(v) - np.log10(lim[0])) / (np.log10(lim[1]) - np.log10(lim[0]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def X(self, x):
        return self.x0 + self.w * self._t(x, self.xlim, self.xlog)

    def Y(self, y):
        return self.y0 + self.h * self._t(y, self.ylim, self.ylog)

    def p(self, x, y):
        return np.array([float(self.X(x)), float(self.Y(y)), 0.0])

    def pts(self, xs, ys):
        return np.stack([self.X(xs), self.Y(ys), np.zeros(len(xs))], -1)

    def axes(self, xticks=(), yticks=(), xlabel=None, ylabel=None, size=30, color=BLACK, tick_len=0.1,
             ylabel_buff=0.25, xlabel_buff=0.45):
        """xticks/yticks: lists of (value, label) where label is a str (dganim.label) or a Mobject."""
        g = VGroup()
        a0 = np.array([self.x0, self.y0, 0])
        g.add(Line(a0, a0 + RIGHT * self.w, stroke_width=2.5, color=color))
        g.add(Line(a0, a0 + UP * self.h, stroke_width=2.5, color=color))
        for v, lab in xticks:
            x = float(self.X(v))
            g.add(Line([x, self.y0, 0], [x, self.y0 - tick_len, 0], stroke_width=2, color=color))
            m = da.label(lab, size) if isinstance(lab, str) else lab
            m.next_to([x, self.y0 - tick_len, 0], DOWN, buff=0.08)
            g.add(m)
        for v, lab in yticks:
            y = float(self.Y(v))
            g.add(Line([self.x0, y, 0], [self.x0 - tick_len, y, 0], stroke_width=2, color=color))
            m = da.label(lab, size) if isinstance(lab, str) else lab
            m.next_to([self.x0 - tick_len, y, 0], LEFT, buff=0.08)
            g.add(m)
        if xlabel is not None:
            m = da.label(xlabel, size) if isinstance(xlabel, str) else xlabel
            m.next_to([self.x0 + self.w / 2, self.y0 - tick_len, 0], DOWN, buff=xlabel_buff)
            g.add(m)
        if ylabel is not None:
            m = da.label(ylabel, size) if isinstance(ylabel, str) else ylabel
            m.next_to([self.x0 - tick_len, self.y0 + self.h / 2, 0], LEFT, buff=ylabel_buff)
            g.add(m)
        return g

    def curve(self, xs, ys, color=BLACK, width=4, clip=True, **kw):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if clip:
            lo, hi = sorted(self.ylim)
            ys = np.clip(ys, lo, hi)
        return polyline(self.pts(xs, ys), color=color, width=width, **kw)


# ---------------------------------------------------------------- scene with a frame check (from _helpers_c)
class Scene2D(da.Scene2D):
    """dganim.Scene2D that asserts, after every play/wait, that all labels (Text, mtex) lie inside the frame
    (margin 0.2, §12.8). With DGANIM_TIMELINE=path it appends "end time  what" lines (for the caption's times)."""

    MARGIN = 0.2

    def _check_frame(self):
        W, H = config["frame_width"] / 2 - self.MARGIN, config["frame_height"] / 2 - self.MARGIN
        for m in self.mobjects:
            for sub in m.get_family():
                if not isinstance(sub, (Text, SVGMobject)):
                    continue
                parts = sub.family_members_with_points()
                if parts and max(q.get_fill_opacity() for q in parts) > 0.05:
                    lo, hi = sub.get_corner(np.array([-1, -1, 0])), sub.get_corner(np.array([1, 1, 0]))
                    assert lo[0] > -W and hi[0] < W and lo[1] > -H and hi[1] < H, \
                        f"label outside the frame: {getattr(sub, 'text', type(sub).__name__)!r} {lo[:2]} {hi[:2]}"

    def _log(self, what):
        path = os.environ.get("DGANIM_TIMELINE")
        if path:
            with open(path, "a") as fh:
                fh.write(f"{self.renderer.time:7.2f}  {what}\n")

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        self._check_frame()
        if args:
            self._log("play " + ", ".join(type(a).__name__ for a in args[:4]))

    def wait(self, *args, **kwargs):
        super().wait(*args, **kwargs)
        self._check_frame()
        self._log(f"wait {args[0] if args else kwargs.get('duration', 1.0)}")
