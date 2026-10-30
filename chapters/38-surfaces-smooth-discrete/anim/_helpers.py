"""Shared helpers for the animations of Chapters 36-39 (GUIDELINES.md 12.8).

Not an animation (the name does not start with ``anim-``, so render_anims.sh skips it). The same file is copied into
chapters/36-*/anim, 37-*/anim, 38-*/anim and 39-*/anim, because an animation must not import another chapter's
helpers. Sources: Cam, MeshView, seg, arrow, dot, polyline, Scene2D (frame check and timeline log) are adapted from
chapters/31-meshes/anim/_helpers_c.py; mt (the space work-around of dganim.mtex) and Plot from
chapters/31-meshes/anim/_helpers_d.py. Additions: Scene2D also asserts that no label enters the bottom band of the
frame (paused control bar, 12.8), and MeshView takes per-face opacities.

Import from an animation script with::

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _helpers as hp
"""

import os
import re

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # this machine freezes under long
    os.environ.setdefault(_k, "4")                                          # all-core loads

import numpy as np  # noqa: E402

import dganim as da  # noqa: E402
from manim import (BLACK, ORIGIN, Arrow, DashedVMobject, Dot, Line, ManimColor, Polygon, SVGMobject,  # noqa: E402
                   Text, UpdateFromAlphaFunc, VGroup, VMobject, config, linear)

C = da.C
HEX = da.COLORS
DOWN = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
LEFT = np.array([-1.0, 0.0, 0.0])
RIGHT = np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------- text
def text(s, size=32, color=BLACK, **kw):
    return da.label(s, size=size, color=color, **kw)


def _mt_spaces(expr):
    """matplotlib writes a plain-text space as an SVG glyph without path data, which svgelements cannot parse.
    Replace spaces outside $...$ by the math space '$\\ $' (from _helpers_d.py of Chapter 31)."""
    parts = re.split(r"(\$[^$]*\$)", expr)
    return "".join(p if p.startswith("$") else p.replace(" ", r"$\ $") for p in parts)


def mt(expr, size=32, color=BLACK):
    """dganim.mtex with the space work-around (text and math may be mixed: 'face $f$')."""
    return da.mtex(_mt_spaces(expr), size, color=color)


def num(x, digits=3):
    """A number as text with a Unicode minus (dganim.mtex note)."""
    s = f"{x:.{digits}f}"
    return s.replace("-", "−")


# ---------------------------------------------------------------- colours
def rgb(color):
    return np.array(ManimColor(color).to_rgb(), float)


def cmap_rgb(values, vmin, vmax, name="RdBu_r"):
    import matplotlib
    matplotlib.use("Agg")
    t = np.clip((np.asarray(values, float) - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.asarray(matplotlib.colormaps[name](t))[..., :3]


def mix(c1, c2, t):
    """Blend two colours (hex or ManimColor) -> ManimColor."""
    return ManimColor(tuple(np.clip((1 - t) * rgb(c1) + t * rgb(c2), 0, 1)))


# ---------------------------------------------------------------- camera
class Cam:
    """Orthographic camera looking from (elev, azim) in degrees (like dgmesh.Ortho / mplot3d view_init).
    World point P -> scene point origin + scale * (P . right, P . up); depth = P . view (toward the viewer)."""

    def __init__(self, elev=20.0, azim=-60.0, scale=1.0, origin=(0.0, 0.0)):
        self.scale = float(scale)
        self.origin = np.array(origin, float)[:2]
        self.set(elev, azim)

    def set(self, elev, azim):
        self.elev, self.azim = float(elev), float(azim)
        e, a = np.radians(elev), np.radians(azim)
        self.view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.right = np.array([-np.sin(a), np.cos(a), 0.0])
        self.up = np.cross(self.view, self.right)
        return self

    def xy(self, P):
        P = np.asarray(P, float)
        return np.stack([P @ self.right, P @ self.up], -1)

    def __call__(self, P):
        q = self.origin + self.scale * self.xy(P)
        return np.concatenate([q, np.zeros(q.shape[:-1] + (1,))], -1)

    def depth(self, P):
        return np.asarray(P, float) @ self.view


def lambert(base_rgb, normals, cam, light=(0.35, 0.45, 0.82), ambient=0.45):
    """Two-sided Lambert shading with a light fixed to the camera (light in (right, up, view) components)."""
    l = light[0] * cam.right + light[1] * cam.up + light[2] * cam.view
    l = l / np.linalg.norm(l)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15)
    k = ambient + (1 - ambient) * np.abs(n @ l)
    base = np.asarray(base_rgb, float)
    if base.ndim == 1:
        base = np.broadcast_to(base, (len(n), 3))
    return np.clip(base * k[:, None], 0, 1)


class MeshView(VGroup):
    """A triangle mesh drawn with an orthographic Cam and the painter's algorithm (adapted from Chapter 31).

    colors: None (one base colour) or (n2, 3) rgb per face. cull: hide faces whose normal points away from the viewer
    (closed, outward oriented, convex meshes). hidden: boolean mask of faces not drawn. face_opacity: None or (n2,).
    extras: list of [mobject, depth] drawn among the faces by depth (arrows, curve pieces); update depth in place.
    """

    def __init__(self, V, F, cam, color=HEX["surface"], colors=None, edge_color="#3a3a3a", edge_width=0.8,
                 opacity=1.0, cull=False, sort=True, shade=True, ambient=0.45, light=(0.35, 0.45, 0.82)):
        super().__init__()
        self.F = np.asarray(F, int)
        self.cam = cam
        self.base = rgb(color)
        self.colors = None if colors is None else np.asarray(colors, float)
        self.edge_color = ManimColor(edge_color)
        self.edge_width = edge_width
        self.opacity = opacity
        self.cull, self.sort, self.shade = cull, sort, shade
        self.ambient, self.light = ambient, light
        self.extras = []
        self.hidden = np.zeros(len(self.F), bool)
        self.face_opacity = None
        self.polys = [Polygon(ORIGIN, RIGHT * 1e-3, UP * 1e-3, stroke_width=edge_width, stroke_color=self.edge_color,
                              fill_opacity=opacity) for _ in range(len(self.F))]
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
        vis = ~self.hidden
        if self.cull:
            vis &= (n @ cam.view) > 0
        order = np.argsort(depth, kind="stable") if self.sort else np.arange(len(F))
        items = [(depth[k], self.polys[k]) for k in order if vis[k]]
        for k in np.nonzero(vis)[0]:
            p = self.polys[k]
            a, b, c = P[F[k]]
            p.set_points_as_corners([a, b, c, a])
            op = self.opacity if self.face_opacity is None else self.face_opacity[k]
            p.set_fill(ManimColor(tuple(col[k])), opacity=op)
            p.set_stroke(self.edge_color, width=self.edge_width, opacity=min(1.0, 1.5 * op))
        if self.extras:
            items += [(d, m) for m, d in self.extras]
            items.sort(key=lambda t: t[0])
        self.submobjects = [m for _, m in items]
        return self


def mesh_mobs(V, F, cam, color=HEX["surface"], colors=None, edge_color="#3a3a3a", edge_width=0.8, opacity=1.0,
              cull=False, shade=True, ambient=0.45, light=(0.35, 0.45, 0.82), hidden=None, extras=()):
    """New Polygons for one frame of a mesh, in painter's order (far first), with `extras` = [(mobject, depth)]
    mixed in by depth. Use inside timeline() events: during one play() manim draws the mobjects that existed at the
    start in their old order, so a reordered MeshView would not be redrawn correctly; fresh mobjects are."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    P = cam(V)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    base = rgb(color) if colors is None else np.asarray(colors, float)
    col = lambert(base, n, cam, light, ambient) if shade else np.broadcast_to(base, (len(F), 3))
    depth = cam.depth(tri.mean(1))
    vis = np.ones(len(F), bool) if hidden is None else ~np.asarray(hidden, bool)
    if cull:
        vis &= (n @ cam.view) > 0
    ec = ManimColor(edge_color)
    items = []
    for k in np.nonzero(vis)[0]:
        a, b, c = P[F[k]]
        op = opacity if np.ndim(opacity) == 0 else opacity[k]
        poly = Polygon(a, b, c, fill_color=ManimColor(tuple(col[k])), fill_opacity=op, stroke_color=ec,
                       stroke_width=edge_width, stroke_opacity=min(1.0, 1.5 * op))
        items.append((depth[k], poly))
    items += list(extras)
    items.sort(key=lambda t: t[0])
    return [m for _, m in items]


# ---------------------------------------------------------------- small drawing helpers
def seg(p, q, color=BLACK, width=3.0, dashed=False, **kw):
    ln = Line(np.asarray(p, float), np.asarray(q, float), color=color, stroke_width=width, **kw)
    if dashed:
        n = max(2, int(np.linalg.norm(np.asarray(q, float) - np.asarray(p, float)) / 0.12))
        return DashedVMobject(ln, num_dashes=n)
    return ln


def arrow(p, q, color=BLACK, width=5.0, tip=0.22):
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = np.linalg.norm(q - p) + 1e-12
    return Arrow(p, q, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=min(0.5, tip / L),
                 max_stroke_width_to_length_ratio=100)


def dot(p, color=BLACK, r=0.07, **kw):
    return Dot(np.asarray(p, float), radius=r, color=color, **kw)


def polyline(points, color=BLACK, width=3.0, closed=False, **kw):
    P = np.asarray(points, float)
    if P.shape[-1] == 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], 1)
    if closed:
        P = np.concatenate([P, P[:1]])
    m = VMobject(stroke_color=color, stroke_width=width, **kw)
    m.set_points_as_corners(P)
    return m


def poly_fill(points, color, opacity=0.5, stroke_width=0.0, stroke_color=None):
    P = np.asarray(points, float)
    if P.shape[-1] == 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], 1)
    return Polygon(*P, fill_color=color, fill_opacity=opacity, stroke_width=stroke_width,
                   stroke_color=stroke_color if stroke_color is not None else color)


def p3(x, y=0.0):
    """Scene point from (x, y) or a length-2 array."""
    if np.ndim(x):
        return np.array([x[0], x[1], 0.0])
    return np.array([x, y, 0.0])


class Plot:
    """A rectangle [x0, x0 + w] x [y0, y0 + h] in scene units showing data xlim x ylim (from Chapter 31)."""

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
        xs = np.asarray(xs, float)
        return np.stack([self.X(xs), self.Y(np.asarray(ys, float)), np.zeros(len(xs))], -1)

    def axes(self, xticks=(), yticks=(), xlabel=None, ylabel=None, size=30, color=BLACK, tick_len=0.1, width=2.5,
             zero_line=None):
        """xticks/yticks: lists of (value, label); label is a str (dganim.label) or a Mobject."""
        g = VGroup()
        a0 = np.array([self.x0, self.y0, 0])
        g.add(Line(a0, a0 + RIGHT * self.w, stroke_width=width, color=color))
        g.add(Line(a0, a0 + UP * self.h, stroke_width=width, color=color))
        if zero_line is not None:
            y = float(self.Y(zero_line))
            g.add(seg([self.x0, y, 0], [self.x0 + self.w, y, 0], color=HEX["aux"], width=1.5, dashed=True))
        for v, lab in xticks:
            x = float(self.X(v))
            g.add(Line([x, self.y0, 0], [x, self.y0 - tick_len, 0], stroke_width=2, color=color))
            if lab is not None:
                m = da.label(lab, size) if isinstance(lab, str) else lab
                g.add(m.next_to([x, self.y0 - tick_len, 0], DOWN, buff=0.08))
        for v, lab in yticks:
            y = float(self.Y(v))
            g.add(Line([self.x0, y, 0], [self.x0 - tick_len, y, 0], stroke_width=2, color=color))
            if lab is not None:
                m = da.label(lab, size) if isinstance(lab, str) else lab
                g.add(m.next_to([self.x0 - tick_len, y, 0], LEFT, buff=0.08))
        if xlabel is not None:
            m = da.label(xlabel, size) if isinstance(xlabel, str) else xlabel
            g.add(m.next_to([self.x0 + self.w, self.y0, 0], RIGHT, buff=0.15))
        if ylabel is not None:
            m = da.label(ylabel, size) if isinstance(ylabel, str) else ylabel
            g.add(m.next_to([self.x0, self.y0 + self.h, 0], UP, buff=0.15))
        return g

    def curve(self, xs, ys, color=BLACK, width=4, clip=True, **kw):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if clip:
            lo, hi = sorted(self.ylim)
            ys = np.clip(ys, lo, hi)
        return polyline(self.pts(xs, ys), color=color, width=width, **kw)


# ---------------------------------------------------------------- one play() with several timed changes
def timeline(group, events, run_time):
    """One animation (hence one partial movie file and one key frame) that runs timed changes.

    group: a VGroup holding every container the events mutate (manim redraws only animated mobjects during play).
    events: list of (t0, t1, fn) in seconds; fn(u) is called with the progress u in [0, 1] on every frame with
    t0 <= t <= t1 and once more with u = 1 when t passes t1 (t1 = t0 gives an instant change).
    Many short play()/wait() calls make the video large (every partial file starts with a key frame), so the
    animations of Chapters 36-39 group the steps of one stage into one timeline."""
    finished = [False] * len(events)

    def upd(_m, a):
        t = a * run_time
        for i, (t0, t1, fn) in enumerate(events):
            if finished[i] or t < t0 - 1e-9:
                continue
            if t >= t1 - 1e-9:
                fn(1.0)
                finished[i] = True
            else:
                fn((t - t0) / max(t1 - t0, 1e-9))

    return UpdateFromAlphaFunc(group, upd, run_time=run_time, rate_func=linear)


def _members(mobs):
    out = []
    for m in mobs:
        out += [q for q in m.get_family() if len(q.points)]
    return out


def fade(mobs, start_hidden=True):
    """fn(u) for timeline(): scale the fill and stroke opacities of mobs (as they are now) by u.
    With start_hidden the mobs are hidden at once (fn(0)); use fn(1 - u) to fade out."""
    mem = _members(mobs)
    base = [(q.get_fill_opacity(), q.get_stroke_opacity()) for q in mem]

    def fn(u):
        for q, (fo, so) in zip(mem, base):
            q.set_fill(opacity=u * fo)
            q.set_stroke(opacity=u * so)
    if start_hidden:
        fn(0.0)
    return fn


def hide(mobs):
    for q in _members(mobs):
        q.set_fill(opacity=0)
        q.set_stroke(opacity=0)


def setter(group, items):
    """fn(u) for timeline(): make `items` the submobjects of `group`. Removed mobjects are also hidden: during a
    play() manim keeps drawing the family that existed when the play started."""
    def fn(u):
        keep = set(map(id, items))
        hide([m for m in group.submobjects if id(m) not in keep])
        group.submobjects = list(items)
    return fn


# ---------------------------------------------------------------- scene with checks
class Scene2D(da.Scene2D):
    """dganim.Scene2D that asserts, after every play/wait, that every visible label (Text, mtex) lies inside the frame
    with margin MARGIN and above the bottom band BOTTOM (fraction of the frame height; 0.15 by 12.8: the paused
    control bar covers about 12% of a 720p frame). With DGANIM_TIMELINE=path it appends "end time  what" lines."""

    MARGIN = 0.2
    BOTTOM = 0.15

    def _check_frame(self):
        W, H = config["frame_width"] / 2 - self.MARGIN, config["frame_height"] / 2 - self.MARGIN
        ybot = -config["frame_height"] / 2 + self.BOTTOM * config["frame_height"]
        for m in self.mobjects:
            for sub in m.get_family():
                if not isinstance(sub, (Text, SVGMobject)):
                    continue
                parts = sub.family_members_with_points()
                if parts and max(q.get_fill_opacity() for q in parts) > 0.05:
                    lo, hi = sub.get_corner(np.array([-1, -1, 0])), sub.get_corner(np.array([1, 1, 0]))
                    name = getattr(sub, "text", type(sub).__name__)
                    assert lo[0] > -W and hi[0] < W and lo[1] > -H and hi[1] < H, \
                        f"label outside the frame: {name!r} {lo[:2]} {hi[:2]}"
                    assert lo[1] > ybot, f"label in the bottom band (y < {ybot:.2f}): {name!r} {lo[:2]}"

    def _log(self, what):
        path = os.environ.get("DGANIM_TIMELINE")
        if path:
            with open(path, "a") as fh:
                fh.write(f"{self.renderer.time:7.2f}  {what}\n")

    def mark(self, what):
        """A named point of the timeline (for writing the time ranges of the caption)."""
        self._log("MARK " + what)

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        self._check_frame()

    def wait(self, *args, **kwargs):
        super().wait(*args, **kwargs)
        self._check_frame()
