"""Shared helpers for the animations of Sections 31.1–31.3 (GUIDELINES.md §12.8).

Not an animation itself (the name does not start with ``anim-``, so render_anims.sh skips it).

Why not ``dganim.mesh_group`` + ``Scene3D``: manim's ThreeDCamera depth-sorts only mobjects with
``shade_in_3d = True`` (camera/three_d_camera.py, ``get_mobjects_to_display``), and mesh_group's polygons do not
set it, so faces are drawn in list order. Here a mesh is projected with an orthographic camera (the same camera
as ``dgmesh.Ortho`` used by the static figures of Chapter 31) and drawn in a 2D scene with the painter's
algorithm, or with back-face culling for closed convex meshes. The polygons are kept and only their points and
colours are updated, so a mesh of a few thousand faces can change every frame.

Import from an animation script with::

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import _helpers_c as hc
    import dgmesh as dm
"""

import numpy as np

import dganim as da
from manim import (BLACK, ORIGIN, Arrow, DashedVMobject, Dot, FadeIn, FadeOut, Line, ManimColor, Polygon, SVGMobject,
                   Text, VGroup, VMobject, config)

C = da.C
HEX = da.COLORS


# ---------------------------------------------------------------- camera
class Cam:
    """Orthographic camera looking from (elev, azim) in degrees, like dgmesh.Ortho / mplot3d view_init.

    right = (-sin a, cos a, 0) is well defined even at elev = 90 (top view; then up = (-cos a, -sin a, 0))."""

    def __init__(self, elev=20.0, azim=-60.0, scale=1.0, origin=ORIGIN):
        self.scale = float(scale)
        self.origin = np.array(origin, float)[:2] if np.ndim(origin) else np.zeros(2)
        self.set(elev, azim)

    def set(self, elev, azim):
        self.elev, self.azim = float(elev), float(azim)
        e, a = np.radians(elev), np.radians(azim)
        self.view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # toward the viewer
        self.right = np.array([-np.sin(a), np.cos(a), 0.0])
        self.up = np.cross(self.view, self.right)
        return self

    def xy(self, P):
        """World points (..., 3) -> 2D camera coordinates (..., 2) (before scale and origin)."""
        P = np.asarray(P, float)
        return np.stack([P @ self.right, P @ self.up], -1)

    def __call__(self, P):
        """World points (..., 3) -> manim scene points (..., 3) with z = 0."""
        q = self.origin + self.scale * self.xy(P)
        return np.concatenate([q, np.zeros(q.shape[:-1] + (1,))], -1)

    def depth(self, P):
        return np.asarray(P, float) @ self.view


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


# ---------------------------------------------------------------- projected mesh
class MeshView(VGroup):
    """A triangle mesh drawn with an orthographic Cam.

    colors: None (one base colour ``color``) or (n2, 3) rgb per face (shaded with Lambert unless shade=False).
    sort: painter's algorithm (far faces first). cull: hide faces whose normal points away from the viewer
    (closed, outward oriented, convex meshes only; then no sorting is needed).
    extras: list of [mobject, depth] drawn among the faces by depth (e.g. normal arrows); update depth in place.
    """

    def __init__(self, V, F, cam, color=HEX["surface"], colors=None, edge_color="#3a3a3a", edge_width=0.8,
                 opacity=1.0, cull=False, sort=True, shade=True, ambient=0.45, light=(0.35, 0.45, 0.82)):
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
        self.extras = []
        self.hidden = np.zeros(len(self.F), bool)
        self.face_opacity = None
        self.polys = [Polygon(ORIGIN, ORIGIN, ORIGIN, stroke_width=edge_width, stroke_color=self.edge_color,
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
        order = np.argsort(depth) if self.sort else np.arange(len(F))
        items = [(depth[k], self.polys[k]) for k in order if vis[k]]
        for k in np.nonzero(vis)[0]:
            p = self.polys[k]
            a, b, c = P[F[k]]
            p.set_points_as_corners([a, b, c, a])
            op = self.opacity if self.face_opacity is None else self.face_opacity[k]
            p.set_fill(ManimColor(tuple(col[k])), opacity=op)
            sc = ManimColor(tuple(col[k])) if self.edge_fill else self.edge_color
            p.set_stroke(sc, width=self.edge_width, opacity=min(1.0, 1.5 * op))
        if self.extras:
            items += [(d, m) for m, d in self.extras]
            items.sort(key=lambda t: t[0])
        self.submobjects = [m for _, m in items]
        return self

    def screen(self, P):
        return self.cam(P)


# ---------------------------------------------------------------- small drawing helpers
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


def polyline(points, color=BLACK, width=3.0):
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners(np.asarray(points, float))
    return m


class Plot:
    """Minimal 2D axes (no LaTeX): data (x, y) -> scene points inside the box [x0, x1] x [y0, y1] (scene units)."""

    def __init__(self, xlim, ylim, box, logx=False, logy=False):
        self.xlim, self.ylim, self.box = xlim, ylim, box
        self.logx, self.logy = logx, logy

    def _t(self, v, lim, log):
        v = np.asarray(v, float)
        if log:
            return (np.log(v) - np.log(lim[0])) / (np.log(lim[1]) - np.log(lim[0]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def __call__(self, x, y):
        (bx0, by0), (bx1, by1) = self.box
        tx, ty = self._t(x, self.xlim, self.logx), self._t(y, self.ylim, self.logy)
        X = bx0 + tx * (bx1 - bx0)
        Y = by0 + ty * (by1 - by0)
        return np.stack([X, Y, np.zeros_like(np.asarray(X, float))], -1)

    def axes(self, color=BLACK, width=2.0, xtick=(), ytick=(), size=30, xlabels=None, ylabels=None):
        (bx0, by0), (bx1, by1) = self.box
        g = VGroup(Line([bx0, by0, 0], [bx1, by0, 0], color=color, stroke_width=width),
                   Line([bx0, by0, 0], [bx0, by1, 0], color=color, stroke_width=width))
        for k, x in enumerate(xtick):
            p = self(x, self.ylim[0])
            g.add(Line(p, p + [0, -0.1, 0], color=color, stroke_width=width))
            lab = xlabels[k] if xlabels else f"{x:g}"
            if lab is not None:
                m = lab if hasattr(lab, "move_to") else da.label(lab, size=size)
                g.add(m.next_to(p, np.array([0, -1.0, 0]), buff=0.18))
        for k, y in enumerate(ytick):
            p = self(self.xlim[0], y)
            g.add(Line(p, p + [-0.1, 0, 0], color=color, stroke_width=width))
            lab = ylabels[k] if ylabels else f"{y:g}"
            if lab is not None:
                m = lab if hasattr(lab, "move_to") else da.label(lab, size=size)
                g.add(m.next_to(p, np.array([-1.0, 0, 0]), buff=0.15))
        return g

    def curve(self, x, y, color=BLACK, width=3.0):
        return polyline(self(np.asarray(x), np.asarray(y)), color=color, width=width)


def text(s, size=32, color=BLACK, **kw):
    return da.label(s, size=size, color=color, **kw)


def math(expr, size=34, color=BLACK):
    return da.mtex(expr, size=size, color=color)


def swap(old, new):
    """Cross-fade two labels (morphing Text glyphs garbles them mid-way)."""
    return [FadeOut(old), FadeIn(new)]


# ---------------------------------------------------------------- rounded cube (Sections 31.1, 31.2)
def rounded_cube(rho, k=8):
    """Triangles (n, 3, 3) and piece labels (0 face, 1 edge cylinder, 2 vertex sphere) of the surface of points at
    distance rho from the cube [-1, 1]^3 (Figure 31.1.4(b), Proposition 31.2.6(b)), every triangle oriented outward.
    k = subdivisions of a quarter circle."""
    tris, lab = [], []
    E = np.eye(3)
    # faces
    for ax in range(3):
        for sg in (-1, 1):
            n = sg * E[ax]
            a, b = E[(ax + 1) % 3], E[(ax + 2) % 3]
            c = n * (1 + rho)
            q = [c - a - b, c + a - b, c + a + b, c - a + b]
            tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            lab += [0, 0]
    # edges: along axis ax, at the corner (s1 e_b, s2 e_c)
    t = np.linspace(0, np.pi / 2, k + 1)
    for ax in range(3):
        b, c = (ax + 1) % 3, (ax + 2) % 3
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                base = s1 * E[b] + s2 * E[c]
                dirs = [s1 * np.cos(tt) * E[b] + s2 * np.sin(tt) * E[c] for tt in t]
                for j in range(k):
                    p = [base + rho * dirs[j] + E[ax], base + rho * dirs[j + 1] + E[ax],
                         base + rho * dirs[j + 1] - E[ax], base + rho * dirs[j] - E[ax]]
                    tris += [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
                    lab += [1, 1]
    # vertices: octant of a sphere, triangular grid on the spherical triangle e_x, e_y, e_z
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner = np.array([sx, sy, sz], float)
                S = np.diag(corner)

                def pt(i, j):
                    v = np.array([k - i - j, i, j], float)
                    v = S @ (v / np.linalg.norm(v))
                    return corner + rho * v
                for i in range(k):
                    for j in range(k - i):
                        tris.append([pt(i, j), pt(i + 1, j), pt(i, j + 1)])
                        lab.append(2)
                        if i + j < k - 1:
                            tris.append([pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)])
                            lab.append(2)
    T = np.array(tris)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = np.einsum("ij,ij->i", n, T.mean(1)) < 0
    T[flip] = T[flip][:, ::-1]
    return T, np.array(lab)


def tri_area(T):
    return 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)


class Scene2D(da.Scene2D):
    """dganim.Scene2D that asserts, after every play/wait, that all labels (Text, mtex) lie inside the frame."""

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
        # with DGANIM_TIMELINE=path, append "end time  what" lines (used to write the time ranges of the captions)
        import os
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
