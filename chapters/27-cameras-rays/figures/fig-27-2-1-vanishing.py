"""Figure 27.2.1: vanishing points and the vanishing line of the ground plane (Section 27.2).

World: right-handed with y_w pointing DOWN (the OpenCV/COLMAP habit), so that the top view (a) with x_w to the right and
z_w up is the scene seen from above (with y_w up the same picture would be a mirror image). Ground plane y_w = 0,
camera center c = (0, -1.5, 0) (height 1.5), yaw 30 deg toward +x_w and pitch 10 deg down;
K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]] (a 640 x 480 image).
Ground grid: family A (blue) = lines parallel to the world x axis, z_w = 3, 4, ..., 12 (x_w in [-300, 300] in the image,
[-6, 6] in the top view); family B (green) = lines parallel to the world z axis, x_w = -3, ..., 3, z_w in [1, 600].
(a) Top view of the scene (x_w, z_w) with the camera center and the horizontal field of view.
(b) The image plane (pixel coordinates, extended beyond the 640 x 480 frame drawn in gray). Black: the vanishing line
    l = K^{-T} R n of the ground plane (n = up = (0, -1, 0)). Blue / green dots: vanishing points v_A = K R e_1, v_B = K R e_3.
Self-checks: v_A and v_B lie on l; long lines of each family pass near their vanishing point; the image of the camera's
horizontal plane through c is l; the camera's right axis points to the right of the viewing direction in the top view
(no mirror); v_B lies left of the image center because +z_w is to the left of the viewing direction.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-2-1-vanishing.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
yaw, pitch = np.radians(30), np.radians(10)
# world: x_w right, y_w down, z_w forward (right-handed). camera axes in world coordinates:
# columns of R_wc are camera x (right), y (down), z (forward)
fwd = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])   # pitch down = +y_w
down_w = np.array([0, 1.0, 0])
right = np.cross(down_w, fwd)          # x = y x z for the right-handed (right, down, forward) frame
right /= np.linalg.norm(right)
down = np.cross(fwd, right)
Rcw = np.column_stack([right, down, fwd])          # camera -> world
R = Rcw.T                                          # world -> camera
assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
# no mirror: in the top view (x_w, z_w) the camera's right axis is the forward direction turned clockwise
assert np.allclose(right[[0, 2]], [fwd[2], -fwd[0]] / np.hypot(fwd[0], fwd[2]))
assert down[1] > 0                                 # camera y points down
cpos = np.array([0, -1.5, 0])
T = -R @ cpos
P = K @ np.hstack([R, T[:, None]])
n = -down_w                                        # up
lvan = np.linalg.inv(K).T @ R @ n
vA = K @ R @ np.array([1.0, 0, 0])
vB = K @ R @ np.array([0, 0, 1.0])
assert abs(lvan @ vA) < 1e-9 and abs(lvan @ vB) < 1e-9
assert vA[0] / vA[2] > 640 and 0 < vB[0] / vB[2] < 320       # v_A right of the frame, v_B left of the center
vA2, vB2 = vA[:2] / vA[2], vB[:2] / vB[2]


def proj(X):
    h = P @ np.append(X, 1)
    return h[:2] / h[2]


far = proj(np.array([3.0, 0, 1e6]))
assert np.linalg.norm(far - vB2) < 1e-2

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(width_ratios=[0.8, 1.35]))
# ---------------------------------------------------------------- (a) top view
for zw in range(3, 13):
    ax.plot([-6, 6], [zw, zw], color=C["tangent"], lw=0.8)
for xw in range(-3, 4):
    ax.plot([xw, xw], [1, 14], color=C["third"], lw=0.8)
hf = np.arctan(320 / 500)
for s in (-1, 1):
    a = yaw + s * hf
    ax.plot([0, 14 * np.sin(a)], [0, 14 * np.cos(a)], color=C["aux"], lw=0.8, ls="--")
ax.annotate("", xy=(3 * np.sin(yaw), 3 * np.cos(yaw)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
ax.plot(0, 0, "ko", ms=5)
ax.text(0.25, -0.7, r"camera $c$", fontsize=11)
ax.text(-5.8, 12.4, "family A", fontsize=11, color=C["tangent"])
ax.text(3.2, 13.2, "family B", fontsize=11, color=C["third"])
ax.set_xlim(-6.2, 6.2)
ax.set_ylim(-1.2, 14)
ax.set_aspect("equal")
ax.set_xlabel(r"$x_w$")
ax.set_ylabel(r"$z_w$")
ax.set_title("(a) ground plane, top view", fontsize=12)

# ---------------------------------------------------------------- (b) image
def draw_line_world(X0, X1, col, lw=0.9):
    s = np.concatenate([np.linspace(0, 1, 4000), [0.5]])
    s = np.sort(s)
    pts = np.array([X0 + si * (X1 - X0) for si in s])
    tz = (R @ pts.T + T[:, None])[2]
    uv = np.array([proj(p) for p in pts])
    uv[tz <= 0.05] = np.nan
    bx.plot(uv[:, 0], uv[:, 1], color=col, lw=lw)


for zw in range(3, 13):
    draw_line_world(np.array([-300, 0, zw]), np.array([300, 0, zw]), C["tangent"])
for xw in range(-3, 4):
    draw_line_world(np.array([xw, 0, 1.0]), np.array([xw, 0, 600.0]), C["third"])
uu = np.linspace(-700, 1900, 10)
bx.plot(uu, -(lvan[0] * uu + lvan[2]) / lvan[1], color="k", lw=1.6)
bx.plot(*vA2, "o", color=C["tangent"], ms=8, zorder=6)
bx.plot(*vB2, "o", color=C["third"], ms=8, zorder=6)
bx.text(vA2[0] + 40, vA2[1] + 30, r"$v_A=\mathsf{K}Re_1$", fontsize=12, color=C["tangent"], ha="left", va="top")
bx.text(vB2[0] - 30, vB2[1] - 60, r"$v_B=\mathsf{K}Re_3$", fontsize=12, color=C["third"], ha="center", va="bottom")
bx.text(1880, -(lvan[0] * 1880 + lvan[2]) / lvan[1] - 25, r"vanishing line $l=\mathsf{K}^{-\mathsf{T}}R\,n$",
        fontsize=12, va="bottom", ha="right")
bx.plot([0, 640, 640, 0, 0], [0, 0, 480, 480, 0], color=C["aux"], lw=1.0)
bx.text(640, -15, "640 x 480 frame", fontsize=10, color=C["aux"], va="bottom", ha="right")
bx.set_xlim(-700, 1900)
bx.set_ylim(900, -200)
bx.set_aspect("equal")
bx.set_xlabel(r"$u_x$ (px)")
bx.set_ylabel(r"$u_y$ (px, down)")
bx.set_title("(b) image: parallel lines meet on $l$", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)
print("vA", vA2, "vB", vB2, "l", lvan / np.linalg.norm(lvan[:2]))
