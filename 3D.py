import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from matplotlib.image import imread
from skimage.transform import resize

W, H = 1000, 1000
img_canvas = np.ones((H, W, 3), dtype=np.float32)

def draw_dot(x, y, clr=(0, 0, 0), size=2):
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < W and 0 <= yy < H:
                img_canvas[H - 1 - yy, xx] = clr
def line_bresenham(xa, ya, xb, yb, clr=(0, 0, 0), size=2, mask=None):
    dx, dy = abs(xb - xa), abs(yb - ya)
    sx, sy = (1 if xa < xb else -1), (1 if ya < yb else -1)
    err = dx - dy
    while True:
        if 0 <= xa < W and 0 <= ya < H:
            if mask is None or not mask[H - 1 - ya, xa]:
                draw_dot(xa, ya, clr, size)
        if xa == xb and ya == yb:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            xa += sx
        if e2 < dx:
            err += dx
            ya += sy

def circle_bresenham(xc, yc, radius, clr=(0, 0, 0), size=2):
    x, y = 0, radius
    d = 3 - 2 * radius
    while y >= x:
        for dx, dy in [(x, y), (y, x), (-x, y), (-y, x),
                       (x, -y), (y, -x), (-x, -y), (-y, -x)]:
            draw_dot(xc + dx, yc + dy, clr, size)
        x += 1
        if d > 0:
            y -= 1
            d += 4 * (x - y) + 10
        else:
            d += 4 * x + 6
def rotate_xy(x, y, cx, cy, ang_deg):
    ang = math.radians(ang_deg)
    xr = cx + (x - cx) * math.cos(ang) - (y - cy) * math.sin(ang)
    yr = cy + (x - cx) * math.sin(ang) + (y - cy) * math.cos(ang)
    return int(round(xr)), int(round(yr))
def is_inside_triangle(px, py, P1, P2, P3):
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    a = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
    b = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
    c = 1 - a - b
    return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)
def scale_triangle(P1, P2, P3, k=1.12):
    cx = (P1[0] + P2[0] + P3[0]) / 3
    cy = (P1[1] + P2[1] + P3[1]) / 3
    def scale_point(p):
        x, y = p
        return (int(cx + (x - cx) * k), int(cy + (y - cy) * k))
    return scale_point(P1), scale_point(P2), scale_point(P3)
def cyrus_beck_clip(p_start, p_end, poly):
    x1, y1 = p_start
    x2, y2 = p_end
    dx, dy = x2 - x1, y2 - y1
    t_in, t_out = 0, 1
    n = len(poly)
    for i in range(n):
        xA, yA = poly[i]
        xB, yB = poly[(i + 1) % n]
        nx, ny = yA - yB, xB - xA
        wx, wy = x1 - xA, y1 - yA
        Dn = dx * nx + dy * ny
        Wn = wx * nx + wy * ny
        if Dn == 0:
            if Wn < 0:
                return None
            continue
        t = -Wn / Dn
        if Dn > 0:
            t_in = max(t_in, t)
        else:
            t_out = min(t_out, t)
        if t_in > t_out:
            return None
    if t_in > 1 or t_out < 0:
        return None
    return t_in, t_out
P1, P2, P3 = (300, 650), (500, 300), (700, 650)
C0 = (500, 500)
r_big, r_small, R_outer = 85, 70, 350
P1, P2, P3 = [rotate_xy(x, y, C0[0], C0[1], 180) for (x, y) in [P1, P2, P3]]
P1s, P2s, P3s = scale_triangle(P1, P2, P3, k=1.12)

img_path = r"C:\Users\fivez\Desktop\photo_3d.jpg"

try:
    texture = imread(img_path)
    if texture.dtype != np.uint8 and texture.max() > 1.0:
        texture = texture / 255.0
    tex_resized = resize(texture, (H, W), anti_aliasing=True)
    mask_tex = np.zeros((H, W), dtype=bool)
    for xx in range(W):
        for yy in range(H):
            if is_inside_triangle(xx, yy, P1, P2, P3):
                dist = math.sqrt((xx - C0[0])**2 + (yy - C0[1])**2)
                if dist > r_big:
                    mask_tex[H - 1 - yy, xx] = True
    img_canvas[mask_tex] = tex_resized[mask_tex]
except FileNotFoundError:
    print("фото не найдено")
line_bresenham(*P1, *P2, size=3)
line_bresenham(*P2, *P3, size=3)
line_bresenham(*P3, *P1, size=3)

def dashed(p1, p2, dash=60, gap=20):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    pos = 0.0
    while pos < L:
        end = min(pos + dash, L)
        t1, t2 = pos / L, end / L
        xs, ys = int(x1 + dx * t1), int(y1 + dy * t1)
        xe, ye = int(x1 + dx * t2), int(y1 + dy * t2)
        line_bresenham(xs, ys, xe, ye, size=2)
        pos += dash + gap

dashed(P1s, P2s)
dashed(P2s, P3s)
dashed(P3s, P1s)
for ang in range(0, 360, 60):
    for a in np.linspace(ang, ang + 30, 300):
        x = int(C0[0] + r_small * math.cos(math.radians(a)))
        y = int(C0[1] + r_small * math.sin(math.radians(a)))
        draw_dot(x, y, size=2)

circle_bresenham(C0[0], C0[1], r_big, size=2)
for a in np.linspace(-50, 180, 600):
    x = int(C0[0] + R_outer * math.cos(math.radians(a + 180)))
    y = int(C0[1] + R_outer * math.sin(math.radians(a + 180)))
    draw_dot(x, y, size=2)
seg_start = (80, 700)
seg_end = (950, 475)
clip_res = cyrus_beck_clip(seg_start, seg_end, [P1, P2, P3])
if clip_res:
    t_in, t_out = clip_res
    dx, dy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]
    entry = (int(seg_start[0] + dx * t_in), int(seg_start[1] + dy * t_in))
    exit_ = (int(seg_start[0] + dx * t_out), int(seg_start[1] + dy * t_out))
    line_bresenham(seg_start[0], seg_start[1], entry[0], entry[1], size=2)
    line_bresenham(exit_[0], exit_[1], seg_end[0], seg_end[1], size=2)
else:
    line_bresenham(*seg_start, *seg_end, size=2)

plt.figure(figsize=(10, 10))
plt.imshow(img_canvas)
plt.axis('off')
plt.savefig("output.png", dpi=300)
