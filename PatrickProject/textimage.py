import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
import matplotlib.image as mpimg
import numpy as np

# ------------------ YOUR IMAGE PATHS ------------------
img_paths = [
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_blue.png",   # target
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_purple.png",
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_pink.png",
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_orange.png",
]

# ------------------ COS SCORES ------------------
cos_scores = [0.92, 0.31, 0.28, 0.25]    # replace with real numbers if you have them
max_idx = int(np.argmax(cos_scores))

# ------------------ FIGURE & CANVAS ------------------
fig = plt.figure(figsize=(9.0, 4.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()

# Dotted rounded container border
outer = FancyBboxPatch((0.02, 0.06), 0.96, 0.88,
                       boxstyle="round,pad=0.012,rounding_size=0.018",
                       linewidth=1.0, edgecolor="#56b06a", facecolor="none",
                       linestyle=(0, (4, 4)), transform=ax.transAxes)
ax.add_patch(outer)

# Title
ax.text(0.5, 0.94,
        "Text–Vision trial (SCDC): same class; size & texture fixed; color varies",
        ha="center", va="center", fontsize=12.5, weight="bold",
        transform=ax.transAxes)

# ------------------ TEXT QUERY BOX ------------------
q_box = FancyBboxPatch((0.10, 0.76), 0.28, 0.12,
                       boxstyle="round,pad=0.02,rounding_size=0.03",
                       linewidth=1.0, edgecolor="#333", facecolor="#f6f6f6",
                       transform=ax.transAxes)
ax.add_patch(q_box)
ax.text(0.24, 0.825, "Text query", ha="center", va="center",
        fontsize=11, transform=ax.transAxes)
ax.text(0.24, 0.785, "“blue ball”", ha="center", va="center",
        fontsize=12, transform=ax.transAxes)

# ------------------ ENCODER TRIANGLES ------------------
def draw_tri(cx, cy, w=0.055, h=0.095, label=r"$f_{\phi}$"):
    pts = np.array([[cx - w/2, cy - h/2],
                    [cx - w/2, cy + h/2],
                    [cx + w/2, cy       ]])
    tri = Polygon(pts, closed=True, ec="#333", fc="white", lw=1.0,
                  transform=ax.transAxes)
    ax.add_patch(tri)
    ax.text(cx - 0.010, cy, label, fontsize=12, va="center", ha="center",
            transform=ax.transAxes)

phi_cx, phi_cy = 0.88, 0.80   # text encoder
theta_cx, theta_cy = 0.88, 0.42  # visual encoder

draw_tri(phi_cx,   phi_cy, label=r"$f_{\phi}$")
draw_tri(theta_cx, theta_cy, label=r"$f_{\theta}$")
ax.text(0.93, phi_cy,  r"$T_1$", fontsize=12, va="center", ha="left", transform=ax.transAxes)
ax.text(0.93, theta_cy, r"$I$",   fontsize=12, va="center", ha="left", transform=ax.transAxes)

# Arrows to encoders (cleaner, centered)
ax.add_patch(FancyArrowPatch((0.38, 0.80), (phi_cx-0.035, phi_cy),
                             arrowstyle="->", mutation_scale=12, linewidth=1.0,
                             color="#333", transform=ax.transAxes))
# From image row to f_theta (use the middle of the row)
# We'll set this after computing the row position below.

# ------------------ IMAGE STRIP ------------------
left_x = 0.17; img_y = 0.36; w_img = 0.15; h_img = 0.19; gap = 0.04
row_w = 4*w_img + 3*gap + 0.02
row_x = left_x - 0.01
row_h = h_img + 0.065
row_y = img_y - 0.03

# Pale green backdrop for the strip
ax.add_patch(FancyBboxPatch((row_x, row_y), row_w, row_h,
                            boxstyle="round,pad=0.012,rounding_size=0.02",
                            ec="#9ad19f", fc="#eaf7ec", lw=1.0,
                            transform=ax.transAxes))

# cos(I, T1) label aligned with the score baseline
ax.text(row_x - 0.075, img_y - 0.038, r"$\cos(I,\,T_1)$",
        fontsize=12, ha="left", va="center", transform=ax.transAxes)

# Draw images with transparent axes & neat borders
for i, (p, score) in enumerate(zip(img_paths, cos_scores)):
    x = left_x + i*(w_img + gap)

    aimg = fig.add_axes([x, img_y, w_img, h_img])
    aimg.set_axis_off()
    aimg.set_facecolor((1, 1, 1, 0))       # << transparent to avoid banding
    aimg.imshow(mpimg.imread(p))

    # thin border; thicker & green for the target
    border_color = "#2ca02c" if i == max_idx else "#222222"
    border_lw    = 2.0 if i == max_idx else 1.4
    ax.add_patch(Rectangle((x-0.005, img_y-0.005),
                           w_img+0.010, h_img+0.010,
                           transform=ax.transAxes, fill=False,
                           ec=border_color, lw=border_lw))

    # centered cosine score below each
    ax.text(x + w_img/2, img_y - 0.038, f"{score:.2f}",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes)

# Arrow from the middle of the strip to f_theta
strip_mid = (row_x + row_w/2, row_y + row_h/2 - 0.01)
ax.add_patch(FancyArrowPatch(strip_mid, (theta_cx-0.035, theta_cy),
                             arrowstyle="->", mutation_scale=12, linewidth=1.0,
                             color="#333", transform=ax.transAxes))

# ------------------ Caption ------------------
ax.text(0.5, 0.085,
        "Text–Vision SCDC: choose the candidate with highest cosine similarity to the text embedding.",
        ha="center", va="center", fontsize=10.2, color="#444", transform=ax.transAxes)

plt.savefig("scdc_textvision_schematic.png", dpi=300, bbox_inches="tight")
plt.savefig("scdc_textvision_schematic.pdf", bbox_inches="tight")
plt.show()
