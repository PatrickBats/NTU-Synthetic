import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
import matplotlib.image as mpimg
import numpy as np

# -----------------------------
# 1) YOUR IMAGE PATHS (Windows)
# -----------------------------
img_paths = [
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_blue.png",   # target
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_purple.png",
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_pink.png",
    r"C:\Users\jbats\Projects\NTU-Synthetic\data\SyntheticKonkle_224\SyntheticKonkle\ball_color\ball_medium_smooth_01_orange.png",
]

# --------------------------------------------
# 2) COSINE SCORES (set your real numbers here)
# --------------------------------------------
cos_scores = [0.92, 0.31, 0.28, 0.25]   # highest will be highlighted

# ------------- Figure canvas ----------------
fig = plt.figure(figsize=(9.2, 4.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()

# Subtle rounded dashed container
outer = FancyBboxPatch(
    (0.02, 0.06), 0.96, 0.88,
    boxstyle="round,pad=0.012,rounding_size=0.018",
    linewidth=1.0, edgecolor="#56b06a", facecolor="none",
    linestyle=(0, (4, 4)), transform=ax.transAxes
)
ax.add_patch(outer)

# ---------------- Title ---------------------
ax.text(0.5, 0.94, "Text–Vision trial (SCDC): same class; size & texture fixed; color varies",
        ha="center", va="center", fontsize=12.5, weight="bold", transform=ax.transAxes)

# ------------- Text query box ---------------
q_box = FancyBboxPatch(
    (0.12, 0.74), 0.28, 0.12,
    boxstyle="round,pad=0.02,rounding_size=0.03",
    linewidth=1.0, edgecolor="#333", facecolor="#f6f6f6",
    transform=ax.transAxes
)
ax.add_patch(q_box)
ax.text(0.26, 0.81, "Text query", ha="center", va="center",
        fontsize=11, transform=ax.transAxes)
ax.text(0.26, 0.76, "“blue ball”", ha="center", va="center",
        fontsize=12, transform=ax.transAxes)

# ----------- Encoders (triangular modules) -----------
# Positions for triangle centers
phi_cx, phi_cy = 0.88, 0.78   # text encoder (top)
theta_cx, theta_cy = 0.88, 0.42  # visual encoder (bottom)

def tri(cx, cy, w=0.06, h=0.10, label=r"$f_{\phi}$"):
    """Right-pointing triangle with a label."""
    pts = np.array([
        [cx - w/2, cy - h/2],
        [cx - w/2, cy + h/2],
        [cx + w/2, cy       ],
    ])
    pg = Polygon(pts, closed=True, ec="#333", fc="white", lw=1.0,
                 transform=ax.transAxes)
    ax.add_patch(pg)
    ax.text(cx - 0.012, cy, label, fontsize=12, va="center", ha="center",
            transform=ax.transAxes)

tri(phi_cx,   phi_cy,  label=r"$f_{\phi}$")
tri(theta_cx, theta_cy, label=r"$f_{\theta}$")

# ----------- Arrows to encoders -----------
# From text query to f_phi
ax.add_patch(FancyArrowPatch((0.40, 0.80), (phi_cx-0.035, phi_cy),
                             arrowstyle="->", mutation_scale=12,
                             linewidth=1.0, color="#333", transform=ax.transAxes))
# From image row (middle) to f_theta
ax.add_patch(FancyArrowPatch((0.50, 0.36), (theta_cx-0.035, theta_cy),
                             arrowstyle="->", mutation_scale=12,
                             linewidth=1.0, color="#333", transform=ax.transAxes))

# Optional: outputs T1 and I to the right of encoders
ax.text(0.93, phi_cy, r"$T_1$", fontsize=12, va="center", ha="left", transform=ax.transAxes)
ax.text(0.93, theta_cy, r"$I$", fontsize=12, va="center", ha="left", transform=ax.transAxes)

# ----------- Image row (same class) -----------
# left margin for the strip & geometry
left_x  = 0.18
img_y   = 0.38
w_img   = 0.15
h_img   = 0.18
gap     = 0.04

# Soft green container behind the row (like the paper style)
row_w = 4*w_img + 3*gap + 0.02
row_x = left_x - 0.01
row_h = h_img + 0.055
row_y = img_y - 0.028
ax.add_patch(FancyBboxPatch(
    (row_x, row_y), row_w, row_h,
    boxstyle="round,pad=0.01,rounding_size=0.02",
    ec="#9ad19f", fc="#eaf7ec", lw=1.0, transform=ax.transAxes)
)

# cos(I,T1) label on the left of the row
ax.text(row_x - 0.07, img_y - 0.04, r"$\cos(I,\,T_1)$",
        fontsize=12, ha="left", va="center", transform=ax.transAxes)

# Load & draw images + scores
max_idx = int(np.argmax(cos_scores))
for i, (p, score) in enumerate(zip(img_paths, cos_scores)):
    x = left_x + i*(w_img + gap)
    # place image in a small axes
    aimg = fig.add_axes([x, img_y, w_img, h_img])
    aimg.imshow(mpimg.imread(p))
    aimg.axis("off")

    # image border (green if max)
    border_color = "#2ca02c" if i == max_idx else "#222"
    ax.add_patch(Rectangle((x-0.005, img_y-0.005),
                           w_img+0.010, h_img+0.010,
                           transform=ax.transAxes, fill=False,
                           ec=border_color, lw=1.8))

    # cosine score under image
    ax.text(x + w_img/2, img_y - 0.035, f"{score:.2f}",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)

# ------------- Footer caption -------------
ax.text(0.5, 0.085,
        "Text–Vision SCDC: choose the candidate with highest cosine similarity to the text embedding.",
        ha="center", va="center", fontsize=10.2, color="#444", transform=ax.transAxes)

# Save
plt.savefig("scdc_textvision_schematic.png", dpi=300, bbox_inches="tight")
plt.savefig("scdc_textvision_schematic.pdf", bbox_inches="tight")
plt.show()
