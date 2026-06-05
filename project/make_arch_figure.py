"""Render the late-fusion architecture diagram to figures/fusion_architecture.png."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

C_IN   = "#DCE6F4"; E_IN   = "#3E6DB5"
C_MOD  = "#ECECEC"; E_MOD  = "#555555"
C_FUSE = "#FBE2C4"; E_FUSE = "#C7791F"
C_OUT  = "#D9EBD6"; E_OUT  = "#3E8E41"
ARROW  = "#444444"

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(-1.6, 14.7)
ax.set_ylim(-3.0, 3.0)
ax.axis("off")

boxes = {}
LSP = 0.40


def box(name, cx, cy, w, h, face, edge, title, subs):
    boxes[name] = (cx, cy, w, h)
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3))
    lines = len(subs) + 1
    top = cy + (lines - 1) / 2 * LSP
    ax.text(cx, top, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#1c1c1c", zorder=4)
    for i, s in enumerate(subs):
        ax.text(cx, top - LSP * (i + 1), s, ha="center", va="center",
                fontsize=10.0, color="#333333", zorder=4)


def arrow(src, dst, src_side="right", dst_side="left"):
    sx, sy, sw, sh = boxes[src]
    dx, dy, dw, dh = boxes[dst]
    p_src = {"right": (sx + sw / 2, sy), "left": (sx - sw / 2, sy)}[src_side]
    p_dst = {"right": (dx + dw / 2, dy), "left": (dx - dw / 2, dy)}[dst_side]
    ax.add_patch(FancyArrowPatch(
        p_src, p_dst, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=ARROW, shrinkA=2, shrinkB=2, zorder=2))


# inputs (left column)
box("text", 0.0,  1.70, 2.8, 1.15, C_IN, E_IN, "Text",    ["caption (EN / ES)"])
box("img",  0.0, -0.55, 2.8, 1.15, C_IN, E_IN, "Image",   ["CLIP feature, 512-d"])
box("phys", 0.0, -1.95, 2.8, 1.30, C_IN, E_IN, "Sensors", ["ET + HR + EEG", "104-d"])

# model branches (middle column)
box("xlmr", 4.85,  1.70, 4.4, 1.35, C_MOD, E_MOD, "XLM-RoBERTa",
    ["fine-tuned", r"$\rightarrow\ p_{\mathrm{text}}$"])
box("lr",   4.85, -1.25, 4.4, 1.35, C_MOD, E_MOD, "Logistic regression",
    [r"img $\oplus$ sens  (616-d)", r"$\rightarrow\ p_{\mathrm{img+sens}}$"])

# fusion + prediction (centred between the two branches)
box("fuse", 9.2,  0.225, 3.2, 1.55, C_FUSE, E_FUSE, "Late fusion",
    [r"$\alpha\, p_{\mathrm{text}}$", r"$+\,(1-\alpha)\, p_{\mathrm{img+sens}}$"])
box("pred", 13.0, 0.225, 2.5, 1.35, C_OUT, E_OUT, "Prediction",
    ["hard + soft"])

# arrows
arrow("text", "xlmr")
arrow("img",  "lr")
arrow("phys", "lr")
arrow("xlmr", "fuse")
arrow("lr",   "fuse")
arrow("fuse", "pred")

out = FIG / "fusion_architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {out}")
