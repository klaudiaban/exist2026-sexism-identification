"""Regenerate the report comparison figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
FIG  = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           12,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.labelsize":      12,
    "axes.edgecolor":      "#333333",
    "axes.linewidth":      0.9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.color":         "#333333",
    "ytick.color":         "#333333",
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     11,
    "legend.frameon":      False,
    "savefig.dpi":         200,
    "savefig.bbox":        "tight",
    "figure.facecolor":    "white",
})

C_BASE  = "#5B7FA8"   # muted blue
C_FINAL = "#C75D2C"   # muted terracotta
C_GREY  = "#8C8C8C"   # text-only CV bar

CV_FILE = ROOT / "cv_phys_results.json"


def annotate(ax, bars, fmt="{:.3f}", weight="normal"):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h),
                    xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10, color="#222222", fontweight=weight)


def cv_phys_plot():
    cv = json.loads(CV_FILE.read_text())
    tasks = ["T2.1", "T2.2", "T2.3"]
    keys  = ["T2.1", "T2.2", "T2.3"]
    confs = [("Text-only",          "Text-only",          C_GREY),
             ("Text + Image",       "Text + Image",       C_BASE),
             ("Text + Image + Phys","Text + Image + Phys",C_FINAL)]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x      = np.arange(len(tasks))
    width  = 0.26

    for i, (cv_key, label, color) in enumerate(confs):
        means = [cv[cv_key][k]["mean"] for k in keys]
        stds  = [cv[cv_key][k]["std"]  for k in keys]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      capsize=3.5, ecolor="#333333",
                      color=color, label=label,
                      edgecolor="white", linewidth=0.7, zorder=3)
        annotate(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} ({n})" for t, n in
                        zip(tasks, ["binary", "type", "category"])])
    ax.set_ylabel("Macro-F1 (5-fold CV)")
    ax.set_ylim(0, 0.72)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("5-fold cross-validation: effect of modality")
    ax.legend(loc="upper right", ncol=1)

    fig.savefig(FIG / "cv_phys_comparison.png")
    plt.close(fig)
    print("  wrote cv_phys_comparison.png")


def task_comparison_plot(name, title, metric_labels, base_vals, final_vals,
                         out_name, ylim=None, legend_loc=None):
    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    bb = ax.bar(x - width/2, base_vals,  width, color=C_BASE,
                edgecolor="white", linewidth=0.7,
                label="XLM-R-base (text-only)", zorder=3)
    bf = ax.bar(x + width/2, final_vals, width, color=C_FINAL,
                edgecolor="white", linewidth=0.7,
                label="Final model", zorder=3)
    annotate(ax, bb)
    annotate(ax, bf, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    if ylim is None:
        all_vals = [v for v in (base_vals + final_vals) if not np.isnan(v)]
        ymax = min(1.08, max(all_vals) + 0.18)
        ymin = max(0, min(all_vals) - 0.05)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, frameon=False, handlelength=1.5,
              columnspacing=1.6, borderaxespad=0.2)
    fig.savefig(FIG / out_name)
    plt.close(fig)
    print(f"  wrote {out_name}")


def main():
    cv_phys_plot()

    task_comparison_plot(
        "T2.1", "Task 2.1 -- Binary sexism detection",
        ["F1-Macro", "F1 (YES)", "Accuracy", "AUC"],
        base_vals=[0.422, 0.734, 0.591, 0.386],
        final_vals=[0.794, 0.821, 0.797, 0.867],
        out_name="comparison_T21.png",
        ylim=(0.30, 1.02),
        legend_loc="lower right",
    )
    task_comparison_plot(
        "T2.2", "Task 2.2 -- Sexism type (3-class)",
        ["F1-Macro", "F1 (sexist)", "Accuracy", "AUC"],
        base_vals=[0.435, 0.304, 0.608, 0.741],
        final_vals=[0.635, 0.579, 0.674, 0.827],
        out_name="comparison_T22.png",
        ylim=(0.20, 0.98),
        legend_loc="upper left",
    )
    task_comparison_plot(
        "T2.3", "Task 2.3 -- Sexism category (multi-label)",
        ["F1-Macro", "Accuracy", "AUC"],
        base_vals=[0.000, 0.895, 0.614],
        final_vals=[0.357, 0.724, 0.821],
        out_name="comparison_T23.png",
        ylim=(0.0, 1.08),
        legend_loc="upper left",
    )


if __name__ == "__main__":
    main()
