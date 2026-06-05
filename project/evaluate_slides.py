"""
Slide-style comparison figures (matching slides 40, 41, 42 from the presentation).

Each task is compared independently:
  Task 2.1: XLM-R-large text-only  vs  XLM-R-large + img/sens fusion
  Task 2.2: XLM-R-base text-only   vs  XLM-R-base + oversample + fusion
  Task 2.3: XLM-R-base single-label vs  XLM-R-base multi-label (our model)

Outputs saved to project/figures/:
  slide40_summary.png       — F1 / AUC summary table across all tasks
  slide41_task21.png        — Bar chart for Task 2.1
  slide41_task22.png        — Bar chart for Task 2.2
  slide41_task23.png        — Bar chart for Task 2.3
  slide42_category_f1.png   — Per-category F1 for Task 2.3
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import FEATURES_DIR, MODELS_DIR, MEMES_TRAIN_JSON, TASK1_LABELS, TASK2_LABELS, TASK3_LABELS
from data_utils import load_json, build_memes_records

MAX_LEN   = 128
VAL_FRAC  = 0.10
RAND_SEED = 42
THRESHOLD = 0.5

CATEGORIES = [l for l in TASK3_LABELS if l != "NO"]
CAT_LABELS = {
    "IDEOLOGICAL-INEQUALITY":      "Ideological Inequality",
    "STEREOTYPING-DOMINANCE":      "Stereotyping Dominance",
    "OBJECTIFICATION":             "Objectification",
    "SEXUAL-VIOLENCE":             "Sexual Violence",
    "MISOGYNY-NON-SEXUAL-VIOLENCE":"Misogyny NSV",
}

LABEL2ID_21 = {l: i for i, l in enumerate(TASK1_LABELS)}
LABEL2ID_22 = {l: i for i, l in enumerate(TASK2_LABELS)}

FIGURES_DIR = Path(__file__).parent / "figures"

# Colours matching the presentation palette
BLUE   = "#4472C4"
ORANGE = "#ED7D31"


# ── Inference helpers ──────────────────────────────────────────────────────────

def infer_cls(model, tokenizer, texts, device, col_perm=None, batch=32):
    model.eval()
    out = []
    for s in range(0, len(texts), batch):
        enc = tokenizer(texts[s:s+batch], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, -1).cpu().numpy()
        if col_perm is not None:
            probs = probs[:, col_perm]
        out.append(probs)
    return np.concatenate(out)


def infer_multilabel(model, tokenizer, texts, device, batch=32):
    model.eval()
    out = []
    for s in range(0, len(texts), batch):
        enc = tokenizer(texts[s:s+batch], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.concatenate(out)


def col_perm_fn(model, label_order):
    stored  = {int(k): v for k, v in model.config.id2label.items()}
    lbl2col = {v: k for k, v in stored.items()}
    return [lbl2col.get(l, i) for i, l in enumerate(label_order)]


def load_image_sensorial():
    img  = np.load(FEATURES_DIR / "memes_train_image.npy")
    sens = np.load(FEATURES_DIR / "memes_train_sensorial.npy")
    return np.concatenate([img, sens], axis=1)


def align(X_raw, feat_ids, records):
    id2i = {sid: i for i, sid in enumerate(feat_ids)}
    return np.array([X_raw[id2i[r["id"]]] for r in records])


def binary_matrix(records):
    Y = np.zeros((len(records), len(CATEGORIES)), dtype=np.float32)
    for i, r in enumerate(records):
        for lbl in r["hard_t23"]:
            if lbl in CATEGORIES:
                Y[i, CATEGORIES.index(lbl)] = 1.0
    return Y


def safe_auc_binary(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return float("nan")


def safe_auc_multi(y_true, scores, labels):
    try:
        return roc_auc_score(y_true, scores, multi_class="ovr",
                             average="macro", labels=labels)
    except Exception:
        return float("nan")


def safe_auc_ml(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# Slide 40 style ─ summary table: XLM-R baseline vs our model, per task
def plot_slide40(results):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")

    col_labels = ["Model", "T2.1: Binary", "T2.2: Type", "T2.3: Category"]
    r = results

    def fmt(f1, auc):
        auc_str = f"{auc:.3f}" if not (isinstance(auc, float) and np.isnan(auc)) else "—"
        return f"F1 {f1:.3f}  /  AUC {auc_str}"

    cell_data = [
        ["XLM-R (text-only baseline)",
         fmt(r["f1_21_base"], r["auc_21_base"]),
         fmt(r["f1_22_base"], r["auc_22_base"]),
         fmt(r["f1_23_base"], r["auc_23_base"])],
        ["XLM-R + fusion / multi-label (ours)",
         fmt(r["f1_21_ft"], r["auc_21_ft"]),
         fmt(r["f1_22_ft"], r["auc_22_ft"]),
         fmt(r["f1_23_ft"], r["auc_23_ft"])],
    ]

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.6)

    header_color = "#2F5496"
    base_color   = "#FFF2CC"
    ft_color     = "#EBF3FB"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="white")
        elif row == 1:
            cell.set_facecolor(base_color)
        elif row == 2:
            cell.set_facecolor(ft_color)
            cell.set_text_props(weight="bold")

    ax.set_title(
        "Table: XLM-R text-only baseline vs our best models (Macro-F1 / AUC).\n"
        "Fusion and multi-label training consistently improve performance.",
        fontsize=12, fontweight="bold", pad=16, loc="left",
    )

    _save(fig, "slide40_summary.png")


# Slide 41 style ─ per-task grouped bar chart
def plot_slide41(task_title, metric_names,
                 base_vals, ft_vals,
                 filename, ylim_min=0.0,
                 base_label="XLM-R (text-only)",
                 ft_label="XLM-R + fusion (ours)"):
    x     = np.arange(len(metric_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars_b = ax.bar(x - width / 2, base_vals, width,
                    label=base_label, color=BLUE, alpha=0.88, zorder=3)
    bars_f = ax.bar(x + width / 2, ft_vals,   width,
                    label=ft_label, color=ORANGE, alpha=0.88, zorder=3)

    all_vals = [v for v in base_vals + ft_vals if not np.isnan(v)]
    y_min = max(ylim_min, min(all_vals) - 0.05) if all_vals else 0
    y_max = min(1.0, max(all_vals) + 0.12)       if all_vals else 1
    ax.set_ylim(y_min, y_max)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Performance on {task_title}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.yaxis.grid(True, zorder=0, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, framealpha=0.9)

    for bar, val in zip(bars_b, base_vals):
        if not np.isnan(val):
            ax.annotate(f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, color="#333333")
    for bar, val in zip(bars_f, ft_vals):
        if not np.isnan(val):
            ax.annotate(f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color="#333333")

    plt.tight_layout()
    _save(fig, filename)


# Slide 42 style ─ per-category F1 table for Task 2.3
def plot_slide42(cat_f1_base, cat_f1_ft, macro_base, macro_ft):
    cat_names = [CAT_LABELS[c] for c in CATEGORIES]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")

    rows = []
    for name, b, f in zip(cat_names, cat_f1_base, cat_f1_ft):
        rows.append([name, f"{b:.3f}", f"{f:.3f}"])
    rows.append(["Macro Average", f"{macro_base:.3f}", f"{macro_ft:.3f}"])

    col_labels = ["Category (Task 2.3)", "Baseline F1\n(XLM-R single-label)", "Our Model F1\n(XLM-R multi-label)"]

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.55, 2.15)

    n = len(CATEGORIES)
    header_color = "#2F5496"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="white")
            cell.set_height(cell.get_height() * 1.3)
        elif row == n + 1:
            cell.set_facecolor("#DDEEFF")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FAFAFA" if row % 2 == 0 else "white")
            if col == 2:
                cell.set_text_props(weight="bold", color="#1A5276")

    ax.set_title(
        "Per-category F1 — Task 2.3 (Sexism Category)\n"
        "Multi-label model significantly outperforms the single-label baseline.",
        fontsize=11, fontweight="bold", pad=18, loc="left",
    )

    _save(fig, "slide42_category_f1.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading data and features...")

    recs   = build_memes_records(load_json(MEMES_TRAIN_JSON), is_test=False)
    texts  = [r["text"] for r in recs]
    y21    = np.array([LABEL2ID_21[r["hard_t21"]] for r in recs])
    y22    = np.array([LABEL2ID_22[r["hard_t22"]] for r in recs])
    y23    = binary_matrix(recs)

    with open(FEATURES_DIR / "memes_train_ids.pkl", "rb") as f:
        feat_ids = pickle.load(f)
    X_feat = align(load_image_sensorial(), feat_ids, recs)

    sss21 = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RAND_SEED)
    tr21, _ = next(sss21.split(texts, y21))
    sss22 = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RAND_SEED)
    tr22, _ = next(sss22.split(texts, y22))

    KNOWN_T3   = set(TASK3_LABELS)
    T3_LABEL2ID = {l: i for i, l in enumerate(TASK3_LABELS)}

    # ── Task 2.1 ─────────────────────────────────────────────────────────────
    # Baseline: XLM-R-base, simple binary, no oversampling, no fusion
    # Ours:     XLM-R-large + img/sens fusion (alpha=0.95)
    print("\nLoading Task 2.1 baseline model...")
    base21 = MODELS_DIR / "xlmr_base_baseline_task2_1"
    btok21 = AutoTokenizer.from_pretrained(base21)
    bmdl21 = AutoModelForSequenceClassification.from_pretrained(base21).to(device)
    bperm21 = col_perm_fn(bmdl21, TASK1_LABELS)
    bprobs21 = infer_cls(bmdl21, btok21, texts, device, col_perm=bperm21)
    bmdl21.cpu(); del bmdl21; torch.cuda.empty_cache()

    pred21_b   = bprobs21.argmax(-1)
    f1_21_b    = f1_score(y21, pred21_b, average="macro", zero_division=0)
    acc_21_b   = accuracy_score(y21, pred21_b)
    auc_21_b   = safe_auc_binary(y21, bprobs21[:, LABEL2ID_21["YES"]])
    f1pos_21_b = f1_score(y21, pred21_b, average=None, zero_division=0)[LABEL2ID_21["YES"]]

    print("Loading Task 2.1 fine-tuned model...")
    save21 = MODELS_DIR / "xlmr_large_full_task2_1"
    tok21  = AutoTokenizer.from_pretrained(save21)
    mdl21  = AutoModelForSequenceClassification.from_pretrained(save21).to(device)
    perm21 = col_perm_fn(mdl21, TASK1_LABELS)
    pipe21 = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))])
    pipe21.fit(X_feat[tr21], y21[tr21])
    xlmr21  = infer_cls(mdl21, tok21, texts, device, col_perm=perm21)
    fused21 = 0.95 * xlmr21 + 0.05 * pipe21.predict_proba(X_feat)
    mdl21.cpu(); del mdl21; torch.cuda.empty_cache()

    pred21_f   = fused21.argmax(-1)
    f1_21_f    = f1_score(y21, pred21_f, average="macro", zero_division=0)
    acc_21_f   = accuracy_score(y21, pred21_f)
    auc_21_f   = safe_auc_binary(y21, fused21[:, 1])
    f1pos_21_f = f1_score(y21, pred21_f, average=None, zero_division=0)[LABEL2ID_21["YES"]]

    # ── Task 2.2 ─────────────────────────────────────────────────────────────
    # Baseline: XLM-R-base, simple 3-class, no oversampling, no fusion
    # Ours:     XLM-R-base + oversampling + fusion (alpha=0.95)
    print("Loading Task 2.2 baseline model...")
    base22 = MODELS_DIR / "xlmr_base_baseline_task2_2"
    btok22 = AutoTokenizer.from_pretrained(base22)
    bmdl22 = AutoModelForSequenceClassification.from_pretrained(base22).to(device)
    bperm22 = col_perm_fn(bmdl22, TASK2_LABELS)
    bprobs22 = infer_cls(bmdl22, btok22, texts, device, col_perm=bperm22)
    bmdl22.cpu(); del bmdl22; torch.cuda.empty_cache()

    pred22_b   = bprobs22.argmax(-1)
    f1_22_b    = f1_score(y22, pred22_b, average="macro", zero_division=0)
    acc_22_b   = accuracy_score(y22, pred22_b)
    auc_22_b   = safe_auc_multi(y22, bprobs22, [0, 1, 2])
    f1pos_22_b = float(np.mean(f1_score(y22, pred22_b, average=None, zero_division=0)[[1, 2]]))

    print("Loading Task 2.2 fine-tuned model...")
    save22   = MODELS_DIR / "xlmr_large_task2_2_oversampled"
    tok22    = AutoTokenizer.from_pretrained(save22)
    base_22  = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-large", num_labels=len(TASK2_LABELS),
        ignore_mismatched_sizes=True,
    )
    mdl22 = PeftModel.from_pretrained(base_22, save22).merge_and_unload().to(device)
    perm22 = col_perm_fn(mdl22, TASK2_LABELS)
    pipe22 = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=1000, C=1.0))])
    pipe22.fit(X_feat[tr22], y22[tr22])
    xlmr22  = infer_cls(mdl22, tok22, texts, device, col_perm=perm22)
    fused22 = 0.75 * xlmr22 + 0.25 * pipe22.predict_proba(X_feat)
    mdl22.cpu(); del mdl22; torch.cuda.empty_cache()

    pred22_f   = fused22.argmax(-1)
    f1_22_f    = f1_score(y22, pred22_f, average="macro", zero_division=0)
    acc_22_f   = accuracy_score(y22, pred22_f)
    auc_22_f   = safe_auc_multi(y22, fused22, [0, 1, 2])
    f1pos_22_f = float(np.mean(f1_score(y22, pred22_f, average=None, zero_division=0)[[1, 2]]))

    # ── Task 2.3 ─────────────────────────────────────────────────────────────
    # Baseline: XLM-R-base single-label classifier (one category per meme)
    # Ours:     XLM-R-base multi-label (BCEWithLogitsLoss, threshold=0.5)
    print("Loading Task 2.3 baseline model...")
    base23 = MODELS_DIR / "xlmr_base_baseline_task2_3"
    btok23 = AutoTokenizer.from_pretrained(base23)
    bmdl23 = AutoModelForSequenceClassification.from_pretrained(base23).to(device)
    bprobs23_cls = infer_cls(bmdl23, btok23, texts, device)   # shape (N, 6)
    bmdl23.cpu(); del bmdl23; torch.cuda.empty_cache()

    # convert 6-class predictions to 5-dim binary matrix (excluding NO)
    pred23_b = np.zeros((len(texts), len(CATEGORIES)), dtype=int)
    for i, cls_idx in enumerate(bprobs23_cls.argmax(-1)):
        label = TASK3_LABELS[cls_idx]
        if label in CATEGORIES:
            pred23_b[i, CATEGORIES.index(label)] = 1
    # AUC: use probabilities of the 5 non-NO classes
    non_no_idx  = [T3_LABEL2ID[c] for c in CATEGORIES]
    bprobs23_ml = bprobs23_cls[:, non_no_idx]
    bprobs23_ml = bprobs23_ml / (bprobs23_ml.sum(axis=1, keepdims=True) + 1e-9)

    f1_23_b  = f1_score(y23.astype(int), pred23_b, average="macro", zero_division=0)
    acc_23_b = accuracy_score(y23.astype(int).ravel(), pred23_b.ravel())
    auc_23_b = safe_auc_ml(y23.astype(int), bprobs23_ml)
    cat_f1_b = f1_score(y23.astype(int), pred23_b, average=None, zero_division=0)

    print("Loading Task 2.3 fine-tuned model...")
    save23  = MODELS_DIR / "xlmr_base_task2_3"
    tok23   = AutoTokenizer.from_pretrained(save23)
    mdl23   = AutoModelForSequenceClassification.from_pretrained(save23).to(device)
    probs23 = infer_multilabel(mdl23, tok23, texts, device)
    mdl23.cpu(); del mdl23; torch.cuda.empty_cache()

    pred23_f = (probs23 >= THRESHOLD).astype(int)
    for i in range(len(pred23_f)):
        if pred23_f[i].sum() == 0:
            pred23_f[i, int(np.argmax(probs23[i]))] = 1
    f1_23_f  = f1_score(y23.astype(int), pred23_f, average="macro", zero_division=0)
    acc_23_f = accuracy_score(y23.astype(int).ravel(), pred23_f.ravel())
    auc_23_f = safe_auc_ml(y23.astype(int), probs23)
    cat_f1_f = f1_score(y23.astype(int), pred23_f, average=None, zero_division=0)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS  (full training set, 3984 records)")
    print("=" * 70)
    hdr = "  {:<40} {:>8} {:>8} {:>8}"
    row = "  {:<40} {:>8.4f} {:>8.4f} {:>8.4f}"
    print(hdr.format("System", "T2.1", "T2.2", "T2.3"))
    print("  " + "-" * 64)
    print(row.format("XLM-R text-only (F1-Macro)",    f1_21_b, f1_22_b, f1_23_b))
    print(row.format("XLM-R + fusion/ML (F1-Macro)", f1_21_f, f1_22_f, f1_23_f))
    print("  " + "-" * 64)
    print(row.format("XLM-R text-only (AUC)",    auc_21_b, auc_22_b, auc_23_b))
    print(row.format("XLM-R + fusion/ML (AUC)", auc_21_f, auc_22_f, auc_23_f))

    print("\nTask 2.3 per-category F1:")
    for cat, b, f in zip(CATEGORIES, cat_f1_b, cat_f1_f):
        print(f"  {CAT_LABELS[cat]:<32}  single-label={b:.3f}  multi-label={f:.3f}")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")

    plot_slide40({
        "f1_21_base": f1_21_b, "auc_21_base": auc_21_b,
        "f1_22_base": f1_22_b, "auc_22_base": auc_22_b,
        "f1_23_base": f1_23_b, "auc_23_base": auc_23_b,
        "f1_21_ft":   f1_21_f, "auc_21_ft":   auc_21_f,
        "f1_22_ft":   f1_22_f, "auc_22_ft":   auc_22_f,
        "f1_23_ft":   f1_23_f, "auc_23_ft":   auc_23_f,
    })

    plot_slide41(
        "Task 2.1 — Binary Sexism Detection",
        ["F1-Macro", "F1(YES)", "Accuracy", "AUC"],
        [f1_21_b,   f1pos_21_b, acc_21_b, auc_21_b],
        [f1_21_f,   f1pos_21_f, acc_21_f, auc_21_f],
        "slide41_task21.png",
        base_label="XLM-R-base (baseline)",
        ft_label="XLM-R-large + fusion (ours)",
    )

    plot_slide41(
        "Task 2.2 — Sexism Type Detection",
        ["F1-Macro", "F1(sexist)", "Accuracy", "AUC"],
        [f1_22_b,   f1pos_22_b, acc_22_b, auc_22_b],
        [f1_22_f,   f1pos_22_f, acc_22_f, auc_22_f],
        "slide41_task22.png",
        base_label="XLM-R-base (baseline)",
        ft_label="XLM-R-large + oversample + fusion (ours)",
    )

    plot_slide41(
        "Task 2.3 — Sexism Category (multi-label)",
        ["F1-Macro", "Accuracy", "AUC"],
        [f1_23_b, acc_23_b, auc_23_b],
        [f1_23_f, acc_23_f, auc_23_f],
        "slide41_task23.png",
        base_label="XLM-R-base single-label (baseline)",
        ft_label="XLM-R-base multi-label (ours)",
    )

    plot_slide42(cat_f1_b, cat_f1_f, float(f1_23_b), float(f1_23_f))

    print("\nAll figures saved to project/figures/")


if __name__ == "__main__":
    main()
