"""
Baseline vs fine-tuned comparison for Tasks 2.1, 2.2, 2.3.

Evaluation set: FULL training set (3984 records, gold labels from JSON).
This matches what the official evaluation baselines use.

Baselines
  - Majority class: always predicts the most frequent class
  - Minority class: always predicts the least frequent class

Fine-tuned models (predictions on the FULL training set)
  - Task 2.1: XLM-RoBERTa-large full fine-tune + image/sensorial LogReg fusion
  - Task 2.2: XLM-RoBERTa-large + LoRA + oversampling + focal loss + fusion
  - Task 2.3: XLM-RoBERTa-base multi-label fine-tune (pos_weight)
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import FEATURES_DIR, MODELS_DIR, MEMES_TRAIN_JSON, TASK1_LABELS, TASK2_LABELS, TASK3_LABELS
from data_utils import load_json, build_memes_records

MAX_LEN    = 128
VAL_FRAC   = 0.10
RAND_SEED  = 42
THRESHOLD  = 0.5
CATEGORIES = [l for l in TASK3_LABELS if l != "NO"]

LABEL2ID_21 = {l: i for i, l in enumerate(TASK1_LABELS)}
LABEL2ID_22 = {l: i for i, l in enumerate(TASK2_LABELS)}
ID2LABEL_21 = {i: l for i, l in enumerate(TASK1_LABELS)}
ID2LABEL_22 = {i: l for i, l in enumerate(TASK2_LABELS)}


# ── Inference helpers ─────────────────────────────────────────────────────────

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


def col_perm(model, label_order):
    stored = {int(k): v for k, v in model.config.id2label.items()}
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


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_task_table(title, rows):
    headers = ["System", "Macro-F1", "Accuracy", "AUC"]
    col_w = [max(len(h), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(f"\n{'=' * (sum(col_w) + 3 * len(col_w) + 1)}")
    print(f"  {title}")
    print(f"{'=' * (sum(col_w) + 3 * len(col_w) + 1)}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading training data and features...")

    recs     = build_memes_records(load_json(MEMES_TRAIN_JSON), is_test=False)
    texts    = [r["text"] for r in recs]
    y21      = np.array([LABEL2ID_21[r["hard_t21"]] for r in recs])
    y22      = np.array([LABEL2ID_22[r["hard_t22"]] for r in recs])
    y23      = binary_matrix(recs)

    with open(FEATURES_DIR / "memes_train_ids.pkl", "rb") as f:
        feat_ids = pickle.load(f)
    X_feat = align(load_image_sensorial(), feat_ids, recs)

    # ── Train split for LogReg (same 90/10 used in training scripts) ──
    # Task 2.1 split (stratified on y21)
    sss21 = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RAND_SEED)
    tr21, _ = next(sss21.split(texts, y21))
    # Task 2.2 split (stratified on y22)
    sss22 = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RAND_SEED)
    tr22, _ = next(sss22.split(texts, y22))

    # ════════════════════════════════════════════════════════════════════════
    # TASK 2.1 -- YES / NO
    # ════════════════════════════════════════════════════════════════════════
    print("\nEvaluating Task 2.1...")
    rows21 = []

    # Majority / minority baselines
    maj21 = Counter(y21).most_common(1)[0][0]
    min21 = Counter(y21).most_common()[-1][0]
    for name, cls in [("Majority class", maj21), ("Minority class", min21)]:
        preds = np.full(len(y21), cls)
        scores = np.full(len(y21), float(cls))   # 0 or 1 as constant score
        rows21.append((
            name,
            f"{f1_score(y21, preds, average='macro', zero_division=0):.4f}",
            f"{accuracy_score(y21, preds):.4f}",
            f"{safe_auc_binary(y21, scores):.4f}",
        ))

    # Fine-tuned
    save21 = MODELS_DIR / "xlmr_large_full_task2_1"
    if save21.exists():
        tok = AutoTokenizer.from_pretrained(save21)
        mdl = AutoModelForSequenceClassification.from_pretrained(save21).to(device)
        perm21 = col_perm(mdl, TASK1_LABELS)

        # LogReg trained on 90% only (same as during training)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                                    class_weight="balanced"))])
        pipe.fit(X_feat[tr21], y21[tr21])

        xlmr_probs   = infer_cls(mdl, tok, texts, device, col_perm=perm21)
        fusion_probs = pipe.predict_proba(X_feat)

        # Use alpha=0.95 (best from training sweep)
        fused = 0.95 * xlmr_probs + 0.05 * fusion_probs
        preds = fused.argmax(-1)
        # AUC: P(NO) as score (label 1 = NO in LABEL2ID_21)
        rows21.append((
            "XLM-R-large + fusion  [fine-tuned]",
            f"{f1_score(y21, preds, average='macro', zero_division=0):.4f}",
            f"{accuracy_score(y21, preds):.4f}",
            f"{safe_auc_binary(y21, fused[:, 1]):.4f}",
        ))
        mdl.cpu(); del mdl; torch.cuda.empty_cache()
    else:
        print("  Task 2.1 model not found.")

    print_task_table("TASK 2.1 -- Binary sexism detection (YES / NO)", rows21)

    # ════════════════════════════════════════════════════════════════════════
    # TASK 2.2 -- NO / DIRECT / JUDGEMENTAL
    # ════════════════════════════════════════════════════════════════════════
    print("\nEvaluating Task 2.2...")
    rows22 = []

    maj22 = Counter(y22).most_common(1)[0][0]
    min22 = Counter(y22).most_common()[-1][0]
    for name, cls in [("Majority class", maj22), ("Minority class", min22)]:
        preds = np.full(len(y22), cls)
        probs_bin = label_binarize(preds, classes=[0, 1, 2])
        rows22.append((
            name,
            f"{f1_score(y22, preds, average='macro', zero_division=0):.4f}",
            f"{accuracy_score(y22, preds):.4f}",
            f"{safe_auc_multi(y22, probs_bin, [0,1,2]):.4f}",
        ))

    save22 = MODELS_DIR / "xlmr_large_task2_2_oversampled"
    if save22.exists():
        tok     = AutoTokenizer.from_pretrained(save22)
        base_22 = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-large", num_labels=len(TASK2_LABELS),
            ignore_mismatched_sizes=True,
        )
        mdl     = PeftModel.from_pretrained(base_22, save22).merge_and_unload().to(device)
        perm22  = col_perm(mdl, TASK2_LABELS)

        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, C=1.0))])
        pipe.fit(X_feat[tr22], y22[tr22])

        xlmr_probs   = infer_cls(mdl, tok, texts, device, col_perm=perm22)
        fusion_probs = pipe.predict_proba(X_feat)

        fused = 0.75 * xlmr_probs + 0.25 * fusion_probs
        preds = fused.argmax(-1)
        rows22.append((
            "XLM-R-large + oversample + fusion  [fine-tuned]",
            f"{f1_score(y22, preds, average='macro', zero_division=0):.4f}",
            f"{accuracy_score(y22, preds):.4f}",
            f"{safe_auc_multi(y22, fused, [0,1,2]):.4f}",
        ))
        mdl.cpu(); del mdl; torch.cuda.empty_cache()
    else:
        print("  Task 2.2 model not found.")

    print_task_table("TASK 2.2 -- Sexism type (NO / DIRECT / JUDGEMENTAL)", rows22)

    # ════════════════════════════════════════════════════════════════════════
    # TASK 2.3 -- Multi-label categories
    # ════════════════════════════════════════════════════════════════════════
    print("\nEvaluating Task 2.3...")
    rows23 = []

    for name, val in [("Majority class (predict no category)", 0),
                      ("Minority class (predict all categories)", 1)]:
        preds = np.full_like(y23, val)
        rows23.append((
            name,
            f"{f1_score(y23.astype(int), preds, average='macro', zero_division=0):.4f}",
            f"{accuracy_score(y23.astype(int).ravel(), preds.ravel()):.4f}",
            "N/A",
        ))

    save23 = MODELS_DIR / "xlmr_base_task2_3"
    if save23.exists():
        tok = AutoTokenizer.from_pretrained(save23)
        mdl = AutoModelForSequenceClassification.from_pretrained(save23).to(device)

        probs = infer_multilabel(mdl, tok, texts, device)
        preds = (probs >= THRESHOLD).astype(int)
        for i in range(len(preds)):
            if preds[i].sum() == 0:
                preds[i, int(np.argmax(probs[i]))] = 1

        rows23.append((
            "XLM-R-base multi-label  [fine-tuned]",
            f"{f1_score(y23.astype(int), preds, average='macro', zero_division=0):.4f}",
            f"{accuracy_score(y23.astype(int).ravel(), preds.ravel()):.4f}",
            f"{safe_auc_ml(y23.astype(int), probs):.4f}",
        ))
        mdl.cpu(); del mdl; torch.cuda.empty_cache()
    else:
        print("  Task 2.3 model not found.")

    print_task_table("TASK 2.3 -- Sexism category (multi-label)", rows23)
    print()


if __name__ == "__main__":
    main()
