"""
Official PyEvALL evaluation on EXIST2026 training set.

Generates gold + prediction files from training data, then runs the
competition's official PyEvALL library with the correct configuration.

Hard metrics : ICM, ICMNorm, FMeasure
Soft metrics : ICMSoft, ICMSoftNorm, CrossEntropy  (T2.1, T2.2)
               ICMSoft, ICMSoftNorm                (T2.3)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import FEATURES_DIR, MODELS_DIR, MEMES_TRAIN_JSON, TASK1_LABELS, TASK2_LABELS, TASK3_LABELS
from data_utils import load_json, build_memes_records

MAX_LEN   = 128
RAND_SEED = 42
VAL_FRAC  = 0.10
THRESHOLD = 0.5
TEST_CASE = "EXIST2026"

CATEGORIES = [l for l in TASK3_LABELS if l != "NO"]
LABEL2ID_21 = {l: i for i, l in enumerate(TASK1_LABELS)}
LABEL2ID_22 = {l: i for i, l in enumerate(TASK2_LABELS)}

TASK2_2_HIERARCHY = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
TASK2_3_HIERARCHY = {
    "YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
            "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"],
    "NO": [],
}

EVAL_DIR = Path(__file__).parent / "eval_pyevall"
GOLD_DIR = EVAL_DIR / "gold"
PRED_DIR = EVAL_DIR / "pred"
for d in [GOLD_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


# ── Gold files ────────────────────────────────────────────────────────────────

VALID_T3 = set(TASK3_LABELS)  # filter stray UNKNOWN labels from annotators


def make_gold(recs):
    g21h, g21s = [], []
    g22h, g22s = [], []
    g23h, g23s = [], []
    for r in recs:
        sid = r["id"]
        g21h.append({"test_case": TEST_CASE, "id": sid, "value": r["hard_t21"]})
        g21s.append({"test_case": TEST_CASE, "id": sid, "value": r["soft_t21"]})
        g22h.append({"test_case": TEST_CASE, "id": sid, "value": r["hard_t22"]})
        g22s.append({"test_case": TEST_CASE, "id": sid, "value": r["soft_t22"]})
        # Filter UNKNOWN labels that some annotators occasionally assign
        hard23 = [l for l in r["hard_t23"] if l in VALID_T3] or ["NO"]
        soft23 = {k: v for k, v in r["soft_t23"].items() if k in VALID_T3}
        g23h.append({"test_case": TEST_CASE, "id": sid, "value": hard23})
        g23s.append({"test_case": TEST_CASE, "id": sid, "value": soft23})
    save_json(g21h, GOLD_DIR / "task2_1_hard.json")
    save_json(g21s, GOLD_DIR / "task2_1_soft.json")
    save_json(g22h, GOLD_DIR / "task2_2_hard.json")
    save_json(g22s, GOLD_DIR / "task2_2_soft.json")
    save_json(g23h, GOLD_DIR / "task2_3_hard.json")
    save_json(g23s, GOLD_DIR / "task2_3_soft.json")
    print("Gold files written.")


# ── Prediction files ──────────────────────────────────────────────────────────

def make_preds_21(recs, texts, X_feat, tr_idx, device):
    save21 = MODELS_DIR / "xlmr_large_full_task2_1"
    tok    = AutoTokenizer.from_pretrained(save21)
    mdl    = AutoModelForSequenceClassification.from_pretrained(save21).to(device)
    perm   = col_perm_fn(mdl, TASK1_LABELS)

    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))])
    pipe.fit(X_feat[tr_idx], [LABEL2ID_21[recs[i]["hard_t21"]] for i in tr_idx])

    xlmr  = infer_cls(mdl, tok, texts, device, col_perm=perm)
    fused = 0.95 * xlmr + 0.05 * pipe.predict_proba(X_feat)
    mdl.cpu(); del mdl; torch.cuda.empty_cache()

    pred_h, pred_s = [], []
    for i, r in enumerate(recs):
        sid = r["id"]
        pred_h.append({"test_case": TEST_CASE, "id": sid,
                        "value": TASK1_LABELS[fused[i].argmax()]})
        pred_s.append({"test_case": TEST_CASE, "id": sid,
                        "value": {TASK1_LABELS[j]: round(float(fused[i, j]), 6)
                                  for j in range(len(TASK1_LABELS))}})
    save_json(pred_h, PRED_DIR / "task2_1_hard.json")
    save_json(pred_s, PRED_DIR / "task2_1_soft.json")
    print("T2.1 predictions written.")
    return fused  # return probs for T2.3 gating


def make_preds_22(recs, texts, X_feat, tr_idx, device):
    save22  = MODELS_DIR / "xlmr_large_task2_2_oversampled"
    tok     = AutoTokenizer.from_pretrained(save22)
    base_22 = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-large", num_labels=len(TASK2_LABELS), ignore_mismatched_sizes=True)
    mdl     = PeftModel.from_pretrained(base_22, save22).merge_and_unload().to(device)
    perm    = col_perm_fn(mdl, TASK2_LABELS)

    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, C=1.0))])
    pipe.fit(X_feat[tr_idx], [LABEL2ID_22[recs[i]["hard_t22"]] for i in tr_idx])

    xlmr  = infer_cls(mdl, tok, texts, device, col_perm=perm)
    fused = 0.75 * xlmr + 0.25 * pipe.predict_proba(X_feat)
    mdl.cpu(); del mdl; torch.cuda.empty_cache()

    pred_h, pred_s = [], []
    for i, r in enumerate(recs):
        sid = r["id"]
        pred_h.append({"test_case": TEST_CASE, "id": sid,
                        "value": TASK2_LABELS[fused[i].argmax()]})
        pred_s.append({"test_case": TEST_CASE, "id": sid,
                        "value": {TASK2_LABELS[j]: round(float(fused[i, j]), 6)
                                  for j in range(len(TASK2_LABELS))}})
    save_json(pred_h, PRED_DIR / "task2_2_hard.json")
    save_json(pred_s, PRED_DIR / "task2_2_soft.json")
    print("T2.2 predictions written.")


def make_preds_23(recs, texts, device, probs21):
    """T2.3 gated by T2.1 for hard; T2.1 soft probs weight T2.3 for soft."""
    save23 = MODELS_DIR / "xlmr_base_task2_3"
    tok    = AutoTokenizer.from_pretrained(save23)
    mdl    = AutoModelForSequenceClassification.from_pretrained(save23).to(device)
    probs  = infer_multilabel(mdl, tok, texts, device)
    mdl.cpu(); del mdl; torch.cuda.empty_cache()

    preds  = (probs >= THRESHOLD).astype(int)
    pred_h, pred_s = [], []
    for i, r in enumerate(recs):
        sid   = r["id"]
        p_no  = float(probs21[i, LABEL2ID_21["NO"]])
        p_yes = float(probs21[i, LABEL2ID_21["YES"]])

        # Hard: T2.1-gated
        if probs21[i].argmax() == LABEL2ID_21["NO"]:
            hard_val = ["NO"]
        else:
            hard_val = [CATEGORIES[j] for j in range(len(CATEGORIES)) if preds[i, j] == 1]
            if not hard_val:
                hard_val = [CATEGORIES[int(np.argmax(probs[i]))]]

        # Soft: T2.1 probability gates T2.3 — coherent joint distribution
        soft_val = {"NO": round(p_no, 6)}
        soft_val.update({c: round(p_yes * float(probs[i, j]), 6)
                         for j, c in enumerate(CATEGORIES)})

        pred_h.append({"test_case": TEST_CASE, "id": sid, "value": hard_val})
        pred_s.append({"test_case": TEST_CASE, "id": sid, "value": soft_val})
    save_json(pred_h, PRED_DIR / "task2_3_hard.json")
    save_json(pred_s, PRED_DIR / "task2_3_soft.json")
    print("T2.3 predictions written.")


# ── PyEvALL runner ────────────────────────────────────────────────────────────

def run(label, pred, gold, metrics, hierarchy=None):
    print(f"\n{'='*60}")
    print(label)
    print("="*60)
    ev = PyEvALLEvaluation()
    params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}
    if hierarchy:
        params[PyEvALLUtils.PARAM_HIERARCHY] = hierarchy
    report = ev.evaluate(str(pred), str(gold), metrics, **params)
    report.print_report()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading training data and features...")
    recs  = build_memes_records(load_json(MEMES_TRAIN_JSON), is_test=False)
    texts = [r["text"] for r in recs]

    with open(FEATURES_DIR / "memes_train_ids.pkl", "rb") as f:
        feat_ids = pickle.load(f)
    X_raw  = np.concatenate([np.load(FEATURES_DIR / "memes_train_image.npy"),
                              np.load(FEATURES_DIR / "memes_train_sensorial.npy")], axis=1)
    id2i   = {sid: i for i, sid in enumerate(feat_ids)}
    X_feat = np.array([X_raw[id2i[r["id"]]] for r in recs])

    sss = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RAND_SEED)
    y21 = [r["hard_t21"] for r in recs]
    y22 = [r["hard_t22"] for r in recs]
    tr21, _ = next(sss.split(texts, y21))
    tr22, _ = next(sss.split(texts, y22))

    print("\nGenerating gold files...")
    make_gold(recs)

    print("\nGenerating T2.1 predictions...")
    probs21 = make_preds_21(recs, texts, X_feat, tr21, device)

    print("\nGenerating T2.2 predictions...")
    make_preds_22(recs, texts, X_feat, tr22, device)

    print("\nGenerating T2.3 predictions (gated by T2.1)...")
    make_preds_23(recs, texts, device, probs21)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    HH = ["ICM", "ICMNorm", "FMeasure"]
    SS21 = ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]
    SS23 = ["ICMSoft", "ICMSoftNorm"]

    run("TASK 2.1 — Hard", PRED_DIR / "task2_1_hard.json", GOLD_DIR / "task2_1_hard.json", HH)
    run("TASK 2.1 — Soft", PRED_DIR / "task2_1_soft.json", GOLD_DIR / "task2_1_soft.json", SS21)
    run("TASK 2.2 — Hard", PRED_DIR / "task2_2_hard.json", GOLD_DIR / "task2_2_hard.json", HH, TASK2_2_HIERARCHY)
    run("TASK 2.2 — Soft", PRED_DIR / "task2_2_soft.json", GOLD_DIR / "task2_2_soft.json", SS21, TASK2_2_HIERARCHY)
    run("TASK 2.3 — Hard", PRED_DIR / "task2_3_hard.json", GOLD_DIR / "task2_3_hard.json", HH, TASK2_3_HIERARCHY)
    run("TASK 2.3 — Soft", PRED_DIR / "task2_3_soft.json", GOLD_DIR / "task2_3_soft.json", SS23, TASK2_3_HIERARCHY)

    print("\nDone.")


if __name__ == "__main__":
    main()
