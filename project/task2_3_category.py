"""Task 2.3: multi-label sexism category classification using XLM-R base with BCE loss."""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (
    MODELS_DIR, RUNS_DIR,
    MEMES_TRAIN_JSON, MEMES_TEST_JSON,
    TASK3_LABELS, TEST_CASE, TEAM_NAME,
)
from data_utils import load_json, build_memes_records

MODEL_ID     = "xlm-roberta-base"
SAVE_DIR     = MODELS_DIR / "xlmr_base_task2_3"
CKPT_DIR     = MODELS_DIR / "xlmr_base_task2_3_ckpt"
CATEGORIES   = [l for l in TASK3_LABELS if l != "NO"]
CAT2IDX      = {c: i for i, c in enumerate(CATEGORIES)}
MAX_LEN      = 128
VAL_FRAC     = 0.10
THRESHOLD    = 0.5
RANDOM_STATE = 42


def records_to_binary_matrix(records) -> np.ndarray:
    """Convert hard_t23 category lists to a binary (n, 5) float matrix."""
    Y = np.zeros((len(records), len(CATEGORIES)), dtype=np.float32)
    for i, r in enumerate(records):
        for lbl in r["hard_t23"]:
            if lbl in CAT2IDX:
                Y[i, CAT2IDX[lbl]] = 1.0
    return Y


class MemeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels":         self.labels[i],
        }


class MultiLabelTrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= THRESHOLD).astype(int)
    for i in range(len(preds)):
        if preds[i].sum() == 0:
            preds[i, int(np.argmax(probs[i]))] = 1
    return {"f1": f1_score(labels.astype(int), preds,
                            average="macro", zero_division=0)}


def get_probs(model, tokenizer, texts, device, batch=32) -> np.ndarray:
    model.eval()
    all_probs = []
    for start in range(0, len(texts), batch):
        enc = tokenizer(
            texts[start:start + batch], padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def write_run(entries, task, mode):
    fname = RUNS_DIR / f"{task}_{mode}_{TEAM_NAME}_1.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(entries)} entries -> {fname}")


def main():
    print("=" * 60)
    print("Task 2.3  --  Multi-label categories  (XLM-R base)")
    print("=" * 60)

    train_recs = build_memes_records(load_json(MEMES_TRAIN_JSON), is_test=False)
    test_recs  = build_memes_records(load_json(MEMES_TEST_JSON),  is_test=True)

    texts_all  = [r["text"] for r in train_recs]
    Y_all      = records_to_binary_matrix(train_recs)
    texts_test = [r["text"] for r in test_recs]
    ids_test   = [r["id"]   for r in test_recs]

    primary = np.argmax(Y_all, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC,
                                  random_state=RANDOM_STATE)
    tr_idx, val_idx = next(sss.split(texts_all, primary))
    texts_tr  = [texts_all[i] for i in tr_idx]
    Y_tr      = Y_all[tr_idx]
    texts_val = [texts_all[i] for i in val_idx]
    Y_val     = Y_all[val_idx]

    if SAVE_DIR.exists() and any(SAVE_DIR.iterdir()):
        print(f"\nLoading saved model from {SAVE_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    else:
        print(f"\nFine-tuning {MODEL_ID}  (multi-label BCE, bf16)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        ds_tr  = MemeDataset(texts_tr,  Y_tr,  tokenizer)
        ds_val = MemeDataset(texts_val, Y_val, tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=len(CATEGORIES),
            problem_type="multi_label_classification",
        )

        args = TrainingArguments(
            output_dir=str(CKPT_DIR),
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            bf16=True,
            optim="adamw_torch_fused",
            report_to="none",
            logging_steps=50,
        )

        n_pos      = Y_tr.sum(axis=0).clip(min=1)
        n_neg      = len(Y_tr) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float)
        print(f"pos_weight per category: {[round(w,1) for w in pos_weight.tolist()]}")

        trainer = MultiLabelTrainer(
            pos_weight=pos_weight,
            model=model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_val,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        model = trainer.model
        trainer.model = None
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"Model saved to {SAVE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\nRunning inference...")
    val_probs  = get_probs(model, tokenizer, texts_val,  device)
    val_preds  = (val_probs >= THRESHOLD).astype(int)
    for i in range(len(val_preds)):
        if val_preds[i].sum() == 0:
            val_preds[i, int(np.argmax(val_probs[i]))] = 1
    val_f1 = f1_score(Y_val.astype(int), val_preds,
                      average="macro", zero_division=0)
    print(f"Val macro-F1: {val_f1:.4f}")

    test_probs = get_probs(model, tokenizer, texts_test, device)
    test_preds = (test_probs >= THRESHOLD).astype(int)

    # Load T2.1 predictions to gate hard and weight soft T2.3 predictions
    with open(RUNS_DIR / f"task2_1_hard_{TEAM_NAME}_1.json", encoding="utf-8") as f:
        t21_hard = {e["id"]: e["value"] for e in json.load(f)}
    with open(RUNS_DIR / f"task2_1_soft_{TEAM_NAME}_1.json", encoding="utf-8") as f:
        t21_soft = {e["id"]: e["value"] for e in json.load(f)}

    hard_entries, soft_entries = [], []
    for i, sid in enumerate(ids_test):
        p_no  = t21_soft[sid]["NO"]
        p_yes = t21_soft[sid]["YES"]

        # Hard: gate by T2.1
        if t21_hard.get(sid) == "NO":
            hard_val = ["NO"]
        else:
            selected = [CATEGORIES[j] for j in range(len(CATEGORIES))
                        if test_preds[i, j] == 1]
            if not selected:
                selected = [CATEGORIES[int(np.argmax(test_probs[i]))]]
            hard_val = selected
        hard_entries.append({"test_case": TEST_CASE, "id": sid, "value": hard_val})

        # Soft: T2.1 probability gates T2.3 for a coherent joint distribution
        soft_val = {"NO": round(p_no, 6)}
        soft_val.update({c: round(p_yes * float(test_probs[i, j]), 6)
                         for j, c in enumerate(CATEGORIES)})
        soft_entries.append({"test_case": TEST_CASE, "id": sid, "value": soft_val})

    write_run(hard_entries, "task2_3", "hard")
    write_run(soft_entries, "task2_3", "soft")

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
