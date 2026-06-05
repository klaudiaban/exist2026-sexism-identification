"""Text-only XLM-R-base baselines for Tasks 2.1, 2.2 and 2.3."""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import MODELS_DIR, MEMES_TRAIN_JSON, TASK1_LABELS, TASK2_LABELS, TASK3_LABELS
from data_utils import load_json, build_memes_records

MODEL_ID  = "xlm-roberta-base"
MAX_LEN   = 128
EPOCHS    = 4
BATCH     = 16
LR        = 2e-5
VAL_FRAC  = 0.10
RAND_SEED = 42

KNOWN_TASK3 = set(TASK3_LABELS)


def primary_label_t23(hard_t23):
    """First non-NO known label, else NO."""
    for lbl in hard_t23:
        if lbl in KNOWN_TASK3 and lbl != "NO":
            return lbl
    return "NO"


class MemeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc    = tokenizer(texts, padding="max_length", truncation=True,
                                max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


def train_one(task_name, save_dir, texts, y, label_list):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"  Labels: {label_list}")
    print(f"  Distribution: {Counter(y.tolist())}")
    print(f"{'='*60}")

    sss = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RAND_SEED)
    tr_idx, val_idx = next(sss.split(texts, y))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(label_list),
        id2label=id2label, label2id=label2id,
    ).to(device)

    tr_ds  = MemeDataset([texts[i] for i in tr_idx],  y[tr_idx].tolist(),  tokenizer)
    val_ds = MemeDataset([texts[i] for i in val_idx], y[val_idx].tolist(), tokenizer)
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for batch, labels in tr_dl:
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            loss   = model(**batch, labels=labels).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for batch, labels in val_dl:
                batch  = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                out    = model(**batch, labels=labels)
                val_loss += out.loss.item()
                correct  += (out.logits.argmax(-1) == labels).sum().item()

        val_acc = correct / len(val_idx)
        print(f"  Epoch {epoch+1}/{EPOCHS}  train={tr_loss/len(tr_dl):.4f}"
              f"  val={val_loss/len(val_dl):.4f}  acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    -> saved ({save_dir.name})")

    model.cpu(); del model; torch.cuda.empty_cache()


def main():
    print("Loading training data...")
    recs  = build_memes_records(load_json(MEMES_TRAIN_JSON), is_test=False)
    texts = [r["text"] for r in recs]

    label2id_21 = {l: i for i, l in enumerate(TASK1_LABELS)}
    label2id_22 = {l: i for i, l in enumerate(TASK2_LABELS)}
    label2id_23 = {l: i for i, l in enumerate(TASK3_LABELS)}

    y21 = np.array([label2id_21[r["hard_t21"]] for r in recs])
    y22 = np.array([label2id_22[r["hard_t22"]] for r in recs])
    y23 = np.array([label2id_23[primary_label_t23(r["hard_t23"])] for r in recs])

    train_one(
        "Task 2.1 — Binary (YES / NO)",
        MODELS_DIR / "xlmr_base_baseline_task2_1",
        texts, y21, TASK1_LABELS,
    )
    train_one(
        "Task 2.2 — Type (NO / DIRECT / JUDGEMENTAL)",
        MODELS_DIR / "xlmr_base_baseline_task2_2",
        texts, y22, TASK2_LABELS,
    )
    train_one(
        "Task 2.3 — Category single-label (NO + 5 categories)",
        MODELS_DIR / "xlmr_base_baseline_task2_3",
        texts, y23, TASK3_LABELS,
    )

    print("\nAll baselines trained.")


if __name__ == "__main__":
    main()
