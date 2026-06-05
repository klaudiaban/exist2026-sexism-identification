"""Task 2.2: sexism type (NO/DIRECT/JUDGEMENTAL) using XLM-R large with LoRA, focal loss and late fusion."""
import gc
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (
    FEATURES_DIR, MODELS_DIR, RUNS_DIR,
    MEMES_TRAIN_JSON, MEMES_TEST_JSON,
    TASK2_LABELS, TEST_CASE, TEAM_NAME,
)
from data_utils import load_json, build_memes_records

MODEL_ID     = "xlm-roberta-large"
SAVE_DIR     = MODELS_DIR / "xlmr_large_task2_2_oversampled"
CKPT_DIR     = MODELS_DIR / "xlmr_large_task2_2_oversampled_ckpt"
LABEL2ID     = {l: i for i, l in enumerate(TASK2_LABELS)}
ID2LABEL     = {i: l for i, l in enumerate(TASK2_LABELS)}
MAX_LEN      = 128
VAL_FRAC     = 0.10
RANDOM_STATE = 42
FOCAL_GAMMA  = 2.0
LORA_R       = 16
LORA_ALPHA   = 32


class MemeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels":         self.labels[i],
        }


class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        ce  = F.cross_entropy(outputs.logits, labels, reduction="none")
        pt  = torch.exp(-ce)
        loss = ((1 - pt) ** FOCAL_GAMMA * ce).mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    f1  = f1_score([ID2LABEL[l] for l in labels],
                   [ID2LABEL[p] for p in preds],
                   average="macro", zero_division=0)
    return {"f1": f1, "accuracy": acc}


def oversample_indices(labels: list, random_state: int = 42) -> list:
    """Upsample minority classes to match majority count. Training split only."""
    rng = np.random.RandomState(random_state)
    labels_arr = np.array(labels)
    counts = np.bincount(labels_arr)
    max_count = int(counts.max())
    all_idx = list(range(len(labels)))
    for cls, cnt in enumerate(counts):
        shortage = max_count - int(cnt)
        if shortage > 0:
            cls_idx = np.where(labels_arr == cls)[0]
            extra   = rng.choice(cls_idx, size=shortage, replace=True)
            all_idx.extend(extra.tolist())
    perm = rng.permutation(len(all_idx))
    return [all_idx[i] for i in perm]


def load_feat_ids(split: str) -> list:
    with open(FEATURES_DIR / f"{split}_ids.pkl", "rb") as f:
        return pickle.load(f)


def load_image_sensorial(split: str) -> np.ndarray:
    image = np.load(FEATURES_DIR / f"{split}_image.npy")
    sens  = np.load(FEATURES_DIR / f"{split}_sensorial.npy")
    return np.concatenate([image, sens], axis=1)


def align_features(X_raw: np.ndarray, feat_ids: list, records: list) -> np.ndarray:
    id_to_idx = {sid: i for i, sid in enumerate(feat_ids)}
    return np.array([X_raw[id_to_idx[r["id"]]] for r in records])


def get_probs(model, tokenizer, texts, device, batch=32, col_perm=None) -> np.ndarray:
    model.eval()
    all_probs = []
    for start in range(0, len(texts), batch):
        enc = tokenizer(
            texts[start:start + batch], padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        if col_perm is not None:
            probs = probs[:, col_perm]
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def model_col_perm(model) -> list:
    """Column permutation so model output[:,i] = P(TASK2_LABELS[i])."""
    stored = {int(k): v for k, v in model.config.id2label.items()}
    label_to_col = {v: k for k, v in stored.items()}
    perm = [label_to_col.get(lbl, i) for i, lbl in enumerate(TASK2_LABELS)]
    if perm != list(range(len(TASK2_LABELS))):
        print(f"  Stored id2label: {stored}  -> reordering columns: {perm}")
    return perm


def write_run(entries, task, mode):
    fname = RUNS_DIR / f"{task}_{mode}_{TEAM_NAME}_1.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(entries)} entries -> {fname}")


def main():
    print("=" * 60)
    print("Task 2.2  --  NO / DIRECT / JUDGEMENTAL")
    print("Model: XLM-R large + LoRA + oversampling + focal loss + fusion")
    print("=" * 60)

    train_recs = build_memes_records(load_json(MEMES_TRAIN_JSON), is_test=False)
    test_recs  = build_memes_records(load_json(MEMES_TEST_JSON),  is_test=True)

    texts_all  = [r["text"] for r in train_recs]
    labels_all = [LABEL2ID[r["hard_t22"]] for r in train_recs]
    texts_test = [r["text"] for r in test_recs]
    ids_test   = [r["id"]   for r in test_recs]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC,
                                  random_state=RANDOM_STATE)
    tr_idx, val_idx = next(sss.split(texts_all, labels_all))
    texts_tr_raw  = [texts_all[i]  for i in tr_idx]
    labels_tr_raw = [labels_all[i] for i in tr_idx]
    texts_val     = [texts_all[i]  for i in val_idx]
    labels_val    = [labels_all[i] for i in val_idx]

    print("\nLoading image/sensorial features...")
    X_raw_train    = load_image_sensorial("memes_train")
    X_raw_test     = load_image_sensorial("memes_test")
    feat_ids_train = load_feat_ids("memes_train")
    feat_ids_test  = load_feat_ids("memes_test")

    X_feat_all    = align_features(X_raw_train, feat_ids_train, train_recs)
    X_feat_test   = align_features(X_raw_test,  feat_ids_test,  test_recs)
    X_feat_tr_raw = X_feat_all[tr_idx]
    X_feat_val    = X_feat_all[val_idx]

    print("\nClass distribution before oversampling:",
          dict(sorted(Counter(labels_tr_raw).items())))
    os_idx    = oversample_indices(labels_tr_raw, random_state=RANDOM_STATE)
    texts_tr  = [texts_tr_raw[i]  for i in os_idx]
    labels_tr = [labels_tr_raw[i] for i in os_idx]
    X_feat_tr = X_feat_tr_raw[os_idx]
    print("Class distribution after oversampling: ",
          dict(sorted(Counter(labels_tr).items())))
    print(f"Training samples: {len(labels_tr_raw)} -> {len(labels_tr)}")

    print("\nTraining fusion LogReg (image + sensorial)...")
    fusion_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])
    fusion_pipe.fit(X_feat_tr, labels_tr)
    fusion_val_probs  = fusion_pipe.predict_proba(X_feat_val)
    fusion_test_probs = fusion_pipe.predict_proba(X_feat_test)

    if SAVE_DIR.exists() and any(SAVE_DIR.iterdir()):
        print(f"\nLoading saved model from {SAVE_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
        base  = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(TASK2_LABELS), ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, SAVE_DIR).merge_and_unload()
    else:
        print(f"\nFine-tuning {MODEL_ID} with LoRA  (r={LORA_R}, focal loss)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        ds_tr  = MemeDataset(texts_tr,  labels_tr,  tokenizer)
        ds_val = MemeDataset(texts_val, labels_val, tokenizer)

        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=len(TASK2_LABELS),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )

        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()

        args = TrainingArguments(
            output_dir=str(CKPT_DIR),
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            seed=42,
            report_to="none",
            logging_steps=50,
        )

        trainer = FocalLossTrainer(
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
        print(f"LoRA adapter saved to {SAVE_DIR}")

        model = model.merge_and_unload()
        print("LoRA weights merged into base model for inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    perm = model_col_perm(model)

    print("\nRunning inference on val + test...")
    xlmr_val_probs  = get_probs(model, tokenizer, texts_val,  device, col_perm=perm)
    xlmr_test_probs = get_probs(model, tokenizer, texts_test, device, col_perm=perm)

    print("\nSweeping fusion alpha...")
    labels_val_arr = np.array(labels_val)
    best_alpha, best_f1 = 0.0, -1.0
    for alpha in np.arange(0.0, 1.01, 0.05):
        fused = alpha * xlmr_val_probs + (1 - alpha) * fusion_val_probs
        preds = np.argmax(fused, axis=-1)
        f1  = f1_score([ID2LABEL[l] for l in labels_val],
                       [ID2LABEL[p] for p in preds],
                       average="macro", zero_division=0)
        acc = (preds == labels_val_arr).mean()
        if f1 > best_f1:
            best_f1, best_alpha = f1, alpha
        print(f"  a={alpha:.2f}  val F1={f1:.4f}  acc={acc:.4f}")

    best_fused = best_alpha * xlmr_val_probs + (1 - best_alpha) * fusion_val_probs
    best_preds = np.argmax(best_fused, axis=-1)
    best_acc   = (best_preds == labels_val_arr).mean()
    print(f"\nBest: a={best_alpha:.2f}  val F1={best_f1:.4f}  acc={best_acc:.4f}")

    fused_test = best_alpha * xlmr_test_probs + (1 - best_alpha) * fusion_test_probs
    hard_preds = np.argmax(fused_test, axis=-1)

    hard_entries, soft_entries = [], []
    for i, sid in enumerate(ids_test):
        hard_entries.append({
            "test_case": TEST_CASE, "id": sid,
            "value": ID2LABEL[hard_preds[i]],
        })
        soft_val = {TASK2_LABELS[j]: round(float(fused_test[i, j]), 6)
                    for j in range(len(TASK2_LABELS))}
        soft_entries.append({"test_case": TEST_CASE, "id": sid, "value": soft_val})

    write_run(hard_entries, "task2_2", "hard")
    write_run(soft_entries, "task2_2", "soft")

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
