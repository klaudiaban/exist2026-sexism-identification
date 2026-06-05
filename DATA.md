# Data access

This repository **does not include the dataset, extracted features, or trained
model weights** - and intentionally so.

## Why the data is not here

The project uses the **EXIST 2026 Memes dataset**, which is:

1. **Released under a usage agreement** that does not permit public
   redistribution. The data must be requested from the task organizers.
2. **Privacy-sensitive.** Beyond the meme images and text, each sample carries
   *physiological sensor signals* (eye-tracking, EEG, heart-rate) and rich
   *annotator demographics*. Re-publishing this would be both a breach of the
   agreement and a privacy risk.

For those reasons the dataset, the derived feature files (`project/features/*.npy`),
and the fine-tuned model weights (`project/models/`) are all git-ignored.

## How to obtain the data

Request access through the official EXIST 2026 lab:

- EXIST shared task: https://nlp.uned.es/exist2026/

## Expected layout

Once you have the data, place it so the paths in
[`project/src/config.py`](project/src/config.py) resolve:

```
<repo root>/
├── EXIST 2026 Memes Dataset/
│   ├── training/
│   │   ├── EXIST2026_training.json
│   │   └── memes/                     # 3,984 training images
│   └── test/
│       ├── EXIST2026_test_clean.json
│       └── memes/                     # 1,053 test images
└── project/
    ├── features/                      # generated locally (see below)
    └── models/                        # written by the training scripts
```

## Feature files

The training scripts expect pre-extracted features in `project/features/`:

| File                                   | Contents                                  |
|----------------------------------------|-------------------------------------------|
| `memes_{train,test}_text.npy`          | Text embeddings                           |
| `memes_{train,test}_image.npy`         | Image embeddings                          |
| `memes_{train,test}_sensorial.npy`     | Physiological sensor features             |
| `memes_{train,test}_ids.pkl`           | Sample-ID order for each feature matrix   |

These are derived from the protected data and must be regenerated in your own
environment after you have been granted access.
