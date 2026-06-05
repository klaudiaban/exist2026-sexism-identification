from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MEMES_TRAIN_JSON = ROOT / "EXIST 2026 Memes Dataset" / "training" / "EXIST2026_training.json"
MEMES_TEST_JSON  = ROOT / "EXIST 2026 Memes Dataset" / "test"     / "EXIST2026_test_clean.json"

FEATURES_DIR = ROOT / "project" / "features"
MODELS_DIR   = ROOT / "project" / "models"
RUNS_DIR     = ROOT / "project" / "runs"

for d in [FEATURES_DIR, MODELS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TASK1_LABELS = ["YES", "NO"]
TASK2_LABELS = ["NO", "DIRECT", "JUDGEMENTAL"]
TASK3_LABELS = [
    "NO", "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE", "SEXUAL-VIOLENCE", "OBJECTIFICATION",
]

TEST_CASE = "EXIST2026"
TEAM_NAME = "KlaudiaBanasiewicz"
