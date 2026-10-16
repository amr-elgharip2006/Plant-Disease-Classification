"""Re-evaluate a trained model on the saved test split.

Example:
    python evaluate.py --artifacts artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras

from plant_disease.config import METADATA_FILE, MODEL_FILE
from plant_disease.data import load_splits, make_dataset
from plant_disease.evaluation import evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tta", action="store_true", help="average flipped views")
    a = p.parse_args()

    meta = json.loads((a.artifacts / METADATA_FILE).read_text())
    model = keras.models.load_model(a.artifacts / MODEL_FILE)
    test_ds = make_dataset(load_splits(a.artifacts)["test"], meta["img_size"], a.batch_size)
    metrics = evaluate(model, test_ds, meta["class_names"], a.artifacts, tta=a.tta)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
