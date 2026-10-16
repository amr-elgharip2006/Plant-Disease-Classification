"""Classify one or more leaf images with a trained model.

Example:
    python predict.py leaf1.jpg leaf2.jpg --top-k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plant_disease.inference import PlantDiseaseClassifier


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--top-k", type=int, default=3)
    a = p.parse_args()

    clf = PlantDiseaseClassifier(a.artifacts)
    for path, preds in zip(a.images, clf.predict_batch(a.images, a.top_k)):
        print(f"\n{path}")
        for i, pred in enumerate(preds, 1):
            status = "healthy" if pred.healthy else "diseased"
            print(f"  {i}. {pred.plant} — {pred.condition} ({status}) {pred.confidence:.1%}")


if __name__ == "__main__":
    main()
