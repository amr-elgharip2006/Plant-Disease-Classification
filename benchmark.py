"""Compare backbones quickly (frozen backbone, few epochs) on the validation set.

Each backbone is built inside the loop and released afterwards, so only one
model is in memory at a time.

Example:
    python benchmark.py --epochs 3 --backbones mobilenetv2 efficientnetb0 efficientnetb2
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import keras

from plant_disease.config import SEED, TrainConfig
from plant_disease.data import make_dataset, prepare_splits
from plant_disease.model import BACKBONES, build_model, compile_model, get_backbone


def main() -> None:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbones", nargs="+", default=sorted(BACKBONES), choices=sorted(BACKBONES))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    a = p.parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    splits, class_names = prepare_splits(a.data_dir, a.output_dir, d.val_frac, d.test_frac)

    results = {}
    for name in a.backbones:
        print(f"\n=== {name} ===")
        keras.utils.set_random_seed(SEED)
        img_size = get_backbone(name).native_size
        train_ds = make_dataset(splits["train"], img_size, a.batch_size, shuffle=True)
        val_ds = make_dataset(splits["val"], img_size, a.batch_size)

        model, _ = build_model(name, len(class_names), img_size)
        compile_model(model, d.head_lr)
        start = time.time()
        history = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs)
        results[name] = {
            "img_size": img_size,
            "params_millions": round(model.count_params() / 1e6, 2),
            "best_val_accuracy": max(history.history["val_accuracy"]),
            "minutes": round((time.time() - start) / 60, 1),
        }

        del model, train_ds, val_ds
        keras.backend.clear_session()
        gc.collect()

    ranking = sorted(results.items(), key=lambda kv: kv[1]["best_val_accuracy"], reverse=True)
    print("\n=== Benchmark results ===")
    print(f"{'backbone':<16}{'val acc':>10}{'params (M)':>12}{'minutes':>10}")
    for name, r in ranking:
        print(f"{name:<16}{r['best_val_accuracy'] * 100:>9.2f}%{r['params_millions']:>12}{r['minutes']:>10}")
    print(f"\nBest: {ranking[0][0]}")
    (a.output_dir / "benchmark.json").write_text(json.dumps(dict(ranking), indent=2))


if __name__ == "__main__":
    main()
