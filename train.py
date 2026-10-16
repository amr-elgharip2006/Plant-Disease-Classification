"""Train the plant disease classifier in two phases and evaluate it on the test set.

Phase 1 trains a new head on a frozen ImageNet backbone.
Phase 2 unfreezes the top of the backbone and fine-tunes with a small learning rate.

Example:
    python train.py --backbone efficientnetb0 --mixed-precision
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from plant_disease.config import HISTORY_FILE, METADATA_FILE, MODEL_FILE, SEED, TrainConfig
from plant_disease.data import make_dataset, prepare_splits
from plant_disease.evaluation import evaluate, plot_history
from plant_disease.model import (
    BACKBONES,
    backbone_of,
    build_model,
    compile_model,
    get_backbone,
    unfreeze_top,
)


def parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default=d.backbone, choices=sorted(BACKBONES))
    p.add_argument("--img-size", type=int, default=d.img_size, help="default: backbone's native size")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--head-epochs", type=int, default=d.head_epochs)
    p.add_argument("--head-lr", type=float, default=d.head_lr)
    p.add_argument("--finetune-epochs", type=int, default=d.finetune_epochs, help="0 to skip fine-tuning")
    p.add_argument("--finetune-lr", type=float, default=d.finetune_lr)
    p.add_argument("--unfreeze-fraction", type=float, default=d.unfreeze_fraction)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=d.class_weights)
    p.add_argument("--mixed-precision", action="store_true", help="faster on modern NVIDIA GPUs")
    p.add_argument("--data-dir", type=Path, default=None, help="local dataset; default: download from Kaggle")
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    p.add_argument("--init-from", type=Path, default=None,
                   help="continue fine-tuning a saved model (skips phase 1)")
    p.add_argument("--tta", action="store_true", help="average flipped views at test time")
    a = p.parse_args()
    return TrainConfig(
        backbone=a.backbone,
        img_size=a.img_size,
        batch_size=a.batch_size,
        head_epochs=a.head_epochs,
        head_lr=a.head_lr,
        finetune_epochs=a.finetune_epochs,
        finetune_lr=a.finetune_lr,
        unfreeze_fraction=a.unfreeze_fraction,
        dropout=a.dropout,
        class_weights=a.class_weights,
        mixed_precision=a.mixed_precision,
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        init_from=a.init_from,
        tta=a.tta,
    )


def callbacks(
    output_dir: Path, patience: int, best_val_loss: float | None = None
) -> list[keras.callbacks.Callback]:
    return [
        # The threshold carries phase 1's best into phase 2, so the checkpoint on
        # disk is always the best model across both phases.
        keras.callbacks.ModelCheckpoint(
            output_dir / MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=best_val_loss,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=max(1, patience // 2), min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(output_dir / HISTORY_FILE, append=True),
    ]


def main() -> None:
    cfg = parse_args()
    keras.utils.set_random_seed(SEED)
    if cfg.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.init_from:
        (out / HISTORY_FILE).unlink(missing_ok=True)

    img_size = cfg.img_size or get_backbone(cfg.backbone).native_size
    splits, class_names = prepare_splits(cfg.data_dir, out, cfg.val_frac, cfg.test_frac)
    train_ds = make_dataset(splits["train"], img_size, cfg.batch_size, shuffle=True)
    val_ds = make_dataset(splits["val"], img_size, cfg.batch_size)
    test_ds = make_dataset(splits["test"], img_size, cfg.batch_size)

    class_weight = None
    if cfg.class_weights:
        y = splits["train"]["label"].to_numpy()
        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight = dict(enumerate(weights.tolist()))

    (out / METADATA_FILE).write_text(
        json.dumps(
            {"class_names": class_names, "img_size": img_size, "config": cfg.to_dict()},
            indent=2,
        )
    )

    if cfg.init_from:
        print(f"\n=== Continuing from {cfg.init_from} (phase 1 skipped) ===")
        model = keras.models.load_model(cfg.init_from)
        base = backbone_of(model)
        compile_model(model, cfg.finetune_lr)
        # Only checkpoint models that beat the one we started from.
        best_val_loss = model.evaluate(val_ds, verbose=0, return_dict=True)["loss"]
        print(f"Starting val_loss: {best_val_loss:.4f}")
        head_epochs_run = 0
    else:
        model, base = build_model(cfg.backbone, len(class_names), img_size, cfg.dropout)

        print(f"\n=== Phase 1: training head on frozen {cfg.backbone} ===")
        compile_model(model, cfg.head_lr)
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.head_epochs,
            class_weight=class_weight,
            callbacks=callbacks(out, patience=3),
        )
        head_epochs_run = len(h1.history["loss"])
        best_val_loss = min(h1.history["val_loss"])

    finetune_start = None
    if cfg.finetune_epochs > 0:
        n = unfreeze_top(base, cfg.unfreeze_fraction)
        print(f"\n=== Phase 2: fine-tuning top {n} backbone layers ===")
        compile_model(model, cfg.finetune_lr)  # recompile so trainable changes apply
        finetune_start = head_epochs_run
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=head_epochs_run,
            epochs=head_epochs_run + cfg.finetune_epochs,
            class_weight=class_weight,
            callbacks=callbacks(out, patience=4, best_val_loss=best_val_loss),
        )

    model = keras.models.load_model(out / MODEL_FILE)  # best checkpoint across both phases

    history = pd.read_csv(out / HISTORY_FILE)
    plot_history(history, out / "training_curves.png", finetune_start)

    print("\n=== Evaluating on held-out test set ===")
    metrics = evaluate(model, test_ds, class_names, out, tta=cfg.tta)
    print(json.dumps({k: v for k, v in metrics.items() if k != "top_confusions"}, indent=2))
    print(f"\nArtifacts saved to {out.resolve()}")


if __name__ == "__main__":
    main()
