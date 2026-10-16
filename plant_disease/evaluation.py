"""Test-set evaluation, reports and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from .config import METRICS_FILE
from .labels import pretty_label


def predict_probs(model, dataset, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, probabilities) over a labelled tf.data dataset.

    With `tta`, predictions are averaged over the image and its horizontal and
    vertical flips.
    """
    y_true, probs = [], []
    for images, labels in dataset:
        p = model.predict_on_batch(images)
        if tta:
            p = p + model.predict_on_batch(images[:, :, ::-1]) + model.predict_on_batch(images[:, ::-1])
            p = p / 3
        probs.append(p)
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(probs)


def evaluate(model, dataset, class_names: list[str], output_dir: Path, tta: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, probs = predict_probs(model, dataset, tta)
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))
    display = [pretty_label(c) for c in class_names]

    metrics = {
        "test_images": int(len(y_true)),
        "tta": tta,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, probs, k=3, labels=labels)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=display, digits=4, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, display, output_dir / "confusion_matrix.png")
    metrics["top_confusions"] = top_confusions(cm, display)

    (output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def top_confusions(cm: np.ndarray, names: list[str], n: int = 10) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0]
    out = []
    for t, p in pairs[:n]:
        if off[t, p] == 0:
            break
        out.append({"true": names[t], "predicted": names[p], "count": int(off[t, p])})
    return out


def plot_confusion_matrix(cm: np.ndarray, names: list[str], path: Path) -> None:
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    size = max(8, len(names) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: pd.DataFrame, path: Path, finetune_start: int | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Row order, not the "epoch" column, so continued runs (--init-from) line up.
    epochs = np.arange(1, len(history) + 1)
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(epochs, history[metric], label="train")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        if finetune_start is not None:
            ax.axvline(finetune_start + 0.5, color="gray", ls="--", label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_title(metric.capitalize())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_distribution(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["class_name"].map(pretty_label).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(6, len(counts) * 0.28)))
    ax.barh(counts.index, counts.values, color="#3a7d44")
    ax.set_xlabel("Images")
    ax.set_title("Images per class")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(frame: pd.DataFrame, path: Path, n: int = 16, seed: int = 0) -> None:
    from PIL import Image

    sample = frame.sample(min(n, len(frame)), random_state=seed)
    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, row) in zip(np.ravel(axes), sample.iterrows()):
        with Image.open(row["path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(pretty_label(row["class_name"]), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
