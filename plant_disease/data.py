"""Dataset download, cleaning, stratified splitting and tf.data pipelines."""

from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import IMAGE_EXTENSIONS, KAGGLE_DATASET, SEED, SPLITS_DIR

AUTOTUNE = tf.data.AUTOTUNE


def locate_dataset(data_dir: Path | None = None) -> Path:
    """Return the PlantVillage `color` folder (one sub-folder per class).

    Downloads the dataset from Kaggle when `data_dir` is not given. Only the
    `color` variant is used; `grayscale` and `segmented` are ignored.
    """
    if data_dir is None:
        import kagglehub

        data_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    data_dir = Path(data_dir)

    if data_dir.name == "color":
        return data_dir
    matches = sorted(p for p in data_dir.rglob("color") if p.is_dir())
    if matches:
        return matches[0]
    # Already a folder of class sub-folders.
    if any(p.is_dir() for p in data_dir.iterdir()):
        return data_dir
    raise FileNotFoundError(f"No class folders found under {data_dir}")


def _inspect(path: Path) -> tuple[str, str | None]:
    """Return (path, md5) for a readable image, or (path, None) if corrupted."""
    try:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        return str(path), hashlib.md5(raw).hexdigest()
    except Exception:
        return str(path), None


def scan_images(root: Path, workers: int = 16) -> tuple[pd.DataFrame, dict]:
    """List every image with its class, dropping corrupted files and exact duplicates."""
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths, labels = [], []
    for label, cls in enumerate(class_names):
        for f in sorted((root / cls).iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(f)
                labels.append(label)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_inspect, paths))

    df = pd.DataFrame(
        {
            "path": [p for p, _ in results],
            "md5": [h for _, h in results],
            "label": labels,
        }
    )
    df["class_name"] = df["label"].map(dict(enumerate(class_names)))

    corrupted = df[df["md5"].isna()]
    df = df.dropna(subset=["md5"])
    # Identical files in the same split inflate scores; across splits they leak.
    duplicates = df[df.duplicated("md5", keep="first")]
    df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)

    report = {
        "num_classes": len(class_names),
        "total_files": len(paths),
        "corrupted": len(corrupted),
        "duplicates_removed": len(duplicates),
        "usable": len(df),
        "corrupted_samples": corrupted["path"].head(10).tolist(),
    }
    return df[["path", "label", "class_name"]], {"class_names": class_names, **report}


def split_dataframe(
    df: pd.DataFrame, val_frac: float, test_frac: float, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split so every class keeps its proportions."""
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_frac / (1 - test_frac),
        stratify=train_val["label"],
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    out = output_dir / SPLITS_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(out / f"{name}.csv", index=False)


def load_splits(output_dir: Path) -> dict[str, pd.DataFrame]:
    out = output_dir / SPLITS_DIR
    return {name: pd.read_csv(out / f"{name}.csv") for name in ("train", "val", "test")}


def prepare_splits(
    data_dir: Path | None, output_dir: Path, val_frac: float, test_frac: float
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Reuse saved splits when present so every run evaluates on the same test set."""
    if (output_dir / SPLITS_DIR / "test.csv").exists():
        splits = load_splits(output_dir)
        class_names = (
            pd.concat(splits.values())
            .drop_duplicates("label")
            .sort_values("label")["class_name"]
            .tolist()
        )
        print(f"Reusing saved splits from {output_dir / SPLITS_DIR}")
        return splits, class_names

    root = locate_dataset(data_dir)
    print(f"Scanning images in {root} ...")
    df, report = scan_images(root)
    print(
        f"{report['total_files']} files | {report['corrupted']} corrupted | "
        f"{report['duplicates_removed']} duplicates removed | {report['usable']} usable | "
        f"{report['num_classes']} classes"
    )
    splits = split_dataframe(df, val_frac, test_frac)
    save_splits(splits, output_dir)
    for name, frame in splits.items():
        print(f"  {name:<5} {len(frame):>6} images")
    return splits, report["class_names"]


def _load_image(path: tf.Tensor, img_size: int) -> tf.Tensor:
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32)  # 0..255; scaling happens inside the model


def make_dataset(
    frame: pd.DataFrame, img_size: int, batch_size: int, shuffle: bool = False
) -> tf.data.Dataset:
    """Stream images from disk. Augmentation lives inside the model, so it is
    re-sampled every epoch and never cached."""
    ds = tf.data.Dataset.from_tensor_slices((frame["path"].values, frame["label"].values))
    if shuffle:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_image(p, img_size), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
