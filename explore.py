"""Download and clean the dataset, create the splits, and save EDA figures.

Example:
    python explore.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plant_disease.config import TrainConfig
from plant_disease.data import prepare_splits
from plant_disease.evaluation import plot_class_distribution, plot_samples


def main() -> None:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    a = p.parse_args()

    splits, class_names = prepare_splits(a.data_dir, a.output_dir, d.val_frac, d.test_frac)
    eda = a.output_dir / "eda"
    eda.mkdir(parents=True, exist_ok=True)

    everything = pd.concat(splits.values(), ignore_index=True)
    plot_class_distribution(everything, eda / "class_distribution.png")
    plot_samples(splits["train"], eda / "samples.png")

    counts = everything["class_name"].value_counts()
    print(f"\n{len(class_names)} classes, {len(everything)} images")
    print(f"Largest class:  {counts.index[0]} ({counts.iloc[0]})")
    print(f"Smallest class: {counts.index[-1]} ({counts.iloc[-1]})")
    print(f"Imbalance ratio: {counts.iloc[0] / counts.iloc[-1]:.1f}x")
    print(f"Figures saved to {eda.resolve()}")


if __name__ == "__main__":
    main()
