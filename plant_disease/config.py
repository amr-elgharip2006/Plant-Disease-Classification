"""Settings shared by training, evaluation and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# File names written inside the artifacts directory.
MODEL_FILE = "model.keras"
METADATA_FILE = "metadata.json"
SPLITS_DIR = "splits"
HISTORY_FILE = "history.csv"
METRICS_FILE = "metrics.json"


@dataclass
class TrainConfig:
    backbone: str = "efficientnetb0"
    img_size: int | None = None  # None -> backbone's native resolution
    batch_size: int = 32
    val_frac: float = 0.15
    test_frac: float = 0.15

    # Phase 1: train only the new classification head on a frozen backbone.
    head_epochs: int = 6
    head_lr: float = 1e-3

    # Phase 2: unfreeze the top of the backbone and fine-tune with a small LR.
    finetune_epochs: int = 20
    finetune_lr: float = 3e-5
    unfreeze_fraction: float = 1.0  # whole backbone (BatchNorm stays frozen)

    dropout: float = 0.4
    class_weights: bool = True
    mixed_precision: bool = False
    data_dir: Path | None = None
    init_from: Path | None = None
    tta: bool = False
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir) if self.data_dir else None
        d["init_from"] = str(self.init_from) if self.init_from else None
        d["output_dir"] = str(self.output_dir)
        return d
