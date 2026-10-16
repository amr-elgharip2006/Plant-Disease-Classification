"""Load a trained model and classify new leaf images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageOps

from .config import METADATA_FILE, MODEL_FILE
from .labels import is_healthy, parse_label


@dataclass
class Prediction:
    class_name: str
    plant: str
    condition: str
    healthy: bool
    confidence: float


class PlantDiseaseClassifier:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        artifacts_dir = Path(artifacts_dir)
        meta = json.loads((artifacts_dir / METADATA_FILE).read_text())
        self.class_names: list[str] = meta["class_names"]
        self.img_size: int = meta["img_size"]
        self.model = keras.models.load_model(artifacts_dir / MODEL_FILE)

    def _prepare(self, image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Center-crop to a square (training images are square), same as the website.
        side = min(image.size)
        image = ImageOps.fit(image, (side, side), centering=(0.5, 0.5))
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32)

    def predict(self, image, top_k: int = 3) -> list[Prediction]:
        probs = self.model.predict(self._prepare(image)[None], verbose=0)[0]
        return self._top(probs, top_k)

    def predict_batch(self, images, top_k: int = 3) -> list[list[Prediction]]:
        batch = np.stack([self._prepare(img) for img in images])
        return [self._top(p, top_k) for p in self.model.predict(batch, verbose=0)]

    def _top(self, probs: np.ndarray, top_k: int) -> list[Prediction]:
        out = []
        for idx in np.argsort(probs)[::-1][:top_k]:
            name = self.class_names[idx]
            plant, condition = parse_label(name)
            out.append(Prediction(name, plant, condition, is_healthy(name), float(probs[idx])))
        return out
