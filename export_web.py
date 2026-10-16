"""Convert the trained model to TensorFlow.js for the in-browser website.

Writes web/model/ (model.json + float16 weight shards) and web/model/metadata.json.

Example:
    python export_web.py --artifacts artifacts
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import keras

from plant_disease.config import METADATA_FILE, MODEL_FILE
from plant_disease.labels import is_healthy, parse_label

# tensorflowjs imports these at load time for decision-forest and JAX models,
# which this converter path never uses; stubs avoid installing heavy packages.
for _name in ("tensorflow_decision_forests", "jax", "jax.experimental", "jax.experimental.jax2tf"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
from tensorflowjs.converters.tf_saved_model_conversion_v2 import convert_tf_saved_model  # noqa: E402


def strip_augmentation(model: keras.Model) -> keras.Model:
    """Rebuild the (linear) model without its augmentation block.

    Augmentation is a no-op at inference, but its random-state variables break
    the TF.js graph freezer.
    """
    inputs = keras.Input(shape=model.input_shape[1:], name="image")
    x = inputs
    for layer in model.layers[1:]:
        if layer.name != "augmentation":
            x = layer(x)
    return keras.Model(inputs, x, name=model.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--out", type=Path, default=Path("web/model"))
    a = p.parse_args()

    meta = json.loads((a.artifacts / METADATA_FILE).read_text())
    model = strip_augmentation(keras.models.load_model(a.artifacts / MODEL_FILE))

    if a.out.exists():
        shutil.rmtree(a.out)
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, format="tf_saved_model", verbose=False)
        convert_tf_saved_model(
            tmp,
            str(a.out),
            signature_def="serving_default",
            saved_model_tags="serve",
            quantization_dtype_map={"float16": "*"},
        )

    classes = [
        {"id": name, "plant": plant, "condition": condition, "healthy": is_healthy(name)}
        for name in meta["class_names"]
        for plant, condition in [parse_label(name)]
    ]
    metrics_file = a.artifacts / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    (a.out / "metadata.json").write_text(
        json.dumps(
            {
                "img_size": meta["img_size"],
                "backbone": meta["config"]["backbone"],
                "classes": classes,
                "test_accuracy": metrics.get("accuracy"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    size_mb = sum(f.stat().st_size for f in a.out.iterdir()) / 1e6
    print(f"Exported to {a.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
