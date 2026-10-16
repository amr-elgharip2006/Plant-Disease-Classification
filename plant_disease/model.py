"""Model construction: ImageNet backbone + augmentation + classification head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import keras
from keras import layers


@dataclass(frozen=True)
class Backbone:
    factory: Callable[..., keras.Model]
    native_size: int
    # Maps raw 0..255 pixels to what the backbone expects. EfficientNet
    # rescales internally, so it needs nothing.
    preprocess: Callable[[], layers.Layer | None]


BACKBONES: dict[str, Backbone] = {
    "mobilenetv2": Backbone(
        keras.applications.MobileNetV2, 224, lambda: layers.Rescaling(1 / 127.5, offset=-1)
    ),
    "efficientnetb0": Backbone(keras.applications.EfficientNetB0, 224, lambda: None),
    "efficientnetb2": Backbone(keras.applications.EfficientNetB2, 260, lambda: None),
}


def get_backbone(name: str) -> Backbone:
    try:
        return BACKBONES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {sorted(BACKBONES)}") from None


def build_augmentation() -> keras.Sequential:
    """Random transforms; active only during training (model.fit)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.25),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ],
        name="augmentation",
    )


def build_model(
    backbone_name: str, num_classes: int, img_size: int, dropout: float = 0.4
) -> tuple[keras.Model, keras.Model]:
    """Return (full model, backbone). The backbone starts frozen."""
    spec = get_backbone(backbone_name)
    base = spec.factory(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = build_augmentation()(inputs)
    preprocess = spec.preprocess()
    if preprocess is not None:
        x = preprocess(x)
    # training=False keeps BatchNorm statistics frozen even while fine-tuning.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(256, activation="swish", name="dense")(x)
    x = layers.Dropout(dropout / 2, name="dropout_2")(x)
    # float32 output keeps softmax stable under mixed precision.
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"plant_disease_{backbone_name}"), base


def backbone_of(model: keras.Model) -> keras.Model:
    """Return the ImageNet backbone nested inside a model built by build_model."""
    return next(
        layer for layer in model.layers
        if isinstance(layer, keras.Model) and layer.name != "augmentation"
    )


def unfreeze_top(base: keras.Model, fraction: float) -> int:
    """Make the top `fraction` of backbone layers trainable, except BatchNorm."""
    base.trainable = True
    cutoff = int(len(base.layers) * (1 - fraction))
    trainable = 0
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
        trainable += layer.trainable
    return trainable


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
