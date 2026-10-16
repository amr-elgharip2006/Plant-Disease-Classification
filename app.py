"""Web demo: upload a leaf photo and get the predicted disease.

Example:
    python app.py --artifacts artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from plant_disease.inference import PlantDiseaseClassifier
from plant_disease.labels import pretty_label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--share", action="store_true", help="create a public Gradio link (useful on Colab)")
    a = p.parse_args()

    clf = PlantDiseaseClassifier(a.artifacts)

    def classify(image):
        if image is None:
            return {}, ""
        preds = clf.predict(image, top_k=5)
        best = preds[0]
        verdict = (
            f"### ✅ {best.plant}: healthy"
            if best.healthy
            else f"### ⚠️ {best.plant}: {best.condition}"
        )
        verdict += f"\nConfidence: **{best.confidence:.1%}**"
        if best.confidence < 0.6:
            verdict += "\n\nLow confidence: try a clearer, closer photo of a single leaf."
        return {pretty_label(p.class_name): p.confidence for p in preds}, verdict

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="numpy", label="Leaf photo"),
        outputs=[gr.Label(num_top_classes=5, label="Top predictions"), gr.Markdown()],
        title="🌿 Plant Disease Classifier",
        description=(
            f"Detects {len(clf.class_names)} plant/disease classes from a single leaf photo. "
            "Trained on PlantVillage (lab photos), so field photos may be less accurate."
        ),
        flagging_mode="never",
    )
    demo.launch(share=a.share)


if __name__ == "__main__":
    main()
