"""Human-readable names for PlantVillage folder names like `Tomato___Early_blight`."""

from __future__ import annotations


def parse_label(folder_name: str) -> tuple[str, str]:
    """Return (plant, condition), e.g. ('Tomato', 'Early blight')."""
    plant, _, condition = folder_name.partition("___")
    plant = plant.replace("_", " ").replace(",", "").strip()
    condition = condition.replace("_", " ").strip() or "unknown"
    condition = " ".join(condition.split())
    return plant, condition[0].upper() + condition[1:]


def pretty_label(folder_name: str) -> str:
    plant, condition = parse_label(folder_name)
    return f"{plant} — {condition}"


def is_healthy(folder_name: str) -> bool:
    return parse_label(folder_name)[1].lower() == "healthy"
