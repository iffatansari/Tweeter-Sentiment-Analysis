from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("models/sarcasm-deberta-v3")

TEMPERATURE = 1.1819
DECISION_THRESHOLD = 0.360
SARCASM_CLASS_INDEX = 1
MAX_LENGTH = 128


@lru_cache(maxsize=1)
def _load_model():
    has_local_weights = (
        (MODEL_DIR / "model.safetensors").exists()
        or (MODEL_DIR / "pytorch_model.bin").exists()
    )
    model_source = str(MODEL_DIR) if has_local_weights else "ppokhrel2109/besstie-sarcasm-deberta-v3"

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=has_local_weights,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        local_files_only=has_local_weights,
    )

    model.eval()
    model.to("cpu")

    return tokenizer, model


def predict_sarcasm(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()

    if not text:
        raise ValueError("text cannot be empty")

    tokenizer, model = _load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    calibrated_logits = logits / TEMPERATURE
    probabilities = torch.softmax(
        calibrated_logits,
        dim=-1,
    )

    sarcasm_probability = float(
        probabilities[SARCASM_CLASS_INDEX].item()
    )

    is_sarcastic = (
        sarcasm_probability >= DECISION_THRESHOLD
    )

    id2label = getattr(
        model.config,
        "id2label",
        {},
    ) or {}

    probability_dict = {
        str(id2label.get(index, f"LABEL_{index}")):
        float(probabilities[index].item())
        for index in range(len(probabilities))
    }

    return {
        "label": (
            "sarcastic"
            if is_sarcastic
            else "not_sarcastic"
        ),
        "is_sarcastic": is_sarcastic,
        "sarcasm_probability": sarcasm_probability,
        "threshold": DECISION_THRESHOLD,
        "temperature": TEMPERATURE,
        "probabilities": probability_dict,
        "model": (
            "ppokhrel2109/"
            "besstie-sarcasm-deberta-v3"
        ),
        "model_status": (
            "local_safetensors_verified_calibrated"
        ),
    }


def model_info() -> dict[str, Any]:
    return {
        "model": (
            "ppokhrel2109/"
            "besstie-sarcasm-deberta-v3"
        ),
        "local_path": str(MODEL_DIR),
        "exists": MODEL_DIR.exists(),
        "weights_exists": (
            MODEL_DIR / "model.safetensors"
        ).exists(),
        "temperature": TEMPERATURE,
        "decision_threshold": DECISION_THRESHOLD,
        "sarcasm_class_index": SARCASM_CLASS_INDEX,
        "max_length": MAX_LENGTH,
        "device": "cpu",
    }
