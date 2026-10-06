from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "sarcasm-detector"
)


class SarcasmDetector:
    """Local trained sarcasm classifier."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
    ):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Sarcasm model directory not found: {self.model_dir}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_safetensors=True,
        )

        self.model.eval()
        self.model.to("cpu")

        self.labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }

        self.sarcastic_index = self._find_sarcasm_index()

    def _find_sarcasm_index(self) -> int:
        """Find the sarcasm class from the model configuration."""

        for index, label in self.labels.items():
            if "sarcas" in label:
                return index

        # Common binary-class fallback.
        return 1

    def predict(self, text: str) -> dict:
        text = str(text).strip()

        if not text:
            raise ValueError("Text cannot be empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predicted_id = int(
            torch.argmax(probabilities).item()
        )

        predicted_label = self.labels.get(
            predicted_id,
            str(predicted_id),
        )

        sarcasm_probability = float(
            probabilities[self.sarcastic_index].item()
        )

        return {
            "label": predicted_label,
            "sarcasm_probability": sarcasm_probability,
            "is_sarcastic": sarcasm_probability >= 0.5,
            "probabilities": {
                self.labels.get(i, str(i)): float(
                    probabilities[i].item()
                )
                for i in range(len(probabilities))
            },
            "model": "vyshnav112233/distilbert-base-sarcasm",
            "model_status": "local_safetensors_verified",
        }


_detector: Optional[SarcasmDetector] = None


def get_sarcasm_detector() -> SarcasmDetector:
    global _detector

    if _detector is None:
        _detector = SarcasmDetector()

    return _detector


def predict_sarcasm(text: str) -> dict:
    return get_sarcasm_detector().predict(text)