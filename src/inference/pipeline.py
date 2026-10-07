from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, RobertaTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "models" / "twitter-roberta-base-sentiment"
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
WHITESPACE_RE = re.compile(r"\s+")

FALLBACK_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def preprocess_twitter_text(text: str) -> str:
    """
    Conservative Twitter preprocessing.

    URLs and usernames are normalized, while emojis and hashtag text
    are preserved because they can carry sentiment information.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()

    text = URL_RE.sub(" <URL> ", text)
    text = USER_RE.sub(" <USER> ", text)

    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    model_status: str
    sarcasm: dict[str, Any] | None = None


class SentimentPipeline:

    def __init__(self, model_dir: Path | None = None):

        env_path = os.getenv("SENTIMENT_MODEL_DIR")

        self.model_dir = (
            Path(env_path).expanduser()
            if env_path
            else (model_dir or DEFAULT_MODEL_DIR)
        )

        self.device = torch.device("cpu")

        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):

        if self._loaded:
            return

        has_local_weights = (
            (self.model_dir / "model.safetensors").exists()
            or (self.model_dir / "pytorch_model.bin").exists()
        )
        model_source = str(self.model_dir) if has_local_weights else "cardiffnlp/twitter-roberta-base-sentiment"

        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_source,
            local_files_only=has_local_weights,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            local_files_only=has_local_weights,
        )

        self.model.to(self.device)
        self.model.eval()

        self._loaded = True

    def _label_name(self, index: int) -> str:

        labels = getattr(
            self.model.config,
            "id2label",
            {},
        )

        raw = labels.get(index)

        if raw:

            normalized = str(raw).strip().lower()

            if normalized in {
                "negative",
                "neutral",
                "positive",
            }:
                return normalized

        return FALLBACK_LABELS.get(
            index,
            f"label_{index}",
        )

    def predict(self, text: str) -> Prediction:

        self.load()

        cleaned = preprocess_twitter_text(text)

        if not cleaned:
            raise ValueError(
                "text must not be empty"
            )

        encoded = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():

            outputs = self.model(**encoded)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

        probs = {
            self._label_name(i): float(
                probabilities[i].item()
            )
            for i in range(
                probabilities.shape[0]
            )
        }

        best_label = max(
            probs,
            key=probs.get,
        )

        return Prediction(
            label=best_label,
            confidence=probs[best_label],
            probabilities=probs,
            model="cardiffnlp/twitter-roberta-base-sentiment",
            model_status="baseline_local_verified",
            sarcasm=None,
        )


_PIPELINE: SentimentPipeline | None = None


def get_pipeline() -> SentimentPipeline:

    global _PIPELINE

    if _PIPELINE is None:
        _PIPELINE = SentimentPipeline()

    return _PIPELINE


def predict_text(
    text: str,
) -> dict[str, Any]:

    prediction = get_pipeline().predict(text)

    return asdict(prediction)