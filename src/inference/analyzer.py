from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.features.emoji_features import (
    extract_emoji_features,
)
from src.fusion.inference import (
    FusionNotReady,
    get_fusion_info,
    predict_fused,
)
from src.inference.pipeline import predict_text
from src.sarcasm.deberta_detector import (
    model_info as sarcasm_model_info,
    predict_sarcasm,
)


SENTIMENT_MODEL_DIR = Path(
    "models/twitter-roberta-base-sentiment"
)


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def analyze_text(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "text cannot be empty"
        )

    sentiment = predict_text(text)
    sarcasm = predict_sarcasm(text)
    emoji = extract_emoji_features(text)

    try:
        final_prediction = predict_fused(
            text,
            sentiment,
            sarcasm,
        )

        pipeline_status = (
            "final_fusion_active"
        )

    except FusionNotReady:
        final_prediction = {
            "label":
                sentiment["label"],
            "confidence":
                sentiment["confidence"],
            "probabilities":
                sentiment["probabilities"],
            "model":
                sentiment["model"],
            "model_status":
                "baseline_fallback_until_fusion_verified",
        }

        pipeline_status = (
            "fusion_not_ready_using_verified_baseline"
        )

    return {
        "text": text,
        "final": final_prediction,
        "sentiment": {
            "label":
                sentiment["label"],
            "confidence":
                sentiment["confidence"],
            "probabilities":
                sentiment["probabilities"],
            "model":
                sentiment["model"],
            "model_status":
                sentiment["model_status"],
        },
        "sarcasm": sarcasm,
        "emoji": emoji,
        "pipeline_status":
            pipeline_status,
    }


def get_backend_model_info():
    return {
        "sentiment_model": {
            "name":
                "cardiffnlp/twitter-roberta-base-sentiment",
            "local_path":
                str(SENTIMENT_MODEL_DIR),
            "ready":
                (
                    SENTIMENT_MODEL_DIR
                    / "model.safetensors"
                ).exists(),
        },
        "sarcasm_model":
            sarcasm_model_info(),
        "fusion_model":
            get_fusion_info(),
        "runtime": {
            "torch":
                _version("torch"),
            "transformers":
                _version("transformers"),
            "fastapi":
                _version("fastapi"),
            "scikit-learn":
                _version("scikit-learn"),
            "openai-whisper":
                _version("openai-whisper"),
        },
    }
