from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features.emoji_features import (
    EMOJI_FEATURE_NAMES,
    extract_emoji_features,
)


# ============================================================
# Fusion-v2 paths
# ============================================================

FUSION_DIR = Path("models/fusion-v2")

MODEL_PATH = FUSION_DIR / "fusion_classifier.joblib"
METADATA_PATH = FUSION_DIR / "metadata.json"


# Fusion-v2 was trained with exactly these 8 features,
# in exactly this order.
DEFAULT_FEATURE_NAMES = [
    "sentiment_negative_probability",
    "sentiment_neutral_probability",
    "sentiment_positive_probability",
    "sarcasm_probability",
    "emoji_positive",
    "emoji_negative",
    "emoji_count",
    "emoji_present",
]


class FusionNotReady(RuntimeError):
    pass


def fusion_ready() -> bool:
    return (
        MODEL_PATH.exists()
        and METADATA_PATH.exists()
    )


@lru_cache(maxsize=1)
def _load_fusion_bundle():
    if not fusion_ready():
        raise FusionNotReady(
            "Fusion-v2 model or metadata is missing."
        )

    # Fusion-v2 is already a complete sklearn Pipeline:
    # StandardScaler -> LogisticRegression
    #
    # Therefore, DO NOT load or apply a separate scaler.
    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    return model, metadata


def build_feature_dict(
    text: str,
    sentiment: dict[str, Any],
    sarcasm: dict[str, Any],
) -> dict[str, float]:
    probabilities = sentiment["probabilities"]

    features = {
        "sentiment_negative_probability":
            float(
                probabilities.get(
                    "negative",
                    0.0,
                )
            ),
        "sentiment_neutral_probability":
            float(
                probabilities.get(
                    "neutral",
                    0.0,
                )
            ),
        "sentiment_positive_probability":
            float(
                probabilities.get(
                    "positive",
                    0.0,
                )
            ),
        "sarcasm_probability":
            float(
                sarcasm["sarcasm_probability"]
            ),
    }

    # Map the extractor's feature names to the exact
    # Fusion-v2 training schema.
    emoji = extract_emoji_features(text)

    features.update({
        "emoji_positive":
            float(
                emoji.get(
                    "positive_emoji_count",
                    0.0,
                )
            ),
        "emoji_negative":
            float(
                emoji.get(
                    "negative_emoji_count",
                    0.0,
                )
            ),
        "emoji_count":
            float(
                emoji.get(
                    "emoji_count",
                    0.0,
                )
            ),
        "emoji_present":
            1.0
            if emoji.get(
                "emoji_count",
                0.0,
            ) > 0
            else 0.0,
    })

    return features


def build_feature_vector(
    text: str,
    sentiment: dict[str, Any],
    sarcasm: dict[str, Any],
    feature_names: list[str] | None = None,
) -> np.ndarray:
    feature_dict = build_feature_dict(
        text,
        sentiment,
        sarcasm,
    )

    names = (
        feature_names
        or DEFAULT_FEATURE_NAMES
    )

    vector = [
        float(
            feature_dict.get(
                name,
                0.0,
            )
        )
        for name in names
    ]

    return np.asarray(
        [vector],
        dtype=np.float32,
    )


def predict_fused(
    text: str,
    sentiment: dict[str, Any],
    sarcasm: dict[str, Any],
) -> dict[str, Any]:
    model, metadata = _load_fusion_bundle()

    feature_names = metadata.get(
        "feature_names",
        DEFAULT_FEATURE_NAMES,
    )

    # Safety check: the deployed feature schema must match
    # the research Fusion-v2 schema.
    if feature_names != DEFAULT_FEATURE_NAMES:
        raise RuntimeError(
            "Fusion-v2 feature schema mismatch. "
            f"Expected {DEFAULT_FEATURE_NAMES}, "
            f"got {feature_names}."
        )

    x = build_feature_vector(
        text,
        sentiment,
        sarcasm,
        feature_names,
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise RuntimeError(
            "Fusion-v2 model does not support "
            "predict_proba()."
        )

    probabilities_array = (
        model.predict_proba(x)[0]
    )

    # Fusion-v2 classifier classes are [0, 1, 2].
    # They correspond to:
    # 0 = negative
    # 1 = neutral
    # 2 = positive
    classes = list(
        getattr(
            model,
            "classes_",
            [],
        )
    )

    # sklearn Pipeline itself may not expose classes_
    # depending on version, so fall back to the final classifier.
    if not classes and hasattr(
        model,
        "named_steps",
    ):
        classifier = model.named_steps.get(
            "classifier"
        )

        if classifier is not None:
            classes = list(
                getattr(
                    classifier,
                    "classes_",
                    [],
                )
            )

    if not classes:
        raise RuntimeError(
            "Fusion-v2 classifier classes are missing."
        )

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    labels = [
        label_map.get(
            int(cls),
            str(cls),
        )
        for cls in classes
    ]

    if len(labels) != len(
        probabilities_array
    ):
        raise RuntimeError(
            "Fusion-v2 class count does not match "
            "model probability output."
        )

    predicted_index = int(
        np.argmax(
            probabilities_array
        )
    )

    return {
        "label":
            labels[predicted_index],
        "confidence":
            float(
                probabilities_array[
                    predicted_index
                ]
            ),
        "probabilities": {
            str(label): float(prob)
            for label, prob in zip(
                labels,
                probabilities_array,
            )
        },
        "model":
            metadata.get(
                "model_name",
                "fusion-v2-logistic-regression",
            ),
        "model_status":
            "fusion_v2_verified",
        "feature_names":
            feature_names,
    }


def get_fusion_info() -> dict[str, Any]:
    info = {
        "directory":
            str(FUSION_DIR),
        "model_path":
            str(MODEL_PATH),
        "metadata_path":
            str(METADATA_PATH),
        "ready":
            fusion_ready(),
        "feature_names":
            DEFAULT_FEATURE_NAMES,
        "model_type":
            "sklearn Pipeline",
        "pipeline":
            [
                "StandardScaler",
                "LogisticRegression",
            ],
    }

    if not fusion_ready():
        return info

    try:
        model, metadata = _load_fusion_bundle()

        classifier = None

        if hasattr(
            model,
            "named_steps",
        ):
            classifier = model.named_steps.get(
                "classifier"
            )

        classes = []

        if classifier is not None:
            classes = [
                int(cls)
                for cls in getattr(
                    classifier,
                    "classes_",
                    [],
                )
            ]

        info.update(
            {
                "model_name":
                    metadata.get(
                        "model_name",
                        "fusion-v2-logistic-regression",
                    ),
                "classes":
                    classes,
                "metadata_feature_names":
                    metadata.get(
                        "feature_names",
                        DEFAULT_FEATURE_NAMES,
                    ),
            }
        )

    except Exception as exc:
        info["load_error"] = str(exc)

    return info
