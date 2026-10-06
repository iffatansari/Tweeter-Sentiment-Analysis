from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.fusion.inference import (
    FusionNotReady,
    _load_fusion_bundle,
    build_feature_vector,
)
from src.inference.pipeline import predict_text
from src.sarcasm.deberta_detector import predict_sarcasm


FUSION_DIR = Path("models/fusion-v2")
X_TRAIN_PATH = FUSION_DIR / "X_train.npy"

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


@lru_cache(maxsize=1)
def _load_shap_explainer():
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed."
        ) from exc

    if not X_TRAIN_PATH.exists():
        raise RuntimeError(
            "Fusion-v2 training features are missing: "
            f"{X_TRAIN_PATH}"
        )

    model, metadata = _load_fusion_bundle()

    if not hasattr(model, "named_steps"):
        raise RuntimeError(
            "Fusion-v2 model must be an sklearn Pipeline."
        )

    scaler = model.named_steps.get("scaler")
    classifier = model.named_steps.get("classifier")

    if scaler is None or classifier is None:
        raise RuntimeError(
            "Fusion-v2 Pipeline must contain scaler "
            "and classifier steps."
        )

    X_train = np.load(
        X_TRAIN_PATH
    ).astype(np.float64)

    background = scaler.transform(
        X_train[:100]
    )

    explainer = shap.LinearExplainer(
        classifier,
        background,
    )

    return (
        explainer,
        scaler,
        classifier,
        metadata,
    )


def _get_class_index(
    classifier: Any,
    predicted_index: int,
) -> int:
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

    predicted_class = int(
        classes[predicted_index]
    )

    return predicted_class


def _normalize_shap_values(
    shap_values: Any,
    class_index: int,
    n_features: int,
) -> np.ndarray:
    """
    Normalize SHAP's possible output formats into
    one feature vector for the selected class.
    """

    values = getattr(
        shap_values,
        "values",
        shap_values,
    )

    if isinstance(values, list):
        # Older SHAP multiclass format:
        # [class0_values, class1_values, class2_values]
        return np.asarray(
            values[class_index][0],
            dtype=np.float64,
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim == 1:
        return values

    if values.ndim == 2:
        # Single output: [samples, features]
        return values[0]

    if values.ndim == 3:
        # Modern multiclass format:
        # [samples, features, classes]
        return values[
            0,
            :,
            class_index,
        ]

    raise RuntimeError(
        "Unexpected SHAP output shape: "
        f"{values.shape}"
    )


def explain_text(
    text: str,
) -> dict[str, Any]:

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "text cannot be empty"
        )

    try:
        (
            explainer,
            scaler,
            classifier,
            metadata,
        ) = _load_shap_explainer()

    except FusionNotReady:
        raise FusionNotReady(
            "Explainability requires the "
            "verified final Fusion-v2 model."
        )

    sentiment = predict_text(text)
    sarcasm = predict_sarcasm(text)

    feature_names = metadata.get(
        "feature_names",
        [],
    )

    if not feature_names:
        raise RuntimeError(
            "Fusion-v2 feature names are missing."
        )

    x = build_feature_vector(
        text,
        sentiment,
        sarcasm,
        feature_names,
    )

    x_scaled = scaler.transform(
        x
    )

    probabilities = classifier.predict_proba(
        x_scaled
    )[0]

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

    labels = [
        LABEL_MAP.get(
            int(cls),
            str(cls),
        )
        for cls in classes
    ]

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_class = _get_class_index(
        classifier,
        predicted_index,
    )

    shap_result = explainer(
        x_scaled
    )

    shap_values = _normalize_shap_values(
        shap_result,
        predicted_index,
        len(feature_names),
    )

    if len(shap_values) != len(
        feature_names
    ):
        raise RuntimeError(
            "SHAP feature count does not match "
            "Fusion-v2 feature count."
        )

    rows = []

    for (
        name,
        value,
        shap_value,
    ) in zip(
        feature_names,
        x[0],
        shap_values,
    ):
        rows.append(
            {
                "feature": name,
                "value": float(value),
                "shap_value": float(
                    shap_value
                ),
                "direction": (
                    "supports_prediction"
                    if shap_value >= 0
                    else "opposes_prediction"
                ),
            }
        )

    rows.sort(
        key=lambda item:
            abs(item["shap_value"]),
        reverse=True,
    )

    base_value = None

    raw_base = getattr(
        shap_result,
        "base_values",
        None,
    )

    if raw_base is not None:
        base_array = np.asarray(
            raw_base
        )

        if base_array.ndim >= 2:
            base_value = float(
                base_array[
                    0,
                    predicted_index,
                ]
            )
        elif base_array.ndim == 1:
            base_value = float(
                base_array[
                    predicted_index
                ]
            )
        elif base_array.ndim == 0:
            base_value = float(
                base_array
            )

    return {
        "text": text,
        "predicted_label":
            labels[predicted_index],
        "confidence":
            float(
                probabilities[
                    predicted_index
                ]
            ),
        "probabilities": {
            str(label): float(prob)
            for label, prob in zip(
                labels,
                probabilities,
            )
        },
        "model":
            metadata.get(
                "model_name",
                "fusion-v2-logistic-regression",
            ),
        "model_status":
            "fusion_v2_verified",
        "explanation_type":
            "shap_linear_explainer",
        "explained_class":
            predicted_class,
        "base_value":
            base_value,
        "top_contributions":
            rows[:10],
    }
