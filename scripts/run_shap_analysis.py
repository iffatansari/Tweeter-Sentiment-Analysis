from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = PROJECT_ROOT / "models" / "fusion-v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shap" / "fusion_v2_robust"

MODEL_PATH = FUSION_DIR / "fusion_classifier.joblib"
METADATA_PATH = FUSION_DIR / "metadata.json"

X_TRAIN_PATH = FUSION_DIR / "X_train.npy"
X_TEST_PATH = FUSION_DIR / "X_test.npy"
Y_TEST_PATH = FUSION_DIR / "y_test.npy"

RANDOM_SEED = 42
BACKGROUND_SIZE = 100
SARCASM_THRESHOLD = 0.36

DEFAULT_FEATURE_NAMES = [
    "roberta_negative_probability",
    "roberta_neutral_probability",
    "roberta_positive_probability",
    "deberta_sarcasm_probability",
    "emoji_positive",
    "emoji_negative",
    "emoji_count",
    "emoji_present",
]

DEFAULT_CLASS_NAMES = [
    "negative",
    "neutral",
    "positive",
]


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_feature_names(metadata: dict) -> list[str]:
    for key in (
        "feature_names",
        "features",
        "feature_columns",
    ):
        value = metadata.get(key)

        if isinstance(value, list) and len(value) > 0:
            return [str(item) for item in value]

    return DEFAULT_FEATURE_NAMES.copy()


def get_class_names(metadata: dict, estimator) -> list[str]:
    for key in (
        "class_names",
        "label_names",
        "classes",
    ):
        value = metadata.get(key)

        if isinstance(value, list) and len(value) > 0:
            return [str(item) for item in value]

    if hasattr(estimator, "classes_"):
        classes = np.asarray(estimator.classes_)

        if np.array_equal(classes, np.array([0, 1, 2])):
            return DEFAULT_CLASS_NAMES.copy()

        return [str(item) for item in classes]

    return DEFAULT_CLASS_NAMES.copy()


def load_model_and_preprocessor():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fusion-v2 model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if hasattr(bundle, "steps"):
        steps = list(bundle.steps)

        if len(steps) < 1:
            raise ValueError("Fusion-v2 pipeline contains no steps.")

        estimator = steps[-1][1]

        transformers = steps[:-1]

        def transform_features(X):
            result = np.asarray(X, dtype=np.float64)

            for _, transformer in transformers:
                if hasattr(transformer, "transform"):
                    result = transformer.transform(result)

            return np.asarray(result, dtype=np.float64)

        return estimator, transform_features

    if hasattr(bundle, "predict_proba"):
        estimator = bundle

        def transform_features(X):
            return np.asarray(X, dtype=np.float64)

        return estimator, transform_features

    raise TypeError(
        "Unsupported Fusion-v2 model artifact. "
        "Expected a sklearn Pipeline or estimator."
    )


def normalize_shap_values(raw_values, n_rows, n_features, n_classes):
    """
    Normalize SHAP output to:
        (rows, features, classes)
    """

    if isinstance(raw_values, list):
        arrays = [
            np.asarray(item, dtype=np.float64)
            for item in raw_values
        ]

        if len(arrays) == 0:
            raise ValueError("SHAP returned an empty list.")

        result = np.stack(arrays, axis=-1)

    else:
        result = np.asarray(raw_values, dtype=np.float64)

    if result.ndim == 2:
        if result.shape != (n_rows, n_features):
            raise ValueError(
                "Unexpected 2D SHAP shape: "
                f"{result.shape}"
            )

        result = result[:, :, None]

    elif result.ndim == 3:

        # Normal expected layout:
        # rows x features x classes
        if (
            result.shape[0] == n_rows
            and result.shape[1] == n_features
        ):
            pass

        # Alternate layout:
        # rows x classes x features
        elif (
            result.shape[0] == n_rows
            and result.shape[2] == n_features
        ):
            result = np.transpose(result, (0, 2, 1))

        else:
            raise ValueError(
                "Unexpected 3D SHAP shape: "
                f"{result.shape}"
            )

    else:
        raise ValueError(
            "Unsupported SHAP dimensionality: "
            f"{result.ndim}"
        )

    if result.shape[0] != n_rows:
        raise ValueError(
            f"SHAP row mismatch: {result.shape}"
        )

    if result.shape[1] != n_features:
        raise ValueError(
            f"SHAP feature mismatch: {result.shape}"
        )

    # Some binary SHAP outputs have one class dimension.
    # Keep it valid rather than inventing a second class.
    if result.shape[2] != n_classes:
        if result.shape[2] == 1:
            return result

        raise ValueError(
            "SHAP class mismatch: "
            f"{result.shape}, expected {n_classes} classes."
        )

    return result


def create_explainer(estimator, X_train_scaled):
    background_size = min(
        BACKGROUND_SIZE,
        len(X_train_scaled),
    )

    rng = np.random.default_rng(RANDOM_SEED)

    indices = rng.choice(
        len(X_train_scaled),
        size=background_size,
        replace=False,
    )

    background = X_train_scaled[indices]

    try:
        explainer = shap.LinearExplainer(
            estimator,
            background,
        )

        explainer_name = "LinearExplainer"

    except Exception as exc:
        print(
            "LinearExplainer failed; trying SHAP auto explainer..."
        )
        print(f"Reason: {exc}")

        explainer = shap.Explainer(
            estimator,
            background,
        )

        explainer_name = "shap.Explainer"

    return explainer, explainer_name


def calculate_importance(shap_values):
    """
    Global mean absolute SHAP importance.
    """

    return np.mean(
        np.abs(shap_values),
        axis=(0, 2),
    )


def save_importance_csv(
    path,
    feature_names,
    values,
    extra_columns=None,
):
    rows = []

    for index, feature in enumerate(feature_names):
        row = {
            "feature": feature,
            "mean_absolute_shap": float(values[index]),
        }

        if extra_columns:
            for key, column in extra_columns.items():
                row[key] = column[index]

        rows.append(row)

    rows.sort(
        key=lambda item: item["mean_absolute_shap"],
        reverse=True,
    )

    import csv

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_bar_chart(
    path,
    feature_names,
    values,
    title,
):
    order = np.argsort(values)

    plt.figure(figsize=(10, 6))

    plt.barh(
        np.array(feature_names)[order],
        np.array(values)[order],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def subgroup_masks(
    X_test,
    predictions,
    y_test,
):
    sarcasm = X_test[:, 3]
    emoji_count = X_test[:, 6]

    error = predictions != y_test

    return {
        "all_test": np.ones(len(X_test), dtype=bool),

        "emoji_present": (
            emoji_count > 0
        ),

        "high_sarcasm": (
            sarcasm >= SARCASM_THRESHOLD
        ),

        "high_sarcasm_errors": (
            (sarcasm >= SARCASM_THRESHOLD)
            & error
        ),

        "emoji_present_errors": (
            (emoji_count > 0)
            & error
        ),
    }


def explain_subset(
    shap_values,
    mask,
):
    selected = shap_values[mask]

    if len(selected) == 0:
        return None

    return calculate_importance(selected)


def run_quick_test(
    explainer,
    X_test_scaled,
    X_test,
    y_test,
    predictions,
    shap_class_names,
    feature_names,
):
    print()
    print("=" * 60)
    print("SHORT SHAP TEST")
    print("=" * 60)

    sarcasm = X_test[:, 3]
    emoji_count = X_test[:, 6]
    errors = predictions != y_test

    candidates = []

    # 1. Emoji example
    emoji_indices = np.where(
        emoji_count > 0
    )[0]

    if len(emoji_indices):
        candidates.append(
            ("emoji_present", int(emoji_indices[0]))
        )

    # 2. High sarcasm error
    sarcasm_error_indices = np.where(
        (sarcasm >= SARCASM_THRESHOLD)
        & errors
    )[0]

    if len(sarcasm_error_indices):
        candidates.append(
            (
                "high_sarcasm_error",
                int(sarcasm_error_indices[0]),
            )
        )

    # 3. Normal example
    normal_indices = np.where(
        (sarcasm < 0.20)
        & (emoji_count == 0)
    )[0]

    if len(normal_indices):
        candidates.append(
            ("normal", int(normal_indices[0]))
        )

    # Deduplicate rows
    seen = set()
    unique_candidates = []

    for label, index in candidates:
        if index not in seen:
            seen.add(index)
            unique_candidates.append(
                (label, index)
            )

    if not unique_candidates:
        raise RuntimeError(
            "Could not find suitable quick-test rows."
        )

    selected_indices = [
        index
        for _, index in unique_candidates
    ]

    quick_scaled = X_test_scaled[selected_indices]

    raw_values = explainer(
        quick_scaled
    )

    values = normalize_shap_values(
        raw_values.values
        if hasattr(raw_values, "values")
        else raw_values,
        len(selected_indices),
        X_test.shape[1],
        len(shap_class_names),
    )

    assert np.isfinite(values).all(), (
        "SHAP contains NaN or infinite values."
    )

    for position, (case, index) in enumerate(
        unique_candidates
    ):
        predicted_class = int(predictions[index])
        true_class = int(y_test[index])

        class_index = min(
            predicted_class,
            values.shape[2] - 1,
        )

        local_values = values[
            position,
            :,
            class_index,
        ]

        order = np.argsort(
            np.abs(local_values)
        )[::-1]

        print()
        print(f"CASE: {case}")
        print(f"ROW: {index}")
        print(
            "TRUE:",
            shap_class_names[true_class]
            if true_class < len(shap_class_names)
            else true_class,
        )
        print(
            "PRED:",
            shap_class_names[predicted_class]
            if predicted_class < len(shap_class_names)
            else predicted_class,
        )
        print(
            "SARCASM PROB:",
            f"{X_test[index, 3]:.4f}",
        )
        print(
            "EMOJI COUNT:",
            f"{X_test[index, 6]:.0f}",
        )

        print("TOP SHAP FEATURES:")

        for feature_index in order[:3]:
            print(
                f"  {feature_names[feature_index]}: "
                f"{local_values[feature_index]:+.6f}"
            )

    print()
    print("SHAP QUICK TEST PASSED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small SHAP sanity test only.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FUSION-V2 ROBUST SHAP ANALYSIS")
    print("=" * 60)

    print()
    print("SHAP version:", shap.__version__)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    required_files = [
        MODEL_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    metadata = load_json(METADATA_PATH)

    feature_names = get_feature_names(metadata)

    X_train = np.load(X_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    X_train = np.asarray(
        X_train,
        dtype=np.float64,
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float64,
    )

    y_test = np.asarray(
        y_test,
        dtype=int,
    )

    if X_train.ndim != 2:
        raise ValueError(
            f"X_train must be 2D: {X_train.shape}"
        )

    if X_test.ndim != 2:
        raise ValueError(
            f"X_test must be 2D: {X_test.shape}"
        )

    if X_test.shape[1] != len(feature_names):
        raise ValueError(
            "Feature-name mismatch:\n"
            f"X_test has {X_test.shape[1]} features\n"
            f"feature_names has {len(feature_names)}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test feature/label size mismatch."
        )

    print()
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Features:", len(feature_names))

    print()
    print("Feature order:")

    for index, feature in enumerate(feature_names):
        print(f"  {index}: {feature}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    estimator, transform_features = (
        load_model_and_preprocessor()
    )

    class_names = get_class_names(
        metadata,
        estimator,
    )

    if hasattr(estimator, "classes_"):
        classes = np.asarray(
            estimator.classes_
        )
        n_classes = len(classes)
    else:
        sample_scaled = transform_features(
            X_test[:1]
        )
        probabilities = estimator.predict_proba(
            sample_scaled
        )
        n_classes = probabilities.shape[1]

    X_train_scaled = transform_features(
        X_train
    )

    X_test_scaled = transform_features(
        X_test
    )

    probabilities = estimator.predict_proba(
        X_test_scaled
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    if hasattr(estimator, "classes_"):
        predictions = np.asarray(
            estimator.classes_
        )[predictions]

    predictions = predictions.astype(int)

    print()
    print("Classes:", class_names)
    print(
        "Test accuracy:",
        f"{np.mean(predictions == y_test):.6f}",
    )

    # --------------------------------------------------------
    # SHAP explainer
    # --------------------------------------------------------

    explainer, explainer_name = create_explainer(
        estimator,
        X_train_scaled,
    )

    print()
    print("Explainer:", explainer_name)
    print(
        "Background size:",
        min(BACKGROUND_SIZE, len(X_train_scaled)),
    )

    # --------------------------------------------------------
    # Quick mode
    # --------------------------------------------------------

    if args.quick:
        run_quick_test(
            explainer,
            X_test_scaled,
            X_test,
            y_test,
            predictions,
            class_names,
            feature_names,
        )
        return

    # --------------------------------------------------------
    # Full test-set explanation
    # --------------------------------------------------------

    print()
    print(
        f"Explaining full test set: "
        f"{len(X_test)} rows..."
    )

    raw_explanation = explainer(
        X_test_scaled
    )

    raw_values = (
        raw_explanation.values
        if hasattr(raw_explanation, "values")
        else raw_explanation
    )

    shap_values = normalize_shap_values(
        raw_values,
        len(X_test),
        X_test.shape[1],
        n_classes,
    )

    if not np.isfinite(shap_values).all():
        raise ValueError(
            "SHAP contains NaN or infinite values."
        )

    print(
        "SHAP matrix:",
        shap_values.shape,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Global importance
    # --------------------------------------------------------

    global_importance = calculate_importance(
        shap_values
    )

    save_importance_csv(
        OUTPUT_DIR / "global_feature_importance.csv",
        feature_names,
        global_importance,
    )

    save_bar_chart(
        OUTPUT_DIR / "global_feature_importance.png",
        feature_names,
        global_importance,
        "Fusion-v2 — Global SHAP Feature Importance",
    )

    # --------------------------------------------------------
    # Class-specific importance
    # --------------------------------------------------------

    class_importance = {}

    for class_index, class_name in enumerate(
        class_names[:shap_values.shape[2]]
    ):
        values = np.mean(
            np.abs(
                shap_values[:, :, class_index]
            ),
            axis=0,
        )

        class_importance[class_name] = values.tolist()

        save_importance_csv(
            OUTPUT_DIR
            / f"class_{class_name}_importance.csv",
            feature_names,
            values,
        )

        save_bar_chart(
            OUTPUT_DIR
            / f"class_{class_name}_importance.png",
            feature_names,
            values,
            (
                "Fusion-v2 — SHAP Importance — "
                f"{class_name.title()} Class"
            ),
        )

    # --------------------------------------------------------
    # Subgroup analysis
    # --------------------------------------------------------

    masks = subgroup_masks(
        X_test,
        predictions,
        y_test,
    )

    subgroup_results = {}

    for subgroup_name, mask in masks.items():
        count = int(mask.sum())

        print()
        print(
            f"SUBGROUP: {subgroup_name} "
            f"({count} rows)"
        )

        if count == 0:
            subgroup_results[subgroup_name] = {
                "count": 0,
                "importance": None,
            }
            continue

        importance = explain_subset(
            shap_values,
            mask,
        )

        subgroup_results[subgroup_name] = {
            "count": count,
            "importance": {
                feature_names[index]: float(
                    importance[index]
                )
                for index in range(
                    len(feature_names)
                )
            },
        }

        save_importance_csv(
            OUTPUT_DIR
            / f"{subgroup_name}_importance.csv",
            feature_names,
            importance,
        )

    # --------------------------------------------------------
    # Local examples
    # --------------------------------------------------------

    high_sarcasm_errors = np.where(
        (X_test[:, 3] >= SARCASM_THRESHOLD)
        & (predictions != y_test)
    )[0]

    emoji_errors = np.where(
        (X_test[:, 6] > 0)
        & (predictions != y_test)
    )[0]

    selected_local = []

    for index in (
        list(high_sarcasm_errors[:2])
        + list(emoji_errors[:2])
        + [0]
    ):
        index = int(index)

        if index not in selected_local:
            selected_local.append(index)

    local_rows = []

    for index in selected_local:
        predicted_class = int(
            predictions[index]
        )

        class_index = min(
            predicted_class,
            shap_values.shape[2] - 1,
        )

        local_values = shap_values[
            index,
            :,
            class_index,
        ]

        order = np.argsort(
            np.abs(local_values)
        )[::-1]

        row = {
            "test_index": index,
            "true_label": int(y_test[index]),
            "predicted_label": predicted_class,
            "sarcasm_probability": float(
                X_test[index, 3]
            ),
            "emoji_count": float(
                X_test[index, 6]
            ),
            "predicted_class_name": (
                class_names[predicted_class]
                if predicted_class < len(class_names)
                else str(predicted_class)
            ),
        }

        for rank, feature_index in enumerate(
            order,
            start=1,
        ):
            row[
                f"rank_{rank}_feature"
            ] = feature_names[feature_index]

            row[
                f"rank_{rank}_shap"
            ] = float(
                local_values[feature_index]
            )

        local_rows.append(row)

    if local_rows:
        import csv

        local_path = (
            OUTPUT_DIR
            / "local_examples.csv"
        )

        fieldnames = list(
            local_rows[0].keys()
        )

        with local_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(local_rows)

    # --------------------------------------------------------
    # Metadata summary
    # --------------------------------------------------------

    summary = {
        "analysis": "Fusion-v2 robust SHAP",
        "explainer": explainer_name,
        "shap_version": shap.__version__,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "class_names": class_names,
        "background_size": int(
            min(
                BACKGROUND_SIZE,
                len(X_train_scaled),
            )
        ),
        "sarcasm_threshold": SARCASM_THRESHOLD,
        "test_accuracy_recomputed": float(
            np.mean(
                predictions == y_test
            )
        ),
        "emoji_present_rows": int(
            np.sum(X_test[:, 6] > 0)
        ),
        "high_sarcasm_rows": int(
            np.sum(
                X_test[:, 3]
                >= SARCASM_THRESHOLD
            )
        ),
        "high_sarcasm_error_rows": int(
            np.sum(
                (X_test[:, 3] >= SARCASM_THRESHOLD)
                & (predictions != y_test)
            )
        ),
        "global_importance": {
            feature_names[index]: float(
                global_importance[index]
            )
            for index in range(
                len(feature_names)
            )
        },
        "subgroups": subgroup_results,
        "shap_matrix_shape": list(
            shap_values.shape
        ),
        "explainer_input": (
            "Fusion-v2 trained preprocessing "
            "output (StandardScaler when present)"
        ),
    }

    with (
        OUTPUT_DIR / "shap_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("=" * 60)
    print("ROBUST SHAP ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print("Top global features:")

    ranking = np.argsort(
        global_importance
    )[::-1]

    for rank, feature_index in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{feature_names[feature_index]} "
            f"= "
            f"{global_importance[feature_index]:.6f}"
        )

    print()
    print("Outputs:")
    print(OUTPUT_DIR)

    print()
    print(
        "Run the short validation next with:"
    )
    print(
        "python scripts/run_shap_analysis.py --quick"
    )


if __name__ == "__main__":
    main()
