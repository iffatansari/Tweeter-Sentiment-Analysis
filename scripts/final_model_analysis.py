from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "models" / "fusion-v2"
OUT = ROOT / "outputs"

OUT.mkdir(parents=True, exist_ok=True)


FEATURE_NAMES = [
    "sentiment_negative_probability",
    "sentiment_neutral_probability",
    "sentiment_positive_probability",
    "sarcasm_probability",
    "emoji_positive",
    "emoji_negative",
    "emoji_count",
    "emoji_present",
]


VARIANTS = {
    "twitter_roberta_direct": [0, 1, 2],
    "sentiment_only_meta": [0, 1, 2],
    "sentiment_plus_sarcasm": [0, 1, 2, 3],
    "sentiment_plus_emoji": [0, 1, 2, 4, 5, 6, 7],
    "sentiment_sarcasm_emoji": list(range(8)),
}


def load():
    return (
        np.load(V2 / "X_train.npy"),
        np.load(V2 / "X_validation.npy"),
        np.load(V2 / "X_test.npy"),
        np.load(V2 / "y_train.npy"),
        np.load(V2 / "y_validation.npy"),
        np.load(V2 / "y_test.npy"),
    )


def build_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])


def metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def main():
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load()

    print("Loaded Fusion-v2 matrices:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # ------------------------------------------------------
    # Emoji availability
    # ------------------------------------------------------

    print("\n=== EMOJI DATA CHECK ===")

    emoji_indices = [4, 5, 6, 7]

    emoji_nonzero = {}

    for index in emoji_indices:
        count = int(
            np.count_nonzero(
                X_test[:, index]
            )
        )

        name = FEATURE_NAMES[index]
        emoji_nonzero[name] = count

        print(
            f"{name}: {count} "
            "non-zero test values"
        )

    # ------------------------------------------------------
    # Ablation
    # ------------------------------------------------------

    all_results = []
    test_predictions = {}

    for variant, columns in VARIANTS.items():

        print("\n" + "=" * 80)
        print("VARIANT:", variant)
        print(
            "FEATURES:",
            [
                FEATURE_NAMES[i]
                for i in columns
            ],
        )
        print("=" * 80)

        if variant == "twitter_roberta_direct":
            val_pred = np.argmax(
                X_val[:, :3],
                axis=1,
            )

            test_pred = np.argmax(
                X_test[:, :3],
                axis=1,
            )

        else:
            model = build_model()

            model.fit(
                X_train[:, columns],
                y_train,
            )

            val_pred = model.predict(
                X_val[:, columns]
            )

            test_pred = model.predict(
                X_test[:, columns]
            )

        val_metrics = metrics(
            y_val,
            val_pred,
        )

        test_metrics = metrics(
            y_test,
            test_pred,
        )

        print(
            "Validation:",
            val_metrics,
        )

        print(
            "Test:",
            test_metrics,
        )

        print("\nTest classification report:")
        print(
            classification_report(
                y_test,
                test_pred,
                target_names=[
                    "negative",
                    "neutral",
                    "positive",
                ],
                digits=4,
                zero_division=0,
            )
        )

        print("Test confusion matrix:")
        print(
            confusion_matrix(
                y_test,
                test_pred,
            )
        )

        all_results.append({
            "variant": variant,
            "validation_accuracy":
                val_metrics["accuracy"],
            "validation_macro_f1":
                val_metrics["macro_f1"],
            "validation_weighted_f1":
                val_metrics["weighted_f1"],
            "test_accuracy":
                test_metrics["accuracy"],
            "test_macro_f1":
                test_metrics["macro_f1"],
            "test_weighted_f1":
                test_metrics["weighted_f1"],
        })

        test_predictions[variant] = test_pred

    # ------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------

    df = pd.DataFrame(
        all_results
    )

    df = df.sort_values(
        "test_macro_f1",
        ascending=False,
    )

    print("\n")
    print("=" * 100)
    print("FINAL ABLATION TABLE")
    print("=" * 100)

    print(
        df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    df.to_csv(
        OUT / "fusion_v2_ablation.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Best variant
    # ------------------------------------------------------

    best_variant = df.iloc[0]["variant"]

    print(
        "\nBest test Macro-F1 variant:",
        best_variant,
    )

    # ------------------------------------------------------
    # Error analysis for final full fusion model
    # ------------------------------------------------------

    final_variant = (
        "sentiment_sarcasm_emoji"
    )

    final_pred = test_predictions[
        final_variant
    ]

    errors = y_test != final_pred

    print("\n")
    print("=" * 80)
    print("FINAL MODEL ERROR ANALYSIS")
    print("=" * 80)

    print(
        "Total test examples:",
        len(y_test),
    )

    print(
        "Correct:",
        int((~errors).sum()),
    )

    print(
        "Errors:",
        int(errors.sum()),
    )

    print(
        "Error rate:",
        float(errors.mean()),
    )

    error_rows = pd.DataFrame({
        "row_index":
            np.arange(len(y_test)),
        "y_true":
            y_test,
        "y_pred":
            final_pred,
        "correct":
            ~errors,
    })

    error_rows = error_rows[
        error_rows["correct"] == False
    ].copy()

    error_rows.to_csv(
        OUT / "fusion_v2_test_errors.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Confusion-category counts
    # ------------------------------------------------------

    category_counts = {}

    for true_label, pred_label in zip(
        y_test[errors],
        final_pred[errors],
    ):
        key = (
            f"{int(true_label)}_to_"
            f"{int(pred_label)}"
        )

        category_counts[key] = (
            category_counts.get(
                key,
                0,
            )
            + 1
        )

    print("\nError categories:")

    for key, count in sorted(
        category_counts.items(),
        key=lambda item:
            item[1],
        reverse=True,
    ):
        print(
            f"{key}: {count}"
        )

    # ------------------------------------------------------
    # Full machine-readable report
    # ------------------------------------------------------

    report = {
        "best_variant":
            best_variant,
        "ablation":
            all_results,
        "emoji_nonzero_test":
            emoji_nonzero,
        "final_error_count":
            int(errors.sum()),
        "final_error_rate":
            float(errors.mean()),
        "error_categories":
            category_counts,
    }

    with (
        OUT / "fusion_v2_analysis.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        "\nSaved:"
    )

    print(
        OUT / "fusion_v2_ablation.csv"
    )

    print(
        OUT / "fusion_v2_test_errors.csv"
    )

    print(
        OUT / "fusion_v2_analysis.json"
    )


if __name__ == "__main__":
    main()
