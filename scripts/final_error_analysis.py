from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]

V2_DIR = ROOT / "models" / "fusion-v2"
OUT_DIR = ROOT / "outputs"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


ERROR_CATEGORY_MAP = {
    (1, 0): "neutral_to_negative",
    (1, 2): "neutral_to_positive",
    (0, 1): "negative_to_neutral",
    (2, 1): "positive_to_neutral",
    (0, 2): "negative_to_positive",
    (2, 0): "positive_to_negative",
}


# ============================================================
# 1. CHECK REQUIRED FUSION-V2 ARTIFACTS
# ============================================================

prediction_path = (
    V2_DIR / "test_predictions.csv"
)

sarcasm_path = (
    V2_DIR / "sarcasm_test.npy"
)

if not prediction_path.exists():
    raise FileNotFoundError(
        f"Missing Fusion-v2 predictions:\n"
        f"{prediction_path}"
    )

if not sarcasm_path.exists():
    raise FileNotFoundError(
        f"Missing Fusion-v2 sarcasm probabilities:\n"
        f"{sarcasm_path}"
    )


# ============================================================
# 2. LOAD TWEETEVAL TEST TEXT
# ============================================================

print("=" * 80)
print("FINAL SENTIMENT ERROR ANALYSIS")
print("=" * 80)

print("\nLoading TweetEval test split...")

dataset = load_dataset(
    "cardiffnlp/tweet_eval",
    "sentiment",
)

texts = list(
    dataset["test"]["text"]
)

print(
    "TweetEval test examples:",
    len(texts),
)


# ============================================================
# 3. LOAD FUSION-V2 PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    prediction_path
)

sarcasm_probabilities = np.load(
    sarcasm_path
)

if len(predictions) != len(texts):
    raise ValueError(
        "Prediction count does not match "
        "TweetEval test count."
    )

if len(sarcasm_probabilities) != len(texts):
    raise ValueError(
        "Sarcasm probability count does not "
        "match TweetEval test count."
    )


# Attach text and sarcasm probability
predictions["text"] = texts

predictions[
    "sarcasm_probability"
] = sarcasm_probabilities

predictions[
    "sarcasm_detected"
] = (
    predictions["sarcasm_probability"]
    >= 0.360
)


# Convert labels to readable names
predictions[
    "true_label_name"
] = predictions[
    "y_true"
].map(LABEL_MAP)

predictions[
    "pred_label_name"
] = predictions[
    "y_pred"
].map(LABEL_MAP)


# ============================================================
# 4. BASIC ERROR STATISTICS
# ============================================================

predictions["correct"] = (
    predictions["y_true"]
    == predictions["y_pred"]
)

errors = predictions[
    ~predictions["correct"]
].copy()

print(
    "\nTotal test examples:",
    len(predictions),
)

print(
    "Correct:",
    int(predictions["correct"].sum()),
)

print(
    "Errors:",
    len(errors),
)

print(
    "Error rate:",
    f"{len(errors) / len(predictions) * 100:.2f}%"
)


# ============================================================
# 5. ERROR CATEGORIES
# ============================================================

def classify_error(row):
    return ERROR_CATEGORY_MAP.get(
        (
            int(row["y_true"]),
            int(row["y_pred"]),
        ),
        "other",
    )


errors[
    "error_category"
] = errors.apply(
    classify_error,
    axis=1,
)

category_counts = (
    errors[
        "error_category"
    ]
    .value_counts()
)


print("\n=== ERROR CATEGORIES ===")

for category, count in category_counts.items():
    print(
        f"{category}: {count}"
    )


# ============================================================
# 6. SARCASM SIGNAL INSIDE SENTIMENT ERRORS
# ============================================================

print(
    "\n=== SARCASM SIGNAL IN SENTIMENT ERRORS ==="
)

high_sarcasm_errors = errors[
    errors["sarcasm_probability"]
    >= 0.360
]

print(
    "Errors with sarcasm >= 36%:",
    len(high_sarcasm_errors),
)

print(
    "Errors with sarcasm >= 50%:",
    int(
        (
            errors["sarcasm_probability"]
            >= 0.50
        ).sum()
    ),
)

print(
    "Errors with sarcasm >= 80%:",
    int(
        (
            errors["sarcasm_probability"]
            >= 0.80
        ).sum()
    ),
)

print(
    "Mean sarcasm probability among errors:",
    f"{errors['sarcasm_probability'].mean() * 100:.2f}%"
)


# ============================================================
# 7. SARCASM-RICH SENTIMENT ERRORS
# ============================================================

sarcasm_rich = errors[
    errors["sarcasm_probability"]
    >= 0.360
].copy()

sarcasm_rich = sarcasm_rich.sort_values(
    "sarcasm_probability",
    ascending=False,
)

print(
    "\n=== TOP SARCASM-RICH SENTIMENT ERRORS ==="
)

for _, row in sarcasm_rich.head(20).iterrows():

    print("\nTEXT:")
    print(row["text"])

    print(
        "TRUE:",
        row["true_label_name"],
    )

    print(
        "PRED:",
        row["pred_label_name"],
    )

    print(
        "SARCASM:",
        f"{row['sarcasm_probability'] * 100:.2f}%"
    )


# ============================================================
# 8. SAVE ALL SENTIMENT ERRORS
# ============================================================

errors_path = (
    OUT_DIR
    / "fusion_v2_sentiment_errors.csv"
)

errors.to_csv(
    errors_path,
    index=False,
)

print(
    "\nSaved sentiment errors:",
    errors_path,
)


# ============================================================
# 9. SAVE ERROR EXAMPLES BY CATEGORY
# ============================================================

examples = {}

for category in category_counts.index:

    subset = errors[
        errors["error_category"]
        == category
    ].copy()

    examples[category] = (
        subset[
            [
                "text",
                "true_label_name",
                "pred_label_name",
                "sarcasm_probability",
                "sarcasm_detected",
            ]
        ]
        .head(20)
        .fillna("")
        .to_dict(
            orient="records"
        )
    )


examples_path = (
    OUT_DIR
    / "fusion_v2_error_examples.json"
)

examples_path.write_text(
    json.dumps(
        examples,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    "Saved error examples:",
    examples_path,
)


# ============================================================
# 10. SARCASM MODEL ERROR ANALYSIS
# ============================================================

print()
print("=" * 80)
print("SARCASM MODEL ERROR ANALYSIS")
print("=" * 80)

sarcasm_prediction_path = (
    OUT_DIR
    / "twitter_sarcasm_test_predictions.json"
)

if sarcasm_prediction_path.exists():

    sarcasm_rows = json.loads(
        sarcasm_prediction_path.read_text(
            encoding="utf-8"
        )
    )

    sarcasm_df = pd.DataFrame(
        sarcasm_rows
    )

    sarcasm_errors = sarcasm_df[
        ~sarcasm_df["correct"]
    ].copy()

    false_positives = sarcasm_df[
        (
            sarcasm_df["true_label"]
            == 0
        )
        &
        (
            sarcasm_df["predicted_label"]
            == 1
        )
    ].copy()

    false_negatives = sarcasm_df[
        (
            sarcasm_df["true_label"]
            == 1
        )
        &
        (
            sarcasm_df["predicted_label"]
            == 0
        )
    ].copy()

    print(
        "Sarcasm test examples:",
        len(sarcasm_df),
    )

    print(
        "Sarcasm errors:",
        len(sarcasm_errors),
    )

    print(
        "False positives:",
        len(false_positives),
    )

    print(
        "False negatives:",
        len(false_negatives),
    )

    fp_path = (
        OUT_DIR
        / "sarcasm_false_positives.csv"
    )

    fn_path = (
        OUT_DIR
        / "sarcasm_false_negatives.csv"
    )

    false_positives.to_csv(
        fp_path,
        index=False,
    )

    false_negatives.to_csv(
        fn_path,
        index=False,
    )

    print(
        "\nSaved sarcasm false positives:",
        fp_path,
    )

    print(
        "Saved sarcasm false negatives:",
        fn_path,
    )

    print(
        "\nTop sarcasm false negatives:"
    )

    print(
        false_negatives[
            [
                "text",
                "sarcasm_probability",
            ]
        ]
        .sort_values(
            "sarcasm_probability"
        )
        .head(15)
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nIndependent sarcasm prediction file "
        "not found; skipping that section."
    )


# ============================================================
# 11. SAVE MACHINE-READABLE SUMMARY
# ============================================================

summary = {
    "sentiment_test_examples":
        int(len(predictions)),

    "sentiment_errors":
        int(len(errors)),

    "sentiment_error_rate":
        float(
            len(errors)
            / len(predictions)
        ),

    "error_categories":
        {
            str(category): int(count)
            for category, count
            in category_counts.items()
        },

    "errors_with_sarcasm_ge_36_percent":
        int(len(high_sarcasm_errors)),

    "errors_with_sarcasm_ge_50_percent":
        int(
            (
                errors["sarcasm_probability"]
                >= 0.50
            ).sum()
        ),

    "errors_with_sarcasm_ge_80_percent":
        int(
            (
                errors["sarcasm_probability"]
                >= 0.80
            ).sum()
        ),

    "mean_sarcasm_probability_errors":
        float(
            errors[
                "sarcasm_probability"
            ].mean()
        ),

    "source_predictions":
        str(prediction_path),

    "source_sarcasm_probabilities":
        str(sarcasm_path),
}


summary_path = (
    OUT_DIR
    / "fusion_v2_error_analysis.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "\nSaved summary:",
    summary_path,
)

print()
print("=" * 80)
print("FINAL ERROR ANALYSIS COMPLETE")
print("=" * 80)
