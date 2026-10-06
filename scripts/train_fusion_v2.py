from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V1_DIR = ROOT / "models" / "fusion"
V2_DIR = ROOT / "models" / "fusion-v2"

V2_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SARCASM_MODEL_DIR = (
    ROOT / "models" / "sarcasm-deberta-v3"
)

SARCASM_MODEL_NAME = (
    "ppokhrel2109/besstie-sarcasm-deberta-v3"
)

TEMPERATURE = 1.1819
SARCASM_CLASS_INDEX = 1
DECISION_THRESHOLD = 0.360

MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16

RANDOM_STATE = 42

# Exact Fusion-v1 feature order, taken from the
# actual train_fusion.py you supplied.
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

# Column indices in the existing Fusion-v1 matrices.
RO_BERTA_COLUMNS = [0, 1, 2]
OLD_SARCASM_COLUMN = 3
EMOJI_COLUMNS = [4, 5, 6, 7]


# ============================================================
# DATA LOADING
# ============================================================

def load_tweeteval():
    print("Loading TweetEval sentiment dataset...")

    return load_dataset(
        "cardiffnlp/tweet_eval",
        "sentiment",
    )


# ============================================================
# EXISTING FUSION-V1 FEATURE VALIDATION
# ============================================================

def load_existing_v1_features():
    required = [
        "X_train.npy",
        "X_validation.npy",
        "X_test.npy",
        "y_train.npy",
        "y_validation.npy",
        "y_test.npy",
    ]

    missing = [
        name
        for name in required
        if not (V1_DIR / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Fusion-v1 feature files missing:\n"
            + "\n".join(missing)
        )

    X_train = np.load(
        V1_DIR / "X_train.npy"
    )

    X_validation = np.load(
        V1_DIR / "X_validation.npy"
    )

    X_test = np.load(
        V1_DIR / "X_test.npy"
    )

    y_train = np.load(
        V1_DIR / "y_train.npy"
    )

    y_validation = np.load(
        V1_DIR / "y_validation.npy"
    )

    y_test = np.load(
        V1_DIR / "y_test.npy"
    )

    for name, matrix in [
        ("X_train", X_train),
        ("X_validation", X_validation),
        ("X_test", X_test),
    ]:
        if matrix.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D."
            )

        if matrix.shape[1] != 8:
            raise ValueError(
                f"{name} expected 8 features, "
                f"got {matrix.shape[1]}."
            )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "Train feature/label length mismatch."
        )

    if (
        X_validation.shape[0]
        != y_validation.shape[0]
    ):
        raise ValueError(
            "Validation feature/label length mismatch."
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            "Test feature/label length mismatch."
        )

    print()
    print("Existing Fusion-v1 matrices:")
    print("  Train:", X_train.shape)
    print("  Validation:", X_validation.shape)
    print("  Test:", X_test.shape)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# DEBERTA LOADING
# ============================================================

def load_sarcasm_model():
    if not SARCASM_MODEL_DIR.exists():
        raise FileNotFoundError(
            "DeBERTa sarcasm model directory not found:\n"
            f"{SARCASM_MODEL_DIR}"
        )

    weights_path = (
        SARCASM_MODEL_DIR
        / "model.safetensors"
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            "Expected SafeTensors weights not found:\n"
            f"{weights_path}"
        )

    print()
    print("Loading DeBERTa sarcasm model...")
    print(
        f"Model: {SARCASM_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(SARCASM_MODEL_DIR),
        local_files_only=True,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(SARCASM_MODEL_DIR),
            local_files_only=True,
        )
    )

    model.eval()
    model.to("cpu")

    id2label = getattr(
        model.config,
        "id2label",
        {},
    ) or {}

    print(
        "Labels:",
        id2label,
    )

    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    if len(id2label) != 2:
        raise RuntimeError(
            "Expected a binary sarcasm classifier."
        )

    if (
        SARCASM_CLASS_INDEX
        not in id2label
    ):
        raise RuntimeError(
            "Sarcasm class index is missing."
        )

    return tokenizer, model


# ============================================================
# BATCHED SARCASM INFERENCE
# ============================================================

def generate_sarcasm_probabilities(
    split_name: str,
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
):
    """
    Generate calibrated sarcasm probabilities.

    This function is resumable.

    The partial probability array is stored as
    a float32 memmap and a small JSON checkpoint records
    the next text index to process.
    """

    final_path = (
        V2_DIR
        / f"sarcasm_{split_name}.npy"
    )

    partial_path = (
        V2_DIR
        / f"sarcasm_{split_name}.partial.dat"
    )

    state_path = (
        V2_DIR
        / f"sarcasm_{split_name}.state.json"
    )

    total = len(texts)

    # --------------------------------------------------------
    # Already complete?
    # --------------------------------------------------------

    if final_path.exists():
        probabilities = np.load(
            final_path
        )

        if probabilities.shape != (
            total,
        ):
            raise ValueError(
                f"Existing {final_path} has "
                f"shape {probabilities.shape}; "
                f"expected {(total,)}."
            )

        print(
            f"{split_name}: existing "
            "DeBERTa probabilities found; "
            "reusing them."
        )

        return probabilities

    # --------------------------------------------------------
    # Resume or initialize checkpoint
    # --------------------------------------------------------

    start_index = 0

    if (
        partial_path.exists()
        and state_path.exists()
    ):
        try:
            state = json.loads(
                state_path.read_text(
                    encoding="utf-8"
                )
            )

            start_index = int(
                state.get(
                    "next_index",
                    0,
                )
            )

            if (
                start_index < 0
                or start_index > total
            ):
                start_index = 0

        except Exception:
            start_index = 0

    if partial_path.exists():
        expected_size = (
            total
            * np.dtype(
                np.float32
            ).itemsize
        )

        actual_size = (
            partial_path.stat().st_size
        )

        if actual_size != expected_size:
            partial_path.unlink(
                missing_ok=True
            )
            state_path.unlink(
                missing_ok=True
            )
            start_index = 0

    if not partial_path.exists():
        mmap = np.memmap(
            partial_path,
            dtype=np.float32,
            mode="w+",
            shape=(total,),
        )

        mmap[:] = 0.0
        mmap.flush()

    else:
        mmap = np.memmap(
            partial_path,
            dtype=np.float32,
            mode="r+",
            shape=(total,),
        )

    print()
    print(
        f"Building DeBERTa sarcasm features "
        f"for {split_name}..."
    )

    if start_index:
        print(
            f"Resuming from "
            f"{start_index}/{total}"
        )

    batch_number = (
        start_index // batch_size
    )

    with torch.inference_mode():

        for batch_start in range(
            start_index,
            total,
            batch_size,
        ):
            batch_end = min(
                batch_start + batch_size,
                total,
            )

            batch_texts = [
                str(text)
                for text in texts[
                    batch_start:batch_end
                ]
            ]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )

            logits = model(
                **encoded
            ).logits

            calibrated_logits = (
                logits / TEMPERATURE
            )

            probabilities = torch.softmax(
                calibrated_logits,
                dim=-1,
            )

            sarcasm_probs = (
                probabilities[
                    :,
                    SARCASM_CLASS_INDEX,
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            mmap[
                batch_start:batch_end
            ] = sarcasm_probs

            mmap.flush()

            batch_number += 1

            if (
                batch_number % 10 == 0
                or batch_end == total
            ):
                print(
                    f"Processed "
                    f"{batch_end}/{total} texts",
                    flush=True,
                )

            # Update checkpoint AFTER writing
            # and flushing the current batch.
            state_path.write_text(
                json.dumps(
                    {
                        "split":
                            split_name,
                        "next_index":
                            batch_end,
                        "total":
                            total,
                        "batch_size":
                            batch_size,
                        "temperature":
                            TEMPERATURE,
                        "sarcasm_class_index":
                            SARCASM_CLASS_INDEX,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    probabilities = np.asarray(
        mmap
    ).copy()

    del mmap

    np.save(
        final_path,
        probabilities,
    )

    state_path.unlink(
        missing_ok=True
    )

    partial_path.unlink(
        missing_ok=True
    )

    print(
        f"Saved: {final_path}"
    )

    return probabilities


# ============================================================
# BUILD V2 MATRICES
# ============================================================

def build_v2_matrix(
    base_matrix: np.ndarray,
    sarcasm_probabilities: np.ndarray,
):
    if base_matrix.shape[0] != (
        sarcasm_probabilities.shape[0]
    ):
        raise ValueError(
            "Feature rows and sarcasm "
            "probabilities do not match."
        )

    X_v2 = base_matrix.copy()

    # Only replace OLD sarcasm feature.
    X_v2[
        :,
        OLD_SARCASM_COLUMN,
    ] = sarcasm_probabilities

    return X_v2.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# FUSION MODEL
# ============================================================

def build_fusion_model():
    """
    Same meta-classifier configuration as Fusion-v1.

    Keeping this fixed lets us attribute differences
    primarily to replacing the sarcasm component.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
):
    predictions = model.predict(X)

    probabilities = None

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(X)
        )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro-F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    print()
    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "negative",
                "neutral",
                "positive",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    cm = confusion_matrix(
        y,
        predictions,
    )

    print(cm)

    return {
        "accuracy":
            float(accuracy),
        "macro_f1":
            float(macro_f1),
        "weighted_f1":
            float(weighted_f1),
        "classification_report":
            classification_report(
                y,
                predictions,
                target_names=[
                    "negative",
                    "neutral",
                    "positive",
                ],
                digits=4,
                zero_division=0,
                output_dict=True,
            ),
        "confusion_matrix":
            cm.tolist(),
        "predictions":
            predictions,
        "probabilities":
            probabilities,
    }


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

def save_test_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
):
    output_path = (
        V2_DIR
        / "test_predictions.csv"
    )

    import csv

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(handle)

        header = [
            "row_index",
            "y_true",
            "y_pred",
            "negative_probability",
            "neutral_probability",
            "positive_probability",
            "correct",
        ]

        writer.writerow(header)

        for index in range(
            len(y_true)
        ):
            if (
                probabilities is not None
                and probabilities.shape[1] >= 3
            ):
                negative_probability = (
                    float(
                        probabilities[
                            index,
                            0,
                        ]
                    )
                )

                neutral_probability = (
                    float(
                        probabilities[
                            index,
                            1,
                        ]
                    )
                )

                positive_probability = (
                    float(
                        probabilities[
                            index,
                            2,
                        ]
                    )
                )

            else:
                negative_probability = ""
                neutral_probability = ""
                positive_probability = ""

            writer.writerow(
                [
                    index,
                    int(y_true[index]),
                    int(predictions[index]),
                    negative_probability,
                    neutral_probability,
                    positive_probability,
                    bool(
                        y_true[index]
                        == predictions[index]
                    ),
                ]
            )

    print(
        f"Saved test predictions: "
        f"{output_path}"
    )


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test(batch_size: int):
    print()
    print("=" * 78)
    print("FUSION-V2 SMOKE TEST")
    print("=" * 78)

    dataset = load_tweeteval()

    texts = list(
        dataset["train"]["text"][:64]
    )

    tokenizer, model = (
        load_sarcasm_model()
    )

    probabilities = (
        generate_sarcasm_probabilities(
            "smoke-test",
            texts,
            tokenizer,
            model,
            batch_size,
        )
    )

    print()
    print(
        "Probability shape:",
        probabilities.shape,
    )

    print(
        "Minimum:",
        float(
            probabilities.min()
        ),
    )

    print(
        "Maximum:",
        float(
            probabilities.max()
        ),
    )

    print(
        "Mean:",
        float(
            probabilities.mean()
        ),
    )

    print(
        "SMOKE TEST PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fusion-v2: replace the "
            "old sarcasm feature from "
            "Fusion-v1 with batched "
            "DeBERTa sarcasm probabilities."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "DeBERTa inference batch size."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run only a 64-example model "
            "inference smoke test."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "batch-size must be at least 1."
        )

    print("=" * 78)
    print("FUSION-V2")
    print("=" * 78)

    print(
        "Using existing Fusion-v1 "
        "features as cache."
    )

    print(
        "Replacing ONLY the old sarcasm "
        "probability."
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Temperature:",
        TEMPERATURE,
    )

    print(
        "Sarcasm threshold:",
        DECISION_THRESHOLD,
        "(used for classification only; "
        "continuous probability is used "
        "as fusion feature)",
    )

    if args.smoke_test:
        smoke_test(
            args.batch_size
        )
        return

    # --------------------------------------------------------
    # Load existing V1 features
    # --------------------------------------------------------

    (
        X_train_v1,
        X_validation_v1,
        X_test_v1,
        y_train,
        y_validation,
        y_test,
    ) = load_existing_v1_features()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_tweeteval()

    train_texts = list(
        dataset["train"]["text"]
    )

    validation_texts = list(
        dataset["validation"]["text"]
    )

    test_texts = list(
        dataset["test"]["text"]
    )

    if len(train_texts) != (
        X_train_v1.shape[0]
    ):
        raise ValueError(
            "TweetEval train size does not "
            "match saved Fusion-v1 features."
        )

    if len(validation_texts) != (
        X_validation_v1.shape[0]
    ):
        raise ValueError(
            "TweetEval validation size does "
            "not match saved Fusion-v1 features."
        )

    if len(test_texts) != (
        X_test_v1.shape[0]
    ):
        raise ValueError(
            "TweetEval test size does "
            "not match saved Fusion-v1 features."
        )

    # --------------------------------------------------------
    # Load DeBERTa ONCE
    # --------------------------------------------------------

    tokenizer, sarcasm_model = (
        load_sarcasm_model()
    )

    # --------------------------------------------------------
    # Generate only NEW sarcasm features
    # --------------------------------------------------------

    sarcasm_train = (
        generate_sarcasm_probabilities(
            "train",
            train_texts,
            tokenizer,
            sarcasm_model,
            args.batch_size,
        )
    )

    sarcasm_validation = (
        generate_sarcasm_probabilities(
            "validation",
            validation_texts,
            tokenizer,
            sarcasm_model,
            args.batch_size,
        )
    )

    sarcasm_test = (
        generate_sarcasm_probabilities(
            "test",
            test_texts,
            tokenizer,
            sarcasm_model,
            args.batch_size,
        )
    )

    # --------------------------------------------------------
    # Build V2 feature matrices
    # --------------------------------------------------------

    print()
    print("Building Fusion-v2 matrices...")

    X_train_v2 = build_v2_matrix(
        X_train_v1,
        sarcasm_train,
    )

    X_validation_v2 = build_v2_matrix(
        X_validation_v1,
        sarcasm_validation,
    )

    X_test_v2 = build_v2_matrix(
        X_test_v1,
        sarcasm_test,
    )

    print(
        "X_train_v2:",
        X_train_v2.shape,
    )

    print(
        "X_validation_v2:",
        X_validation_v2.shape,
    )

    print(
        "X_test_v2:",
        X_test_v2.shape,
    )

    # --------------------------------------------------------
    # Save V2 feature matrices
    # --------------------------------------------------------

    np.save(
        V2_DIR / "X_train.npy",
        X_train_v2,
    )

    np.save(
        V2_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        V2_DIR / "X_validation.npy",
        X_validation_v2,
    )

    np.save(
        V2_DIR / "y_validation.npy",
        y_validation,
    )

    np.save(
        V2_DIR / "X_test.npy",
        X_test_v2,
    )

    np.save(
        V2_DIR / "y_test.npy",
        y_test,
    )

    print(
        "Fusion-v2 matrices saved."
    )

    # --------------------------------------------------------
    # Train NEW fusion classifier
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING FUSION-V2 CLASSIFIER")
    print("=" * 78)

    fusion_model = build_fusion_model()

    fusion_model.fit(
        X_train_v2,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------------

    validation_result = evaluate(
        "Fusion-v2 — VALIDATION",
        fusion_model,
        X_validation_v2,
        y_validation,
    )

    # --------------------------------------------------------
    # Evaluate test
    # --------------------------------------------------------

    test_result = evaluate(
        "Fusion-v2 — TEST",
        fusion_model,
        X_test_v2,
        y_test,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        V2_DIR
        / "fusion_classifier.joblib"
    )

    joblib.dump(
        fusion_model,
        model_path,
    )

    print(
        f"Saved model: {model_path}"
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    save_test_predictions(
        y_test,
        test_result["predictions"],
        test_result["probabilities"],
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "fusion_version":
            "v2",
        "task":
            "3-class Twitter sentiment classification",
        "dataset":
            "cardiffnlp/tweet_eval",
        "splits": {
            "train": len(train_texts),
            "validation": len(validation_texts),
            "test": len(test_texts),
        },
        "feature_names":
            FEATURE_NAMES,
        "feature_count":
            len(FEATURE_NAMES),
        "feature_source": {
            "sentiment_probabilities":
                "reused from verified Fusion-v1 cache",
            "emoji_features":
                "reused from verified Fusion-v1 cache",
            "sarcasm_probability":
                "regenerated with DeBERTa-v3 BESSTIE",
        },
        "sarcasm_model":
            SARCASM_MODEL_NAME,
        "sarcasm_model_path":
            str(SARCASM_MODEL_DIR),
        "sarcasm_temperature":
            TEMPERATURE,
        "sarcasm_class_index":
            SARCASM_CLASS_INDEX,
        "sarcasm_decision_threshold":
            DECISION_THRESHOLD,
        "sarcasm_probability_used_as_feature":
            True,
        "fusion_model":
            "StandardScaler + LogisticRegression",
        "class_weight":
            "balanced",
        "random_state":
            RANDOM_STATE,
        "max_iterations":
            1000,
        "max_sequence_length":
            MAX_LENGTH,
        "batch_size":
            args.batch_size,
        "test_set_used_for_training":
            False,
        "note":
            (
                "Fusion-v2 replaces only the "
                "sarcasm feature from Fusion-v1 "
                "while preserving the same "
                "8-feature schema and "
                "Logistic Regression configuration."
            ),
    }

    metadata_path = (
        V2_DIR
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "validation": {
            key: value
            for key, value
            in validation_result.items()
            if key not in {
                "predictions",
                "probabilities",
            }
        },
        "test": {
            key: value
            for key, value
            in test_result.items()
            if key not in {
                "predictions",
                "probabilities",
            }
        },
    }

    metrics_path = (
        V2_DIR
        / "metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    summary_path = (
        V2_DIR
        / "results_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            [
                "FUSION-V2 RESULTS",
                "",
                "Validation:",
                (
                    f"Accuracy: "
                    f"{validation_result['accuracy']:.4f}"
                ),
                (
                    f"Macro-F1: "
                    f"{validation_result['macro_f1']:.4f}"
                ),
                (
                    f"Weighted-F1: "
                    f"{validation_result['weighted_f1']:.4f}"
                ),
                "",
                "Test:",
                (
                    f"Accuracy: "
                    f"{test_result['accuracy']:.4f}"
                ),
                (
                    f"Macro-F1: "
                    f"{test_result['macro_f1']:.4f}"
                ),
                (
                    f"Weighted-F1: "
                    f"{test_result['weighted_f1']:.4f}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("FUSION-V2 COMPLETE")
    print("=" * 78)

    print(
        "Validation Macro-F1:",
        f"{validation_result['macro_f1']:.4f}",
    )

    print(
        "Test Macro-F1:",
        f"{test_result['macro_f1']:.4f}",
    )

    print()
    print(
        "Artifacts:",
        V2_DIR,
    )


if __name__ == "__main__":
    main()
