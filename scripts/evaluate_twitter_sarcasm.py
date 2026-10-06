from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "sarcasm-deberta-v3"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = (
    "ppokhrel2109/besstie-sarcasm-deberta-v3"
)

TEMPERATURE = 1.1819
THRESHOLD = 0.360
SARCASM_CLASS_INDEX = 1

MAX_LENGTH = 128
BATCH_SIZE = 16


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model.eval()
    model.to("cpu")

    return tokenizer, model


def main():
    print("=" * 80)
    print("TWITTER SARCASM EVALUATION")
    print("=" * 80)

    print("\nLoading independent Twitter sarcasm dataset...")

    dataset = load_dataset(
        "shiv213/Automatic-Sarcasm-Detection-Twitter"
    )

    print(dataset)

    test = dataset["test"]

    texts = [
        str(text).strip()
        for text in test["response"]
    ]

    raw_labels = test["label"]

    y_true = np.array(
        [
            1 if label == "SARCASM" else 0
            for label in raw_labels
        ],
        dtype=np.int64,
    )

    print("\nTest examples:", len(texts))
    print(
        "Sarcastic:",
        int(y_true.sum()),
    )
    print(
        "Not sarcastic:",
        int((y_true == 0).sum()),
    )

    print("\nLoading DeBERTa...")

    tokenizer, model = load_model()

    probabilities = []

    print("\nRunning batched inference...")

    with torch.inference_mode():

        for start in range(
            0,
            len(texts),
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                len(texts),
            )

            batch = texts[start:end]

            encoded = tokenizer(
                batch,
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

            probs = torch.softmax(
                calibrated_logits,
                dim=-1,
            )

            sarcasm_probs = (
                probs[
                    :,
                    SARCASM_CLASS_INDEX,
                ]
                .cpu()
                .numpy()
            )

            probabilities.extend(
                sarcasm_probs.tolist()
            )

            if (
                end % 200 == 0
                or end == len(texts)
            ):
                print(
                    f"Processed {end}/{len(texts)}",
                    flush=True,
                )

    sarcasm_prob = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    y_pred = (
        sarcasm_prob >= THRESHOLD
    ).astype(np.int64)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    print(
        f"Accuracy:    {accuracy:.4f}"
    )
    print(
        f"Macro-F1:    {macro_f1:.4f}"
    )
    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )
    print(
        f"Sarcasm precision: {precision:.4f}"
    )
    print(
        f"Sarcasm recall:    {recall:.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "not_sarcastic",
                "sarcastic",
            ],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("Confusion matrix:")
    print(cm)

    # ------------------------------------------------------
    # Save prediction-level results
    # ------------------------------------------------------

    prediction_rows = []

    for i, text in enumerate(texts):
        prediction_rows.append({
            "index": i,
            "text": text,
            "true_label": int(y_true[i]),
            "predicted_label": int(y_pred[i]),
            "sarcasm_probability": float(
                sarcasm_prob[i]
            ),
            "correct": bool(
                y_true[i] == y_pred[i]
            ),
        })

    predictions_path = (
        OUTPUT_DIR
        / "twitter_sarcasm_test_predictions.json"
    )

    predictions_path.write_text(
        json.dumps(
            prediction_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # Save summary metrics
    # ------------------------------------------------------

    metrics = {
        "dataset":
            "shiv213/Automatic-Sarcasm-Detection-Twitter",
        "split":
            "test",
        "examples":
            len(texts),
        "model":
            MODEL_NAME,
        "temperature":
            TEMPERATURE,
        "threshold":
            THRESHOLD,
        "accuracy":
            float(accuracy),
        "macro_f1":
            float(macro_f1),
        "weighted_f1":
            float(weighted_f1),
        "sarcasm_precision":
            float(precision),
        "sarcasm_recall":
            float(recall),
        "confusion_matrix":
            cm.tolist(),
    }

    metrics_path = (
        OUTPUT_DIR
        / "twitter_sarcasm_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # Show high-confidence mistakes
    # ------------------------------------------------------

    mistakes = [
        row
        for row in prediction_rows
        if not row["correct"]
    ]

    mistakes.sort(
        key=lambda row:
            abs(
                row["sarcasm_probability"]
                - THRESHOLD
            ),
        reverse=True,
    )

    print("\nHigh-confidence mistakes:")

    for row in mistakes[:20]:
        print(
            "\nTRUE:",
            "sarcastic"
            if row["true_label"]
            else "not sarcastic",
        )
        print(
            "PRED:",
            "sarcastic"
            if row["predicted_label"]
            else "not sarcastic",
        )
        print(
            "PROB:",
            f"{row['sarcasm_probability'] * 100:.2f}%"
        )
        print(
            "TEXT:",
            row["text"],
        )

    print("\nSaved:")
    print(predictions_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
