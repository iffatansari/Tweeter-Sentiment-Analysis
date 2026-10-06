from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.pipeline import predict_text
from src.sarcasm.detector import predict_sarcasm


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_TRAIN = None
MAX_VALIDATION = None
MAX_TEST = None

RANDOM_STATE = 42


# ---------------------------------------------------------
# Emoji feature extraction
# ---------------------------------------------------------

POSITIVE_EMOJIS = {
    "😀", "😃", "😄", "😁", "😆", "😅",
    "😂", "🤣", "😊", "😇", "🙂", "🙃",
    "😉", "😌", "😍", "🥰", "😘", "😗",
    "😙", "😚", "🤗", "🤩", "🥳",
    "❤️", "❤", "💕", "💖", "💗", "💓",
    "💞", "💘", "👍", "👏", "🙌", "✨",
}

NEGATIVE_EMOJIS = {
    "😞", "😔", "😟", "😕", "🙁", "☹️",
    "😣", "😖", "😫", "😩", "🥺",
    "😢", "😭", "😤", "😠", "😡",
    "🤬", "🤢", "🤮", "🤧", "😰",
    "😨", "😱", "💔", "👎", "😒",
    "😑", "😐",
}


def extract_emoji_features(text: str) -> dict:
    text = str(text)

    positive_count = sum(
        text.count(emoji)
        for emoji in POSITIVE_EMOJIS
    )

    negative_count = sum(
        text.count(emoji)
        for emoji in NEGATIVE_EMOJIS
    )

    total_emoji = positive_count + negative_count

    return {
        "emoji_positive": float(positive_count),
        "emoji_negative": float(negative_count),
        "emoji_count": float(total_emoji),
        "emoji_present": float(total_emoji > 0),
    }


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def load_tweeteval():
    print("Loading TweetEval sentiment dataset...")

    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    return dataset


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def build_features(dataset_split):
    texts = dataset_split["text"]
    labels = np.asarray(dataset_split["label"], dtype=np.int64)

    features = []

    total = len(texts)

    for index, text in enumerate(texts, start=1):

        sentiment = predict_text(text)
        sarcasm = predict_sarcasm(text)
        emoji = extract_emoji_features(text)

        row = [
            sentiment["probabilities"]["negative"],
            sentiment["probabilities"]["neutral"],
            sentiment["probabilities"]["positive"],
            sarcasm["sarcasm_probability"],
            emoji["emoji_positive"],
            emoji["emoji_negative"],
            emoji["emoji_count"],
            emoji["emoji_present"],
        ]

        features.append(row)

        if index % 250 == 0 or index == total:
            print(
                f"Processed {index}/{total} texts",
                flush=True,
            )

    return np.asarray(features, dtype=np.float32), labels


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(name, model, X, y):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )
    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Macro-F1:   {macro_f1:.4f}")
    print(f"Weighted-F1:{weighted_f1:.4f}")

    print("\nClassification report:")
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
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    dataset = load_tweeteval()

    train = dataset["train"]
    validation = dataset["validation"]
    test = dataset["test"]

    if MAX_TRAIN:
        train = train.select(range(min(MAX_TRAIN, len(train))))

    if MAX_VALIDATION:
        validation = validation.select(
            range(min(MAX_VALIDATION, len(validation)))
        )

    if MAX_TEST:
        test = test.select(
            range(min(MAX_TEST, len(test)))
        )

    print("\nDataset sizes:")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    # -----------------------------------------------------
    # Feature generation
    # -----------------------------------------------------

    print("\nBuilding TRAIN features...")
    X_train, y_train = build_features(train)

    print("\nBuilding VALIDATION features...")
    X_validation, y_validation = build_features(validation)

    print("\nBuilding TEST features...")
    X_test, y_test = build_features(test)

    # -----------------------------------------------------
    # Save features
    # -----------------------------------------------------

    output_dir = ROOT / "models" / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)

    np.save(output_dir / "X_validation.npy", X_validation)
    np.save(output_dir / "y_validation.npy", y_validation)

    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    print("\nFeature matrices saved to:")
    print(output_dir)

    # -----------------------------------------------------
    # Proposed fusion model
    # -----------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining fusion classifier...")

    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    evaluate(
        "Fusion model — validation",
        model,
        X_validation,
        y_validation,
    )

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    results = evaluate(
        "Fusion model — TEST",
        model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    import joblib

    model_path = output_dir / "fusion_classifier.joblib"

    joblib.dump(model, model_path)

    print("\nSaved fusion model:")
    print(model_path)

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    pd.DataFrame([results]).to_csv(
        output_dir / "fusion_results.csv",
        index=False,
    )

    print("\nFusion training complete.")


if __name__ == "__main__":
    main()
    