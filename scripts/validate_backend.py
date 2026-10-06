from __future__ import annotations

import compileall
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]


REQUIRED = [
    "src/inference/pipeline.py",
    "src/inference/analyzer.py",
    "src/sarcasm/deberta_detector.py",
    "src/sarcasm/detector.py",
    "src/features/emoji_features.py",
    "src/fusion/inference.py",
    "src/explainability/fusion_explainer.py",
    "src/whisper/transcriber.py",
    "api/main.py",
    "tests/test_emoji_features.py",
    "tests/test_analyzer.py",
    "tests/test_api.py",
]


def main():
    missing = []

    print("=== FILE CHECK ===")

    for relative in REQUIRED:
        path = ROOT / relative

        if path.exists():
            print("OK  ", relative)
        else:
            print("MISS", relative)
            missing.append(relative)

    print("\n=== MODEL CHECK ===")

    sentiment = (
        ROOT
        / "models/twitter-roberta-base-sentiment"
        / "model.safetensors"
    )

    sarcasm = (
        ROOT
        / "models/sarcasm-deberta-v3"
        / "model.safetensors"
    )

    fusion = (
        ROOT
        / "models/fusion/model.joblib"
    )

    print(
        "Twitter-RoBERTa:",
        "READY" if sentiment.exists()
        else "MISSING",
    )

    print(
        "DeBERTa sarcasm:",
        "READY" if sarcasm.exists()
        else "MISSING",
    )

    print(
        "Final fusion:",
        "READY" if fusion.exists()
        else "NOT READY",
    )

    print("\n=== PYTHON COMPILE ===")

    src_ok = compileall.compile_dir(
        str(ROOT / "src"),
        quiet=1,
    )

    api_ok = compileall.compile_dir(
        str(ROOT / "api"),
        quiet=1,
    )

    tests_ok = compileall.compile_dir(
        str(ROOT / "tests"),
        quiet=1,
    )

    compile_ok = (
        src_ok
        and api_ok
        and tests_ok
    )

    print(
        "Compilation:",
        "OK" if compile_ok
        else "FAILED",
    )

    if missing or not compile_ok:
        print(
            "\nVALIDATION FAILED"
        )
        sys.exit(1)

    print(
        "\nSTATIC VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
