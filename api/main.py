from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.explainability.fusion_explainer import (
    explain_text,
)
from src.fusion.inference import FusionNotReady
from src.inference.analyzer import (
    analyze_text,
    get_backend_model_info,
)
from src.whisper.transcriber import (
    transcribe_media,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "twitter-sentiment-api"
)


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "*",
    ).strip()

    if not raw:
        return ["*"]

    return [
        value.strip()
        for value in raw.split(",")
        if value.strip()
    ]


MAX_UPLOAD_MB = int(
    os.getenv(
        "MAX_UPLOAD_MB",
        "100",
    )
)

MAX_UPLOAD_BYTES = (
    MAX_UPLOAD_MB * 1024 * 1024
)

ALLOWED_MEDIA_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".mov",
    ".webm",
    ".aac",
    ".flac",
    ".ogg",
    ".aiff",
    ".avi",
    ".mkv",
}


app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description=(
        "Backend for Twitter sentiment, "
        "sarcasm, emoji features, "
        "Whisper transcription, "
        "and fusion inference."
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(
    request,
    call_next,
):
    start = time.perf_counter()

    response = await call_next(request)

    response.headers[
        "X-Process-Time"
    ] = (
        f"{time.perf_counter() - start:.4f}"
    )

    return response


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=10000,
    )


@app.get("/")
def root():
    return {
        "service":
            "twitter-sentiment-analysis-api",
        "status":
            "ok",
        "docs":
            "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service":
            "twitter-sentiment-analysis-api",
    }


@app.get("/model-info")
def model_info():
    return get_backend_model_info()


@app.post("/predict")
def predict(
    request: PredictRequest,
):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    try:
        return analyze_text(text)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Text analysis failed. "
                "Check backend logs."
            ),
        ) from exc


async def _save_upload(
    file: UploadFile,
):
    filename = file.filename or ""
    suffix = Path(
        filename
    ).suffix.lower()

    if suffix not in (
        ALLOWED_MEDIA_EXTENSIONS
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported media type."
            ),
        )

    temp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    )

    path = Path(temp.name)
    total = 0

    try:
        while True:
            chunk = await file.read(
                1024 * 1024
            )

            if not chunk:
                break

            total += len(chunk)

            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Uploaded file exceeds "
                        f"{MAX_UPLOAD_MB} MB."
                    ),
                )

            temp.write(chunk)

        temp.flush()

    finally:
        temp.close()

    if total == 0:
        path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    return path, total


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
):
    path, size = await _save_upload(file)

    try:
        result = transcribe_media(path)

        return {
            "filename":
                file.filename,
            "size_bytes":
                size,
            **result,
        }

    except Exception as exc:
        logger.exception(
            "Transcription failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Transcription failed. "
                "Check backend logs."
            ),
        ) from exc

    finally:
        path.unlink(
            missing_ok=True
        )


@app.post("/analyze-media")
async def analyze_media(
    file: UploadFile = File(...),
):
    path, size = await _save_upload(file)

    try:
        transcription = (
            transcribe_media(path)
        )

        transcript = str(
            transcription.get(
                "text",
                "",
            )
        ).strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Whisper produced "
                    "an empty transcript."
                ),
            )

        analysis = analyze_text(
            transcript
        )

        return {
            "filename":
                file.filename,
            "size_bytes":
                size,
            "transcription":
                transcription,
            "analysis":
                analysis,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Media analysis failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Media analysis failed. "
                "Check backend logs."
            ),
        ) from exc

    finally:
        path.unlink(
            missing_ok=True
        )


@app.post("/explain")
def explain(
    request: PredictRequest,
):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    try:
        return explain_text(text)

    except FusionNotReady as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Explainability failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Explainability failed. "
                "Check backend logs."
            ),
        ) from exc
