from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import whisper
import imageio_ffmpeg


class WhisperTranscriber:
    """
    Local Whisper transcription service.

    Audio/video is converted/transcribed locally and the resulting
    transcript can be passed to the same sentiment pipeline used for text.
    """

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            # Make bundled FFmpeg available to Whisper.
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = str(Path(ffmpeg_path).parent)

            current_path = os.environ.get("PATH", "")
            if ffmpeg_dir not in current_path.split(os.pathsep):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

            self._model = whisper.load_model(self.model_name)

        return self._model

    def transcribe(
        self,
        media_path: str | Path,
        language: Optional[str] = None,
    ) -> dict:
        media_path = Path(media_path)

        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        model = self._ensure_model()

        kwargs = {}

        if language:
            kwargs["language"] = language

        result = model.transcribe(
            str(media_path),
            **kwargs,
        )

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language"),
            "segments": result.get("segments", []),
            "model": f"whisper-{self.model_name}",
        }


_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber(model_name: str = "tiny") -> WhisperTranscriber:
    global _transcriber

    if _transcriber is None:
        _transcriber = WhisperTranscriber(model_name)

    return _transcriber


def transcribe_media(
    media_path: str | Path,
    language: Optional[str] = None,
    model_name: str = "tiny",
) -> dict:
    transcriber = get_transcriber(model_name)
    return transcriber.transcribe(media_path, language)
