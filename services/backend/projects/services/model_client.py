"""Client for the model server.

Django owns the durable transcript; the model server only caches a voice
profile. That asymmetry is deliberate — it means a model pod can be restarted or
scaled out at any moment and the worst case is one re-seed, which this client
performs transparently on a 409.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests
from django.conf import settings

log = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """The model server could not be reached, or refused the request."""

    def __init__(self, message: str, status: int = 502, code: str = "model_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def _url(path: str) -> str:
    return f"{settings.VOXDOCS['MODEL_URL']}{path}"


def _timeout() -> float:
    return settings.VOXDOCS["MODEL_TIMEOUT"]


def _raise_for(response: requests.Response) -> ModelError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    return ModelError(
        body.get("message") or body.get("error") or f"model server returned {response.status_code}",
        503 if response.status_code == 503 else 502,
        body.get("error", "model_error"),
    )


def _request(method: str, path: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, _url(path), timeout=_timeout(), **kwargs)
    except requests.Timeout as exc:
        raise ModelError("model server timed out", 504, "model_timeout") from exc
    except requests.RequestException as exc:
        raise ModelError(
            f"cannot reach model server at {settings.VOXDOCS['MODEL_URL']}",
            503, "model_unreachable",
        ) from exc


def health() -> dict:
    response = _request("GET", "/health")
    if not response.ok:
        raise _raise_for(response)
    return response.json()


def transcribe(audio_path: Path, project_id: str, language: str | None = None) -> dict:
    """Transcribe a file, seeding the voice profile as a side effect."""
    data = {"project_id": project_id}
    if language:
        data["language"] = language
    with open(audio_path, "rb") as handle:
        response = _request(
            "POST", "/transcribe",
            files={"audio": (Path(audio_path).name, handle)},
            data=data,
        )
    if not response.ok:
        raise _raise_for(response)
    return response.json()


def put_voice_profile(project_id: str, words: list[dict], duration: float,
                      audio_path: Path | None = None) -> dict:
    """Re-seed a voice profile the model server has evicted."""
    data = {
        "project_id": project_id,
        "words": json.dumps(words),
        "duration": str(duration),
    }
    if audio_path is not None:
        with open(audio_path, "rb") as handle:
            response = _request("POST", "/voice-profile",
                                files={"audio": ("audio.wav", handle)}, data=data)
    else:
        response = _request("POST", "/voice-profile", data=data)

    if not response.ok:
        raise _raise_for(response)
    return response.json()


def synthesize_batch(project_id: str, items: list[dict], reseed) -> dict:
    """Resolve every insertion in one round trip.

    On a 409 the profile has been evicted; ``reseed`` restores it from the
    transcript Django holds and the request is retried exactly once.
    """
    if not items:
        return {"results": []}

    payload = {"project_id": project_id, "items": items}
    response = _request("POST", "/synthesize/batch", json=payload)
    if response.status_code == 409:
        log.info("voice profile for %s was evicted; re-seeding", project_id)
        reseed()
        response = _request("POST", "/synthesize/batch", json=payload)

    if not response.ok:
        raise _raise_for(response)
    return response.json()
