"""RVC (Retrieval-based Voice Conversion) serving for the Pro tier.

A per-speaker RVC model, trained on Kaggle (see the backend's voice_train
pipeline), converts generic TTS output into the exact timbre of the project's
speaker. This module owns two things on the model server:

  - a per-project on-disk cache of the trained artifacts (.pth + .index),
    populated by the backend after a model is promoted, and
  - lazy RVC inference over those artifacts.

RVC inference depends on `rvc-python`, whose `fairseq` dependency does not
install on Python 3.12+. The model server's container is Python 3.11 (see its
Dockerfile), where it installs fine; on a newer local interpreter the import
simply fails and `available()` returns False, so the Pro tier transparently
falls through to XTTS. Nothing here crashes a request for lack of RVC.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

from . import audio as audio_util

log = logging.getLogger(__name__)

# Where promoted models are cached, keyed by project id. Kept on disk so a
# model survives a worker restart; re-uploaded by the backend on a cache miss.
CACHE_DIR = Path(os.environ.get("VOXDOCS_RVC_CACHE", "/tmp/voxdocs-rvc"))


def _project_dir(project_id: str) -> Path:
    return CACHE_DIR / project_id


def store_model(project_id: str, model_bytes: bytes, index_bytes: bytes | None) -> None:
    """Cache a promoted model's artifacts for this project."""
    d = _project_dir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pth").write_bytes(model_bytes)
    if index_bytes:
        (d / "added.index").write_bytes(index_bytes)


def has_model(project_id: str) -> bool:
    return (_project_dir(project_id) / "model.pth").exists()


class RvcConverter:
    """Lazily-loaded per-project RVC inference."""

    sample_rate = 40000

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import rvc_python  # noqa: F401
            return True
        except Exception:  # noqa: BLE001 - any import failure means unavailable
            return False

    def _engine(self, project_id: str):
        engine = self._engines.get(project_id)
        if engine is not None:
            return engine
        with self._lock:
            engine = self._engines.get(project_id)
            if engine is not None:
                return engine
            d = _project_dir(project_id)
            model = d / "model.pth"
            if not model.exists():
                raise RuntimeError(f"no RVC model cached for project {project_id}")
            from rvc_python.infer import RVCInference

            index = d / "added.index"
            log.info("loading RVC model for %s", project_id)
            engine = RVCInference(model_path=str(model),
                                  index_path=str(index) if index.exists() else "")
            self._engines[project_id] = engine
            return engine

    def convert(self, samples: np.ndarray, sample_rate: int, project_id: str) -> tuple[np.ndarray, int]:
        """Convert `samples` (any TTS output) into the project speaker's timbre.

        Raises if no model is cached for the project — the caller (the tier
        engine chain) treats that as "engine unavailable here" and falls back.
        """
        import tempfile
        import soundfile as sf

        engine = self._engine(project_id)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.wav")
            dst = os.path.join(tmp, "out.wav")
            sf.write(src, samples.astype(np.float32), sample_rate)
            engine.infer_file(src, dst)
            out, out_sr = sf.read(dst, dtype="float32")
        if out.ndim > 1:
            out = out.mean(axis=1)
        return out, out_sr
