"""Per-speaker RVC training orchestration — the MLOps pipeline.

Drives one training run end to end and moves a `VoiceModel` row through its
lifecycle: version the speaker's audio as a Kaggle Dataset, push a GPU training
Kernel, poll it, pull the artifacts, run the evaluation gate on the reported
speaker-similarity, and promote the model to serving only if it clears the bar.

Each stage updates `VoiceModel.status` so the client polling the row (and the
UI) sees exactly where the run is — the same status-driven pattern the render
and translation pipelines use.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from django.conf import settings

from ..models import VoiceModel
from . import kaggle_client, model_client
from .rvc_kernel_template import build_kernel_script

log = logging.getLogger(__name__)


def _cfg(key: str):
    return settings.VOXDOCS[key]


def _set(model: VoiceModel, **fields) -> None:
    for k, v in fields.items():
        setattr(model, k, v)
    model.save(update_fields=[*fields.keys(), "updated_at"])


def train(model: VoiceModel) -> VoiceModel:
    """Run the full ingest → train → pull → gate → promote sequence."""
    project = model.project
    if not project.master_path.exists():
        raise FileNotFoundError(f"master audio missing for {project.id}")

    slug = f"voxdocs-{project.id}".lower()
    work = Path(tempfile.mkdtemp(prefix="voxdocs-train-"))
    try:
        # 1. Data ingestion + versioning -----------------------------------
        _set(model, status=VoiceModel.Status.UPLOADING)
        data_dir = work / "data"
        data_dir.mkdir()
        shutil.copy(project.master_path, data_dir / "speaker.wav")
        dataset_ref = kaggle_client.dataset_push(data_dir, slug, f"VoxDocs {project.id} voice")

        # 2. Training job (Kaggle GPU kernel) ------------------------------
        _set(model, status=VoiceModel.Status.TRAINING, kaggle_dataset=dataset_ref)
        kernel_dir = work / "kernel"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.py").write_text(
            build_kernel_script(project.id, dataset_ref,
                                epochs=_cfg("RVC_EPOCHS"), sample_rate=_cfg("RVC_SAMPLE_RATE")),
            encoding="utf-8",
        )
        kernel_ref = kaggle_client.kernel_push(
            kernel_dir / "kernel.py", slug, f"VoxDocs train {project.id}", dataset_ref)
        _set(model, kaggle_kernel=kernel_ref)

        outcome = kaggle_client.kernel_wait(
            kernel_ref, timeout_seconds=_cfg("KAGGLE_KERNEL_TIMEOUT"))
        if outcome != "complete":
            raise RuntimeError(f"Kaggle kernel finished with status: {outcome}")

        # 3. Pull artifacts ------------------------------------------------
        _set(model, status=VoiceModel.Status.PULLING)
        out_dir = work / "out"
        kaggle_client.kernel_output(kernel_ref, out_dir)
        metrics = _read_metrics(out_dir)
        model_src = out_dir / "model.pth"
        if not model_src.exists():
            raise RuntimeError("training produced no model.pth (see kernel logs)")

        # 4. Evaluation gate ------------------------------------------------
        _set(model, status=VoiceModel.Status.EVALUATING, metrics=metrics)
        pro = metrics.get("pro_similarity")
        threshold = _cfg("RVC_MIN_SIMILARITY")
        if pro is None or pro < threshold:
            _set(model, status=VoiceModel.Status.REJECTED,
                 error=f"speaker similarity {pro} below threshold {threshold}")
            return model

        # 5. Register + promote --------------------------------------------
        model.directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_src, model.directory / "model.pth")
        index_src = out_dir / "added.index"
        has_index = index_src.exists()
        if has_index:
            shutil.copy(index_src, model.directory / "added.index")
        _set(model, model_file="model.pth", index_file="added.index" if has_index else "")

        model_client.upload_voice_model(
            project.id, model.version,
            model.directory / "model.pth",
            (model.directory / "added.index") if has_index else None,
        )

        # Only one model serves at a time — deactivate older ones, activate this.
        VoiceModel.objects.filter(project=project, is_active=True).update(is_active=False)
        _set(model, status=VoiceModel.Status.READY, is_active=True, error="")
        log.info("voice model %s promoted (pro_similarity=%s)", model.id, pro)
        return model
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _read_metrics(out_dir: Path) -> dict:
    import json
    path = out_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
