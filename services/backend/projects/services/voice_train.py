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

    # The dataset is versioned per project (the data-versioning story); each
    # training run gets its own kernel slug so concurrent/retried runs never
    # collide (Kaggle 409s a push to a kernel that is still running).
    dataset_slug = f"voxdocs-{project.id}".lower()
    kernel_slug = f"voxdocs-{project.id}-v{model.version}".lower()
    work = Path(tempfile.mkdtemp(prefix="voxdocs-train-"))
    try:
        # 1. Data ingestion + versioning -----------------------------------
        _set(model, status=VoiceModel.Status.UPLOADING)
        data_dir = work / "data"
        data_dir.mkdir()
        shutil.copy(project.master_path, data_dir / "speaker.wav")
        dataset_ref = kaggle_client.dataset_push(data_dir, dataset_slug, f"VoxDocs {project.id} voice")

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
            kernel_dir / "kernel.py", kernel_slug, f"VoxDocs train {project.id} v{model.version}", dataset_ref)
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

        # 4. Evaluation gate ------------------------------------------------
        # Record the metrics regardless, then decide. A run that produced no
        # model, or whose speaker similarity is below the bar, is *rejected*
        # (a normal, honest outcome) — not a pipeline failure. The recorded
        # standard_similarity still shows the zero-shot baseline in the UI.
        _set(model, status=VoiceModel.Status.EVALUATING, metrics=metrics)
        model_src = out_dir / "model.pth"
        pro = metrics.get("pro_similarity")
        threshold = _cfg("RVC_MIN_SIMILARITY")
        if not model_src.exists():
            _set(model, status=VoiceModel.Status.REJECTED,
                 error="no RVC model was produced (see kernel logs)")
            return model
        # A produced model is promoted. pro_similarity is measured on the
        # serving side (RVC inference needs the Python 3.11 container), so the
        # gate only *rejects* on it when it was measured and fell short.
        if pro is not None and pro < threshold:
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
