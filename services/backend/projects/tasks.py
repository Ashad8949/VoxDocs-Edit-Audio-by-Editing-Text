"""Celery tasks.

Thin by design: they own status transitions and failure reporting, while the
actual work lives in ``services.pipeline`` where it can be tested without a
broker.
"""

from __future__ import annotations

import logging

from celery import shared_task

from .models import Project, Render
from .services import pipeline

log = logging.getLogger(__name__)


@shared_task(bind=True, name="projects.ingest_project")
def ingest_project(self, project_id: str, language: str | None = None) -> str:
    """Transcode and transcribe an upload."""
    try:
        project = Project.objects.get(pk=project_id)
    except Project.DoesNotExist:
        log.warning("ingest for unknown project %s; it was probably deleted", project_id)
        return "missing"

    try:
        pipeline.ingest(project, language)
        return project.status
    except Exception as exc:
        log.exception("ingest failed for %s", project_id)
        # Report the failure on the row the client is polling, rather than only
        # in the worker log where nobody will see it.
        Project.objects.filter(pk=project_id).update(
            status=Project.Status.FAILED, error=str(exc)[:2000]
        )
        raise


@shared_task(bind=True, name="projects.render_edit")
def render_edit(self, render_id: str) -> str:
    """Render one edit to a finished file."""
    try:
        render = Render.objects.select_related("project").get(pk=render_id)
    except Render.DoesNotExist:
        log.warning("render for unknown id %s; it was probably deleted", render_id)
        return "missing"

    Render.objects.filter(pk=render_id).update(status=Render.Status.RENDERING)
    render.refresh_from_db()

    try:
        pipeline.perform_render(render)
        return render.status
    except Exception as exc:
        log.exception("render failed for %s", render_id)
        Render.objects.filter(pk=render_id).update(
            status=Render.Status.FAILED, error=str(exc)[:2000]
        )
        raise


@shared_task(bind=True, name="projects.translate_project")
def translate_project(self, translation_id: str) -> str:
    """Translate a project's transcript to target language."""
    try:
        from .models import Translation
        translation = Translation.objects.select_related("project").get(pk=translation_id)
    except Exception:
        log.warning("translate for unknown id %s; it was probably deleted", translation_id)
        return "missing"

    Translation.objects.filter(pk=translation_id).update(status=Translation.Status.TRANSLATING)
    translation.refresh_from_db()

    try:
        from .services import translation as translation_svc
        result = translation_svc.translate_transcript(
            translation.project,
            translation.target_language,
        )
        Translation.objects.filter(pk=translation_id).update(
            status=Translation.Status.READY,
            translated_text=result.get("segments_text", {}),
            error="",
        )
        return "ready"
    except Exception as exc:
        log.exception("translation failed for %s", translation_id)
        Translation.objects.filter(pk=translation_id).update(
            status=Translation.Status.FAILED,
            error=str(exc)[:2000]
        )
        raise


@shared_task(bind=True, name="projects.train_voice_model")
def train_voice_model(self, voice_model_id: str) -> str:
    """Train a per-speaker RVC model on Kaggle GPU and gate/promote it."""
    try:
        from .models import VoiceModel
        model = VoiceModel.objects.select_related("project").get(pk=voice_model_id)
    except Exception:
        log.warning("train for unknown voice model %s; it was probably deleted", voice_model_id)
        return "missing"

    try:
        from .services import voice_train
        result = voice_train.train(model)
        return result.status
    except Exception as exc:
        log.exception("voice-model training failed for %s", voice_model_id)
        from .models import VoiceModel
        VoiceModel.objects.filter(pk=voice_model_id).update(
            status=VoiceModel.Status.FAILED, error=str(exc)[:2000]
        )
        raise


@shared_task(bind=True, name="projects.extract_voice")
def extract_voice(self, project_id: str) -> str:
    """Extract voice profile from project's original audio."""
    try:
        project = Project.objects.get(pk=project_id)
    except Project.DoesNotExist:
        log.warning("voice extract for unknown project %s", project_id)
        return "missing"

    try:
        from .services import translation as translation_svc
        profile = translation_svc.extract_voice_profile(project)
        return "ready"
    except Exception as exc:
        log.exception("voice extraction failed for %s", project_id)
        raise


@shared_task(bind=True, name="projects.render_dub")
def render_dub(self, dub_render_id: str) -> str:
    """Render a dubbed version (translated + edited audio with original video)."""
    try:
        from .models import DubRender
        dub_render = DubRender.objects.select_related("translation__project").get(pk=dub_render_id)
    except Exception:
        log.warning("dub render for unknown id %s", dub_render_id)
        return "missing"

    DubRender.objects.filter(pk=dub_render_id).update(status=DubRender.Status.RENDERING)
    dub_render.refresh_from_db()

    try:
        from .services import dub_render as dub_render_svc
        dub_render_svc.perform_dub_render(dub_render)
        return dub_render.status
    except Exception as exc:
        log.exception("dub render failed for %s", dub_render_id)
        DubRender.objects.filter(pk=dub_render_id).update(
            status=DubRender.Status.FAILED,
            error=str(exc)[:2000]
        )
        raise

