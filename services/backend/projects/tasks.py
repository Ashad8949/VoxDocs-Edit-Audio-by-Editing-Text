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
