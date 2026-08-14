"""HTTP surface.

Ingest and render both run on Celery, because transcribing or re-rendering an
hour of audio takes minutes and no browser request should be held open for that.
Both return 202 and the client polls the resulting row.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path

from django.conf import settings
from django.db.models import Count
from django.http import FileResponse, Http404, HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from .models import Project, Render
from .serializers import (
    EditRequestSerializer,
    ProjectDetailSerializer,
    ProjectSummarySerializer,
    RenderSerializer,
)
from .services import model_client, pipeline
from .services.ffmpeg import FfmpegError
from .tasks import ingest_project, render_edit

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv",
}
# Ids reach the filesystem, so keep them unambiguous.
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


def exception_handler(exc, context):
    """Map service-layer errors onto sensible status codes."""
    if isinstance(exc, pipeline.ValidationError):
        return Response(
            {"error": "validation_error", "message": str(exc)},
            status=getattr(exc, "status", 400),
        )
    if isinstance(exc, model_client.ModelError):
        return Response({"error": exc.code, "message": str(exc)}, status=exc.status)
    if isinstance(exc, FfmpegError):
        log.error("ffmpeg failure: %s", exc)
        return Response({"error": "render_failed", "message": str(exc)}, status=500)

    response = drf_exception_handler(exc, context)
    if response is not None and isinstance(response.data, dict):
        response.data.setdefault("error", "error")
    return response


def get_project(project_id: str) -> Project:
    if not ID_PATTERN.match(project_id or ""):
        raise Http404("invalid project id")
    try:
        return Project.objects.get(pk=project_id)
    except Project.DoesNotExist as exc:
        raise Http404("project not found") from exc


def require_ready(project: Project) -> None:
    if project.status == Project.Status.FAILED:
        raise pipeline.ValidationError(
            f"this project failed to import: {project.error or 'unknown error'}"
        )
    if project.status != Project.Status.READY:
        error = pipeline.ValidationError("the transcript is still being prepared")
        error.status = 409
        raise error


# ------------------------------------------------------------------ health

@api_view(["GET"])
def health(request):
    model_health = None
    model_error = None
    try:
        model_health = model_client.health()
    except model_client.ModelError as exc:
        model_error = str(exc)
    return Response({
        "status": "ok" if model_health else "degraded",
        "model": model_health,
        "modelError": model_error,
        "renderSampleRate": settings.VOXDOCS["RENDER_SAMPLE_RATE"],
    })


@api_view(["GET"])
def ready(request):
    # Must not touch the model server, or a rolling deploy stalls behind it.
    return Response({"ready": True})


# ---------------------------------------------------------------- projects

@api_view(["GET", "POST"])
@parser_classes([MultiPartParser, FormParser])
def project_list(request):
    if request.method == "GET":
        projects = Project.objects.annotate(word_count=Count("words"))
        return Response({"projects": ProjectSummarySerializer(projects, many=True).data})

    upload = request.FILES.get("file")
    if upload is None:
        raise pipeline.ValidationError('expected a "file" upload')
    if upload.size > settings.VOXDOCS["MAX_UPLOAD_BYTES"]:
        return Response(
            {"error": "file_too_large", "limitBytes": settings.VOXDOCS["MAX_UPLOAD_BYTES"]},
            status=413,
        )

    extension = Path(upload.name or "").suffix.lower()
    if extension and extension not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        raise pipeline.ValidationError(f'unsupported file type "{extension}"')

    project = Project.objects.create(
        name=(request.data.get("name") or upload.name or "Untitled")[:255],
        source_file=f"source{extension or '.bin'}",
        status=Project.Status.QUEUED,
    )
    project.directory.mkdir(parents=True, exist_ok=True)
    destination = project.directory / project.source_file
    with open(destination, "wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)

    ingest_project.delay(project.id, request.data.get("language") or None)
    return Response(
        {"project": ProjectSummarySerializer(project).data},
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["GET", "DELETE"])
def project_detail(request, project_id: str):
    project = get_project(project_id)
    if request.method == "DELETE":
        pipeline.delete_project_files(project)
        project.delete()
        return Response({"deleted": True})

    project = (
        Project.objects.prefetch_related("words", "segments", "renders")
        .annotate(word_count=Count("words", distinct=True))
        .get(pk=project.pk)
    )
    return Response({"project": ProjectDetailSerializer(project).data})


@api_view(["GET"])
def project_envelope(request, project_id: str):
    project = get_project(project_id)
    envelope = pipeline.load_envelope(project)
    if not envelope:
        raise Http404("no envelope for this project")

    rms = envelope.get("rms") or []
    try:
        points = max(0, min(20000, int(request.query_params.get("points", 0))))
    except ValueError:
        points = 0

    if not points or points >= len(rms):
        return Response(envelope)

    # The editor draws a few thousand pixels at most. Downsample by peak so
    # transients survive rather than being averaged away.
    bucket = len(rms) / points
    peaks = []
    for i in range(points):
        start = int(i * bucket)
        end = min(len(rms), int((i + 1) * bucket) + 1)
        peaks.append(round(max(rms[start:end], default=0.0), 5))

    return Response({
        "fps": envelope.get("fps", 100) * (points / len(rms)),
        "rms": peaks,
        "downsampled": True,
    })


# ------------------------------------------------------------------- media

@api_view(["GET"])
def project_media(request, project_id: str):
    project = get_project(project_id)

    # The preview is a small AAC copy; the original is served when the client
    # needs the video track or the preview never got made.
    path = project.preview_path
    if request.query_params.get("original") == "1" or project.has_video or project.preview_failed:
        path = project.source_path() or path
    if not path or not Path(path).exists():
        path = project.source_path()
    if not path or not Path(path).exists():
        raise Http404("media not found")

    return _serve_file(request, Path(path))


@api_view(["GET"])
def render_download(request, project_id: str, render_id: str):
    project = get_project(project_id)
    try:
        render = project.renders.get(pk=render_id)
    except Render.DoesNotExist as exc:
        raise Http404("render not found") from exc

    if render.status != Render.Status.READY or not render.path or not render.path.exists():
        raise Http404("render is not ready")

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(project.name).stem) or "voxdocs"
    response = _serve_file(request, render.path)
    response["content-disposition"] = (
        f'attachment; filename="{safe_name[:60]}-edited.{render.format}"'
    )
    return response


def _serve_file(request, path: Path) -> HttpResponse:
    """Serve a file with range support, which media elements need to seek."""
    size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    range_header = request.headers.get("range", "")
    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if match:
        start = int(match.group(1)) if match.group(1) else 0
        end = int(match.group(2)) if match.group(2) else size - 1
        if start >= size or start > end:
            response = HttpResponse(status=416)
            response["content-range"] = f"bytes */{size}"
            return response

        end = min(end, size - 1)
        handle = open(path, "rb")
        handle.seek(start)
        response = FileResponse(handle, status=206, content_type=content_type)
        response["content-length"] = str(end - start + 1)
        response["content-range"] = f"bytes {start}-{end}/{size}"
        response["accept-ranges"] = "bytes"
        return response

    response = FileResponse(open(path, "rb"), content_type=content_type)
    response["content-length"] = str(size)
    response["accept-ranges"] = "bytes"
    return response


# ------------------------------------------------------------------ edits

@api_view(["POST"])
def project_plan(request, project_id: str):
    """Cost and duration of an edit, without rendering it."""
    project = get_project(project_id)
    require_ready(project)

    body = EditRequestSerializer(data=request.data)
    body.is_valid(raise_exception=True)

    tokens = pipeline.resolve_tokens(project, request.data)
    edl = pipeline.plan_edit(project, tokens)

    payload = {"stats": edl.stats.to_json()}
    if body.validated_data.get("includeTokens"):
        payload["tokens"] = tokens
    if body.validated_data.get("includeSegments"):
        payload["segments"] = [_segment_json(s) for s in edl.segments]
    return Response(payload)


def _segment_json(segment) -> dict:
    if hasattr(segment, "word_ids"):
        return {
            "kind": "copy",
            "start": round(segment.start, 4),
            "end": round(segment.end, 4),
            "wordIds": segment.word_ids,
        }
    return {
        "kind": "synth",
        "text": segment.text,
        "contextBefore": segment.context_before,
        "contextAfter": segment.context_after,
        "estimatedDuration": round(segment.estimated_duration, 4),
    }


@api_view(["POST"])
def project_render(request, project_id: str):
    """Queue a render. Returns 202; the client polls the render row."""
    project = get_project(project_id)
    require_ready(project)

    body = EditRequestSerializer(data=request.data)
    body.is_valid(raise_exception=True)

    fmt = (body.validated_data.get("format") or "wav").lower()
    want_video = bool(body.validated_data.get("video")) and project.has_video
    if not want_video and fmt not in pipeline.AUDIO_FORMATS:
        raise pipeline.ValidationError(f'unsupported format "{fmt}" (use wav, mp3 or m4a)')

    tokens = pipeline.resolve_tokens(project, request.data)
    render = Render.objects.create(
        project=project, tokens=tokens, format=fmt, with_video=want_video,
        status=Render.Status.PENDING,
    )
    async_result = render_edit.delay(render.id)
    Render.objects.filter(pk=render.id).update(task_id=async_result.id)
    render.refresh_from_db()

    return Response(
        {"render": RenderSerializer(render).data,
         "statusUrl": f"/api/projects/{project.id}/renders/{render.id}/status"},
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["GET"])
def render_status(request, project_id: str, render_id: str):
    """Poll a queued render."""
    project = get_project(project_id)
    try:
        render = project.renders.get(pk=render_id)
    except Render.DoesNotExist as exc:
        raise Http404("render not found") from exc
    return Response({"render": RenderSerializer(render).data})
