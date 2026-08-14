"""Data model.

A project owns its transcript, and the transcript is the durable record of what
was said and when. The model server only ever caches a copy of it, which is what
lets model pods be evicted freely.

Words are real rows rather than a JSON blob. They are queried as a whole set on
every edit, so a blob would have been defensible, but rows give indexed access,
make transcript search possible later, and keep the timings typed rather than
trusting whatever the ASR happened to emit.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from django.conf import settings
from django.db import models


def new_id() -> str:
    """A short, URL- and filesystem-safe identifier."""
    return secrets.token_urlsafe(12)


class Project(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        TRANSCRIBING = "transcribing", "Transcribing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    # A natural string key, because it appears in URLs and in directory names.
    id = models.CharField(primary_key=True, max_length=64, default=new_id, editable=False)
    name = models.CharField(max_length=255, default="Untitled")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    error = models.TextField(blank=True, default="")

    source_file = models.CharField(max_length=255, blank=True, default="")
    duration = models.FloatField(default=0.0)

    has_video = models.BooleanField(default=False)
    video_width = models.IntegerField(default=0)
    video_height = models.IntegerField(default=0)
    video_fps = models.FloatField(default=0.0)

    language = models.CharField(max_length=16, blank=True, default="")
    asr_backend = models.CharField(max_length=64, blank=True, default="")
    voice = models.JSONField(null=True, blank=True)
    preview_failed = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["-created_at"])]

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"

    # ------------------------------------------------------------ paths

    @property
    def directory(self) -> Path:
        return Path(settings.MEDIA_ROOT) / "projects" / self.id

    @property
    def master_path(self) -> Path:
        """Canonical mono float32 audio; every cut is taken from this."""
        return self.directory / "master.wav"

    @property
    def preview_path(self) -> Path:
        """Small AAC copy the browser streams while editing."""
        return self.directory / "preview.m4a"

    @property
    def envelope_path(self) -> Path:
        """Waveform energy. A large float array, so it stays a file."""
        return self.directory / "envelope.json"

    @property
    def renders_dir(self) -> Path:
        return self.directory / "renders"

    def source_path(self) -> Path | None:
        """The original upload, whose extension varies."""
        if self.source_file:
            candidate = self.directory / self.source_file
            if candidate.exists():
                return candidate
        if self.directory.exists():
            for entry in sorted(self.directory.iterdir()):
                if entry.name.startswith("source."):
                    return entry
        return None

    def edl_words(self) -> list[dict]:
        """Words in the shape the EDL compiler expects."""
        return [
            {"id": w["token_id"], "text": w["text"], "start": w["start"], "end": w["end"]}
            for w in self.words.values("token_id", "text", "start", "end")
        ]


class Word(models.Model):
    """One recognised word and the span of samples it occupies."""

    project = models.ForeignKey(Project, related_name="words", on_delete=models.CASCADE)
    index = models.IntegerField()
    # Stable id the editor holds onto; independent of database keys so a
    # transcript can be re-imported without invalidating a client's edit.
    token_id = models.CharField(max_length=32)
    text = models.CharField(max_length=255)
    start = models.FloatField()
    end = models.FloatField()
    confidence = models.FloatField(default=1.0)
    segment = models.IntegerField(default=0)

    class Meta:
        ordering = ["index"]
        constraints = [
            models.UniqueConstraint(fields=["project", "index"], name="unique_word_index"),
            models.UniqueConstraint(fields=["project", "token_id"], name="unique_word_token"),
        ]
        indexes = [models.Index(fields=["project", "index"])]

    def __str__(self) -> str:
        return f"{self.text} [{self.start:.2f}–{self.end:.2f}]"


class Segment(models.Model):
    """A sentence-ish grouping, used for paragraph layout in the editor."""

    project = models.ForeignKey(Project, related_name="segments", on_delete=models.CASCADE)
    index = models.IntegerField()
    start = models.FloatField()
    end = models.FloatField()
    text = models.TextField(blank=True, default="")
    first_word = models.IntegerField()
    last_word = models.IntegerField()

    class Meta:
        ordering = ["index"]
        constraints = [
            models.UniqueConstraint(fields=["project", "index"], name="unique_segment_index"),
        ]


class Render(models.Model):
    """One rendered output of an edit."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RENDERING = "rendering", "Rendering"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    id = models.CharField(primary_key=True, max_length=64, default=new_id, editable=False)
    project = models.ForeignKey(Project, related_name="renders", on_delete=models.CASCADE)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")

    # The edit itself is kept so a render is reproducible and auditable.
    tokens = models.JSONField(default=list)
    format = models.CharField(max_length=8, default="wav")
    with_video = models.BooleanField(default=False)

    file = models.CharField(max_length=255, blank=True, default="")
    bytes = models.BigIntegerField(default=0)
    duration = models.FloatField(default=0.0)
    pieces = models.IntegerField(default=0)

    stats = models.JSONField(default=dict)
    warnings = models.JSONField(default=list)
    synthesis = models.JSONField(default=dict)

    task_id = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["project", "-created_at"])]

    @property
    def directory(self) -> Path:
        return self.project.renders_dir / self.id

    @property
    def path(self) -> Path | None:
        return self.directory / self.file if self.file else None

    def __str__(self) -> str:
        return f"{self.id} ({self.status})"
