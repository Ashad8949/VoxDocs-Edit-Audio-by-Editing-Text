"""Serializers.

The JSON shape is deliberately camelCase and identical to what the previous Node
service emitted, so the React client is unaffected by the backend rewrite.
"""

from __future__ import annotations

from rest_framework import serializers

from .models import Project, Render, Segment, Word


class WordSerializer(serializers.ModelSerializer):
    # The editor's stable handle on a word; not the database key.
    id = serializers.CharField(source="token_id")

    class Meta:
        model = Word
        fields = ("id", "text", "start", "end", "confidence")


class SegmentSerializer(serializers.ModelSerializer):
    first_word = serializers.IntegerField()
    last_word = serializers.IntegerField()

    class Meta:
        model = Segment
        fields = ("start", "end", "text", "first_word", "last_word")


class RenderSerializer(serializers.ModelSerializer):
    downloadUrl = serializers.SerializerMethodField()
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)

    class Meta:
        model = Render
        fields = (
            "id", "status", "error", "format", "file", "bytes", "duration",
            "pieces", "stats", "warnings", "synthesis", "createdAt", "downloadUrl",
        )

    def get_downloadUrl(self, obj: Render) -> str | None:
        if obj.status != Render.Status.READY:
            return None
        return f"/api/projects/{obj.project_id}/renders/{obj.id}"


class ProjectSummarySerializer(serializers.ModelSerializer):
    """The listing view: everything except the bulky per-word arrays."""

    createdAt = serializers.DateTimeField(source="created_at", read_only=True)
    updatedAt = serializers.DateTimeField(source="updated_at", read_only=True)
    hasVideo = serializers.BooleanField(source="has_video", read_only=True)
    wordCount = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = (
            "id", "name", "status", "error", "duration", "hasVideo",
            "language", "wordCount", "createdAt", "updatedAt",
        )

    def get_wordCount(self, obj: Project) -> int:
        # Annotated by the queryset where available, to avoid a query per row.
        count = getattr(obj, "word_count", None)
        return count if count is not None else obj.words.count()


class TranscriptSerializer(serializers.Serializer):
    words = WordSerializer(many=True)
    segments = SegmentSerializer(many=True)
    language = serializers.CharField()
    backend = serializers.CharField()


class ProjectDetailSerializer(ProjectSummarySerializer):
    """The editing view, including the full transcript."""

    transcript = serializers.SerializerMethodField()
    renders = serializers.SerializerMethodField()
    video = serializers.SerializerMethodField()

    class Meta(ProjectSummarySerializer.Meta):
        fields = ProjectSummarySerializer.Meta.fields + ("transcript", "renders", "video", "voice")

    def get_transcript(self, obj: Project) -> dict:
        return {
            "words": WordSerializer(obj.words.all(), many=True).data,
            "segments": SegmentSerializer(obj.segments.all(), many=True).data,
            "language": obj.language,
            "backend": obj.asr_backend,
        }

    def get_renders(self, obj: Project) -> list:
        return RenderSerializer(obj.renders.all()[:25], many=True).data

    def get_video(self, obj: Project) -> dict | None:
        if not obj.has_video:
            return None
        return {"width": obj.video_width, "height": obj.video_height, "fps": obj.video_fps}


class EditRequestSerializer(serializers.Serializer):
    """Body accepted by /plan and /render."""

    tokens = serializers.ListField(child=serializers.DictField(), required=False)
    text = serializers.CharField(required=False, allow_blank=True, trim_whitespace=False)
    format = serializers.CharField(required=False, default="wav")
    video = serializers.BooleanField(required=False, default=False)
    includeSegments = serializers.BooleanField(required=False, default=False)
    includeTokens = serializers.BooleanField(required=False, default=False)
