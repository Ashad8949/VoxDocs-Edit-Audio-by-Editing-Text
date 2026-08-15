"""Admin registrations.

Worth having: when a transcription goes wrong, being able to inspect the actual
word timings that produced a bad cut is the fastest way to understand why.
"""

from django.contrib import admin

from .models import DubRender, Project, Render, Segment, Translation, VoiceProfile, Word


class WordInline(admin.TabularInline):
    model = Word
    extra = 0
    fields = ("index", "token_id", "text", "start", "end", "confidence")
    readonly_fields = fields
    can_delete = False
    max_num = 0


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "id", "status", "duration", "has_video", "created_at")
    list_filter = ("status", "has_video", "language")
    search_fields = ("id", "name")
    readonly_fields = ("id", "created_at", "updated_at", "voice")
    inlines = [WordInline]


@admin.register(Render)
class RenderAdmin(admin.ModelAdmin):
    list_display = ("id", "project", "status", "format", "duration", "pieces", "created_at")
    list_filter = ("status", "format")
    search_fields = ("id", "project__id")
    readonly_fields = ("id", "created_at", "updated_at", "stats", "synthesis", "warnings")


@admin.register(Translation)
class TranslationAdmin(admin.ModelAdmin):
    list_display = ("id", "project", "source_language", "target_language", "status", "created_at")
    list_filter = ("status", "source_language", "target_language")
    search_fields = ("id", "project__id")
    readonly_fields = ("id", "created_at", "updated_at")
    fields = ("id", "project", "source_language", "target_language", "status", "error", "translated_text", "edits", "created_at", "updated_at")


@admin.register(DubRender)
class DubRenderAdmin(admin.ModelAdmin):
    list_display = ("id", "translation", "status", "format", "duration", "created_at")
    list_filter = ("status", "format")
    search_fields = ("id", "translation__id", "translation__project__id")
    readonly_fields = ("id", "created_at", "updated_at", "stats", "synthesis", "warnings")
    fields = ("id", "translation", "status", "error", "format", "file", "bytes", "duration", "stats", "synthesis", "warnings", "created_at", "updated_at")


@admin.register(VoiceProfile)
class VoiceProfileAdmin(admin.ModelAdmin):
    list_display = ("project", "created_at", "updated_at")
    search_fields = ("project__id",)
    readonly_fields = ("created_at", "updated_at")
    fields = ("project", "embedding", "pitch_info", "spectral_features", "supported_languages", "created_at", "updated_at")


admin.site.register(Segment)
