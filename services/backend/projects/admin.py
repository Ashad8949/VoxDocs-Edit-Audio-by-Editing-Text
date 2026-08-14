"""Admin registrations.

Worth having: when a transcription goes wrong, being able to inspect the actual
word timings that produced a bad cut is the fastest way to understand why.
"""

from django.contrib import admin

from .models import Project, Render, Segment, Word


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


admin.site.register(Segment)
