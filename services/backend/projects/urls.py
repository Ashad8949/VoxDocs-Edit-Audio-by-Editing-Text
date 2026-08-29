"""API routes.

Paths mirror the previous Node service exactly, so the React client needed no
change beyond the new async render flow.
"""

from django.urls import path

from . import views

urlpatterns = [
    path("health", views.health, name="health"),
    path("ready", views.ready, name="ready"),

    path("projects", views.project_list, name="project-list"),
    path("projects/<str:project_id>", views.project_detail, name="project-detail"),
    path("projects/<str:project_id>/envelope", views.project_envelope, name="project-envelope"),
    path("projects/<str:project_id>/media", views.project_media, name="project-media"),
    path("projects/<str:project_id>/plan", views.project_plan, name="project-plan"),
    path("projects/<str:project_id>/render", views.project_render, name="project-render"),
    path(
        "projects/<str:project_id>/renders/<str:render_id>",
        views.render_download, name="render-download",
    ),
    path(
        "projects/<str:project_id>/renders/<str:render_id>/status",
        views.render_status, name="render-status",
    ),

    # Translation endpoints
    path("projects/<str:project_id>/translate", views.project_translate, name="project-translate"),
    path("projects/<str:project_id>/translations", views.translations_list, name="translations-list"),
    path("projects/<str:project_id>/translations/<str:translation_id>", views.translation_detail, name="translation-detail"),
    path("projects/<str:project_id>/translations/<str:translation_id>/edit", views.translation_edit, name="translation-edit"),

    # Dub render endpoints
    path("projects/<str:project_id>/translations/<str:translation_id>/dub", views.translation_dub_render, name="translation-dub-render"),
    path(
        "projects/<str:project_id>/translations/<str:translation_id>/dubs/<str:dub_render_id>",
        views.dub_render_download, name="dub-render-download",
    ),
    path(
        "projects/<str:project_id>/translations/<str:translation_id>/dubs/<str:dub_render_id>/status",
        views.dub_render_status, name="dub-render-status",
    ),

    # Pro-tier voice model (RVC training)
    path("projects/<str:project_id>/voice-model", views.project_voice_model, name="project-voice-model"),
]

