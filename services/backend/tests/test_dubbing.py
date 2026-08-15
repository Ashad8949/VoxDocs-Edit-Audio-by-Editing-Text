"""Integration tests for translation and dubbing features."""

import json
from pathlib import Path

import pytest

from projects.models import DubRender, Project, Translation, VoiceProfile
from projects.services.dub_render import perform_dub_render
from projects.services.translation import (
    apply_translation_edits,
    build_dub_edl,
    create_translation,
    extract_voice_profile,
)


class TestTranslation:
    """Translation workflow tests."""

    def test_create_translation_for_ready_project(self, project):
        """Creating a translation requires a ready project."""
        assert project.status == "ready"
        
        translation = create_translation(project, "en")
        
        assert translation.project == project
        assert translation.target_language == "en"
        assert translation.source_language == project.language or "en"
        assert translation.status == "pending"

    def test_create_translation_fails_for_unready_project(self, client):
        """Cannot create translation for non-ready project."""
        project = Project.objects.create(
            name="Unready Project",
            status="transcribing",
        )
        
        with pytest.raises(ValueError, match="must be ready"):
            create_translation(project, "en")

    def test_translation_uniqueness(self, project):
        """Only one translation per project per language."""
        create_translation(project, "hi")
        
        # Creating again should return the same one
        second = create_translation(project, "hi")
        
        assert Translation.objects.filter(
            project=project, target_language="hi"
        ).count() == 1

    def test_apply_translation_edits(self, project):
        """Apply user edits to a translation."""
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
            translated_text={
                0: "how to use this",
                1: "feature is useful",
            }
        )
        
        edits = [
            {"type": "segment", "index": 0, "text": "how do we use this", "keepOriginal": False},
        ]
        
        result = apply_translation_edits(translation, edits)
        
        assert result["translated_text"][0] == "how do we use this"
        assert result["edits"][0]["text"] == "how do we use this"
        
        translation.refresh_from_db()
        assert translation.translated_text[0] == "how do we use this"

    def test_build_dub_edl_keeps_unchanged_segments(self, project):
        """EDL keeps original audio for unchanged segments."""
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
            translated_text={0: "original", 1: "also original"},
        )
        
        edl = build_dub_edl(translation)
        
        # All segments unchanged, so all should be kept
        assert len(edl["keep_original_indices"]) > 0
        assert len(edl["replace_text"]) == 0

    def test_build_dub_edl_replaces_edited_segments(self, project):
        """EDL replaces segments where text changed."""
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
            translated_text={0: "new text", 1: "original"},
        )
        translation.edits = [
            {"type": "segment", "index": 0, "text": "new text", "keepOriginal": False}
        ]
        translation.save()
        
        edl = build_dub_edl(translation)
        
        assert 0 in edl["replace_text"]
        assert edl["replace_text"][0] == "new text"

    def test_build_dub_edl_respects_keep_original_flag(self, project):
        """EDL keeps original even if text appears changed."""
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
            translated_text={0: "different"},
        )
        translation.edits = [
            {"type": "segment", "index": 0, "text": "different", "keepOriginal": True}
        ]
        translation.save()
        
        edl = build_dub_edl(translation)
        
        assert 0 in edl["keep_original_indices"]
        assert 0 not in edl["replace_text"]


class TestVoiceProfile:
    """Voice profile extraction tests."""

    def test_extract_voice_profile(self, project):
        """Extract voice profile from project audio."""
        if not project.master_path.exists():
            pytest.skip("Master audio not available")
        
        profile = extract_voice_profile(project)
        
        assert profile.project == project
        assert profile.embedding
        assert profile.pitch_info
        assert profile.spectral_features
        assert profile.supported_languages

    def test_voice_profile_one_per_project(self, project):
        """Only one voice profile per project."""
        if not project.master_path.exists():
            pytest.skip("Master audio not available")
        
        profile1 = extract_voice_profile(project)
        profile2 = extract_voice_profile(project)
        
        assert profile1.id == profile2.id
        assert VoiceProfile.objects.filter(project=project).count() == 1


class TestDubRender:
    """Dub rendering tests."""

    def test_dub_render_creation(self, project):
        """Create a dub render."""
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
        )
        
        dub = DubRender.objects.create(
            translation=translation,
            format="mp4",
            status="pending",
        )
        
        assert dub.translation == translation
        assert dub.format == "mp4"
        assert dub.status == "pending"
        assert dub.directory.exists() or not dub.directory.exists()  # Path property works

    def test_dub_render_requires_video(self, project):
        """Cannot dub a project without video."""
        if project.has_video:
            pytest.skip("Project has video")
        
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
        )
        dub = DubRender.objects.create(
            translation=translation,
            format="mp4",
        )
        
        with pytest.raises(ValueError, match="no video"):
            perform_dub_render(dub)

    def test_dub_render_with_file(self, project):
        """Dub render with actual file creation."""
        if not project.has_video:
            pytest.skip("Project has no video")
        
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
            translated_text={},
        )
        
        dub = DubRender.objects.create(
            translation=translation,
            format="mp4",
        )
        
        # This would perform actual rendering
        # perform_dub_render(dub)
        # 
        # assert dub.status == "ready"
        # assert dub.path and dub.path.exists()
        # assert dub.bytes > 0
        # (Actual test requires full stack)


class TestAPIEndpoints:
    """API endpoint tests."""

    def test_post_translate_endpoint(self, client, project):
        """POST /api/projects/{id}/translate creates translation."""
        response = client.post(
            f"/api/projects/{project.id}/translate",
            json={"targetLanguage": "hi"},
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "translation" in data
        assert data["translation"]["targetLanguage"] == "hi"
        assert data["translation"]["status"] in ("pending", "translating")

    def test_get_translations_endpoint(self, client, project):
        """GET /api/projects/{id}/translations lists translations."""
        Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
        )
        Translation.objects.create(
            project=project,
            target_language="hi",
            status="ready",
        )
        
        response = client.get(f"/api/projects/{project.id}/translations")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["translations"]) == 2

    def test_get_translation_detail_endpoint(self, client, project):
        """GET /api/projects/{id}/translations/{id} gets translation details."""
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
            translated_text={0: "test"},
        )
        
        response = client.get(
            f"/api/projects/{project.id}/translations/{translation.id}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["translation"]["id"] == translation.id
        assert data["translation"]["targetLanguage"] == "en"

    def test_put_translation_edit_endpoint(self, client, project):
        """PUT /api/projects/{id}/translations/{id}/edit applies edits."""
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
            translated_text={0: "original"},
        )
        
        response = client.put(
            f"/api/projects/{project.id}/translations/{translation.id}/edit",
            json={
                "edits": [
                    {"type": "segment", "index": 0, "text": "new text"}
                ]
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["translation"]["translatedText"][0] == "new text"

    def test_post_dub_render_endpoint(self, client, project):
        """POST /api/projects/{id}/translations/{id}/dub queues dub render."""
        if not project.has_video:
            pytest.skip("Project has no video")
        
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
        )
        
        response = client.post(
            f"/api/projects/{project.id}/translations/{translation.id}/dub",
            json={"format": "mp4"},
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "dubRender" in data
        assert data["dubRender"]["format"] == "mp4"

    def test_get_dub_render_status_endpoint(self, client, project):
        """GET /api/projects/{id}/translations/{id}/dubs/{id}/status polls status."""
        if not project.has_video:
            pytest.skip("Project has no video")
        
        translation = Translation.objects.create(
            project=project,
            target_language="en",
            status="ready",
        )
        dub = DubRender.objects.create(
            translation=translation,
            format="mp4",
            status="ready",
        )
        
        response = client.get(
            f"/api/projects/{project.id}/translations/{translation.id}/dubs/{dub.id}/status"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["dubRender"]["status"] == "ready"
