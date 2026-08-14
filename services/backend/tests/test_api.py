"""API and pipeline tests.

The model server is stubbed so the whole ingest and render path runs
deterministically without loading an ASR model, but ffmpeg is real: these tests
cut actual audio and check what came out.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from django.urls import reverse

from projects.models import Project, Render, Word
from projects.services import model_client, pipeline

pytestmark = pytest.mark.django_db


FAKE_WORDS = [
    {"id": "w0", "text": "alpha", "start": 0.05, "end": 0.9, "confidence": 0.9},
    {"id": "w1", "text": "bravo", "start": 1.05, "end": 1.9, "confidence": 0.9},
    {"id": "w2", "text": "charlie", "start": 2.05, "end": 2.9, "confidence": 0.9},
    {"id": "w3", "text": "delta", "start": 3.05, "end": 3.9, "confidence": 0.9},
]


@pytest.fixture
def tone_file(tmp_path) -> Path:
    """Four seconds of tone, matching FAKE_WORDS."""
    path = tmp_path / "input.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=4:sample_rate=44100",
        "-ac", "1", "-c:a", "pcm_s16le", str(path),
    ], check=True)
    return path


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the model server with an in-process double."""
    calls = {"transcribe": 0, "synthesize": 0, "reseed": 0}
    state = {"fail_next_with_409": False}

    def transcribe(audio_path, project_id, language=None):
        calls["transcribe"] += 1
        return {
            "words": FAKE_WORDS,
            "segments": [{"start": 0, "end": 3.9, "text": "alpha bravo charlie delta",
                          "first_word": 0, "last_word": 3}],
            "language": "en",
            "backend": "stub",
            "duration": 4.0,
            "envelope": {"fps": 10, "rms": [0.3] * 40},
            "voice": {"median_f0": 120, "speech_rms": 0.1, "peak": 0.8, "sample_rate": 16000},
        }

    def synthesize_batch(project_id, items, reseed):
        if state["fail_next_with_409"]:
            state["fail_next_with_409"] = False
            reseed()
        calls["synthesize"] += 1
        return {"results": [
            # Pretend every insertion is covered by the bank, lifting second 1.
            {"units": [{"type": "source", "start": 1.0, "end": 1.5, "gain": 1,
                        "word": item["text"]}],
             "backends": ["voice-bank"], "covered": item["text"].split(),
             "generated": [], "missing": [], "coverage": 1}
            for item in items
        ]}

    def put_voice_profile(project_id, words, duration, audio_path=None):
        calls["reseed"] += 1
        return {"project_id": project_id, "words": len(words)}

    monkeypatch.setattr(model_client, "transcribe", transcribe)
    monkeypatch.setattr(model_client, "synthesize_batch", synthesize_batch)
    monkeypatch.setattr(model_client, "put_voice_profile", put_voice_profile)
    monkeypatch.setattr(model_client, "health", lambda: {"status": "ok"})
    return calls, state


@pytest.fixture
def project(client, tone_file, stub_model, settings, tmp_path):
    """An ingested, ready project."""
    settings.MEDIA_ROOT = tmp_path / "media"
    settings.CELERY_TASK_ALWAYS_EAGER = True
    with open(tone_file, "rb") as handle:
        response = client.post("/api/projects", {"file": handle, "name": "Tone Test"})
    assert response.status_code == 202
    return Project.objects.get(pk=response.json()["project"]["id"])


# ------------------------------------------------------------------ health

def test_health_reports_the_model_server(client, stub_model):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["renderSampleRate"] == 48000


def test_ready_does_not_touch_the_model_server(client):
    assert client.get("/api/ready").json() == {"ready": True}


# ------------------------------------------------------------------ ingest

def test_upload_creates_a_project_and_transcribes_it(project, stub_model):
    calls, _ = stub_model
    assert project.status == Project.Status.READY
    assert calls["transcribe"] == 1
    assert project.words.count() == 4
    assert project.segments.count() == 1
    assert project.duration == pytest.approx(4.0, abs=0.1)


def test_words_keep_their_stable_ids_and_order(project):
    words = list(project.words.all())
    assert [w.token_id for w in words] == ["w0", "w1", "w2", "w3"]
    assert [w.index for w in words] == [0, 1, 2, 3]
    assert words[0].text == "alpha"


def test_the_master_and_preview_are_written(project):
    assert project.master_path.exists()
    assert project.preview_path.exists()
    assert project.envelope_path.exists()


def test_project_detail_includes_the_transcript(client, project):
    body = client.get(f"/api/projects/{project.id}").json()["project"]
    assert body["status"] == "ready"
    assert len(body["transcript"]["words"]) == 4
    assert body["transcript"]["words"][0]["id"] == "w0"
    assert body["hasVideo"] is False
    assert body["wordCount"] == 4


def test_listing_omits_the_bulky_transcript(client, project):
    projects = client.get("/api/projects").json()["projects"]
    entry = next(p for p in projects if p["id"] == project.id)
    assert entry["wordCount"] == 4
    assert "transcript" not in entry


def test_unsupported_file_types_are_rejected(client, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path / "media"
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    with open(notes, "rb") as handle:
        response = client.post("/api/projects", {"file": handle})
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["message"]


def test_uploading_nothing_is_a_400(client):
    assert client.post("/api/projects", {}).status_code == 400


def test_a_file_without_audio_fails_the_project(client, tmp_path, settings, stub_model):
    settings.MEDIA_ROOT = tmp_path / "media"
    settings.CELERY_TASK_ALWAYS_EAGER = True
    # Match production: a worker failure is recorded on the row the client
    # polls, it does not travel back into the upload request. Without this,
    # eager mode would re-raise inside the view and mask the recorded state.
    settings.CELERY_TASK_EAGER_PROPAGATES = False

    silent = tmp_path / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
        "-i", "color=c=black:s=64x64:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(silent),
    ], check=True)

    with open(silent, "rb") as handle:
        response = client.post("/api/projects", {"file": handle})
    assert response.status_code == 202

    project = Project.objects.get(pk=response.json()["project"]["id"])
    assert project.status == Project.Status.FAILED
    assert "no audio track" in project.error

    # And the client is told why when it asks, rather than being left waiting.
    detail = client.get(f"/api/projects/{project.id}").json()["project"]
    assert detail["status"] == "failed"
    assert "no audio track" in detail["error"]


# ------------------------------------------------------------------- plan

def test_planning_an_unchanged_transcript_reports_no_edits(client, project):
    body = client.post(f"/api/projects/{project.id}/plan",
                       data=json.dumps({}), content_type="application/json").json()
    assert body["stats"]["keptWords"] == 4
    assert body["stats"]["deletedWords"] == 0
    assert body["stats"]["cuts"] == 0


def test_planning_a_deletion_predicts_a_shorter_result(client, project):
    body = client.post(f"/api/projects/{project.id}/plan",
                       data=json.dumps({"text": "alpha delta"}),
                       content_type="application/json").json()
    assert body["stats"]["deletedWords"] == 2
    assert body["stats"]["cuts"] == 1
    assert body["stats"]["estimatedDuration"] < body["stats"]["sourceDuration"]


def test_planning_accepts_an_explicit_token_list(client, project):
    body = client.post(
        f"/api/projects/{project.id}/plan",
        data=json.dumps({"tokens": [{"ref": "w0"}, {"insert": "new words"}, {"ref": "w3"}],
                         "includeSegments": True}),
        content_type="application/json",
    ).json()
    assert body["stats"]["keptWords"] == 2
    assert body["stats"]["insertedWords"] == 2
    kinds = [s["kind"] for s in body["segments"]]
    assert kinds == ["copy", "synth", "copy"]


def test_a_malformed_token_list_is_rejected(client, project):
    response = client.post(f"/api/projects/{project.id}/plan",
                           data=json.dumps({"tokens": [{"bogus": True}]}),
                           content_type="application/json")
    assert response.status_code == 400


def test_editing_a_project_still_importing_is_a_409(client, project):
    Project.objects.filter(pk=project.id).update(status=Project.Status.TRANSCRIBING)
    response = client.post(f"/api/projects/{project.id}/plan",
                           data=json.dumps({}), content_type="application/json")
    assert response.status_code == 409


# ----------------------------------------------------------------- render

def test_rendering_a_deletion_shortens_the_audio(client, project, settings):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    response = client.post(f"/api/projects/{project.id}/render",
                           data=json.dumps({"text": "alpha delta", "format": "wav"}),
                           content_type="application/json")
    assert response.status_code == 202, "renders are queued, not awaited"

    render = Render.objects.get(pk=response.json()["render"]["id"])
    assert render.status == Render.Status.READY
    assert render.stats["deletedWords"] == 2
    assert render.pieces == 2
    assert render.duration < 3.0
    assert render.bytes > 1000
    assert render.path.exists()


def test_a_ready_render_downloads_with_a_filename(client, project, settings):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    created = client.post(f"/api/projects/{project.id}/render",
                          data=json.dumps({"text": "alpha delta"}),
                          content_type="application/json").json()
    render_id = created["render"]["id"]

    response = client.get(f"/api/projects/{project.id}/renders/{render_id}")
    assert response.status_code == 200
    assert "Tone-Test-edited.wav" in response["content-disposition"]
    assert len(b"".join(response.streaming_content)) > 1000


def test_render_status_is_pollable(client, project, settings):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    created = client.post(f"/api/projects/{project.id}/render",
                          data=json.dumps({"text": "alpha delta"}),
                          content_type="application/json").json()
    body = client.get(created["statusUrl"]).json()["render"]
    assert body["status"] == "ready"
    assert body["downloadUrl"].endswith(created["render"]["id"])


def test_an_insertion_consults_the_model_server(client, project, settings, stub_model):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    calls, _ = stub_model
    before = calls["synthesize"]

    created = client.post(
        f"/api/projects/{project.id}/render",
        data=json.dumps({"tokens": [{"ref": "w0"}, {"insert": "echo"}, {"ref": "w3"}]}),
        content_type="application/json",
    ).json()

    assert calls["synthesize"] == before + 1
    render = Render.objects.get(pk=created["render"]["id"])
    assert render.synthesis["fromVoiceBank"] == 1
    assert render.synthesis["coverage"] == 1
    assert render.warnings == []


def test_an_evicted_voice_profile_is_reseeded(client, project, settings, stub_model):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    calls, state = stub_model
    state["fail_next_with_409"] = True
    before = calls["reseed"]

    client.post(f"/api/projects/{project.id}/render",
                data=json.dumps({"tokens": [{"ref": "w0"}, {"insert": "echo"}]}),
                content_type="application/json")
    assert calls["reseed"] == before + 1


def test_an_unsupported_format_is_refused_before_queueing(client, project):
    response = client.post(f"/api/projects/{project.id}/render",
                           data=json.dumps({"format": "ogg"}),
                           content_type="application/json")
    assert response.status_code == 400
    assert "unsupported format" in response.json()["message"]
    assert Render.objects.count() == 0, "a rejected render must not leave a row behind"


def test_mp3_export_works(client, project, settings):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    created = client.post(f"/api/projects/{project.id}/render",
                          data=json.dumps({"text": "alpha bravo charlie delta",
                                           "format": "mp3"}),
                          content_type="application/json").json()
    render = Render.objects.get(pk=created["render"]["id"])
    assert render.format == "mp3"
    assert render.bytes > 500


def test_an_empty_edit_renders_without_error(client, project, settings):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    created = client.post(f"/api/projects/{project.id}/render",
                          data=json.dumps({"tokens": []}),
                          content_type="application/json").json()
    render = Render.objects.get(pk=created["render"]["id"])
    assert render.status == Render.Status.READY
    assert render.duration < 0.2


def test_a_failed_render_records_why(client, project, settings, monkeypatch):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    settings.CELERY_TASK_EAGER_PROPAGATES = False

    def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(pipeline, "perform_render", boom)
    created = client.post(f"/api/projects/{project.id}/render",
                          data=json.dumps({"text": "alpha"}),
                          content_type="application/json").json()

    render = Render.objects.get(pk=created["render"]["id"])
    assert render.status == Render.Status.FAILED
    assert "ffmpeg exploded" in render.error


def test_downloading_an_unfinished_render_is_a_404(client, project):
    render = Render.objects.create(project=project, tokens=[], status=Render.Status.PENDING)
    assert client.get(f"/api/projects/{project.id}/renders/{render.id}").status_code == 404


# ------------------------------------------------------------------ media

def test_media_supports_range_requests(client, project):
    full = client.get(f"/api/projects/{project.id}/media")
    assert full.status_code == 200
    assert full["accept-ranges"] == "bytes"
    total = int(full["content-length"])

    partial = client.get(f"/api/projects/{project.id}/media", headers={"range": "bytes=0-99"})
    assert partial.status_code == 206
    assert partial["content-length"] == "100"
    assert partial["content-range"] == f"bytes 0-99/{total}"


def test_an_unsatisfiable_range_is_a_416(client, project):
    response = client.get(f"/api/projects/{project.id}/media",
                          headers={"range": "bytes=99999999-"})
    assert response.status_code == 416


def test_the_envelope_downsamples_on_request(client, project):
    full = client.get(f"/api/projects/{project.id}/envelope").json()
    assert len(full["rms"]) == 40

    small = client.get(f"/api/projects/{project.id}/envelope?points=10").json()
    assert len(small["rms"]) == 10
    assert small["downsampled"] is True


# --------------------------------------------------------------- lifecycle

def test_ids_that_could_escape_the_media_root_are_refused(client):
    for bad in ["..", "a%2Fb", "short"]:
        assert client.get(f"/api/projects/{bad}").status_code == 404


def test_deleting_a_project_removes_its_rows_and_files(client, project):
    directory = project.directory
    assert directory.exists()

    assert client.delete(f"/api/projects/{project.id}").status_code == 200
    assert client.get(f"/api/projects/{project.id}").status_code == 404
    assert not directory.exists()
    assert Word.objects.filter(project_id=project.id).count() == 0


def test_deleting_twice_is_a_404(client, project):
    client.delete(f"/api/projects/{project.id}")
    assert client.delete(f"/api/projects/{project.id}").status_code == 404
