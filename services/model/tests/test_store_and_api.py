"""Cache eviction and HTTP contract tests."""

import json

import numpy as np
import pytest

from voxdocs.audio import VoiceStats, encode_wav
from voxdocs.store import ProfileStore
from voxdocs.synth import ProfileWord, VoiceProfile
from voxdocs.app import create_app


def make_profile(pid: str, audio_seconds: float = 0.0) -> VoiceProfile:
    words = [ProfileWord(text="hello", start=0.0, end=0.4)]
    samples = (
        np.zeros(int(audio_seconds * 16000), dtype=np.float32) if audio_seconds else None
    )
    profile = VoiceProfile(
        project_id=pid,
        words=words,
        stats=VoiceStats(120.0, 0.1, 0.5, 16000),
        duration=1.0,
        sample_rate=16000,
        samples=samples,
    )
    profile.build_index()
    return profile


# ------------------------------------------------------------------ store

def test_store_round_trip():
    store = ProfileStore()
    store.put(make_profile("a"))
    assert store.get("a") is not None
    assert store.get("missing") is None


def test_store_evicts_least_recently_used():
    store = ProfileStore(max_entries=2)
    store.put(make_profile("a"))
    store.put(make_profile("b"))
    store.get("a")               # refresh 'a' so 'b' becomes the coldest
    store.put(make_profile("c"))

    assert store.get("a") is not None
    assert store.get("b") is None, "the least recently used profile should be gone"
    assert store.get("c") is not None


def test_store_honours_its_ttl(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("voxdocs.store.time.monotonic", lambda: clock["now"])

    store = ProfileStore(ttl_seconds=10.0)
    store.put(make_profile("a"))
    clock["now"] += 5
    assert store.get("a") is not None
    clock["now"] += 20
    assert store.get("a") is None


def test_memory_pressure_drops_waveforms_before_profiles():
    # Each profile carries 1 s of float32 audio = 64 kB.
    store = ProfileStore(max_entries=10, max_audio_bytes=100_000)
    store.put(make_profile("a", audio_seconds=1.0))
    store.put(make_profile("b", audio_seconds=1.0))

    # Both profiles survive; the audio is what gets shed.
    assert store.get("a") is not None and store.get("b") is not None
    assert store.stats()["audio_bytes"] <= 100_000
    assert (store.get("a").samples is None) or (store.get("b").samples is None)


def test_store_drop_and_stats():
    store = ProfileStore()
    store.put(make_profile("a"))
    assert store.drop("a") is True
    assert store.drop("a") is False
    assert store.stats()["profiles"] == 0


# -------------------------------------------------------------------- api

@pytest.fixture()
def client():
    return create_app().test_client()


def test_health_reports_backend_availability(client):
    body = client.get("/health").get_json()
    assert body["status"] in ("ok", "degraded")
    assert "voice_bank" in body["synthesis"]
    assert isinstance(body["synthesis"]["tts"], list)


def test_ready_does_not_load_a_model(client):
    assert client.get("/ready").get_json() == {"ready": True}


def test_transcribe_requires_a_file(client):
    response = client.post("/transcribe", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "missing_file"


def test_transcribe_rejects_undecodable_media(client):
    import io

    data = {"audio": (io.BytesIO(b"definitely not audio"), "x.wav")}
    response = client.post("/transcribe", data=data, content_type="multipart/form-data")
    assert response.status_code == 415
    assert response.get_json()["error"] == "decode_failed"


def test_synthesize_without_a_profile_asks_the_caller_to_reseed(client):
    response = client.post("/synthesize", json={"project_id": "nope", "text": "hello"})
    assert response.status_code == 409
    assert response.get_json()["error"] == "voice_profile_missing"


def test_synthesize_rejects_empty_text(client):
    response = client.post("/synthesize", json={"project_id": "x", "text": "  "})
    assert response.status_code == 400


def test_voice_profile_upload_then_synthesize_from_the_bank(client):
    words = [
        {"text": "four", "start": 0.0, "end": 0.4},
        {"text": "score", "start": 0.5, "end": 0.9},
        {"text": "years", "start": 1.0, "end": 1.4},
    ]
    response = client.post(
        "/voice-profile",
        data={"project_id": "p", "words": json.dumps(words), "duration": "2.0"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["vocabulary"] == 3

    response = client.get("/voice-profile/p")
    assert response.status_code == 200
    assert response.get_json()["words"] == 3

    response = client.post("/synthesize", json={"project_id": "p", "text": "four score"})
    body = response.get_json()
    assert response.status_code == 200
    assert body["coverage"] == 1.0
    assert body["units"][0]["type"] == "source"
    assert body["units"][0]["word"] == "four score"

    assert client.delete("/voice-profile/p").get_json()["deleted"] is True
    assert client.get("/voice-profile/p").status_code == 404


def test_voice_profile_accepts_audio_and_measures_the_speaker(client):
    import io

    rate = 16000
    t = np.arange(2 * rate) / rate
    saw = ((2 * (t * 140.0 % 1.0) - 1.0) * 0.4).astype(np.float32)
    wav = encode_wav(saw, rate)

    data = {
        "project_id": "withaudio",
        "words": json.dumps([{"text": "hi", "start": 0.0, "end": 0.4}]),
        "audio": (io.BytesIO(wav), "a.wav"),
    }
    response = client.post("/voice-profile", data=data, content_type="multipart/form-data")
    assert response.status_code == 200

    body = client.get("/voice-profile/withaudio").get_json()
    assert body["voice"]["median_f0"] == pytest.approx(140.0, rel=0.1)
    assert body["duration"] == pytest.approx(2.0, abs=0.05)


def test_voice_profile_requires_a_project_id(client):
    response = client.post("/voice-profile", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_batch_synthesis_resolves_every_item_in_one_call(client):
    words = [
        {"text": "alpha", "start": 0.0, "end": 0.4},
        {"text": "bravo", "start": 0.5, "end": 0.9},
    ]
    client.post(
        "/voice-profile",
        data={"project_id": "b", "words": json.dumps(words), "duration": "1.5"},
        content_type="multipart/form-data",
    )
    response = client.post(
        "/synthesize/batch",
        json={"project_id": "b", "items": [{"text": "alpha"}, {"text": "bravo"}, {"text": " "}]},
    )
    results = response.get_json()["results"]
    assert len(results) == 3
    assert results[0]["coverage"] == 1.0
    assert results[2]["units"] == []


def test_batch_synthesis_reports_a_missing_profile(client):
    response = client.post("/synthesize/batch", json={"project_id": "ghost", "items": []})
    assert response.status_code == 409
