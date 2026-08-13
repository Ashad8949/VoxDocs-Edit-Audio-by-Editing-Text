"""VoxDocs model server.

Isolated from the API server because it has a completely different resource
profile: it wants CPU (or a GPU) and a warm model in memory, while the API
server wants disk and sockets. Splitting them lets each scale on its own metric.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time

import numpy as np
from flask import Flask, jsonify, request

from . import audio as audio_util
from .asr import ASR_SAMPLE_RATE, select_backend
from .store import ProfileStore
from .synth import ProfileWord, Synthesizer, VoiceProfile

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.environ.get("VOXDOCS_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
ENVELOPE_FPS = 100


def _median_gap(words: list[ProfileWord]) -> float:
    gaps = [
        words[i].start - words[i - 1].end
        for i in range(1, len(words))
        if 0 <= words[i].start - words[i - 1].end < 1.0
    ]
    return float(np.median(gaps)) if gaps else 0.08


def _sec_per_word(words: list[ProfileWord]) -> float:
    if len(words) < 4:
        return 0.34
    spoken = sum(max(0.0, w.end - w.start) for w in words)
    if spoken <= 0:
        return 0.34
    span = words[-1].end - words[0].start
    gap_share = max(0.0, span - spoken) / len(words)
    return spoken / len(words) + min(gap_share, 0.12)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    asr_preference = os.environ.get("VOXDOCS_ASR_BACKEND", "auto")
    keep_audio = os.environ.get("VOXDOCS_KEEP_AUDIO", "auto")
    store = ProfileStore(
        max_entries=int(os.environ.get("VOXDOCS_PROFILE_CACHE", "64")),
        ttl_seconds=float(os.environ.get("VOXDOCS_PROFILE_TTL", "3600")),
    )
    synthesizer = Synthesizer(
        enable_voice_bank=os.environ.get("VOXDOCS_VOICE_BANK", "1") not in ("0", "false", "no")
    )

    state: dict = {"asr": None, "asr_error": None}

    def get_asr():
        if state["asr"] is None and state["asr_error"] is None:
            try:
                state["asr"] = select_backend(asr_preference)
            except RuntimeError as exc:
                state["asr_error"] = str(exc)
        if state["asr"] is None:
            raise RuntimeError(state["asr_error"] or "ASR unavailable")
        return state["asr"]

    def wants_audio() -> bool:
        """Cache the waveform only when a backend could actually use it."""
        if keep_audio in ("1", "true", "yes"):
            return True
        if keep_audio in ("0", "false", "no"):
            return False
        return any(b.available() for b in synthesizer.tts_backends if b.name == "paddlespeech")

    # ---------------------------------------------------------------- health

    @app.get("/health")
    def health():
        try:
            backend = get_asr()
            asr_info = {"available": True, "backend": backend.name}
        except RuntimeError as exc:
            asr_info = {"available": False, "error": str(exc)}
        return jsonify({
            "status": "ok" if asr_info["available"] else "degraded",
            "asr": asr_info,
            "synthesis": synthesizer.describe(),
            "cache": store.stats(),
        })

    @app.get("/ready")
    def ready():
        # Readiness must not force a model load, or a rolling deploy stalls.
        return jsonify({"ready": True})

    # ------------------------------------------------------------ transcribe

    @app.post("/transcribe")
    def transcribe():
        upload = request.files.get("audio") or request.files.get("file")
        if upload is None:
            return jsonify({"error": "missing_file", "message": "expected an 'audio' file part"}), 400

        project_id = request.form.get("project_id") or ""
        language = request.form.get("language") or None
        if language in ("auto", ""):
            language = None

        suffix = os.path.splitext(upload.filename or "")[1][:10] or ".bin"
        started = time.monotonic()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            upload.save(tmp.name)
            try:
                samples = audio_util.decode(tmp.name, ASR_SAMPLE_RATE)
            except audio_util.AudioError as exc:
                return jsonify({"error": "decode_failed", "message": str(exc)}), 415
            container_duration = audio_util.probe_duration(tmp.name)

        if samples.size == 0:
            return jsonify({"error": "empty_audio", "message": "no audio samples decoded"}), 415

        duration = container_duration or samples.size / ASR_SAMPLE_RATE

        try:
            backend = get_asr()
        except RuntimeError as exc:
            return jsonify({"error": "asr_unavailable", "message": str(exc)}), 503

        transcript = backend.transcribe(samples, language)
        envelope = audio_util.rms_envelope(samples, ASR_SAMPLE_RATE, ENVELOPE_FPS)
        stats = audio_util.voice_stats(samples, ASR_SAMPLE_RATE)

        profile_words = [
            ProfileWord(text=w.text, start=w.start, end=w.end, confidence=w.confidence)
            for w in transcript.words
        ]
        if project_id:
            profile = VoiceProfile(
                project_id=project_id,
                words=profile_words,
                stats=stats,
                duration=duration,
                sample_rate=ASR_SAMPLE_RATE,
                median_gap=_median_gap(profile_words),
                sec_per_word=_sec_per_word(profile_words),
                samples=samples if wants_audio() else None,
            )
            profile.build_index()
            store.put(profile)

        return jsonify({
            "project_id": project_id,
            "language": transcript.language,
            "backend": transcript.backend,
            "duration": round(duration, 3),
            "words": [w.to_json(i) for i, w in enumerate(transcript.words)],
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text,
                    "first_word": s.first_word,
                    "last_word": s.last_word,
                }
                for s in transcript.segments
            ],
            "envelope": {"fps": ENVELOPE_FPS, "rms": [round(float(v), 5) for v in envelope]},
            "voice": stats.to_json(),
            "elapsed": round(time.monotonic() - started, 3),
        })

    # --------------------------------------------------------- voice profile

    @app.post("/voice-profile")
    def put_voice_profile():
        """Re-seed a profile after a cache miss, without re-running ASR."""
        project_id = request.form.get("project_id") or ""
        if not project_id:
            return jsonify({"error": "missing_project_id"}), 400

        try:
            words_raw = json.loads(request.form.get("words") or "[]")
        except json.JSONDecodeError as exc:
            return jsonify({"error": "bad_words", "message": str(exc)}), 400

        words = [
            ProfileWord(
                text=str(w.get("text", "")),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                confidence=float(w.get("confidence", 1.0)),
            )
            for w in words_raw
        ]
        duration = float(request.form.get("duration") or 0.0)

        samples = None
        stats = audio_util.VoiceStats(0.0, 0.0, 0.0, ASR_SAMPLE_RATE)
        upload = request.files.get("audio")
        if upload is not None:
            suffix = os.path.splitext(upload.filename or "")[1][:10] or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                upload.save(tmp.name)
                try:
                    samples = audio_util.decode(tmp.name, ASR_SAMPLE_RATE)
                except audio_util.AudioError as exc:
                    return jsonify({"error": "decode_failed", "message": str(exc)}), 415
                duration = duration or audio_util.probe_duration(tmp.name)
            stats = audio_util.voice_stats(samples, ASR_SAMPLE_RATE)

        profile = VoiceProfile(
            project_id=project_id,
            words=words,
            stats=stats,
            duration=duration or (words[-1].end if words else 0.0),
            sample_rate=ASR_SAMPLE_RATE,
            median_gap=_median_gap(words),
            sec_per_word=_sec_per_word(words),
            samples=samples if (samples is not None and wants_audio()) else None,
        )
        profile.build_index()
        store.put(profile)
        return jsonify({
            "project_id": project_id,
            "words": len(words),
            "vocabulary": profile.vocabulary_size(),
            "has_audio": profile.samples is not None,
        })

    @app.get("/voice-profile/<project_id>")
    def get_voice_profile(project_id: str):
        profile = store.get(project_id)
        if profile is None:
            return jsonify({"error": "not_found", "project_id": project_id}), 404
        return jsonify({
            "project_id": project_id,
            "words": len(profile.words),
            "vocabulary": profile.vocabulary_size(),
            "duration": round(profile.duration, 3),
            "has_audio": profile.samples is not None,
            "voice": profile.stats.to_json(),
        })

    @app.delete("/voice-profile/<project_id>")
    def delete_voice_profile(project_id: str):
        return jsonify({"deleted": store.drop(project_id)})

    # ------------------------------------------------------------ synthesise

    @app.post("/synthesize")
    def synthesize():
        payload = request.get_json(silent=True) or {}
        project_id = str(payload.get("project_id") or "")
        text = str(payload.get("text") or "")

        if not text.strip():
            return jsonify({"error": "empty_text"}), 400

        profile = store.get(project_id) if project_id else None
        if profile is None:
            # The caller owns the durable transcript; ask it to re-seed us.
            return jsonify({
                "error": "voice_profile_missing",
                "project_id": project_id,
                "message": "re-seed the profile via POST /voice-profile and retry",
            }), 409

        result = synthesizer.synthesize(
            profile,
            text,
            context_before=payload.get("context_before"),
            context_after=payload.get("context_after"),
            lead_gap=float(payload.get("lead_gap") or 0.0),
            trail_gap=float(payload.get("trail_gap") or 0.0),
        )
        return jsonify(result.to_json())

    @app.post("/synthesize/batch")
    def synthesize_batch():
        """Resolve every insertion in one render with a single round trip."""
        payload = request.get_json(silent=True) or {}
        project_id = str(payload.get("project_id") or "")
        items = payload.get("items") or []

        profile = store.get(project_id) if project_id else None
        if profile is None:
            return jsonify({
                "error": "voice_profile_missing",
                "project_id": project_id,
                "message": "re-seed the profile via POST /voice-profile and retry",
            }), 409

        results = []
        for item in items:
            text = str(item.get("text") or "")
            if not text.strip():
                results.append({"units": [], "backends": [], "covered": [],
                                "generated": [], "missing": [], "coverage": 1.0})
                continue
            result = synthesizer.synthesize(
                profile,
                text,
                context_before=item.get("context_before"),
                context_after=item.get("context_after"),
                lead_gap=float(item.get("lead_gap") or 0.0),
                trail_gap=float(item.get("trail_gap") or 0.0),
            )
            results.append(result.to_json())
        return jsonify({"project_id": project_id, "results": results})

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({
            "error": "file_too_large",
            "limit_bytes": MAX_UPLOAD_BYTES,
        }), 413

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
