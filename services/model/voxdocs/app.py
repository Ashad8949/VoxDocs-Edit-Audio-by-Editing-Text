"""VoxDocs model server.

Isolated from the API server because it has a completely different resource
profile: it wants CPU (or a GPU) and a warm model in memory, while the API
server wants disk and sockets. Splitting them lets each scale on its own metric.
"""

from __future__ import annotations

import base64
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
from .synth import DEFAULT_TIER, ProfileWord, Synthesizer, VoiceProfile
from .translate import IndicTrans2Translator
from . import voice_rvc
from .voice_clone import XttsVoiceCloner

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


def _extract_pitch_info(samples: np.ndarray, sr: int = 16000) -> tuple[float, float, float]:
    """Extract pitch statistics from audio.
    
    Returns: (mean_pitch_hz, std_dev, pitch_range)
    """
    # Stub: use simple spectral analysis
    # In production, use a pitch extraction library like librosa.piptrack or pyin
    try:
        # Compute power spectrum
        from scipy import signal
        freqs, power = signal.periodogram(samples, sr)
        # Find dominant frequency (very rough pitch estimate)
        dominant_idx = np.argmax(power)
        mean_pitch = freqs[dominant_idx] if dominant_idx < len(freqs) else 120.0
        # Assume some variation
        std_dev = mean_pitch * 0.15  # 15% variation
        pitch_range = mean_pitch * 0.5  # 50% range
        return float(np.clip(mean_pitch, 50, 400)), float(std_dev), float(pitch_range)
    except Exception:
        # Fallback: typical male voice ~120Hz, female ~220Hz
        return 130.0, 20.0, 80.0


def _extract_mfcc(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract MFCC (Mel-Frequency Cepstral Coefficients) from audio.
    
    Returns: Array of MFCC coefficients
    """
    # Stub: return zeros
    # In production, use librosa.feature.mfcc
    try:
        # Compute a basic spectrogram as proxy
        from scipy import signal
        f, t, Sxx = signal.spectrogram(samples, sr)
        # Return simplified features
        return np.mean(np.log(Sxx + 1e-9), axis=1)[:13]  # 13 coefficients
    except Exception:
        return np.zeros(13, dtype=np.float32)


def _generate_speaker_embedding(samples: np.ndarray, sr: int = 16000) -> list[float]:
    """Generate a speaker embedding vector from audio.
    
    In production, this would use a proper speaker encoder model like:
    - SpeakerNet
    - X-Vector (Kaldi)
    - Resemblyzer (based on VoxCeleb)
    """
    # Stub: generate random but deterministic embedding based on audio
    # In production, use model inference
    try:
        # Use audio statistics as a simple embedding
        embedding = []
        for chunk_size in [512, 1024, 2048]:
            for i in range(0, len(samples) - chunk_size, chunk_size):
                chunk = samples[i:i+chunk_size]
                embedding.append(float(np.mean(chunk)))
                embedding.append(float(np.std(chunk)))
        
        # Normalize to reasonable length (e.g., 192-dim for speaker embeddings)
        if len(embedding) > 192:
            embedding = embedding[:192]
        else:
            embedding.extend([0.0] * (192 - len(embedding)))
        
        return embedding[:192]
    except Exception:
        return [0.0] * 192


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
    translator = IndicTrans2Translator()
    voice_cloner = XttsVoiceCloner()

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
        if voice_cloner.available():
            return True
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
        # delete=False, closed before reuse: NamedTemporaryFile's own handle
        # holds an exclusive lock on Windows, so reopening the same path via
        # upload.save() while that handle is still open raises PermissionError
        # there (POSIX allows it, which is why this only shows up on Windows).
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            upload.save(tmp_path)
            try:
                samples = audio_util.decode(tmp_path, ASR_SAMPLE_RATE)
            except audio_util.AudioError as exc:
                return jsonify({"error": "decode_failed", "message": str(exc)}), 415
            container_duration = audio_util.probe_duration(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

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
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
            try:
                upload.save(tmp_path)
                try:
                    samples = audio_util.decode(tmp_path, ASR_SAMPLE_RATE)
                except audio_util.AudioError as exc:
                    return jsonify({"error": "decode_failed", "message": str(exc)}), 415
                duration = duration or audio_util.probe_duration(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
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

    # -------------------------------------------------------- voice model (RVC)

    @app.post("/voice-model/rvc")
    def put_voice_model():
        """Cache a promoted RVC model for a project so the Pro tier can serve
        it. The backend uploads this after training passes the eval gate."""
        project_id = request.form.get("project_id") or ""
        if not project_id:
            return jsonify({"error": "missing_project_id"}), 400
        model = request.files.get("model")
        if model is None:
            return jsonify({"error": "missing_model"}), 400
        index = request.files.get("index")
        voice_rvc.store_model(
            project_id,
            model.read(),
            index.read() if index is not None else None,
        )
        return jsonify({
            "project_id": project_id,
            "version": request.form.get("version") or "",
            "cached": True,
        })

    @app.get("/voice-model/rvc/<project_id>")
    def get_voice_model(project_id: str):
        return jsonify({
            "project_id": project_id,
            "cached": voice_rvc.has_model(project_id),
        })

    # ------------------------------------------------------------ translation

    @app.post("/translate")
    def translate():
        """Translate text segments to the target language.

        Uses IndicTrans2 (en<->hi) for real translation; "hinglish" is Hindi
        plus a loanword substitution pass so it reads the way people actually
        speak rather than textbook Hindi. Falls back to passthrough (with a
        warning) if the translator can't load, so a model-server hiccup never
        turns into a 500 the caller has to handle specially.
        """
        payload = request.get_json(silent=True) or {}
        project_id = str(payload.get("project_id") or "")
        segments = [str(s) for s in (payload.get("segments") or [])]
        source_language = str(payload.get("source_language") or "en")
        target_language = str(payload.get("target_language") or "en")

        warning = None
        try:
            translations = translator.translate(segments, source_language, target_language)
        except Exception as exc:  # model load/inference failures shouldn't 500 the caller
            log.exception("translation failed for %s (%s -> %s)", project_id, source_language, target_language)
            translations = list(segments)
            warning = str(exc)[:200]

        return jsonify({
            "project_id": project_id,
            "source_language": source_language,
            "target_language": target_language,
            "translations": translations,
            "confidence": 0.0 if warning else 0.9,
            "metadata": {
                "translator": translator.name,
                "warning": warning,
            }
        })

    # ------------------------------------------------------------ voice profile extraction

    @app.post("/voice-profile/extract")
    def extract_voice_profile():
        """Extract speaker voice profile and embeddings from audio.
        
        Analyzes the uploaded audio to extract:
        - Voice embedding (for voice cloning)
        - Pitch information (mean, std, range)
        - Spectral features (MFCC, formants)
        - Supported languages for synthesis
        """
        if "audio" not in request.files:
            return jsonify({
                "error": "missing_audio",
                "message": "audio file required"
            }), 400

        project_id = str(request.form.get("project_id") or "")
        if not project_id:
            return jsonify({
                "error": "missing_project_id",
                "message": "project_id required"
            }), 400

        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        audio_file.save(tmp_path)

        try:
            # Decode audio
            samples = audio_util.decode(tmp_path, sample_rate=16000)
            
            # Extract basic voice features
            # In production, use a proper speaker embedding model
            # Example: from resemblyzer import VoiceEncoder
            
            # Compute pitch
            pitch_mean, pitch_std, pitch_range = _extract_pitch_info(samples, sr=16000)
            
            # Compute spectral features (MFCC as proxy)
            mfcc = _extract_mfcc(samples, sr=16000)
            
            # Stub embedding (in production: use speaker encoder)
            embedding = _generate_speaker_embedding(samples, sr=16000)
            
            return jsonify({
                "project_id": project_id,
                "embedding": {
                    "model": "stub-speaker-encoder",
                    "embedding": embedding,
                    "speaker_id": f"speaker_{project_id[:8]}"
                },
                "pitch_info": {
                    "mean_hz": float(pitch_mean),
                    "std_dev": float(pitch_std),
                    "range_hz": float(pitch_range)
                },
                "spectral_features": {
                    "mfcc": mfcc.tolist() if hasattr(mfcc, 'tolist') else mfcc,
                    "energy": float(np.mean(np.abs(samples) ** 2))
                },
                "supported_languages": ["en", "hi", "hinglish"]
            })
        except Exception as exc:
            log.exception("voice profile extraction failed for %s", project_id)
            return jsonify({
                "error": "extraction_failed",
                "message": str(exc)[:200]
            }), 500
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------ voice-preserving synthesis

    @app.post("/synthesize/voice")
    def synthesize_voice():
        """Synthesize text using the project's extracted voice profile.
        
        This endpoint generates speech in a target language while preserving
        the speaker's voice characteristics. It can either:
        - Return the audio directly (type="audio")
        - Reference original audio (type="source") if text was already spoken
        """
        payload = request.get_json(silent=True) or {}
        project_id = str(payload.get("project_id") or "")
        text = str(payload.get("text") or "")
        target_language = str(payload.get("target_language") or "en")
        tier = str(payload.get("quality") or DEFAULT_TIER)

        if not project_id:
            return jsonify({
                "error": "missing_project_id",
                "message": "project_id required"
            }), 400

        if not text.strip():
            return jsonify({
                "type": "audio",
                "data": None,
                "word": text
            })

        profile = store.get(project_id)
        if profile is None:
            return jsonify({
                "error": "voice_profile_missing",
                "project_id": project_id,
                "message": "re-seed the profile via POST /voice-profile and retry",
            }), 409

        try:
            # One unified path: the chosen tier's engine chain regenerates the
            # whole segment as one cross-lingual utterance, falling through to
            # eSpeak so a dub segment is never silently dropped.
            produced = synthesizer.clone_phrase(profile, text, tier, target_language)
            if produced is None:
                return jsonify({
                    "type": "audio",
                    "data": None,
                    "word": text,
                    "error": "synthesis_failed",
                }), 500

            samples, sample_rate, engine = produced
            audio_bytes = audio_util.encode_wav(samples, sample_rate=sample_rate)
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
            # "cloned" = came from a real voice-cloning engine, not the eSpeak
            # formant fallback; the client uses this to warn on low fidelity.
            voice_cloned = engine not in ("espeak-ng",)
            return jsonify({
                "type": "audio",
                "data": audio_b64,
                "word": text,
                "sample_rate": sample_rate,
                "engine": engine,
                "voice_cloned": voice_cloned,
                "confidence": 0.9 if voice_cloned else 0.5,
            })
        except Exception as exc:
            log.exception("voice synthesis failed for %s", project_id)
            return jsonify({
                "error": "synthesis_failed",
                "message": str(exc)[:200]
            }), 500

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
            tier=str(payload.get("quality") or DEFAULT_TIER),
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

        # A batch is one render, so its tier is uniform; take it from the top
        # level, falling back to the first item for older callers.
        batch_tier = str(payload.get("quality") or (items[0].get("quality") if items else None) or DEFAULT_TIER)

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
                tier=str(item.get("quality") or batch_tier),
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
