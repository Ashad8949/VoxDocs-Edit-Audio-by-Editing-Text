"""In-memory voice-profile cache.

The model server is deliberately stateless-ish: profiles are a cache, never the
source of truth. The API server owns the durable copy of every transcript, so a
model pod can be evicted, restarted or scaled out and the worst outcome is one
re-upload. That is what lets the two services scale independently, which was
the point of splitting them.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from .synth import VoiceProfile


class ProfileStore:
    """LRU cache of voice profiles with a TTL and a memory ceiling."""

    def __init__(self, max_entries: int = 64, ttl_seconds: float = 3600.0,
                 max_audio_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.max_audio_bytes = max_audio_bytes
        self._entries: OrderedDict[str, tuple[VoiceProfile, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _audio_bytes(self) -> int:
        return sum(
            profile.samples.nbytes
            for profile, _ in self._entries.values()
            if profile.samples is not None
        )

    def _evict_locked(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, stamp) in self._entries.items() if now - stamp > self.ttl_seconds]
        for key in expired:
            self._entries.pop(key, None)

        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

        # Under memory pressure, drop cached waveforms before dropping profiles:
        # unit selection only needs timings, so a profile without audio is still
        # fully useful for the common path.
        while self._audio_bytes() > self.max_audio_bytes and self._entries:
            for key in list(self._entries.keys()):
                profile, stamp = self._entries[key]
                if profile.samples is not None:
                    profile.samples = None
                    break
            else:
                break

    def put(self, profile: VoiceProfile) -> None:
        with self._lock:
            self._entries.pop(profile.project_id, None)
            self._entries[profile.project_id] = (profile, time.monotonic())
            self._evict_locked()

    def get(self, project_id: str) -> VoiceProfile | None:
        with self._lock:
            entry = self._entries.get(project_id)
            if entry is None:
                return None
            profile, stamp = entry
            if time.monotonic() - stamp > self.ttl_seconds:
                self._entries.pop(project_id, None)
                return None
            self._entries.move_to_end(project_id)
            self._entries[project_id] = (profile, time.monotonic())
            return profile

    def drop(self, project_id: str) -> bool:
        with self._lock:
            return self._entries.pop(project_id, None) is not None

    def stats(self) -> dict:
        with self._lock:
            return {
                "profiles": len(self._entries),
                "max_entries": self.max_entries,
                "audio_bytes": self._audio_bytes(),
                "max_audio_bytes": self.max_audio_bytes,
                "ttl_seconds": self.ttl_seconds,
            }
