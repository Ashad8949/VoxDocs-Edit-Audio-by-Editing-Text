import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run the full Django backend pipeline verification against a live stack."

    def add_arguments(self, parser):
        parser.add_argument("--api", default=os.environ.get("VOXDOCS_API_URL", "http://localhost:3000"))
        parser.add_argument("--file", default="")
        parser.add_argument("--timeout", type=float, default=600.0)

    def handle(self, *args, **options):
        base = options["api"].rstrip("/")
        timeout = options["timeout"]
        sample = options["file"]
        sentence = (
            "Four score and seven years ago our fathers brought forth on this continent "
            "a new nation, conceived in liberty and dedicated to the proposition that "
            "all men are created equal."
        )

        self.stdout.write(f"VoxDocs Django verification against {base}\n")

        health = self._api(base, "/api/health", method="GET")
        self._check("api is reachable", True)
        self._check("model server is reachable", health.get("status") == "ok", health.get("modelError") or "")
        if health.get("status") != "ok":
            raise RuntimeError("the model server is not available")

        if not sample:
            with tempfile.TemporaryDirectory(prefix="voxdocs-verify-") as tmpdir:
                sample_path = Path(tmpdir) / "sample.wav"
                subprocess.run(["espeak-ng", "-w", str(sample_path), sentence], check=True)
                self._verify_stack(base, sample_path, timeout)
            return

        self._verify_stack(base, Path(sample), timeout)

    def _verify_stack(self, base: str, sample_path: Path, timeout: float):
        created = self._api(base, "/api/projects", method="POST", files={"file": (sample_path.name, sample_path.read_bytes(), "application/octet-stream")}, data={"name": "verify"})
        project_id = created.get("project", {}).get("id")
        self._check("upload accepted", bool(project_id), str(project_id or ""))
        if not project_id:
            raise RuntimeError("upload failed")

        deadline = time.time() + timeout
        while True:
            project = self._api(base, f"/api/projects/{project_id}", method="GET").get("project", {})
            if project.get("status") in {"ready", "failed"}:
                break
            if time.time() > deadline:
                raise TimeoutError("timed out waiting for transcription")
            time.sleep(1.5)

        self._check("transcription finished", project.get("status") == "ready", project.get("error") or "")
        if project.get("status") != "ready":
            raise RuntimeError(project.get("error") or "transcription failed")

        words = project.get("transcript", {}).get("words", [])
        self._check("transcript has word-level timings", len(words) > 5, f"{len(words)} words")
        self._check(
            "word timings are monotonic and inside the media",
            all(
                w.get("start", 0) < w.get("end", 0) and (idx == 0 or w.get("start", 0) >= words[idx - 1].get("start", 0))
                for idx, w in enumerate(words)
            ) and len(words) > 0 and words[-1].get("end", 0) <= float(project.get("duration", 0)) + 0.05,
        )

        years_index = next((idx for idx, word in enumerate(words) if "years".lower() in (word.get("text") or "").lower()), -1)
        self._check("found the word to cut at", years_index > 0, f"index {years_index}")
        kept = [{"ref": word["id"]} for word in words[years_index:]]

        plan = self._api(base, f"/api/projects/{project_id}/plan", method="POST", json={"tokens": kept})
        self._check(
            "plan reports the deletion",
            plan.get("stats", {}).get("deletedWords", 0) == years_index,
            f"{plan.get('stats', {}).get('deletedWords', 0)} deleted",
        )
        self._check(
            "plan predicts a shorter result",
            plan.get("stats", {}).get("estimatedDuration", 0) < plan.get("stats", {}).get("sourceDuration", 0),
            f"{plan.get('stats', {}).get('estimatedDuration', 0):.2f}s of {plan.get('stats', {}).get('sourceDuration', 0):.2f}s",
        )

        queued = self._api(base, f"/api/projects/{project_id}/render", method="POST", json={"tokens": kept, "format": "wav"})
        render = queued.get("render", {})
        self._check("render accepted", bool(render.get("id")))
        render_id = render.get("id")
        if not render_id:
            raise RuntimeError("queueing render failed")

        render_deadline = time.time() + timeout
        while True:
            status = self._api(base, f"/api/projects/{project_id}/renders/{render_id}/status", method="GET").get("render", {})
            if status.get("status") in {"ready", "failed"}:
                render = status
                break
            if time.time() > render_deadline:
                raise TimeoutError("timed out waiting for render")
            time.sleep(0.7)

        self._check("render succeeded", render.get("status") == "ready", render.get("error") or "")
        if render.get("status") != "ready":
            raise RuntimeError(render.get("error") or "render failed")

        download = render.get("downloadUrl")
        if not download:
            raise RuntimeError("render has no download URL")

        response = requests.get(f"{base}{download}", timeout=60)
        response.raise_for_status()
        output = Path(tempfile.mkdtemp(prefix="voxdocs-render-")) / "rendered.wav"
        output.write_bytes(response.content)

        verdict = self._transcribe(base, output)
        self._check("the cut words are gone from the audio", "four score" not in verdict.lower(), verdict[:70])
        self._check("the kept words survive in the audio", "liberty" in verdict.lower(), verdict[:70])

        inserted = self._api(base, f"/api/projects/{project_id}/render", method="POST", json={"tokens": [{"insert": "246"}, *kept], "format": "wav"})
        insert_render = inserted.get("render", {})
        insert_id = insert_render.get("id")
        self._check("insertion rendered", bool(insert_id))
        if insert_id:
            deadline = time.time() + timeout
            while True:
                status = self._api(base, f"/api/projects/{project_id}/renders/{insert_id}/status", method="GET").get("render", {})
                if status.get("status") in {"ready", "failed"}:
                    insert_render = status
                    break
                if time.time() > deadline:
                    raise TimeoutError("timed out waiting for insertion render")
                time.sleep(0.7)
            self._check("the inserted word was synthesised, not dropped", insert_render.get("status") == "ready", insert_render.get("error") or "")
            insert_response = requests.get(f"{base}{insert_render.get('downloadUrl')}", timeout=60)
            insert_response.raise_for_status()
            insert_out = Path(tempfile.mkdtemp(prefix="voxdocs-insert-")) / "insert.wav"
            insert_out.write_bytes(insert_response.content)
            insert_verdict = self._transcribe(base, insert_out)
            self._check("the inserted word is audible in the result", any(token in insert_verdict.lower() for token in ("246", "two hundred", "two forty")), insert_verdict[:70])

        self._api(base, f"/api/projects/{project_id}", method="DELETE")
        self._check("project deleted", True)

        self.stdout.write(self.style.SUCCESS("\nAll Django verification checks passed."))

    def _transcribe(self, base: str, audio_path: Path):
        with audio_path.open("rb") as fh:
            payload = {"file": (audio_path.name, fh.read(), "application/octet-stream"), "name": "verify"}
        created = self._api(base, "/api/projects", method="POST", files=payload)
        project_id = created.get("project", {}).get("id")
        if not project_id:
            raise RuntimeError("verification upload failed")

        deadline = time.time() + 300
        while True:
            project = self._api(base, f"/api/projects/{project_id}", method="GET").get("project", {})
            if project.get("status") == "ready":
                transcript = project.get("transcript", {}).get("words", [])
                text = " ".join(word.get("text", "") for word in transcript)
                self._api(base, f"/api/projects/{project_id}", method="DELETE")
                return text
            if project.get("status") == "failed":
                raise RuntimeError(project.get("error") or "verification transcription failed")
            if time.time() > deadline:
                raise TimeoutError("verification transcribe timed out")
            time.sleep(1.2)

    @staticmethod
    def _check(label: str, ok: bool, detail: str = ""):
        status = "ok" if ok else "FAIL"
        if detail:
            print(f"  {status}  {label} — {detail}")
        else:
            print(f"  {status}  {label}")
        if not ok:
            raise AssertionError(label)

    @staticmethod
    def _api(base: str, path: str, method: str = "GET", **kwargs):
        url = f"{base}{path}"
        response = requests.request(method, url, timeout=60, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {response.status_code} {body.get('message') or body.get('error') or response.text}")
        return body
