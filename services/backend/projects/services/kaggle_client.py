"""Thin wrapper over the Kaggle public API, for GPU training orchestration.

Kaggle's free GPU runs inside Kernels, not an SSH box, so we drive training the
way the platform supports: version a Dataset holding the speaker's audio, push a
GPU Kernel (a generated Python script) that reads that dataset and writes model
artifacts to its output, poll it to completion, and pull the output back.

All credentials come from ~/.kaggle/kaggle.json (or the KAGGLE_USERNAME /
KAGGLE_KEY environment variables). Nothing here logs the token.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


class KaggleError(RuntimeError):
    """The Kaggle API could not be reached or refused the request."""


_api = None
_username: str | None = None


def _client():
    """Authenticate once and cache the client. Raises KaggleError if the
    credentials are missing so callers can surface a clear setup message."""
    global _api, _username
    if _api is None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            _username = api.get_config_value(api.CONFIG_NAME_USER)
        except Exception as exc:  # noqa: BLE001 - want a single clear failure
            raise KaggleError(
                "Kaggle authentication failed — place your token at "
                "~/.kaggle/kaggle.json (Account → Create New API Token). "
                f"Underlying error: {exc}"
            ) from exc
        _api = api
    return _api


def username() -> str:
    _client()
    return _username or ""


def dataset_push(folder: Path, slug: str, title: str) -> str:
    """Create the dataset, or add a new version if it already exists.

    `folder` must contain the data files; we write the required
    dataset-metadata.json into it. Returns the ``user/slug`` reference.
    """
    api = _client()
    ref = f"{username()}/{slug}"
    (folder / "dataset-metadata.json").write_text(
        json.dumps({"title": title[:50], "id": ref, "licenses": [{"name": "CC0-1.0"}]}),
        encoding="utf-8",
    )
    try:
        if _dataset_exists(ref):
            api.dataset_create_version(str(folder), version_notes="voxdocs update", dir_mode="zip")
        else:
            api.dataset_create_new(str(folder), dir_mode="zip", public=False)
    except Exception as exc:  # noqa: BLE001
        raise KaggleError(f"dataset push failed: {exc}") from exc
    return ref


def _dataset_exists(ref: str) -> bool:
    api = _client()
    try:
        owner, slug = ref.split("/", 1)
        for d in api.dataset_list(user=owner, search=slug):
            if str(d) == ref:
                return True
    except Exception:  # noqa: BLE001 - treat lookup failure as "not found"
        return False
    return False


def kernel_push(script_path: Path, slug: str, title: str, dataset_ref: str,
                enable_gpu: bool = True, enable_internet: bool = True) -> str:
    """Push a script kernel that reads `dataset_ref`. Returns ``user/slug``."""
    api = _client()
    ref = f"{username()}/{slug}"
    workdir = script_path.parent
    (workdir / "kernel-metadata.json").write_text(
        json.dumps({
            "id": ref,
            "title": title[:50],
            "code_file": script_path.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": enable_gpu,
            "enable_internet": enable_internet,
            "dataset_sources": [dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
        }),
        encoding="utf-8",
    )
    try:
        resp = api.kernels_push(str(workdir))
    except Exception as exc:  # noqa: BLE001
        raise KaggleError(f"kernel push failed: {exc}") from exc
    # Use the ref Kaggle actually assigned (it may normalise the slug) rather
    # than the one we constructed, so status/output target the right kernel.
    return getattr(resp, "ref", None) or ref


def kernel_status(ref: str) -> str:
    """Current kernel run status as a lowercase string."""
    api = _client()
    try:
        res = api.kernels_status(ref)
    except Exception as exc:  # noqa: BLE001
        raise KaggleError(f"kernel status failed: {exc}") from exc
    status = getattr(res, "status", res)
    return str(status).lower()


def kernel_wait(ref: str, poll_seconds: float = 30.0, timeout_seconds: float = 3600.0) -> str:
    """Block until the kernel finishes. Returns 'complete' or 'error'.

    A freshly pushed kernel isn't queryable for a few seconds, so transient
    status errors early on are treated as "not ready yet" rather than fatal.
    """
    start = time.monotonic()
    consecutive_errors = 0
    while True:
        try:
            status = kernel_status(ref)
            consecutive_errors = 0
        except KaggleError:
            consecutive_errors += 1
            # Tolerate the post-push registration lag; only give up if the
            # error persists well beyond it.
            if consecutive_errors > 6:
                raise
            time.sleep(poll_seconds)
            continue
        if "complete" in status:
            return "complete"
        if "error" in status or "cancel" in status:
            return "error"
        if time.monotonic() - start > timeout_seconds:
            raise KaggleError(f"kernel {ref} timed out (last status: {status})")
        time.sleep(poll_seconds)


def kernel_output(ref: str, dest: Path) -> None:
    """Download the kernel's output files into `dest`."""
    api = _client()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_output(ref, str(dest))
    except Exception as exc:  # noqa: BLE001
        raise KaggleError(f"kernel output pull failed: {exc}") from exc
