"""Celery entry point.

Transcription and rendering are minutes-long, CPU-bound jobs. Running them in a
web worker would tie up a request slot and lose the job on any restart, so they
go through a broker where they can be retried, observed, and scaled separately.
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "voxdocs.settings")

app = Celery("voxdocs")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
