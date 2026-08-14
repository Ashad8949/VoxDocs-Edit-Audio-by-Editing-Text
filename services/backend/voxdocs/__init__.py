"""VoxDocs backend.

Importing the Celery app here ensures @shared_task decorators bind to it as soon
as Django starts, whether the process is a web worker or a Celery worker.
"""

from .celery import app as celery_app

__all__ = ("celery_app",)
