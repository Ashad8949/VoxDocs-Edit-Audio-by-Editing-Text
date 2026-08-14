#!/bin/sh
# Apply migrations before serving, but only for the web tier: a Celery worker
# starting at the same moment must not race the web pod for the schema lock.
set -e

case "$1" in
  gunicorn|python)
    echo "[entrypoint] applying migrations"
    python manage.py migrate --noinput
    ;;
esac

exec "$@"
