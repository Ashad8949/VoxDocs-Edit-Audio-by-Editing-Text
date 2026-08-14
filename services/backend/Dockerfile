# The Django backend. Runs as either the web tier or a Celery worker depending
# on the command, so both always share exactly one image and one set of
# dependencies.
FROM python:3.11-slim

# ffmpeg does all decoding, cutting and encoding.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DJANGO_SETTINGS_MODULE=voxdocs.settings
ENV PORT=3000
ENV VOXDOCS_DATA_DIR=/data

EXPOSE 3000
VOLUME ["/data"]

# Migrations run on start so a fresh deploy is self-sufficient. For multi-replica
# rollouts, move this to an init container or a release job so only one pod
# migrates at a time.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Long uploads and long renders both need a generous timeout; the real work is
# on the worker, but ingest still streams a large file in.
CMD ["gunicorn", "--bind", "0.0.0.0:3000", \
     "--workers", "3", "--threads", "4", \
     "--timeout", "600", "--graceful-timeout", "30", \
     "voxdocs.wsgi:application"]
