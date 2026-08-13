# The model server. Sized for CPU inference; set VOXDOCS_ASR_DEVICE=cuda and use
# an nvidia base image to run it on a GPU instead.
FROM python:3.11-slim

# ffmpeg is the only system dependency: it does all decoding and resampling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY voxdocs ./voxdocs

# Bake the ASR weights into the image. Without this the first request after
# every pod start pays a model download, which makes autoscaling useless.
ARG WHISPER_MODEL=base.en
ENV VOXDOCS_WHISPER_MODEL=${WHISPER_MODEL}
ENV HF_HOME=/app/.cache
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# One worker per process with threads: the model is large and holding several
# copies in memory costs far more than it gains. Scale with replicas instead.
# The timeout is generous because transcribing a long file is a single request.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--workers", "1", "--threads", "4", \
     "--timeout", "1800", "--graceful-timeout", "30", \
     "voxdocs.app:app"]
