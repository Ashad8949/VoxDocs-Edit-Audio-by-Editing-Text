"""Django settings for the VoxDocs backend."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)

ALLOWED_HOSTS = [h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",") if h]
CSRF_TRUSTED_ORIGINS = [
    o for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "projects",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Transport security. Enabled whenever DEBUG is off, since every real deployment
# terminates TLS at the ingress; set VOXDOCS_INSECURE=1 for the rare case of
# running a non-debug build over plain HTTP.
SECURE_ENABLED = not DEBUG and not env_bool("VOXDOCS_INSECURE", False)
if SECURE_ENABLED:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = env_int("DJANGO_HSTS_SECONDS", 31536000)
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    # TLS ends at the ingress, so trust its forwarded scheme rather than
    # redirect-looping on a request that is already secure.
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

ROOT_URLCONF = "voxdocs.urls"
WSGI_APPLICATION = "voxdocs.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


def database_from_url(url: str) -> dict:
    """Minimal DATABASE_URL parser, so no extra dependency is needed."""
    parsed = urlparse(url)
    if parsed.scheme in ("postgres", "postgresql"):
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": unquote(parsed.path.lstrip("/")),
            "USER": unquote(parsed.username or ""),
            "PASSWORD": unquote(parsed.password or ""),
            "HOST": parsed.hostname or "",
            "PORT": str(parsed.port or ""),
            "CONN_MAX_AGE": 600,
        }
    if parsed.scheme == "sqlite":
        return {"ENGINE": "django.db.backends.sqlite3", "NAME": parsed.path or ":memory:"}
    raise ValueError(f"unsupported DATABASE_URL scheme: {parsed.scheme!r}")


_database_url = os.environ.get("DATABASE_URL")
DATABASES = {
    "default": database_from_url(_database_url) if _database_url else {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "data" / "db.sqlite3",
        # Long transcription writes must not trip over a reader.
        "OPTIONS": {"timeout": 30},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------- media

MEDIA_ROOT = Path(os.environ.get("VOXDOCS_DATA_DIR", BASE_DIR / "data" / "media"))
MEDIA_URL = "/media/"

# Uploads are whole media files. Streaming them to a temporary file rather than
# buffering in memory is the only workable choice at this size.
FILE_UPLOAD_MAX_MEMORY_SIZE = 4 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 32 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FIELDS = 5000
FILE_UPLOAD_TEMP_DIR = os.environ.get("VOXDOCS_TMP_DIR") or None

# ------------------------------------------------------------- voxdocs

VOXDOCS = {
    "MODEL_URL": os.environ.get("VOXDOCS_MODEL_URL", "http://localhost:8000").rstrip("/"),
    # Transcribing a long file is a single request; the ceiling is generous.
    "MODEL_TIMEOUT": env_float("VOXDOCS_MODEL_TIMEOUT", 30 * 60),
    "MAX_UPLOAD_BYTES": env_int("VOXDOCS_MAX_UPLOAD_MB", 1024) * 1024 * 1024,
    # Everything renders at one rate so segments concatenate sample-exactly.
    "RENDER_SAMPLE_RATE": env_int("VOXDOCS_RENDER_RATE", 48000),
    # Long enough to kill clicks at a seam, short enough to be inaudible.
    "SEAM_FADE": env_float("VOXDOCS_SEAM_FADE", 0.008),
    "MAX_SEGMENTS_PER_PASS": env_int("VOXDOCS_MAX_SEGMENTS_PER_PASS", 400),
    "FFMPEG": os.environ.get("VOXDOCS_FFMPEG", "ffmpeg"),
    "FFPROBE": os.environ.get("VOXDOCS_FFPROBE", "ffprobe"),
    "ENVELOPE_FPS": 100,
}

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
    ],
    "EXCEPTION_HANDLER": "projects.views.exception_handler",
    "UNAUTHENTICATED_USER": None,
}
if not DEBUG:
    REST_FRAMEWORK["DEFAULT_RENDERER_CLASSES"] = ["rest_framework.renderers.JSONRenderer"]

# ----------------------------------------------------------------- cors

CORS_ALLOW_ALL_ORIGINS = env_bool("VOXDOCS_CORS_ALLOW_ALL", DEBUG)
CORS_ALLOWED_ORIGINS = [o for o in os.environ.get("VOXDOCS_CORS_ORIGINS", "").split(",") if o]

# --------------------------------------------------------------- celery

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", False)
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_ACKS_LATE = True
# A worker that prefetches long jobs starves its siblings; take one at a time.
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_TIME_LIMIT = env_int("CELERY_TASK_TIME_LIMIT", 3 * 3600)
CELERY_TASK_SOFT_TIME_LIMIT = env_int("CELERY_TASK_SOFT_TIME_LIMIT", 3 * 3600 - 60)
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    "root": {"handlers": ["console"], "level": os.environ.get("DJANGO_LOG_LEVEL", "INFO")},
}
