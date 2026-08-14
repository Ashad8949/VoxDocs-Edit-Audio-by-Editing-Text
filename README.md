# VoxDocs — edit audio by editing text

Delete a word from the transcript and it disappears from the recording. Type a
new word and it is spoken in the same voice. Works on video too.

The premise: finding a mistake in an hour of audio is slow, and cutting it out
cleanly is slower. Reading is fast. So the transcript becomes the timeline, and
every edit to the text is compiled into an edit on the waveform.

```
                        ┌──────────────────────┐
┌──────────┐            │  Django + DRF        │        ┌──────────────────┐
│  React   │──/api─────▶│  projects, ORM,      │──HTTP─▶│  Model server    │
│  editor  │◀───────────│  transcripts, media  │◀───────│  ASR + synthesis │
└──────────┘            └──────────┬───────────┘        └──────────────────┘
                                   │ Redis
                        ┌──────────▼───────────┐
                        │  Celery workers      │
                        │  transcode, render   │
                        └──────────┬───────────┘
                                   │
                             media volume
```

Three tiers, because their appetites differ. The model server wants a warm model
resident in memory and as much CPU (or GPU) as it can get. The workers want CPU
for ffmpeg. The web tier wants sockets and does almost no work. Separating them
lets each scale on its own signal, and stops a transcription backlog from
starving the request path.

---

## What actually happens when you delete a word

This is the interesting part, so it is worth being precise.

**1. Transcription produces word-level timings.** Not sentence timings — every
word gets a `start` and an `end`. Every cut the system ever makes is derived
from these, so a backend that cannot produce them cannot drive the editor.

**2. The editor keeps word identity.** The transcript is *not* a text box. Each
word is a block carrying the id that ties it to its span of samples. A
`contenteditable` would hand back a flat string on every keystroke, throwing
away the very mapping the renderer needs. Deleting sets a flag rather than
removing the block, which is what makes deletion reversible and reviewable.

**3. The edit compiles to an Edit Decision List.** Surviving runs of words
become `copy` segments holding a source time range; typed text becomes `synth`
segments holding the text and its neighbours. Consecutive words that were
already adjacent in the source collapse into a single uninterrupted segment —
so an edit far away never disturbs audio near you.

**4. Cuts land in silence, not mid-phoneme.** ASR word boundaries mark roughly
where the vowel energy is. Cutting exactly on `word.end` clips the release of a
final consonant and produces the chopped sound that gives naive text-based
editing away. Instead VoxDocs cuts at the *midpoint of the gap* between words,
bounded so deleting a word never drags a long silence along with it, and
optionally snaps to the quietest point nearby using the audio's energy envelope.
Snapping is conservative on purpose: it only moves a cut when there is a clearly
quieter place to put it, and prefers the smallest movement.

**5. Seams get a few milliseconds of fade.** Butt-joining two unrelated pieces
of a waveform steps the signal discontinuously and clicks. An 8 ms fade on each
side removes it, well below audibility. Only real seams are faded — the outer
edges of the render keep their natural attack and decay.

**6. The whole thing renders as one ffmpeg graph,** passed via
`-filter_complex_script` because a heavily edited hour becomes a filter graph
longer than the OS argument limit. Very large edits fall back to batched passes
that are joined afterwards.

### Synthesising words that were never said

Three strategies, in order:

**Unit selection from the speaker's own recording.** If the words you typed were
already said somewhere in the file, the best possible synthesis is the speaker
actually saying them. Longest-match n-gram selection keeps naturally
coarticulated runs intact — type "seven years ago" and if that phrase exists it
is lifted as *one* piece, not three splices. Candidates are ranked by whether
their neighbours match your context, by recognition confidence, and by how
close their duration is to the speaker's normal rate.

This costs no model, no GPU and no download, and the result is indistinguishable
from the surrounding audio because it *is* the surrounding audio. Crucially,
these come back from the model server as a *plan* — a time range — not as audio,
so the renderer lifts them from the same full-quality master as every other
segment rather than from a resampled copy shipped over HTTP.

**Neural voice cloning.** For words never spoken, the PaddleSpeech stack is
wired up: a GE2E speaker embedding conditions a FastSpeech2 acoustic model,
vocoded by Parallel WaveGAN. See *Backend status* below for what this requires.

**Formant-matched fallback.** With no neural model installed, eSpeak NG
synthesises the words and the result is pitch- and level-matched to the speaker
(median F0 is measured by autocorrelation, then corrected with an
`asetrate`/`atempo` shift). It does not sound like them and is not meant to: it
keeps the edit audible and reviewable instead of silently dropping words. Words
that nothing can produce are reported as warnings, never quietly discarded.

### Video

The same EDL drives the picture. Copy segments trim video exactly as they trim
audio. Inserted speech has no picture to accompany it, so the preceding shot is
frozen for the length of the insertion. Freezing is the honest choice: it keeps
sound and picture in sync and makes the edit visible, rather than pretending the
speaker's lips match words they never said.

---

## Running it

### Docker (everything, one command)

```bash
docker compose up --build      # then open http://localhost:8080
```

The first build bakes the ASR weights into the model image, so pods start warm.

### Locally, for development

Prerequisites: **ffmpeg** and **ffprobe** on `PATH`, Python 3.11+, Node 20+, and
a Redis. `espeak-ng` is optional but enables the fallback synthesiser.

```bash
redis-server &                             # the Celery broker

# model server
cd services/model
pip install -r requirements.txt
python -m voxdocs.app                      # :8000

# Django backend
cd services/backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:3000    # :3000

# Celery worker, in another shell, same directory
celery -A voxdocs worker -l info

# editor
npm install && npm run dev:web             # :5173
```

Open <http://localhost:5173>.

By default the backend uses SQLite, which is fine for one process. Set
`DATABASE_URL=postgres://user:pass@host/db` for anything with more than one
replica.

### Using it

Drop in an audio or video file and wait for the transcript. Then:

| Action | How |
| --- | --- |
| Select a word | Click it |
| Select a phrase | Drag across it, or shift-click the far end |
| Cut the selection | <kbd>Delete</kbd> / <kbd>Backspace</kbd> |
| Add new speech | Click the gap between two words, then type |
| Replace words | Select them and start typing |
| Restore a cut | Select the struck-through words, press Restore |
| Undo / redo | <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Z</kbd>, add <kbd>Shift</kbd> to redo |
| Play / pause | <kbd>Space</kbd> |
| Jump to a word | Double-click it |

Cut regions are painted onto the waveform, and the header tracks what the
result will be before you render it.

---

## Backend status

Being explicit about what has been run, since "supported" and "verified" are
different claims:

| Component | Status |
| --- | --- |
| `faster-whisper` ASR (word timings) | **Verified** — default backend |
| Unit selection from the speaker's voice | **Verified** |
| eSpeak NG fallback, pitch/level matched | **Verified** |
| ffmpeg render pipeline, audio and video | **Verified** |
| Django + DRF + Celery/Redis end to end | **Verified** |
| Silero ASR | Implemented, not run here — needs `torch`; timings are coarser than Whisper's |
| PaddleSpeech voice cloning | Implemented, not run here — see below |

PaddleSpeech ships pretrained voice-cloning weights for Chinese; English needs
an acoustic model trained on an English corpus, with the checkpoint supplied via
`VOXDOCS_PADDLE_AM` / `VOXDOCS_PADDLE_VOC`. It is off unless
`VOXDOCS_ENABLE_PADDLE=1`, and the adapter is written against the documented
`TTSExecutor` / `VectorExecutor` API but has not been executed in this
environment. Treat it as a wired-up integration point, not a tested path.

Other current limits: **one speaker per file** (multi-speaker needs
diarisation before the transcript is built), and no background-music separation
— cutting a word cuts whatever else was in that moment too.

---

## Tests

```bash
cd services/backend && python -m pytest    # EDL, render pipeline, API, Celery
cd services/model   && python -m pytest    # ASR, unit selection, DSP
npm test                                   # the editor's document model
node scripts/e2e.mjs                       # against a running stack
```

The unit suites cover the parts that are easy to get subtly wrong: the diff and
alignment, cut-point placement, and the render graph. The render tests assert on
actual samples — they build a master of distinct tones and check by frequency
that the right slices survived, and measure the waveform across a seam to prove
it is continuous.

`scripts/e2e.mjs` goes further and asserts on meaning: it edits a real recording
through the real API, then **transcribes the rendered output** to confirm the
deleted words are gone and the inserted ones are audible. A pipeline can pass
every unit test and still emit silence.

```
ok   the cut words are gone from the audio — years ago, our farmers brought forth…
ok   the inserted word is audible in the result — 246 years ago our father brought…
ok   repeated words are lifted from the speaker's own recording — 5/5 from the voice bank
```

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/projects` | Upload media; returns `202`, transcription runs on a worker |
| `GET` | `/api/projects/:id` | Project, status and transcript |
| `GET` | `/api/projects/:id/envelope?points=N` | Waveform, peak-downsampled |
| `GET` | `/api/projects/:id/media` | Source media, with range requests |
| `POST` | `/api/projects/:id/plan` | Cost and duration of an edit, without rendering |
| `POST` | `/api/projects/:id/render` | Queue a render; returns `202` |
| `GET` | `/api/projects/:id/renders/:rid/status` | Poll a queued render |
| `GET` | `/api/projects/:id/renders/:rid` | Download a finished render |
| `DELETE` | `/api/projects/:id` | Delete the project and its media |

Both long operations return `202` and are polled, because transcribing or
re-rendering an hour of audio takes minutes and no browser request should be
held open for that.

Edits are posted either as `tokens` — `[{ref: "w12"}, {insert: "246"}]`, what the
editor sends, exact and unambiguous — or as plain `text`, which is aligned
against the original transcript with a Myers diff (patience-anchored for large
documents) to recover which words survived. The editor uses the first; pastes
and scripts use the second.

---

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | SQLite | Postgres URL; required for more than one replica |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Broker for the long jobs |
| `DJANGO_SECRET_KEY` | dev key | **Must** be set in production |
| `DJANGO_DEBUG` | `1` | Turn off in production |
| `VOXDOCS_MODEL_URL` | `http://localhost:8000` | Where the backend finds the model server |
| `VOXDOCS_DATA_DIR` | `data/media` | Project and media storage |
| `VOXDOCS_MAX_UPLOAD_MB` | `1024` | Upload ceiling |
| `VOXDOCS_RENDER_RATE` | `48000` | Canonical render sample rate |
| `VOXDOCS_SEAM_FADE` | `0.008` | Fade length at each seam, seconds |
| `VOXDOCS_WHISPER_MODEL` | `base.en` | Whisper size: `tiny.en` … `large-v3` |
| `VOXDOCS_ASR_BACKEND` | `auto` | `faster-whisper`, `silero`, or `auto` |
| `VOXDOCS_ASR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `VOXDOCS_VOICE_BANK` | `1` | Unit selection from the speaker's own audio |
| `VOXDOCS_ENABLE_PADDLE` | `0` | Enable PaddleSpeech voice cloning |
| `VOXDOCS_PROFILE_TTL` | `3600` | Voice-profile cache lifetime, seconds |

---

## Deploying

```bash
kubectl apply -f k8s/
```

Manifests cover all four tiers with their own HPAs and disruption budgets. A few
choices worth knowing about:

- **Migrations run as a Job, not on pod start.** Otherwise a multi-replica
  rollout has several pods racing for the same schema lock.
- **Web and worker share one image**, differing only in command, so they cannot
  drift apart in dependencies or configuration.
- **Readiness probes hit `/api/ready`, which does not force a model load.**
  Probing an endpoint that triggers inference makes every rollout stall behind
  the first request.
- **Voice profiles are a cache, never state.** Django owns the durable
  transcript; the model server only memoises it. So a model pod can be evicted,
  restarted or scaled at any moment, and the worst case is a single re-seed,
  which the backend performs transparently on a `409`.
- **Workers ack late and get a 300 s grace period**, so a render in flight
  finishes rather than being killed mid-file — and if it is killed anyway, the
  job is redelivered rather than lost.

Media lives on a `ReadWriteMany` claim, since both the web tier and every worker
read and write it. On a cluster without an RWX storage class, move media to
object storage.

---

## Credit

Based on VoxDocs by Daniel Zeng, Tony Sun and Andrew Gaut. This is an
independent implementation of the system described in their talk, and follows
its architecture: separate model and application servers, ASR with word
timings, and GE2E-conditioned FastSpeech2 voice cloning for inserted words.

MIT licensed.
