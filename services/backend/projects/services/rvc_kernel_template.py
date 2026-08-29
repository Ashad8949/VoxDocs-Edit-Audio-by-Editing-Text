"""Builds the Kaggle GPU training kernel for a per-speaker RVC model.

`build_kernel_script(...)` returns a standalone Python script that Kaggle runs
on a GPU. It follows the canonical RVC-Project training flow and, crucially for
the evaluation gate, measures speaker similarity of a zero-shot XTTS clone vs
the trained RVC conversion against a held-out slice of the real speaker — the
`standard → pro` number the app promotes on and shows in the UI.

The script is deliberately self-contained (installs its own deps, no repo-local
imports) because it executes in Kaggle's environment, not ours. It reads its
one input dataset from /kaggle/input and writes artifacts to /kaggle/working:
``model.pth``, ``added.index``, and ``metrics.json``.

This is the one artifact that runs on hardware we can't test here, so it is
written to be read and iterated against real kernel logs — the normal way ML
training infra is shaken out.
"""

from __future__ import annotations

# A short, speaker-neutral line used to compare voices content-independently.
EVAL_TEXT = "This is a short sample used to measure how closely the voice matches."


def build_kernel_script(project_id: str, dataset_ref: str,
                        epochs: int = 100, sample_rate: int = 40000) -> str:
    """Render the kernel source. `dataset_ref` is 'user/slug'; the audio lands
    at /kaggle/input/<slug>/speaker.wav."""
    dataset_slug = dataset_ref.split("/", 1)[-1]
    replacements = {
        "__PROJECT_ID__": project_id,
        "__DATASET_SLUG__": dataset_slug,
        "__EPOCHS__": str(int(epochs)),
        "__SR__": str(int(sample_rate)),
        "__EVAL_TEXT__": EVAL_TEXT.replace('"', "'"),
    }
    script = _TEMPLATE
    for token, value in replacements.items():
        script = script.replace(token, value)
    return script


_TEMPLATE = r'''
"""VoxDocs RVC training kernel (auto-generated). Runs on Kaggle GPU."""
import json, os, subprocess, sys, glob, shutil

PROJECT_ID = "__PROJECT_ID__"
DATASET_SLUG = "__DATASET_SLUG__"
EPOCHS = __EPOCHS__
SR = __SR__
EVAL_TEXT = "__EVAL_TEXT__"

WORK = "/kaggle/working"      # only small final artifacts go here (it's the output)
SCRATCH = "/tmp/voxdocs"      # heavy intermediates (repo, separation, features)
os.makedirs(SCRATCH, exist_ok=True)
EXP = os.path.join(SCRATCH, "exp")
os.makedirs(EXP, exist_ok=True)


def sh(cmd):
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find_input_audio():
    for root in ("/kaggle/input",):
        for f in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
            return f
    raise SystemExit("no input .wav found under /kaggle/input")


# --- 1. Dependencies -------------------------------------------------------
sh(f"{sys.executable} -m pip install -q demucs resemblyzer soundfile librosa faiss-cpu")
sh(f"{sys.executable} -m pip install -q coqui-tts")

import soundfile as sf
import numpy as np
import librosa

raw = find_input_audio()
print("input audio:", raw, flush=True)

# Kaggle often assigns a P100 (compute capability sm_60), which the container's
# default PyTorch no longer ships kernels for — every CUDA call then fails with
# "no kernel image is available". The light models (Demucs, Resemblyzer, the
# eval) run fine on CPU, so we pin them to CPU and reserve the GPU for RVC
# training, which handles its own device.
DEVICE = "cpu"

# --- 2. Data prep: vocal separation (Demucs, CPU) --------------------------
# Isolate the speaker's voice from any music/background so RVC trains clean.
sep_dir = os.path.join(SCRATCH, "sep")
os.makedirs(sep_dir, exist_ok=True)
try:
    sh(f"{sys.executable} -m demucs -d cpu --two-stems vocals -o {sep_dir} \"{raw}\"")
    vocals = glob.glob(os.path.join(sep_dir, "**", "vocals.wav"), recursive=True)
    clean = vocals[0] if vocals else raw
except Exception as e:
    print("demucs failed, using raw audio:", e, flush=True)
    clean = raw

# Standardise to mono at the training sample rate.
y, _ = librosa.load(clean, sr=SR, mono=True)
# Hold out the last 20% of real speech for an honest, unseen eval clip.
split = int(len(y) * 0.8)
train_y, holdout_y = y[:split], y[split:]
train_dir = os.path.join(EXP, "audio")
os.makedirs(train_dir, exist_ok=True)
sf.write(os.path.join(train_dir, "speaker.wav"), train_y, SR)
real_holdout = os.path.join(WORK, "real_holdout.wav")
sf.write(real_holdout, holdout_y, SR)

# --- 3. Baseline eval (before RVC touches torch) ---------------------------
# Zero-shot XTTS clone vs a held-out REAL clip, compared with content-agnostic
# speaker embeddings (Resemblyzer). Done first so the RVC step's torch reinstall
# can't disturb this number.
from resemblyzer import VoiceEncoder, preprocess_wav
enc = VoiceEncoder("cpu")


def emb(path):
    return enc.embed_utterance(preprocess_wav(path))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

metrics = {"epochs": EPOCHS, "sample_rate": SR, "trained": False}
speaker_wav = os.path.join(train_dir, "speaker.wav")
xtts_out = os.path.join(WORK, "xtts_out.wav")
try:
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=EVAL_TEXT, speaker_wav=speaker_wav, language="en", file_path=xtts_out)
    real_e = emb(real_holdout)
    metrics["standard_similarity"] = round(cos(real_e, emb(xtts_out)), 4)
except Exception as e:
    print("baseline eval failed:", repr(e), flush=True)
    metrics["eval_error"] = str(e)[:300]

# --- 4. RVC training -------------------------------------------------------
# Invocations mirror the repo's own webui.py exactly (train/ layout). The repo
# ships a cu118 requirements set whose torch supports the P100 (sm_60), which
# the default container torch does not — installing it is what makes GPU
# training work here.
RVC_DIR = os.path.join(SCRATCH, "rvc")
model_out = os.path.join(WORK, "model.pth")
index_out = os.path.join(WORK, "added.index")
trained = False
try:
    subprocess.run(
        f"git clone --depth 1 "
        f"https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI {RVC_DIR}",
        shell=True, check=True)
    os.chdir(RVC_DIR)
    print("RVC repo top-level:", os.listdir("."), flush=True)

    # P100-compatible torch first: the cu118 wheels bundle their own cudnn, so
    # we avoid the repo requirements' pinned nvidia-cudnn-cu11 (not on PyPI).
    sh(f"{sys.executable} -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    # Install the rest of the repo's deps, but strip the torch/cudnn pins that
    # would either override our cu118 torch or fail to resolve — a single bad
    # pin in a requirements file aborts the whole install otherwise.
    req = "requirments_cu118_py312.txt"
    if os.path.exists(req):
        subprocess.run(
            "grep -viE 'nvidia-cudnn|^torch==|^torchaudio==|^torchvision==' "
            f"{req} > /tmp/rvc_req.txt", shell=True)
        subprocess.run(f"{sys.executable} -m pip install -q -r /tmp/rvc_req.txt", shell=True)
    # ffmpeg-python is what infer/audio.py imports as `ffmpeg`; make sure it's in.
    sh(f"{sys.executable} -m pip install -q ffmpeg-python")
    # The training scripts import the repo's top-level `infer` package, so the
    # repo root must be on PYTHONPATH (python puts the script's own dir first).
    os.environ["PYTHONPATH"] = RVC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

    base = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
    for asset, url in [
        ("assets/hubert/hubert_base.pt", f"{base}/hubert_base.pt"),
        ("assets/rmvpe/rmvpe.pt", f"{base}/rmvpe.pt"),
        ("assets/pretrained_v2/f0G40k.pth", f"{base}/pretrained_v2/f0G40k.pth"),
        ("assets/pretrained_v2/f0D40k.pth", f"{base}/pretrained_v2/f0D40k.pth"),
    ]:
        os.makedirs(os.path.dirname(asset), exist_ok=True)
        if not os.path.exists(asset):
            subprocess.run(f"wget -q -O {asset} {url}", shell=True)

    EXP_NAME = "voxdocs_" + PROJECT_ID
    sr_tag = "40k" if SR == 40000 else "48k"
    exp_path = os.path.join(RVC_DIR, "logs", EXP_NAME)
    # Exact CLI from the repo's webui.py (new ./train/ layout).
    sh(f'{sys.executable} train/preprocess.py "{train_dir}" {SR} 2 "{exp_path}" False 3.0')
    sh(f'{sys.executable} train/dataset/extract_f0.py cpu "{exp_path}" 2 rmvpe')
    sh(f'{sys.executable} train/dataset/extract_hubert_feature.py cpu 1 0 "{exp_path}" v2 False')
    sh(f'{sys.executable} train/train.py -e "{EXP_NAME}" -sr {sr_tag} -f0 1 -bs 8 -g 0 '
       f'-te {EPOCHS} -se 50 -pg assets/pretrained_v2/f0G40k.pth '
       f'-pd assets/pretrained_v2/f0D40k.pth -l 0 -c 0 -sw 1 -v v2')
    subprocess.run(f'{sys.executable} train/train_index.py "{EXP_NAME}" v2 "{exp_path}" 4 auto',
                   shell=True)

    for src in glob.glob(f"assets/weights/{EXP_NAME}*.pth"):
        shutil.copy(src, model_out); trained = True; break
    for src in glob.glob(os.path.join(exp_path, "added_*.index")):
        shutil.copy(src, index_out); break
    print("RVC trained:", trained, flush=True)
except Exception as e:
    print("RVC training step failed:", repr(e), flush=True)
finally:
    os.chdir(WORK)

metrics["trained"] = trained

# --- 5. Write results ------------------------------------------------------
# pro_similarity (RVC vs real) is measured on the serving side (the Python 3.11
# container), where RVC inference runs; the kernel's job is to produce the
# model and the zero-shot baseline.
with open(os.path.join(WORK, "metrics.json"), "w") as fh:
    json.dump(metrics, fh)
print("METRICS", json.dumps(metrics), flush=True)
'''
