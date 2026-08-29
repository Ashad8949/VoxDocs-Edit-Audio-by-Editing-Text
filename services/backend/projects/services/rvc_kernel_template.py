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

WORK = "/kaggle/working"
EXP = os.path.join(WORK, "exp")
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
sep_dir = os.path.join(WORK, "sep")
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

# --- 3. RVC training -------------------------------------------------------
# Canonical RVC-Project flow: preprocess -> extract F0/features -> train ->
# build faiss index. CLI arg shapes vary by RVC fork; iterate against logs.
RVC_DIR = os.path.join(WORK, "rvc")
model_out = os.path.join(WORK, "model.pth")
index_out = os.path.join(WORK, "added.index")
trained = False
try:
    # No --recurse-submodules: the repo's realtime-VST submodule drags in a
    # huge tree of C++ audio-plugin SDKs we don't need (and which blows up the
    # clone). The Python training scripts live in the main repo's infer/ tree,
    # which the glob below locates.
    subprocess.run(
        f"git clone --depth 1 "
        f"https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI {RVC_DIR}",
        shell=True, check=True)
    os.chdir(RVC_DIR)
    print("RVC repo top-level:", os.listdir("."), flush=True)

    # The repo layout drifts between revisions, so locate the training entry
    # points by name instead of hard-coding paths; skip RVC gracefully (the
    # eval below still runs) if this revision doesn't ship them.
    def find(name):
        hits = glob.glob(f"./**/{name}", recursive=True)
        return hits[0] if hits else None

    preprocess = find("preprocess.py")
    extract_f0 = find("extract_f0_rmvpe.py")
    extract_feat = find("extract_feature_print.py")
    train_py = find("train.py")
    print("RVC scripts:", preprocess, extract_f0, extract_feat, train_py, flush=True)

    if all([preprocess, extract_f0, extract_feat, train_py]):
        req = find("requirements.txt")
        if req:
            subprocess.run(f"{sys.executable} -m pip install -q -r {req}", shell=True)
        for asset, url in [
            ("assets/hubert/hubert_base.pt",
             "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"),
            ("assets/rmvpe/rmvpe.pt",
             "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"),
        ]:
            if not os.path.exists(asset):
                os.makedirs(os.path.dirname(asset), exist_ok=True)
                subprocess.run(f"wget -q -O {asset} {url}", shell=True)

        EXP_NAME = "voxdocs_" + PROJECT_ID
        sr_tag = "40k" if SR == 40000 else "48k"
        sh(f"{sys.executable} {preprocess} {train_dir} {SR} 2 ./logs/{EXP_NAME} False 3.0")
        sh(f"{sys.executable} {extract_f0} 1 0 0 ./logs/{EXP_NAME} True")
        sh(f"{sys.executable} {extract_feat} cuda:0 1 0 ./logs/{EXP_NAME} v2 True")
        sh(f"{sys.executable} {train_py} -e {EXP_NAME} -sr {sr_tag} "
           f"-f0 1 -bs 8 -te {EPOCHS} -se 50 -sw 1 -v v2 -l 0 -c 0")
        idx = find("train_index.py")
        if idx:
            subprocess.run(f"{sys.executable} {idx} {EXP_NAME} v2", shell=True)

        for src in (glob.glob(f"./assets/weights/{EXP_NAME}*.pth")
                    + sorted(glob.glob(f"./logs/{EXP_NAME}/G_*.pth"))):
            shutil.copy(src, model_out); trained = True; break
        for src in glob.glob(f"./logs/{EXP_NAME}/added_*.index"):
            shutil.copy(src, index_out); break
    else:
        print("RVC training scripts not found in this revision — skipping RVC; "
              "eval will still report standard (zero-shot) similarity.", flush=True)
except Exception as e:
    print("RVC training step failed:", repr(e), flush=True)
finally:
    os.chdir(WORK)

# --- 4. Evaluation gate: speaker similarity --------------------------------
# Content-independent speaker embeddings (Resemblyzer); cosine similarity of a
# held-out REAL clip vs (a) zero-shot XTTS and (b) trained RVC output.
from resemblyzer import VoiceEncoder, preprocess_wav
enc = VoiceEncoder(DEVICE)


def emb(path):
    return enc.embed_utterance(preprocess_wav(path))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

metrics = {"epochs": EPOCHS, "sample_rate": SR, "trained": trained}
try:
    from TTS.api import TTS
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    xtts_out = os.path.join(WORK, "xtts_out.wav")
    tts.tts_to_file(text=EVAL_TEXT, speaker_wav=os.path.join(train_dir, "speaker.wav"),
                    language="en", file_path=xtts_out)
    real_e = emb(real_holdout)
    metrics["standard_similarity"] = round(cos(real_e, emb(xtts_out)), 4)

    # RVC conversion of the XTTS output, if a model was produced.
    if trained and os.path.exists(model_out):
        try:
            sh(f"{sys.executable} -m pip install -q rvc-python")
            from rvc_python.infer import RVCInference
            rvc = RVCInference(model_path=model_out,
                               index_path=index_out if os.path.exists(index_out) else "")
            rvc_out = os.path.join(WORK, "rvc_out.wav")
            rvc.infer_file(xtts_out, rvc_out)
            metrics["pro_similarity"] = round(cos(real_e, emb(rvc_out)), 4)
        except Exception as e:
            print("RVC eval inference failed:", e, flush=True)
            metrics["pro_error"] = str(e)[:300]
except Exception as e:
    print("eval failed:", e, flush=True)
    metrics["eval_error"] = str(e)[:300]

with open(os.path.join(WORK, "metrics.json"), "w") as fh:
    json.dump(metrics, fh)
print("METRICS", json.dumps(metrics), flush=True)
'''
