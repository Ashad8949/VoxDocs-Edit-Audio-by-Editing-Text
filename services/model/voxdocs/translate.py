"""IndicTrans2-backed translation for the en<->hi pair VoxDocs supports.

Uses AI4Bharat's distilled 200M models directly through `transformers` rather
than their official `IndicTransToolkit`, because that package ships a Cython
extension that needs a C++ compiler to build. This reimplements the parts of
its preprocessing that matter for plain conversational text (Devanagari
normalization, trivial tokenization, the literal FLORES-200 language-tag
prefix the model was trained on) and skips the entity-masking step (numbers,
URLs, emails), which will only be missed on transcripts that contain a lot of
those.

Model load is expensive and not thread-safe, so it happens once, lazily, the
same way asr.py loads Whisper.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)


def _ensure_onnx_shim() -> None:
    """IndicTrans2's `trust_remote_code` config module imports
    `transformers.onnx`, a submodule dropped in newer `transformers`
    releases. It only defines an ONNX-export config class we never use
    (we only call `.generate()`), so a bare stub satisfies the import
    without pulling in the real thing or downgrading `transformers`.
    """
    import sys
    import types

    if "transformers.onnx" in sys.modules:
        return

    class OnnxConfig:
        default_fixed_batch = 2
        default_fixed_sequence = 8

    class OnnxSeq2SeqConfigWithPast(OnnxConfig):
        pass

    def compute_effective_axis_dimension(*args, **kwargs):
        raise NotImplementedError("ONNX export is not supported; this stub only unblocks import")

    stub = types.ModuleType("transformers.onnx")
    stub.__path__ = []  # marks it as a package so `transformers.onnx.utils` resolves
    stub.OnnxConfig = OnnxConfig
    stub.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast

    utils_stub = types.ModuleType("transformers.onnx.utils")
    utils_stub.compute_effective_axis_dimension = compute_effective_axis_dimension
    stub.utils = utils_stub

    sys.modules["transformers.onnx"] = stub
    sys.modules["transformers.onnx.utils"] = utils_stub

# FLORES-200 codes IndicTrans2 was trained on.
_FLORES = {"en": "eng_Latn", "hi": "hin_Deva"}

# Common English loanwords an urban Hindi speaker actually says, in place of
# the formal/Sanskritized word a translation model produces by default. This
# is what turns a plain Hindi translation into something that reads as
# Hinglish; anything this table misses is a manual per-segment edit away in
# the TranslationEditor.
_HINGLISH_LOANWORDS = {
    "सुविधा": "फीचर",
    "अद्यतन": "अपडेट",
    "समस्या": "प्रॉब्लम",
    "संदेश": "मैसेज",
    "दूरभाष": "फोन",
    "जालस्थल": "वेबसाइट",
    "अंतरजाल": "इंटरनेट",
    "संगणक": "कंप्यूटर",
    "अनुप्रयोग": "ऐप",
    "उपयोगकर्ता": "यूज़र",
    "व्यक्तिगत": "पर्सनल",
    "मूल रूप से": "बेसिकली",
    "वास्तव में": "actually",
    "स्पष्ट रूप से": "obviously",
    "निश्चित रूप से": "definitely",
    "विशेष रूप से": "स्पेशली",
}


def to_hinglish(text: str) -> str:
    """Swap formal Hindi terms for the English loanword most Indian speakers
    would actually use. A heuristic default, not a real code-mixing model.
    """
    for formal, loan in _HINGLISH_LOANWORDS.items():
        text = text.replace(formal, loan)
    return text


class IndicTrans2Translator:
    """Lazily-loaded en<->hi translator using AI4Bharat's distilled models."""

    name = "indictrans2"

    def __init__(self, device: str | None = None) -> None:
        self.device = device or os.environ.get("VOXDOCS_TRANSLATE_DEVICE", "cpu")
        self._bundles: dict[str, tuple] = {}
        self._normalizers: dict[str, object] = {}
        self._moses_tok = None
        self._moses_detok = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------ loading

    def _load_direction(self, direction: str):
        """direction is 'en-indic' or 'indic-en'."""
        bundle = self._bundles.get(direction)
        if bundle is not None:
            return bundle
        with self._lock:
            bundle = self._bundles.get(direction)
            if bundle is not None:
                return bundle
            _ensure_onnx_shim()
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = (
                "ai4bharat/indictrans2-en-indic-dist-200M"
                if direction == "en-indic"
                else "ai4bharat/indictrans2-indic-en-dist-200M"
            )
            log.info("loading %s on %s", model_name, self.device)
            hf_token = os.environ.get("HF_TOKEN") or None
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
            model = model.to(self.device).eval()
            bundle = (tokenizer, model)
            self._bundles[direction] = bundle
            return bundle

    def _indic_normalizer(self, lang: str):
        norm = self._normalizers.get(lang)
        if norm is None:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            norm = IndicNormalizerFactory().get_normalizer(lang)
            self._normalizers[lang] = norm
        return norm

    def _moses(self):
        if self._moses_tok is None:
            import sacremoses
            self._moses_tok = sacremoses.MosesTokenizer(lang="en")
            self._moses_detok = sacremoses.MosesDetokenizer(lang="en")
        return self._moses_tok, self._moses_detok

    # --------------------------------------------------------- pre/post

    def _preprocess(self, text: str, lang: str) -> str:
        text = " ".join(text.split())
        if lang == "hi":
            from indicnlp.tokenize import indic_tokenize
            normalized = self._indic_normalizer("hi").normalize(text)
            return " ".join(indic_tokenize.trivial_tokenize(normalized, "hi"))
        tok, _ = self._moses()
        return " ".join(tok.tokenize(text, escape=False))

    def _postprocess(self, text: str, lang: str) -> str:
        if lang == "hi":
            # Devanagari punctuation reads fine space-joined; the model's own
            # spacing around danda/comma is already close enough without a
            # dedicated indic detokenizer.
            return text.replace(" ।", "।").replace(" ,", ",").replace(" ?", "?").strip()
        _, detok = self._moses()
        return detok.detokenize(text.split())

    # -------------------------------------------------------------- run

    def _run(self, direction: str, texts: list[str], src: str, tgt: str) -> list[str]:
        import torch

        tokenizer, model = self._load_direction(direction)
        tagged = [f"{_FLORES[src]} {_FLORES[tgt]} {self._preprocess(t, src)}" for t in texts]
        inputs = tokenizer(tagged, truncation=True, padding="longest", return_tensors="pt").to(self.device)
        with torch.no_grad():
            # use_cache=False: this custom (trust_remote_code) model predates
            # transformers' Cache-class KV cache and indexes it as a legacy
            # tuple, which crashes on the newer default. Recomputing attention
            # each step is slower but correct, and fine for an async job.
            generated = model.generate(**inputs, use_cache=False, min_length=0, max_length=256, num_beams=5)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [self._postprocess(d, tgt) for d in decoded]

    def translate(self, texts: list[str], source_language: str, target_language: str) -> list[str]:
        if not texts:
            return []

        src = source_language if source_language in _FLORES else "hi"
        # "hinglish" is not a model target; it is Hindi plus a loanword pass
        # applied after generation (see to_hinglish).
        model_target = "hi" if target_language == "hinglish" else target_language

        if src == model_target or model_target not in _FLORES:
            translated = list(texts)
        elif src == "en" and model_target == "hi":
            translated = self._run("en-indic", texts, "en", "hi")
        elif src != "en" and model_target == "en":
            translated = self._run("indic-en", texts, "hi", "en")
        else:
            # No direct model for indic-indic pairs (e.g. hi -> ta); pass through.
            translated = list(texts)

        if target_language == "hinglish":
            translated = [to_hinglish(t) for t in translated]
        return translated
