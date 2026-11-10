"""
Central configuration for the NEC Real-Time Translator.

Everything tuneable lives here.  Each subsystem (video / audio / whisper /
translate / tts / stream) reads only the constants it needs, so you can flip a
single ENABLE_* flag to bring a subsystem up or down in isolation — which is
exactly how you verify one milestone at a time.
"""

import os

# Windows: Hugging Face's cache uses symlinks, which need Developer Mode or
# admin.  Silence the repeated warning (the real fix is enabling Developer Mode).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
FACES_DIR  = os.path.join(DATA_DIR, "faces")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Master subsystem switches ────────────────────────────────────────
# Turn pieces off to test milestones independently.
ENABLE_VIDEO  = True     # camera + face recognition + overlay window
ENABLE_AUDIO  = True     # microphone + VAD
ENABLE_WHISPER = True    # speech-to-text
ENABLE_TRANSLATE = True  # machine translation
ENABLE_TTS    = True     # Piper text-to-speech
ENABLE_STREAM = True     # audio output (local speaker and/or RTP)

# ── Language routing ─────────────────────────────────────────────────
# Source is auto-detected by Whisper.  Everything is translated INTO this:
TARGET_LANGUAGE = "Japanese"     # must be a key in LANGUAGE_CODES below
# If you want a fixed source instead of auto-detect, set e.g. "English".
SOURCE_LANGUAGE = "English"

# ── Video / Camera ───────────────────────────────────────────────────
CAMERA_INDEX        = 0
CAMERA_WIDTH        = 1280
CAMERA_HEIGHT       = 720
VIDEO_TARGET_FPS    = 30
SHOW_WINDOW         = True       # cv2 display window (main thread)
WINDOW_NAME         = "ESA Live Translator"

# ── Face Detection / Recognition (OpenCV) ────────────────────────────
FACE_DETECTION_SCALE_FACTOR  = 1.3
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE      = (80, 80)
FACE_RECOGNITION_THRESHOLD   = 85        # LBPH distance: lower = better match
FACE_SAMPLE_SIZE             = (200, 200)
FACE_CAPTURE_COUNT           = 20        # frames captured during registration

# ── Audio capture ────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE   = 16_000             # 16 kHz mono — required by Whisper + Silero
AUDIO_CHANNELS      = 1
VAD_FRAME_SIZE      = 512                 # Silero needs 512 samples @ 16 kHz (32 ms)
MIC_MUTE_COOLDOWN_S = 1.0                # ignore mic this long after TTS stops (echo guard)

# ── Voice Activity Detection (Silero) ────────────────────────────────
VAD_BACKEND             = "silero"       # "silero" | "energy"
VAD_THRESHOLD           = 0.5            # Silero speech probability threshold
VAD_MIN_SPEECH_MS       = 350           # discard blips shorter than this
VAD_MIN_SILENCE_MS      = 600           # silence that ends an utterance
VAD_SPEECH_PAD_MS       = 150           # padding kept around each segment
VAD_MAX_SPEECH_S        = 15.0          # force-flush a very long utterance
ENERGY_VAD_THRESHOLD    = 0.015          # only used when VAD_BACKEND == "energy"

# ── Whisper (Speech-to-Text) ─────────────────────────────────────────
WHISPER_MODEL_SIZE   = "small"          # tiny | base | small | medium | large-v3
WHISPER_DEVICE       = "auto"            # auto | cpu | cuda
WHISPER_COMPUTE_TYPE = "int8"            # auto | int8 | float16 | float32
WHISPER_BEAM_SIZE    = 5

# ── Translation (NLLB-200) ───────────────────────────────────────────
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ── Text-to-Speech (multi-engine, routed by language) ────────────────
# Everything is translated INTO the target language, so the TTS stage only
# ever speaks one language per run — TTS_ROUTES picks the engine for it:
#   • Japanese        → Style-Bert-VITS2 (most natural JA; also EN/ZH)
#   • English / Hindi → Piper           (fast local; Piper has no JA voice)
#
# Each engine has its own native sample rate, so every output is resampled to
# OUTPUT_SAMPLE_RATE before reaching the stream — the streamer opens once.
TTS_ROUTES = {
    "Japanese": "sbv2",
    "English":  "piper",
    "Hindi":    "piper",
    # "Chinese":  "sbv2",
}
OUTPUT_SAMPLE_RATE = 48_000              # common rate fed to speaker / RTP

# Piper voices (download with: python scripts/download_piper_voice.py <name>)
PIPER_VOICES = {
    "English":   os.path.join(MODELS_DIR, "piper", "en_US-amy-medium.onnx"),
    "Hindi":     os.path.join(MODELS_DIR, "piper", "hi_IN-pratham-medium.onnx"),
}

# Style-Bert-VITS2 (download with: python scripts/download_sbv2_model.py)
SBV2_DEVICE      = "cuda"                 # auto | cpu | cuda
SBV2_ASSETS_DIR  = os.path.join(MODELS_DIR, "sbv2", "model_assets")
SBV2_MODEL_NAME  = "jvnv-F1-jp"          # default voice from litagin/style_bert_vits2_jvnv
SBV2_MODEL_FILE  = "jvnv-F1-jp_e160_s14000.safetensors"
SBV2_CONFIG_FILE = "config.json"
SBV2_STYLE_FILE  = "style_vectors.npy"
SBV2_STYLE       = "Neutral"             # speaking style baked into the model
# BERT models each language needs (auto-downloaded from Hugging Face on first use)
SBV2_BERT = {
    "Japanese": "ku-nlp/deberta-v2-large-japanese-char-wwm",
    "English":  "microsoft/deberta-v3-large",
    # "Chinese":  "hfl/chinese-roberta-wwm-ext-large",
}

# ── Audio output / streaming ─────────────────────────────────────────
# "local" → play through the default speaker (most reliable for a demo)
# "rtp"   → push to an RTP endpoint via FFmpeg (open in VLC / OBS)
# "both"  → do both
STREAM_MODE = "both"     # "rtp" = VLC only; "both" = VLC + your speaker
RTP_HOST      = "127.0.0.1"
RTP_PORT      = 5004
RTP_CODEC     = "libopus"                # libopus | pcm_mulaw | aac
FFMPEG_BIN    = "ffmpeg"                 # full path if not on PATH
# One SDP per target language so each config gets its own file for VLC,
# e.g. stream_Japanese.sdp, stream_Hindi.sdp (no overwriting between runs).
SDP_OUTPUT    = os.path.join(BASE_DIR, f"stream_{TARGET_LANGUAGE}.sdp")

# ── Queues ───────────────────────────────────────────────────────────
# Small + drop-oldest = stay near real-time.  On slow (CPU) hardware this means
# when you fall behind, stale utterances are dropped instead of queued up and
# replayed — you always hear the MOST RECENT speech, not a backlog.
AUDIO_QUEUE_MAX      = 2
TRANSCRIPT_QUEUE_MAX = 2
TRANSLATION_QUEUE_MAX = 2
PCM_QUEUE_MAX        = 2

# ── Logging ──────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"                       # DEBUG | INFO | WARNING | ERROR

# ── Language Registry ────────────────────────────────────────────────
# name → codes for Whisper, NLLB, and a human flag.
LANGUAGE_CODES = {
    "Japanese":  {"whisper": "ja", "nllb": "jpn_Jpan", "flag": "JP"},
    "English":   {"whisper": "en", "nllb": "eng_Latn", "flag": "EN"},
    # "Chinese":   {"whisper": "zh", "nllb": "zho_Hans", "flag": "CN"},
    # "Korean":    {"whisper": "ko", "nllb": "kor_Hang", "flag": "KR"},
    "Hindi":     {"whisper": "hi", "nllb": "hin_Deva", "flag": "IN"},
    #"Spanish":   {"whisper": "es", "nllb": "spa_Latn", "flag": "ES"},
    # "French":    {"whisper": "fr", "nllb": "fra_Latn", "flag": "FR"},
    # "German":    {"whisper": "de", "nllb": "deu_Latn", "flag": "DE"},
    # "Portuguese":{"whisper": "pt", "nllb": "por_Latn", "flag": "PT"},
    # "Arabic":    {"whisper": "ar", "nllb": "arb_Arab", "flag": "AR"},
}


def whisper_to_name(code: str) -> str:
    """Map a Whisper language code ('ja') back to a human name ('Japanese')."""
    for name, codes in LANGUAGE_CODES.items():
        if codes["whisper"] == code:
            return name
    return code
