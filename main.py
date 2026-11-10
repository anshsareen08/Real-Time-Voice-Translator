"""
NEC Real-Time Translator — entry point.

The main thread does almost nothing: it builds the queues and shared state,
starts each subsystem's thread, then drives the OpenCV display window.  All the
real work happens in six independent threads connected by queues:

    Video    : camera → face recognition → overlay
    Audio    : mic → Silero VAD → audio_q
    Whisper  : audio_q → transcript_q
    Translate: transcript_q → translation_q
    TTS      : translation_q → pcm_q   (Piper)
    Stream   : pcm_q → speaker / RTP

Each subsystem can be toggled in config.py (ENABLE_*) so you can bring up one
milestone at a time.  Press 'q' or ESC in the window (or Ctrl-C) to quit.
"""

import time
import signal

import cv2
import config
from utils.queues import Pipeline
from utils.state import SharedState
from utils.logging_utils import get_logger

log = get_logger("Main")


class App:
    def __init__(self) -> None:
        self.pipe = Pipeline()
        self.state = SharedState()
        self.components = []          # started, in start order (stopped in reverse)
        self.video = None
        self._running = False
        self._window_ready = False

        # Create the window immediately so it's responsive during model loading.
        if config.ENABLE_VIDEO and config.SHOW_WINDOW:
            cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            # Show a black placeholder frame while models load.
            blank = __import__('numpy').zeros(
                (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3),
                dtype=__import__('numpy').uint8,
            )
            cv2.putText(blank, "Loading models, please wait...",
                        (40, config.CAMERA_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 2)
            cv2.imshow(config.WINDOW_NAME, blank)
            cv2.waitKey(1)
            self._window_ready = True

    # ── helpers ───────────────────────────────────────────────────────
    def _pump_window(self) -> None:
        """Call between heavy loading steps to keep the window responsive."""
        if self._window_ready:
            cv2.waitKey(1)

    # ── build ────────────────────────────────────────────────────────
    def build(self) -> None:
        pipe, state = self.pipe, self.state

        # Consumers first, producers last, so nothing overflows at startup.
        tts = None
        if config.ENABLE_TTS:
            from tts.engine import TTS
            tts = TTS(pipe.translation, pipe.pcm, state)
            self._pump_window()

        if config.ENABLE_STREAM and tts is not None:
            from stream.ffmpeg import AudioStreamer
            # All engines resample to a fixed rate, so the streamer opens once.
            streamer = AudioStreamer(pipe.pcm, sample_rate=tts.sample_rate, state=state)
            self.components.append(("Stream", streamer))
            self._pump_window()

        if tts is not None:
            self.components.append(("TTS", tts))

        if config.ENABLE_TRANSLATE:
            from translate.translator import Translator
            self.components.append(("Translate", Translator(pipe.transcript, pipe.translation, state)))
            self._pump_window()

        if config.ENABLE_WHISPER:
            from whisper.engine import WhisperEngine
            self.components.append(("Whisper", WhisperEngine(pipe.audio, pipe.transcript, state)))
            self._pump_window()

        if config.ENABLE_AUDIO:
            from audio.capture import AudioCapture
            self.components.append(("Audio", AudioCapture(pipe.audio, state)))

        if config.ENABLE_VIDEO:
            from video.camera import VideoCapture
            self.video = VideoCapture(state, config.CAMERA_INDEX)

    # ── run ──────────────────────────────────────────────────────────
    def start(self) -> None:
        log.info("Starting subsystems (target language: %s)…", config.TARGET_LANGUAGE)
        for name, comp in self.components:
            log.info("→ starting %s", name)
            comp.start()
            self._pump_window()
        if self.video is not None:
            if not self.video.start():
                log.error("Video failed to start: %s", self.video.error)
                # Show error on the window instead of stale "Loading" text.
                if self._window_ready:
                    import numpy as np
                    err_frame = np.zeros(
                        (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8,
                    )
                    cv2.putText(err_frame, "Camera failed to open.",
                                (40, config.CAMERA_HEIGHT // 2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(err_frame, "Close other apps using the camera & restart.",
                                (40, config.CAMERA_HEIGHT // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                    cv2.imshow(config.WINDOW_NAME, err_frame)
                    cv2.waitKey(1)
                self.video = None
        self._running = True
        log.info("All subsystems running. Press 'q'/ESC in the window or Ctrl-C to quit.")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        log.info("Shutting down…")
        if self.video is not None:
            self.video.stop()
        for name, comp in reversed(self.components):
            try:
                comp.stop()
            except Exception as exc:
                log.error("Error stopping %s: %s", name, exc)
        cv2.destroyAllWindows()
        log.info("Bye.")

    # ── main-thread display loop ─────────────────────────────────────
    def loop(self) -> None:
        if not self._window_ready and self.video is None:
            # Truly headless: just idle until interrupted.
            while self._running:
                time.sleep(0.2)
            return

        last_frame = None
        while self._running:
            if self.video is not None:
                frame = self.video.get_frame()
                if frame is not None and frame is not last_frame:
                    cv2.imshow(config.WINDOW_NAME, frame)
                    last_frame = frame
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):     # q or ESC
                break
        cv2.destroyAllWindows()


def main() -> None:
    app = App()

    def _sigint(_sig, _frm):
        app._running = False
    signal.signal(signal.SIGINT, _sigint)

    app.build()
    app.start()
    try:
        app.loop()
    finally:
        app.stop()


if __name__ == "__main__":
    main()
