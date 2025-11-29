"""Camera helpers for streaming frames from a USB device."""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2


class CameraNotReadyError(Exception):
    """Raised when a frame is requested before the camera is ready."""


class USBCameraStream:
    """Continuously captures frames from a USB camera on a background thread."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return

        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            raise RuntimeError("Unable to open USB camera")

        # Configure resolution + FPS to stay in the VGA / low-latency envelope.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            self._thread = None

        if self._capture:
            self._capture.release()
            self._capture = None

        with self._frame_lock:
            self._latest_frame = None

    def _reader_loop(self) -> None:
        assert self._capture is not None
        wait_time = 1.0 / max(self.fps, 1)
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.1)
                continue

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            with self._frame_lock:
                self._latest_frame = buffer.tobytes()

            time.sleep(wait_time)

    def get_frame(self) -> bytes:
        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            raise CameraNotReadyError("Camera warming up")

        return frame
