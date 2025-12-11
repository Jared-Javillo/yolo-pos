"""Camera helpers for streaming frames from a USB device."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import queue

import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO
try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

from . import config

# Configure logging
config.LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TTSAnnouncer:
    """Background TTS announcer using pyttsx3.

    Uses a worker thread and message queue so speaking doesn't block camera loop.
    """

    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Do not init pyttsx3 engine on the main thread; initialize on worker thread
        self._engine = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Put sentinel to unblock queue.get
        try:
            self._queue.put_nowait("")
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        if self._engine is not None:
            try:
                # Some engines require explicit stop/cleanup
                self._engine.stop()
            except Exception:
                pass

    def speak(self, text: str) -> None:
        # Non-blocking enqueue
        if not text:
            return
        logger.debug("Enqueue TTS: %s", text)
        try:
            self._queue.put_nowait(text)
        except Exception:
            pass

    def _worker(self) -> None:
        # Initialize engine on worker thread to avoid cross-thread issues
        if pyttsx3 is not None:
            try:
                self._engine = pyttsx3.init()
                # sensible defaults
                try:
                    self._engine.setProperty('rate', 150)
                    self._engine.setProperty('volume', 1.0)
                except Exception:
                    pass
                logger.debug("TTS engine initialized on worker thread")
            except Exception:
                logger.exception("Failed to initialize TTS engine on worker thread")
                self._engine = None

        while self._running:
            try:
                # Use timeout so we can exit promptly when stopping
                text = self._queue.get(timeout=0.5)
            except Exception:
                # timeout or queue error - loop again if still running
                continue

            if not self._running:
                break
            if not text:
                continue

            logger.debug("TTS speaking: %s", text)

            # If engine unavailable, just log the phrase
            if self._engine is None:
                logger.info("TTS would say: %s", text)
                continue

            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                logger.exception("TTS engine failed during speak; will attempt re-init")
                # Drop engine so we try to re-init on next loop
                try:
                    self._engine = None
                    if pyttsx3 is not None:
                        self._engine = pyttsx3.init()
                except Exception:
                    logger.exception("Re-init of TTS engine failed")

# Check GPU availability
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
else:
    logger.warning("Running on CPU only - No CUDA GPU detected")


class CameraNotReadyError(Exception):
    """Raised when a frame is requested before the camera is ready."""


class USBCameraStream:
    """Continuously captures frames from a USB camera on a background thread."""

    def __init__(
        self,
        device_index: int = config.CAMERA_DEVICE_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
        model_path: Optional[str] = None,
        inference_interval: int = config.INFERENCE_INTERVAL,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        cooldown_seconds: float = config.COOLDOWN_SECONDS,
        per_class_thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.inference_interval = inference_interval
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # YOLO detection tracking
        self._model: Optional[YOLO] = None
        self._frame_count = 0
        self._detection_lock = threading.Lock()
        self._last_detection_time: dict[str, float] = {}
        self._detection_queue: deque[dict] = deque(maxlen=config.DETECTION_QUEUE_MAX_SIZE)
        self._last_added_item: Optional[str] = None  # Track last item added to prevent consecutive duplicates
        self._per_class_thresholds: dict[str, float] = per_class_thresholds or {}
        # Running total for the current recording transaction
        self._transaction_total: int = 0
        self._transaction_items: list[dict] = []
        
        # Load YOLO model if path provided
        if model_path:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                self._model = YOLO(str(model_path_obj))
                # Force GPU if available
                if torch.cuda.is_available():
                    self._model.to('cuda')
                    logger.info("YOLO model loaded on GPU")
                else:
                    logger.info("YOLO model loaded on CPU")
            else:
                raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        # Hand gesture detection
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=config.GESTURE_STATIC_IMAGE_MODE,
            max_num_hands=config.GESTURE_MAX_NUM_HANDS,
            min_detection_confidence=config.GESTURE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.GESTURE_MIN_TRACKING_CONFIDENCE
        )
        self._recording_enabled = False
        self._gesture_history: deque[str] = deque(maxlen=config.GESTURE_HISTORY_MAX_SIZE)
        # Text-to-speech announcer (background worker)
        self._tts = TTSAnnouncer()

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
        # Start TTS worker
        try:
            self._tts.start()
        except Exception:
            logger.exception("Failed to start TTS announcer")
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        # Reset transaction state on start
        with self._detection_lock:
            self._transaction_total = 0
            self._transaction_items.clear()
            self._detection_queue.clear()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            self._thread = None

        if self._capture:
            self._capture.release()
            self._capture = None
        
        if self._hands:
            self._hands.close()

        # Stop TTS worker
        try:
            self._tts.stop()
        except Exception:
            logger.exception("Failed to stop TTS announcer")

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

            self._frame_count += 1
            
            # Detect hand gestures every frame for responsiveness
            gesture = self._detect_hand_gesture(frame)
            self._update_recording_state(gesture)
            
            # Draw recording status
            status_text = config.STATUS_TEXT_RECORDING if self._recording_enabled else config.STATUS_TEXT_PAUSED
            status_color = config.STATUS_COLOR_RECORDING if self._recording_enabled else config.STATUS_COLOR_PAUSED
            cv2.putText(frame, status_text, config.STATUS_POSITION, config.STATUS_FONT, 
                       config.STATUS_FONT_SCALE, status_color, config.STATUS_FONT_THICKNESS)
            
            # Run YOLO inference every Nth frame
            if self._model and self._frame_count % self.inference_interval == 0:
                frame = self._process_detections(frame)
            
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
    
    def _process_detections(self, frame):
        """Run YOLO inference and draw bounding boxes on frame."""
        results = self._model(frame, verbose=False)[0]
        current_time = time.time()
        
        for box in results.boxes:
            # Extract detection info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Check cooldown period first
            last_time = self._last_detection_time.get(class_name, 0)
            time_since_last = current_time - last_time
            in_cooldown = time_since_last < self.cooldown_seconds
            
            # Determine threshold (per-class override falls back to default)
            threshold = self._per_class_thresholds.get(class_name, self.confidence_threshold)
            # Check if confidence meets threshold
            meets_threshold = confidence >= threshold
            
            # Check if this is the same as the last added item (prevent consecutive duplicates)
            is_duplicate_of_last = class_name == self._last_added_item
            
            # Add to detection queue if meets threshold, not in cooldown, not a consecutive duplicate, AND recording is enabled
            if meets_threshold and not in_cooldown and not is_duplicate_of_last and self._recording_enabled:
                price = config.ITEM_PRICES.get(class_name, 0)
                with self._detection_lock:
                    self._detection_queue.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "used_threshold": threshold,
                        "timestamp": current_time,
                        "price": price
                    })
                    # Update running transaction total and items so subtotal is reliable
                    try:
                        self._transaction_total += int(price or 0)
                        self._transaction_items.append({"class_name": class_name, "price": price})
                    except Exception:
                        logger.exception("Failed to update transaction total")
                    self._last_detection_time[class_name] = current_time
                    self._last_added_item = class_name  # Update last added item
                    # Announce the detected item via TTS (friendly name)
                    try:
                        friendly_name = class_name.replace("_", " ").replace("-", " ").title()
                        self._tts.speak(f"{friendly_name}")
                    except Exception:
                        logger.exception("Failed to enqueue TTS announcement for detection")
                
                # Log the detection with price
                logger.info(f"Detected: {class_name} | Confidence: {confidence:.2%} | Threshold: {threshold:.2%} | Price: {price}")
            elif meets_threshold and not in_cooldown and is_duplicate_of_last:
                # Reset last_added_item if a different item is detected (allows next item to be added)
                if self._last_added_item and class_name != self._last_added_item:
                    with self._detection_lock:
                        self._last_added_item = None
            
            # Draw bounding box - green if ready to add, orange if in cooldown, gray if consecutive duplicate or below threshold
            if meets_threshold:
                if is_duplicate_of_last:
                    color = config.BBOX_COLOR_DUPLICATE
                elif in_cooldown:
                    color = config.BBOX_COLOR_COOLDOWN
                else:
                    color = config.BBOX_COLOR_READY
            else:
                color = config.BBOX_COLOR_LOW_CONFIDENCE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)
            
            # Draw label with confidence
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, config.LABEL_FONT, config.LABEL_FONT_SCALE, config.LABEL_FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - config.LABEL_PADDING), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), config.LABEL_FONT, config.LABEL_FONT_SCALE, config.LABEL_TEXT_COLOR, config.LABEL_FONT_THICKNESS)
        
        return frame
    
    def _detect_hand_gesture(self, frame):
        """Detect hand gestures (fist or open palm) using MediaPipe."""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Count extended fingers
                landmarks = hand_landmarks.landmark
                
                # Finger tip and base indices
                finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
                finger_bases = [6, 10, 14, 18]  # Corresponding bases
                thumb_tip = 4
                thumb_base = 2
                
                extended_fingers = 0
                
                # Check thumb (different logic - x-axis for most hand orientations)
                if abs(landmarks[thumb_tip].x - landmarks[thumb_base].x) > config.GESTURE_FINGER_EXTENSION_THRESHOLD:
                    extended_fingers += 1
                
                # Check other fingers (y-axis - tip above base means extended)
                for tip, base in zip(finger_tips, finger_bases):
                    if landmarks[tip].y < landmarks[base].y - config.GESTURE_FINGER_EXTENSION_THRESHOLD:
                        extended_fingers += 1
                
                # Classify gesture
                if extended_fingers >= config.GESTURE_PALM_MIN_FINGERS:
                    return "palm"
                elif extended_fingers <= config.GESTURE_FIST_MAX_FINGERS:
                    return "fist"
        
        return None
    
    def _update_recording_state(self, gesture):
        """Update recording state based on gesture sequence."""
        if gesture:
            self._gesture_history.append(gesture)
            
            # Check for gesture sequence: palm → fist
            if len(self._gesture_history) == 2:
                if list(self._gesture_history) == ["palm", "fist"]:
                    # Toggle recording state
                    self._recording_enabled = not self._recording_enabled
                    if self._recording_enabled:
                        logger.info("Recording STARTED - Items will be added to order")
                    else:
                        logger.info("Recording STOPPED - Items will not be added")
                        # Reset last added item when stopping recording for new transaction
                        with self._detection_lock:
                            self._last_added_item = None
                            # Use running transaction total (more reliable than queue that may be cleared by API consumers)
                            try:
                                subtotal = int(self._transaction_total or 0)
                                # Friendly announcement (assumed local currency)
                                self._tts.speak(f"Subtotal {subtotal} pesos")
                            except Exception:
                                logger.exception("Failed to compute or speak subtotal")
                            # Reset transaction state for next transaction
                            try:
                                self._transaction_total = 0
                                self._transaction_items.clear()
                                self._detection_queue.clear()
                            except Exception:
                                logger.exception("Failed to reset transaction state")
                    self._gesture_history.clear()
    
    def get_detections(self) -> list[dict]:
        """Get and clear all detections from the queue."""
        with self._detection_lock:
            detections = list(self._detection_queue)
            self._detection_queue.clear()
        return detections
    
    def is_recording(self) -> bool:
        """Get current recording state."""
        return self._recording_enabled
