"""Camera helpers for streaming frames from a USB device."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO

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
        
        if self._hands:
            self._hands.close()

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
                    self._last_detection_time[class_name] = current_time
                    self._last_added_item = class_name  # Update last added item
                
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
