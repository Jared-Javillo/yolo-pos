"""Configuration settings for YOLO POS system."""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "best.pt"

# ============================================================================
# LOGGING
# ============================================================================
LOG_FILE = LOG_DIR / "detections.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(message)s"

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_DEVICE_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ============================================================================
# YOLO DETECTION SETTINGS
# ============================================================================
# How often to run inference (every N frames)
INFERENCE_INTERVAL = 3

# Minimum confidence score to consider a detection valid (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.78

# Optional per-class confidence thresholds. If a class is present here its
# value will be used instead of `CONFIDENCE_THRESHOLD`. Values are in the
# 0.0 - 1.0 range. Example:
# PER_CLASS_CONFIDENCE_THRESHOLDS = {"Coke-in-can": 0.85, "coffee_nescafe": 0.70}
PER_CLASS_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "coffee_nescafe": 0.85,
    "coffee_kopiko": 0.84,
    "lucky-me-pancit-canton": 0.88,
    "Coke-in-can":0.72,
    "alaska_milk": 0.70,
    "Century-Tuna": 0.82,
    "VCut-Spicy-Barbeque": 0.80,
    "Selecta-Cornetto": 0.50,
    "nestleyogurt": 0.70,
    "Femme-Bathroom-Tissue": 0.82,
    "maya-champorado": 0.80,
    "jnj-potato-chips": 0.85,
    "Nivea-Deodorant": 0.82,
    "UFC-Canned-Mushroom": 0.85,
    "Libbys-Vienna-Sausage-can": 0.85,
    "Stik-O": 0.85,
    "nissin_cup_noodles": 0.78,
    "dewberry-strawberry": 0.70,
    "Smart-C": 0.80,
    "pineapple-juice-can": 0.80,
    "nestle_chuckie": 0.75,
    "Delight-Probiotic-Drink": 0.85,
    "Summit-Drinking-Water": 0.85,
    "almond_milk": 0.45,
    "Piknik": 0.60,
    "Bactidol": 0.85,
    "head&shoulders_shampoo": 0.85,
    "irish-spring-soap": 0.80,
    "c2_na_green": 0.87,
    "colgate_toothpaste": 0.50,
    "555-sardines-tomato": 0.2,
    "meadows_truffle_chips": 0.80,
    "double-black": 0.85,
    "NongshimCupNoodles": 0.55,
}
# Seconds to wait before allowing the same item to be detected again
COOLDOWN_SECONDS = 2.0

# Maximum number of detections to store in queue
DETECTION_QUEUE_MAX_SIZE = 100

# ============================================================================
# HAND GESTURE DETECTION SETTINGS
# ============================================================================
GESTURE_STATIC_IMAGE_MODE = False
GESTURE_MAX_NUM_HANDS = 1
GESTURE_MIN_DETECTION_CONFIDENCE = 0.85
GESTURE_MIN_TRACKING_CONFIDENCE = 0.5
GESTURE_HISTORY_MAX_SIZE = 2

# Gesture recognition thresholds
GESTURE_FINGER_EXTENSION_THRESHOLD = 0.05  # Distance threshold for finger extension
GESTURE_PALM_MIN_FINGERS = 4  # Minimum extended fingers for palm gesture
GESTURE_FIST_MAX_FINGERS = 1  # Maximum extended fingers for fist gesture

# ============================================================================
# UI DISPLAY SETTINGS
# ============================================================================
# Recording status display
STATUS_TEXT_RECORDING = "REC"
STATUS_TEXT_PAUSED = "PAUSED"
STATUS_COLOR_RECORDING = (0, 0, 255)  # BGR: Red
STATUS_COLOR_PAUSED = (128, 128, 128)  # BGR: Gray
STATUS_POSITION = (10, 30)
STATUS_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
STATUS_FONT_SCALE = 0.8
STATUS_FONT_THICKNESS = 2

# Bounding box colors (BGR format)
BBOX_COLOR_READY = (0, 255, 0)  # Green - ready to add
BBOX_COLOR_COOLDOWN = (0, 200, 255)  # Orange - in cooldown
BBOX_COLOR_DUPLICATE = (128, 128, 128)  # Gray - consecutive duplicate
BBOX_COLOR_LOW_CONFIDENCE = (128, 128, 128)  # Gray - below threshold
BBOX_THICKNESS = 2

# Label settings
LABEL_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 2
LABEL_TEXT_COLOR = (0, 0, 0)  # BGR: Black
LABEL_PADDING = 10

# ============================================================================
# ITEM PRICES (in local currency)
# ============================================================================
ITEM_PRICES = {
    "coffee_nescafe": 10,
    "coffee_kopiko": 10,
    "lucky-me-pancit-canton": 14,
    "Coke-in-can": 30,
    "alaska_milk": 28,
    "Century-Tuna": 38,
    "VCut-Spicy-Barbeque": 20,
    "Selecta-Cornetto": 35,
    "nestleyogurt": 25,
    "Femme-Bathroom-Tissue": 20,
    "maya-champorado": 94,
    "jnj-potato-chips": 20,
    "Nivea-Deodorant": 100,
    "UFC-Canned-Mushroom": 35,
    "Libbys-Vienna-Sausage-can": 23,
    "Stik-O": 12,
    "nissin_cup_noodles": 22,
    "dewberry-strawberry": 18,
    "Smart-C": 22,
    "pineapple-juice-can": 30,
    "nestle_chuckie": 15,
    "Delight-Probiotic-Drink": 12,
    "Summit-Drinking-Water": 15,
    "almond_milk": 90,
    "Piknik": 25,
    "Bactidol": 85,
    "head&shoulders_shampoo": 210,
    "irish-spring-soap": 40,
    "c2_na_green": 20,
    "colgate_toothpaste": 70,
    "555-sardines-tomato": 26,
    "meadows_truffle_chips": 90,
    "double-black": 1000,
    "NongshimCupNoodles": 50,
}
