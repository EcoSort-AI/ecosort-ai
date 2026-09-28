import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# --- LOGS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ecosort")

# --- API & CLOUD ---
API_URL = os.getenv("API_URL")
API_BASE_URL = os.getenv("API_BASE_URL", API_URL.replace("/trash-events", "") if API_URL else "http://localhost:3000/api/v1")
BIN_ID = os.getenv("BIN_ID", "smart_bin_01")
DEVICE_TOKEN = str(os.getenv("DEVICE_TOKEN", "")).strip()

R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

# --- MODEL & CAMERA ---
MODEL_PATH = os.getenv("MODEL_PATH", "best_ncnn_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.3.0")
camera_env = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(camera_env) if camera_env.isdigit() else camera_env

# --- THRESHOLDS ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
HIGH_CONF_THRESHOLD = float(os.getenv("HIGH_CONF_THRESHOLD", 0.8))

# --- SYSTEM ---
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))
TRIGGER_FILE = "trigger.txt"
SPOOL_DIR = os.getenv("SPOOL_DIR", "/app/data/spool/")
os.makedirs(SPOOL_DIR, exist_ok=True)

# --- DISPLAY ---
DISPLAY_MODE = os.getenv("DISPLAY_MODE", "prod").lower()
ASSETS_DIR = os.getenv("ASSETS_DIR", "/app/assets")

DISPLAY_TIME_SUCCESS = float(os.getenv("DISPLAY_TIME_SUCCESS", 7.0))
DISPLAY_TIME_UNSURE = float(os.getenv("DISPLAY_TIME_UNSURE", 4.0))

RECYCLING_COLORS = {
    "paper": (255, 0, 0), "cardboard": (255, 0, 0), "plastic": (0, 0, 255),
    "white-glass": (0, 200, 0), "green-glass": (0, 200, 0), "brown-glass": (0, 200, 0),
    "metal": (0, 255, 255), "unsure": (128, 128, 128)
}