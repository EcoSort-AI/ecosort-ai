import os
import sys
import logging
import cv2
import requests
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "https://ecosort-ai-nine.vercel.app/api/v1/trash-events")
BIN_ID = os.getenv("BIN_ID", "smart_bin_01")
MODEL_PATH = os.getenv("MODEL_PATH", "best_ncnn_model")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))

camera_env = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(camera_env) if camera_env.isdigit() else camera_env

# --- TRIGGER FILE (HEADLESS) ---
# Python will look for this file in the /app folder to sort it
TRIGGER_FILE = "trigger.txt"

# --- LOGS ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def send_classification_to_api(class_name: str, confidence: float) -> None:
    payload = {
        "bin_id": BIN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "detection": {"class_name": class_name, "confidence": round(confidence, 4)}
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Data successfully sent to the backend. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending data to the backend: {e}")

def main():
    logger.info("Initializing EcoSort Headless Mode...")
    logger.info("Watchtower Test")
    
    if os.path.exists(TRIGGER_FILE):
        os.remove(TRIGGER_FILE)
        
    try:
        model = YOLO(MODEL_PATH, task="classify")
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    logger.info("Initializing real time camera...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        logger.critical(f"Error connecting to the camera.: {CAMERA_SOURCE}")
        sys.exit(1)

    logger.info(f"Smart Bin '{BIN_ID}' is active.")
    logger.info(f"==> Para classificar, rode este comando via SSH: touch {TRIGGER_FILE} <==")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # --- CROP METHOD ---
            height, width, _ = frame.shape
            fraction = 0.6
            side = int(min(height, width) * fraction)
            y_center, x_center = height // 2, width // 2

            y_min = y_center - side // 2
            x_min = x_center - side // 2

            cropped_frame = frame[y_min:y_min + side, x_min:x_min + side]
            # --- END OF CROP ---
                
            cv2.imshow("EcoSort Preview (Headless Mode)", cropped_frame) # use cropped_frame instead of frame

            cv2.waitKey(30)
            
            # --- VIRTUAL TRIGGER ---
            if os.path.exists(TRIGGER_FILE):
                logger.info("SSH signal received! Starting classification...")
                os.remove(TRIGGER_FILE)
                
                results = model.predict(source=cropped_frame, conf=CONFIDENCE_THRESHOLD, verbose=False) # use cropped_frame instead of frame
                res = results[0]
                
                if res.probs is not None:
                    top_index = res.probs.top1
                    class_name = res.names[top_index]
                    confidence = float(res.probs.top1conf)

                    if confidence >= CONFIDENCE_THRESHOLD:
                        logger.info(f"Classification Result: {class_name.upper()} ({confidence:.2%})")
                        send_classification_to_api(class_name, confidence)
                    else:
                        logger.warning(f"inconclusive: {class_name} a {confidence:.2%} (below the threshold). Nothing sent.")
                else:
                    logger.warning("No object recognized with sufficient confidence.")
                    

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        logger.info("Releasing camera and shutting down...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()
