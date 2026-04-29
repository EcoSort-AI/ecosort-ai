import os
import sys
import logging
import cv2
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from ultralytics import YOLO
from requests.exceptions import RequestException

load_dotenv()

# --- CONFIGURATION (Environment Variables) ---
API_URL = os.getenv("API_URL", "https://your-domain.vercel.app/api/classify")
BIN_ID = os.getenv("BIN_ID", "smart_bin_01")
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def send_classification_to_api(class_name: str, confidence: float) -> None:
    """
    Sends the classification data to the web API.
    """
    payload = {
        "bin_id": BIN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "detection": {
            "class_name": class_name,
            "confidence": round(confidence, 4)
        }
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Successfully sent data to API. API Response: {response.status_code}")
    except RequestException as e:
        logger.error(f"Failed to send data to API. Error: {e}")
        logger.error(f"Vercel error details: {e.response.text}")
    except RequestException as e:
        logger.error(f"Network error: {e}")

def main():
    """
    Main loop for the Smart Bin Edge AI.
    """
    logger.info("Initializing EcoSort Edge AI...")
    
    try:
        model = YOLO(MODEL_PATH, task="classify")
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    logger.info("Initializing real time camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        logger.critical(f"Error: Could not access camera at index {CAMERA_INDEX}.")
        sys.exit(1)

    logger.info(f"Smart Bin '{BIN_ID}' is active and waiting for events...")

    try:
        while True:
            # In production, change the input() to a GPIO sensor trigger
            # input("\n[SIMULATION] Press ENTER to trigger object detection...")
            
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera. Retrying...")
                continue

            cv2.imshow("EcoSort Preview - Pressione 'C' para classificar", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Encerrando o sistema a pedido do usuário...")
                break
            elif key == ord('c') or key == 13:
                logger.info("Imagem capturada. Rodando inferência...")

            results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            res = results[0]
                
            logger.info("Image captured. Running inference...")
            
            
            # Correção CRÍTICA: Avaliar 'probs' (Classificação) em vez de 'boxes' (Detecção)
            if res.probs is not None:
                top_index = res.probs.top1
                class_name = res.names[top_index]
                confidence = float(res.probs.top1conf)
                
                logger.info(f"Classification Result: {class_name.upper()} ({confidence:.2%} confidence)")
                
                send_classification_to_api(class_name, confidence)
            else:
                logger.warning("No classification result found for this frame.")

    except KeyboardInterrupt:
        logger.info("Shutdown signal received (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        logger.info("Releasing camera and shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()