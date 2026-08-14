import os
import sys
import logging
import random
import cv2
import requests
import time
import uuid
import boto3
import threading
import psutil
from botocore.exceptions import ClientError
from datetime import datetime, timezone
from dotenv import load_dotenv
from ultralytics import YOLO
load_dotenv()
# --- SETTINGS ---
API_URL = os.getenv("API_URL")
API_BASE_URL = os.getenv("API_BASE_URL", API_URL.replace("/trash-events", "") if API_URL else "http://localhost:3000/api/v1")
BIN_ID = os.getenv("BIN_ID", "smart_bin_01")
MODEL_PATH = os.getenv("MODEL_PATH", "best_ncnn_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
HIGH_CONF_THRESHOLD = float(os.getenv("HIGH_CONF_THRESHOLD", 0.8))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))
camera_env = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(camera_env) if camera_env.isdigit() else camera_env
TRIGGER_FILE = "trigger.txt"
# --- CLOUDFLARE R2 ---
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
s3_client = boto3.client(
    service_name='s3',
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    region_name="auto"
)
# --- LOGS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# ==========================================
# BACKGROUND WORKER (TELEMETRIA E CONFIG)
# ==========================================
def background_worker():
    """Roda em paralelo para não travar o feed de vídeo da câmera"""
    global CONFIDENCE_THRESHOLD
    auth_header = {
        "Authorization": f"Bearer ecotoken_{BIN_ID}",
        "Content-Type": "application/json"
    }
    while True:
       
        try:
            config_res = requests.get(f"{API_BASE_URL}/device/config?device={BIN_ID}", headers=auth_header, timeout=REQUEST_TIMEOUT)
            if config_res.status_code == 200:
                config_data = config_res.json()
                new_threshold = float(config_data.get('confidence_threshold', 80)) / 100.0
                if new_threshold != CONFIDENCE_THRESHOLD:
                    logger.info(f"[CONFIG] Limiar de confiança atualizado remotamente para: {new_threshold:.1%}")
                    CONFIDENCE_THRESHOLD = new_threshold
        except Exception as e:
            logger.debug(f"[CONFIG] Falha ao sincronizar configurações: {e}")
       
        try:
           
            cpu_usage = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read()) / 1000.0
            except FileNotFoundError:
                temp = 0.0
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                uptime_hours = int(uptime_seconds // 3600)
            telemetry_data = {
                "device_name": BIN_ID,
                "cpu_usage": round(cpu_usage, 1),
                "ram_usage": f"{ram.percent}%",
                "disk_free": f"{round(disk.free / (1024**3), 1)}GB",
                "temperature": round(temp, 1),
                "uptime": f"{uptime_hours}h"
            }
            tele_res = requests.post(f"{API_BASE_URL}/device/telemetry", headers=auth_header, json=telemetry_data, timeout=REQUEST_TIMEOUT)
            if tele_res.status_code not in [200, 201]:
                logger.debug(f"[TELEMETRY] Servidor rejeitou telemetria. Status: {tele_res.status_code}")
        except Exception as e:
            logger.debug(f"[TELEMETRY] Falha ao enviar telemetria: {e}")
        # ==========================================
        # REMOTE COMMANDS
        # ==========================================
        try:
            cmd_res = requests.get(f"{API_BASE_URL}/device/commands?device={BIN_ID}&status=pending", headers=auth_header, timeout=REQUEST_TIMEOUT)
            if cmd_res.status_code == 200:
                commands = cmd_res.json()
                
                if commands:
                    logger.info(f"[COMMANDS] Resposta da API: {commands}")
                
                latest_cmd = commands[0] if isinstance(commands, list) and len(commands) > 0 else (commands if isinstance(commands, dict) else {})
                if latest_cmd and latest_cmd.get("command") in ["restart", "restart_docker"]:
                    logger.warning("Comando de REINICIALIZAÇÃO recebido da nuvem!")
                    
                    cmd_id = latest_cmd.get("id") or latest_cmd.get("command_id")
                    if cmd_id:
                        
                        payload = {
                            "command_id": cmd_id,
                            "status": "completed"
                        }
                        patch_res = requests.patch(f"{API_BASE_URL}/device/commands", headers=auth_header, json=payload, timeout=2)
                        if patch_res.status_code == 200:
                            logger.info(f"[COMMANDS] Comando {cmd_id} concluído no banco de dados com sucesso.")
                        else:
                            logger.error(f"[COMMANDS] Falha ao marcar comando como concluído. API retornou: {patch_res.status_code}")
                    logger.warning("Derrubando processo para forçar o reinício pelo Docker...")
                    time.sleep(1)
                    os._exit(1) 
            else:
                
                logger.error(f"[COMMANDS] API recusou a busca de comandos. Status: {cmd_res.status_code}")
        except Exception as e:
            logger.error(f"[COMMANDS] Erro de conexão ao buscar comandos: {e}")
        
        time.sleep(30)
# ==========================================
# CLASSIFICATION AND UPLOAD FUNCTIONS
# ==========================================
def upload_to_r2(local_file_path: str, r2_object_path: str) -> bool:
    try:
        s3_client.upload_file(local_file_path, R2_BUCKET_NAME, r2_object_path)
        logger.info(f"Upload completed successfully: {r2_object_path}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading to R2: {e}")
        return False
def send_classification_to_api(class_name: str, confidence: float, image_path: str, event_id: str) -> None:
    payload = {
        "bin_id": BIN_ID,
        "source_event_id": event_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "model_version": MODEL_VERSION,
        "detection": {
            "class_name": class_name.lower(),
            "confidence": round(confidence, 4)
        },
    }
    if image_path:
        payload["image_path"] = image_path
    auth_header = {
        "Authorization": f"Bearer ecotoken_{BIN_ID}"
    }
    try:
        response = requests.post(API_URL, json=payload, headers=auth_header, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Data successfully sent to the backend. Status: {response.status_code}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Error sending data to the backend: {e}")
        logger.error(f"Backend rejection details: {e.response.text if hasattr(e, 'response') else 'No details'}")
    except Exception as e:
        logger.error(f"Unexpected connection error: {e}")
# ==========================================
# MAIN LOOP 
# ==========================================
def main():
    logger.info("Initializing EcoSort Visual Validation Mode...")
    
    telemetry_thread = threading.Thread(target=background_worker, daemon=True)
    telemetry_thread.start()
    logger.info("Background Telemetry & Config worker started.")
    if os.path.exists(TRIGGER_FILE):
        os.remove(TRIGGER_FILE)
    try:
        model = YOLO(MODEL_PATH, task="classify")
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Failed to load YOLO model: {e}")
        sys.exit(1)
    logger.info("Initializing real-time camera...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        logger.warning(f"Failed to open CAMERA_SOURCE {CAMERA_SOURCE}. Trying fallback to 0...")
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_class = "Waiting..."
    last_conf = 0.0
    text_color = (255, 255, 255)
    logger.info(f"Smart Bin '{BIN_ID}' is active.")
    logger.info(f"==> DICA: Para classificar, rode o comando SSH: docker exec ecosort-edge touch /app/{TRIGGER_FILE} <==")
    # --- FULLSCREEN SETTINGS ---
    window_name = "EcoSort - Validation HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera signal lost. Attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(CAMERA_SOURCE)
                continue
            # --- CROP ---
            height, width, _ = frame.shape
            fraction = 0.6
            side = int(min(height, width) * fraction)
            y_center, x_center = height // 2, width // 2
            y_min = y_center - side // 2
            x_min = x_center - side // 2
            y_max = y_min + side
            x_max = x_min + side
            # --- (Heads-Up Display) ---
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_frame, "AI ANALYSIS ZONE", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Class: {last_class.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
            if last_conf > 0:
                cv2.putText(display_frame, f"Confidence: {last_conf:.1%}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            cv2.putText(display_frame, f"Current Threshold: {CONFIDENCE_THRESHOLD:.1%}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(display_frame, "[SPACE] to Classify | [ESC] to Exit", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            try:
                cv2.imshow(window_name, display_frame)
            except cv2.error:
                pass
            key = cv2.waitKey(30)
            # --- TRIGGER ACTION ---
            if os.path.exists(TRIGGER_FILE) or key == 32:
                logger.info("Triggering neural network...")
                if os.path.exists(TRIGGER_FILE):
                    os.remove(TRIGGER_FILE)
                cropped_frame_for_ai = frame[y_min:y_max, x_min:x_max]
                
                results = model.predict(source=cropped_frame_for_ai, conf=0.01, verbose=False)
                res = results[0]
                if res.probs is not None:
                    top_index = res.probs.top1
                    class_name = res.names[top_index]
                    confidence = float(res.probs.top1conf)
                    event_uuid = str(uuid.uuid4())
                   
                    if confidence >= HIGH_CONF_THRESHOLD:
                        logger.info(f"HIGH CONFIDENCE: {class_name.upper()} ({confidence:.2%}). Saving to dataset.")
                        last_class = class_name
                        text_color = (0, 255, 0) # Verde
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        logger.info(f"REVIEW REQUIRED: {class_name.upper()} ({confidence:.2%}). Saving to dataset.")
                        last_class = f"Review ({class_name})"
                        text_color = (0, 165, 255) # Laranja
                    else:
                        logger.warning(f"INCONCLUSIVE: {class_name.upper()} ({confidence:.2%}). Saving to dataset.")
                        last_class = f"Unsure ({class_name})"
                        text_color = (0, 0, 255) # Vermelho
                    last_conf = confidence
                    # Upload of all detections
                    temp_filename = f"temp_{event_uuid}.jpg"
                    cv2.imwrite(temp_filename, frame)
                    r2_path = f"pending/{event_uuid}.jpg"
                    upload_success = upload_to_r2(temp_filename, r2_path)
                    # Send to API
                    final_image_path = r2_path if upload_success else None
                    send_classification_to_api(class_name, confidence, image_path=final_image_path, event_id=event_uuid)
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                else:
                    logger.warning("No object recognized.")
                    last_class = "Not recognized"
                    last_conf = 0.0
                    text_color = (0, 0, 255)
            elif key == 27:
                break
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
