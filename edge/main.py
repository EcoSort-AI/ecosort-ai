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
import json
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
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.1.0")
DEVICE_TOKEN = str(os.getenv("DEVICE_TOKEN", "")).strip()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
HIGH_CONF_THRESHOLD = float(os.getenv("HIGH_CONF_THRESHOLD", 0.8))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))
camera_env = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(camera_env) if camera_env.isdigit() else camera_env
TRIGGER_FILE = "trigger.txt"

# --- SPOOL ---
SPOOL_DIR = os.getenv("SPOOL_DIR", "/app/data/spool/")
os.makedirs(SPOOL_DIR, exist_ok=True)

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
# BACKGROUND WORKERS (TELEMETRIA E FILA)
# ==========================================
def background_worker():
    """Roda a cada 2 minutos enviando telemetria e recebendo configs e comandos"""
    global CONFIDENCE_THRESHOLD
        
    auth_header = {
        "Authorization": f"Bearer {DEVICE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    while True:
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
                
            payload = {
                "device_name": BIN_ID,
                "cpu_usage": round(cpu_usage, 1),
                "ram_usage": f"{ram.percent}%",
                "disk_free": f"{round(disk.free / (1024**3), 1)}GB",
                "temperature": round(temp, 1),
                "uptime": f"{uptime_hours}h"
            }

            sync_res = requests.post(f"{API_BASE_URL}/device/sync", headers=auth_header, json=payload, timeout=REQUEST_TIMEOUT)

            if sync_res.status_code in [200, 201]:
                data = sync_res.json()

                config_data = data.get("config", {})
                if config_data:
                    new_threshold = float(config_data.get('confidence_threshold', 80)) / 100.0
                    if new_threshold != CONFIDENCE_THRESHOLD:
                        logger.info(f"[SYNC] Limiar atualizado remotamente para: {new_threshold:.1%}")
                        CONFIDENCE_THRESHOLD = new_threshold

                commands = data.get("commands", [])
                latest_cmd = commands[0] if isinstance(commands, list) and len(commands) > 0 else (commands if isinstance(commands, dict) else {})

                if latest_cmd and latest_cmd.get("command") in ["restart", "restart_docker"]:
                    logger.warning("Comando de REINICIALIZAÇÃO recebido no Sync!")
                    cmd_id = latest_cmd.get("id") or latest_cmd.get("command_id")

                    if cmd_id:
                        patch_payload = {"command_id": cmd_id, "status": "completed"}
                        requests.patch(f"{API_BASE_URL}/device/commands", headers=auth_header, json=patch_payload, timeout=2)

                    logger.warning("Derrubando processo para forçar o reinício pelo Docker...")
                    time.sleep(1)
                    os._exit(1)
            else:
                logger.debug(f"[SYNC] Servidor rejeitou sincronização. Status: {sync_res.status_code}")

        except Exception as e:
            logger.error(f"[SYNC] Erro no ciclo de background: {e}")

        time.sleep(120)

def queue_worker():
    """Fila local persistente. Lê o disco e envia eventos pendentes para a nuvem de forma idempotente."""
    while True:
        try:
            arquivos = os.listdir(SPOOL_DIR)
            for filename in arquivos:
                if filename.endswith(".json"):
                    event_id = filename.replace(".json", "")
                    json_path = os.path.join(SPOOL_DIR, filename)
                    img_path = os.path.join(SPOOL_DIR, f"{event_id}.jpg")
                    
                    if not os.path.exists(img_path):
                        os.remove(json_path) 
                        continue
                        
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        
                    r2_path = f"pending/{event_id}.jpg"
                    
                    if upload_to_r2(img_path, r2_path):
                        if send_classification_to_api(data["class_name"], data["confidence"], r2_path, event_id, data["timestamp"]):
                            os.remove(img_path)
                            os.remove(json_path)
                            logger.info(f"[QUEUE] Evento {event_id} processado com sucesso e removido do disco.")
                        else:
                            logger.warning(f"[QUEUE] Falha na API para {event_id}. Mantendo no spool para retentativa.")
                    else:
                        logger.warning(f"[QUEUE] Falha no R2 para {event_id}. Mantendo no spool para retentativa.")
                        
        except Exception as e:
            logger.error(f"[QUEUE] Erro ao processar a fila persistente: {e}")
            
        time.sleep(10)

# ==========================================
# CLASSIFICATION AND UPLOAD FUNCTIONS
# ==========================================
def enqueue_event(event_id: str, frame, class_name: str, confidence: float):
    """Salva frame original e metadados no disco para processamento resiliente."""
    img_path = os.path.join(SPOOL_DIR, f"{event_id}.jpg")
    json_path = os.path.join(SPOOL_DIR, f"{event_id}.json")
    
    cv2.imwrite(img_path, frame)
    
    payload = {
        "event_id": event_id,
        "class_name": class_name,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }
    
    with open(json_path, "w") as f:
        json.dump(payload, f)
    logger.info(f"[SPOOL] Evento {event_id} salvo na fila local.")

def upload_to_r2(local_file_path: str, r2_object_path: str) -> bool:
    try:
        s3_client.upload_file(local_file_path, R2_BUCKET_NAME, r2_object_path)
        return True
    except ClientError as e:
        return False

def send_classification_to_api(class_name: str, confidence: float, image_path: str, event_id: str, timestamp: str) -> bool:
    payload = {
        "bin_id": BIN_ID,
        "source_event_id": event_id,
        "timestamp": timestamp,
        "model_version": MODEL_VERSION,
        "detection": {
            "class_name": class_name.lower(),
            "confidence": round(confidence, 4)
        },
    }
    if image_path:
        payload["image_path"] = image_path    
        
    auth_header = {
        "Authorization": f"Bearer {DEVICE_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, json=payload, headers=auth_header, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"Erro HTTP da API (Rejeição): {e.response.text if hasattr(e, 'response') else e}")
        return False
    except Exception as e:
        logger.error(f"Erro de conexão com API: {e}")
        return False

# ==========================================
# MAIN LOOP 
# ==========================================
def main():
    logger.info("Initializing EcoSort Visual Validation Mode...")
    
    threading.Thread(target=background_worker, daemon=True).start()
    threading.Thread(target=queue_worker, daemon=True).start()
    
    logger.info("Background Telemetry & Persistent Queue workers started.")
    
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
                
            height, width, _ = frame.shape
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Class: {last_class.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
            
            if last_conf > 0:
                cv2.putText(display_frame, f"Confidence: {last_conf:.1%}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            cv2.putText(display_frame, f"Current Threshold: {CONFIDENCE_THRESHOLD:.1%}", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
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
                
                results = model.predict(source=frame, conf=0.01, verbose=False)
                res = results[0]
                
                if res.probs is not None:
                    top_index = res.probs.top1
                    class_name = res.names[top_index]
                    confidence = float(res.probs.top1conf)
                    event_uuid = str(uuid.uuid4())
                   
                    if confidence >= HIGH_CONF_THRESHOLD:
                        logger.info(f"HIGH CONFIDENCE: {class_name.upper()} ({confidence:.2%}).")
                        last_class = class_name
                        text_color = (0, 255, 0) 
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        logger.info(f"REVIEW REQUIRED: {class_name.upper()} ({confidence:.2%}).")
                        last_class = f"Review ({class_name})"
                        text_color = (0, 165, 255)
                    else:
                        logger.warning(f"INCONCLUSIVE: {class_name.upper()} ({confidence:.2%}).")
                        last_class = f"Unsure ({class_name})"
                        text_color = (0, 0, 255)
                        
                    last_conf = confidence
                    
                    enqueue_event(event_uuid, frame, class_name, confidence)
                    
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