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
import numpy as np
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
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.3.0")
DEVICE_TOKEN = str(os.getenv("DEVICE_TOKEN", "")).strip()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
HIGH_CONF_THRESHOLD = float(os.getenv("HIGH_CONF_THRESHOLD", 0.8))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))
camera_env = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(camera_env) if camera_env.isdigit() else camera_env
TRIGGER_FILE = "trigger.txt"

# --- DISPLAY SETTINGS ---
DISPLAY_MODE = os.getenv("DISPLAY_MODE", "prod").lower() # 'dev' ou 'prod'
ASSETS_DIR = os.getenv("ASSETS_DIR", "/app/assets")

DISPLAY_TIME_SUCCESS = float(os.getenv("DISPLAY_TIME_SUCCESS", 7.0))
DISPLAY_TIME_UNSURE = float(os.getenv("DISPLAY_TIME_UNSURE", 4.0))

RECYCLING_COLORS = {
    "paper": (255, 0, 0),       # Azul
    "cardboard": (255, 0, 0),   # Azul
    "plastic": (0, 0, 255),     # Vermelho
    "white-glass": (0, 200, 0), # Verde
    "green-glass": (0, 200, 0), # Verde
    "brown-glass": (0, 200, 0), # Verde
    "metal": (0, 255, 255),     # Amarelo
    "unsure": (128, 128, 128)   # Cinza (Não reconhecido)
}

# --- SPOOL / FILA PERSISTENTE ---
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
# BACKGROUND WORKERS
# ==========================================
def background_worker():
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
                        logger.info(f"[SYNC] Limiar remoto: {new_threshold:.1%}")
                        CONFIDENCE_THRESHOLD = new_threshold

                commands = data.get("commands", [])
                latest_cmd = commands[0] if isinstance(commands, list) and len(commands) > 0 else (commands if isinstance(commands, dict) else {})

                if latest_cmd and latest_cmd.get("command") in ["restart", "restart_docker"]:
                    cmd_id = latest_cmd.get("id") or latest_cmd.get("command_id")
                    if cmd_id:
                        requests.patch(f"{API_BASE_URL}/device/commands", headers=auth_header, json={"command_id": cmd_id, "status": "completed"}, timeout=2)
                    time.sleep(1)
                    os._exit(1)
        except Exception as e:
            pass
        time.sleep(120)

def queue_worker():
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
        except Exception as e:
            pass
        time.sleep(10)

def enqueue_event(event_id: str, frame, class_name: str, confidence: float):
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

def upload_to_r2(local_file_path: str, r2_object_path: str) -> bool:
    try:
        s3_client.upload_file(local_file_path, R2_BUCKET_NAME, r2_object_path)
        return True
    except ClientError:
        return False

def send_classification_to_api(class_name: str, confidence: float, image_path: str, event_id: str, timestamp: str) -> bool:
    payload = {
        "bin_id": BIN_ID,
        "source_event_id": event_id,
        "timestamp": timestamp,
        "model_version": MODEL_VERSION,
        "detection": {"class_name": class_name.lower(), "confidence": round(confidence, 4)},
        "image_path": image_path
    }
    try:
        response = requests.post(API_URL, json=payload, headers={"Authorization": f"Bearer {DEVICE_TOKEN}"}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return True
    except:
        return False

# ==========================================
# MAIN LOOP AND INTERFACE
# ==========================================
def main():
    logger.info(f"Initializing EcoSort - Display Mode: {DISPLAY_MODE.upper()}")

    threading.Thread(target=background_worker, daemon=True).start()
    threading.Thread(target=queue_worker, daemon=True).start()

    if os.path.exists(TRIGGER_FILE): os.remove(TRIGGER_FILE)

    try:
        model = YOLO(MODEL_PATH, task="classify")
    except Exception as e:
        logger.critical(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_class = "Waiting..."
    last_conf = 0.0
    text_color = (255, 255, 255)
    show_ui_until = 0.0
    ui_detected_class = ""

    window_name = "EcoSort UI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap = cv2.VideoCapture(CAMERA_SOURCE)
                continue

            height, width, _ = frame.shape
            key = cv2.waitKey(30)

            if os.path.exists(TRIGGER_FILE) or key == 32:
                if os.path.exists(TRIGGER_FILE): os.remove(TRIGGER_FILE)

                results = model.predict(source=frame, conf=0.01, verbose=False)
                res = results[0]

                if res.probs is not None:
                    class_name = res.names[res.probs.top1]
                    confidence = float(res.probs.top1conf)
                    event_uuid = str(uuid.uuid4())

                    if confidence >= CONFIDENCE_THRESHOLD:
                        last_class = class_name
                        text_color = (0, 255, 0) if confidence >= HIGH_CONF_THRESHOLD else (0, 165, 255)
                        ui_detected_class = class_name
                        show_ui_until = time.time() + DISPLAY_TIME_SUCCESS
                    else:
                        last_class = f"Unsure ({class_name})"
                        text_color = (0, 0, 255)
                        ui_detected_class = "unsure"
                        show_ui_until = time.time() + DISPLAY_TIME_UNSURE

                    last_conf = confidence
                    enqueue_event(event_uuid, frame, class_name, confidence)

            # --- DISPLAY RENDER ---
            if DISPLAY_MODE == "dev":
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Class: {last_class.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
                if last_conf > 0:
                    cv2.putText(display_frame, f"Conf: {last_conf:.1%}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

                cv2.putText(display_frame, f"Current Threshold: {CONFIDENCE_THRESHOLD:.1%}", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.putText(display_frame, "[SPACE] to Classify | [ESC] to Exit", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            else:
                display_frame = np.zeros((height, width, 3), dtype=np.uint8)

                if time.time() < show_ui_until:
                    bg_color = RECYCLING_COLORS.get(ui_detected_class, (50, 50, 50))
                    display_frame[:] = bg_color

                    msg = "NAO RECONHECIDO. Tente novamente." if ui_detected_class == "unsure" else f"DETECTADO: {ui_detected_class.upper()}"
                    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
                    text_x = (width - text_size[0]) // 2
                    cv2.putText(display_frame, msg, (text_x, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

                    asset_path_png = os.path.join(ASSETS_DIR, f"{ui_detected_class}.png")
                    asset_path_jpg = os.path.join(ASSETS_DIR, f"{ui_detected_class}.jpg")
                    asset_img_path = asset_path_png if os.path.exists(asset_path_png) else asset_path_jpg

                    if os.path.exists(asset_img_path):
                        img_asset = cv2.imread(asset_img_path, cv2.IMREAD_UNCHANGED)
                        if img_asset is not None:
                            img_asset = cv2.resize(img_asset, (400, 400))
                            start_y = (height - 400) // 2 + 50
                            start_x = (width - 400) // 2
                            roi = display_frame[start_y:start_y+400, start_x:start_x+400]

                            if len(img_asset.shape) == 3 and img_asset.shape[2] == 4:
                                alpha = img_asset[:, :, 3]
                                mask_inv = cv2.bitwise_not(alpha)
                                white_icon = np.full((400, 400, 3), 255, dtype=np.uint8)
                                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                                icon_fg = cv2.bitwise_and(white_icon, white_icon, mask=alpha)
                                display_frame[start_y:start_y+400, start_x:start_x+400] = cv2.add(roi_bg, icon_fg)
                            else:
                                img_gray = cv2.cvtColor(img_asset, cv2.COLOR_BGR2GRAY)
                                _, mask = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
                                mask_inv = cv2.bitwise_not(mask)
                                white_icon = np.full((400, 400, 3), 255, dtype=np.uint8)
                                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                                icon_fg = cv2.bitwise_and(white_icon, white_icon, mask=mask)
                                display_frame[start_y:start_y+400, start_x:start_x+400] = cv2.add(roi_bg, icon_fg)
                else:
                    display_frame[:] = (0, 0, 0)
                    logo_path = os.path.join(ASSETS_DIR, "logo.jpg")

                    if os.path.exists(logo_path):
                        bg_img = cv2.imread(logo_path)
                        if bg_img is not None:
                            scale = width / bg_img.shape[1]
                            new_h = int(bg_img.shape[0] * scale)

                            if new_h >= height:
                                resized_bg = cv2.resize(bg_img, (width, new_h))
                                start_y = (new_h - height) // 2
                                display_frame[:] = resized_bg[start_y:start_y+height, :]
                            else:
                                scale = height / bg_img.shape[0]
                                new_w = int(bg_img.shape[1] * scale)
                                resized_bg = cv2.resize(bg_img, (new_w, height))
                                start_x = (new_w - width) // 2
                                display_frame[:] = resized_bg[:, start_x:start_x+width]
                    else:
                        cv2.putText(display_frame, "ECOSORT AI", ((width - 300) // 2, height // 2 - 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 4)

            try:
                cv2.imshow(window_name, display_frame)
            except cv2.error:
                pass

            if key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()