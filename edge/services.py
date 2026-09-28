import os
import time
import json
import psutil
import requests
import boto3
import cv2
from botocore.exceptions import ClientError
from datetime import datetime, timezone
import config

s3_client = boto3.client(
    service_name='s3',
    endpoint_url=f"https://{config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=config.R2_ACCESS_KEY_ID,
    aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
    region_name="auto"
)

def background_worker():
    auth_header = {
        "Authorization": f"Bearer {config.DEVICE_TOKEN}",
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
                "device_name": config.BIN_ID,
                "cpu_usage": round(cpu_usage, 1),
                "ram_usage": f"{ram.percent}%",
                "disk_free": f"{round(disk.free / (1024**3), 1)}GB",
                "temperature": round(temp, 1),
                "uptime": f"{uptime_hours}h"
            }

            sync_res = requests.post(f"{config.API_BASE_URL}/device/sync", headers=auth_header, json=payload, timeout=config.REQUEST_TIMEOUT)

            if sync_res.status_code in [200, 201]:
                data = sync_res.json()
                config_data = data.get("config", {})
                if config_data:
                    new_threshold = float(config_data.get('confidence_threshold', 80)) / 100.0
                    if new_threshold != config.CONFIDENCE_THRESHOLD:
                        config.logger.info(f"[SYNC] Limiar remoto: {new_threshold:.1%}")
                        config.CONFIDENCE_THRESHOLD = new_threshold

                commands = data.get("commands", [])
                latest_cmd = commands[0] if isinstance(commands, list) and len(commands) > 0 else (commands if isinstance(commands, dict) else {})

                if latest_cmd and latest_cmd.get("command") in ["restart", "restart_docker"]:
                    cmd_id = latest_cmd.get("id") or latest_cmd.get("command_id")
                    if cmd_id:
                        requests.patch(f"{config.API_BASE_URL}/device/commands", headers=auth_header, json={"command_id": cmd_id, "status": "completed"}, timeout=2)
                    time.sleep(1)
                    os._exit(1)
        except Exception:
            pass
        time.sleep(120)

def queue_worker():
    while True:
        try:
            arquivos = os.listdir(config.SPOOL_DIR)
            for filename in arquivos:
                if filename.endswith(".json"):
                    event_id = filename.replace(".json", "")
                    json_path = os.path.join(config.SPOOL_DIR, filename)
                    img_path = os.path.join(config.SPOOL_DIR, f"{event_id}.jpg")

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
        except Exception:
            pass
        time.sleep(10)

def enqueue_event(event_id: str, frame, class_name: str, confidence: float):
    img_path = os.path.join(config.SPOOL_DIR, f"{event_id}.jpg")
    json_path = os.path.join(config.SPOOL_DIR, f"{event_id}.json")
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
        s3_client.upload_file(local_file_path, config.R2_BUCKET_NAME, r2_object_path)
        return True
    except ClientError:
        return False

def send_classification_to_api(class_name: str, confidence: float, image_path: str, event_id: str, timestamp: str) -> bool:
    payload = {
        "bin_id": config.BIN_ID,
        "source_event_id": event_id,
        "timestamp": timestamp,
        "model_version": config.MODEL_VERSION,
        "detection": {"class_name": class_name.lower(), "confidence": round(confidence, 4)},
        "image_path": image_path
    }
    try:
        response = requests.post(config.API_URL, json=payload, headers={"Authorization": f"Bearer {config.DEVICE_TOKEN}"}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        return True
    except:
        return False