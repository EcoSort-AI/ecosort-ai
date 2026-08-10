import os
import sys
import logging
import random
import cv2
import requests
import time
import uuid
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# --- CONFIGURAÇÕES ---
API_URL = os.getenv("API_URL", "https://www.ecosort.com.br/api/v1/trash-events")
BIN_ID = os.getenv("BIN_ID", "smart_bin_01")
MODEL_PATH = os.getenv("MODEL_PATH", "best_ncnn_model") 
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
HIGH_CONF_THRESHOLD = float(os.getenv("HIGH_CONF_THRESHOLD", 0.8))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))

camera_env = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(camera_env) if camera_env.isdigit() else camera_env

# --- TRIGGER FILE (SSH / Docker) ---
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

def upload_to_r2(local_file_path: str, r2_object_path: str) -> bool:
    try:
        s3_client.upload_file(local_file_path, R2_BUCKET_NAME, r2_object_path)
        logger.info(f"Upload concluído com sucesso: {r2_object_path}")
        return True
    except ClientError as e:
        logger.error(f"Erro no upload para o R2: {e}")
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
        logger.info(f"Dados enviados com sucesso. Status: {response.status_code}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Erro ao enviar dados para a API: {e}")
        logger.error(f"Detalhes da rejeição: {e.response.text if hasattr(e, 'response') else 'Sem detalhes'}")
    except Exception as e:
        logger.error(f"Erro inesperado de conexão: {e}")

def main():
    logger.info("Inicializando EcoSort AI - Edge Mode com HUD...")
    
    if os.path.exists(TRIGGER_FILE):
        os.remove(TRIGGER_FILE)
        
    try:
        model = YOLO(MODEL_PATH, task="classify")
        logger.info(f"Modelo carregado com sucesso de {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Falha ao carregar o modelo YOLO: {e}")
        sys.exit(1)

    logger.info("Inicializando câmera em tempo real...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        logger.warning(f"Falha ao abrir CAMERA_SOURCE {CAMERA_SOURCE}. Tentando fallback para 0...")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    last_class = "Aguardando..."
    last_conf = 0.0
    text_color = (255, 255, 255) 

    logger.info(f"Lixeira '{BIN_ID}' está ativa.")
    logger.info(f"==> DICA: Para classificar, rode o comando SSH: docker exec ecosort-edge touch /app/{TRIGGER_FILE} <==")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Sinal da câmera perdido. Tentando reconectar...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(CAMERA_SOURCE)
                continue

            # --- (CROP) ---
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
            cv2.putText(display_frame, "ZONA DE ANALISE IA", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Classe: {last_class.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
            if last_conf > 0:
                cv2.putText(display_frame, f"Confianca: {last_conf:.1%}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            cv2.putText(display_frame, "[ESC] para Sair (Testes Locais)", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            try:
                cv2.imshow("EcoSort - Validation HUD", display_frame)
            except cv2.error:
                pass

            key = cv2.waitKey(30)
            
            # --- TRIGGER ACTION ---
            if os.path.exists(TRIGGER_FILE):
                logger.info("Acionando a rede neural...")
                os.remove(TRIGGER_FILE)
                
                cropped_frame_for_ai = frame[y_min:y_max, x_min:x_max]
                
                results = model.predict(source=cropped_frame_for_ai, conf=CONFIDENCE_THRESHOLD, verbose=False)
                res = results[0]
                
                if res.probs is not None:
                    top_index = res.probs.top1
                    class_name = res.names[top_index]
                    confidence = float(res.probs.top1conf)

                    if confidence >= CONFIDENCE_THRESHOLD:
                        event_uuid = str(uuid.uuid4())
                        
                        is_audit = random.random() <= 0.20

                        if confidence >= HIGH_CONF_THRESHOLD and not is_audit:
                            logger.info(f"ALTA CONFIANCA: {class_name.upper()} ({confidence:.2%})")
                            send_classification_to_api(class_name, confidence, image_path=None, event_id=event_uuid)
                            
                            last_class = class_name
                            last_conf = confidence
                            text_color = (0, 255, 0) 
                            
                        else:
                            if is_audit and confidence >= HIGH_CONF_THRESHOLD:
                                logger.info(f"AUDITORIA (Sorteio): {class_name.upper()} ({confidence:.2%}). Enviando foto.")
                            else:
                                logger.info(f"REVISAO NECESSARIA: {class_name.upper()} ({confidence:.2%}). Enviando foto.")
                            
                            temp_filename = f"temp_{event_uuid}.jpg"
                            cv2.imwrite(temp_filename, frame)
                            
                            r2_path = f"pending/{event_uuid}.jpg"
                            
                            upload_success = upload_to_r2(temp_filename, r2_path)
                            
                            if upload_success:
                                send_classification_to_api(class_name, confidence, image_path=r2_path, event_id=event_uuid)
                            
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
                            
                            if is_audit and confidence >= HIGH_CONF_THRESHOLD:
                                last_class = f"Audit ({class_name})"
                            else:
                                last_class = f"Revisao ({class_name})"
                                
                            last_conf = confidence
                            text_color = (0, 165, 255)

                    else:
                        logger.warning(f"Inconclusivo: {class_name} com {confidence:.2%}")
                        last_class = f"Incerto ({class_name})"
                        last_conf = confidence
                        text_color = (0, 165, 255) 
                else:
                    logger.warning("Nenhum objeto reconhecido na zona de analise.")
                    last_class = "Nao reconhecido"
                    last_conf = 0.0
                    text_color = (0, 0, 255) 
                    
            elif key == 27: 
                break

    except KeyboardInterrupt:
        logger.info("Sinal de desligamento recebido.")
    finally:
        logger.info("Liberando a camera e encerrando...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()