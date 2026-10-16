import os
import sys
import time
import uuid
import cv2
import numpy as np
import threading
import torch

# --- PATCH PARA PYTORCH 2.6+ ---
# Força o PyTorch a permitir carregar checkpoints sem restrição de segurança
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
# ------------------------------

from ultralytics import YOLO

import config
import services
import interface
from mechatronics import Mechatronics

# --- CONFIGURAÇÕES DO SENSOR AUTO-TRIGGER ---
AUTO_TRIGGER_ENABLED = os.getenv("AUTO_TRIGGER", "true").lower() == "true"
PRESENCE_THRESHOLD = 25       # Sensibilidade de diferença contra o prato vazio (0 a 255)
MIN_OBJECT_AREA = 10000       # Área mínima (pixels na ROI) para confirmar que há um objeto no prato
MOTION_SETTLE_THRESHOLD = 30  # Tolerância a variações de luz/reflexo entre frames seguidos
MAX_MOTION_AREA = 12000       # Permite pequenas oscilações do objeto sem zerar o timer
STABILIZATION_TIME = 1.5      # Segundos que o objeto precisa ficar sem movimento brusco
MAX_WAIT_FOR_SETTLE = 3.0     # Timeout: se o objeto continuar oscilando após 3s no prato, classifica mesmo assim!
COOLDOWN_TIME = 5.5           # Pausa após classificar (3.8s dos servos + 1.7s para estabilizar câmera)
STARTUP_GRACE_PERIOD = 5.0    # Tempo inicial para a câmera estabilizar brilho/foco ao ligar
reset_baseline_file = "reset_bg.txt" # Permite resetar o fundo via: touch reset_bg.txt

def main():
    config.logger.info(f"Initializing EcoSort - Mode: {'AUTO + MANUAL' if AUTO_TRIGGER_ENABLED else 'MANUAL ONLY'}")

    threading.Thread(target=services.background_worker, daemon=True).start()
    threading.Thread(target=services.queue_worker, daemon=True).start()

    mechanism = Mechatronics()

    # Limpa qualquer gatilho residual ao iniciar
    if os.path.exists(config.TRIGGER_FILE):
        os.remove(config.TRIGGER_FILE)
    if os.path.exists(reset_baseline_file):
        os.remove(reset_baseline_file)

    try:
        model = YOLO(config.MODEL_PATH, task="classify")
    except Exception as e:
        config.logger.critical(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- WARM-UP DA CÂMERA ---
    # Lê e descarta 30 frames para forçar o ajuste automático de exposição e foco da câmera
    config.logger.info("Aquecendo sensor da camera (ajustando auto-exposicao)...")
    for _ in range(30):
        cap.read()
        time.sleep(0.04)

    last_class = "Waiting..."
    last_conf = 0.0
    text_color = (255, 255, 255)
    show_ui_until = 0.0
    ui_detected_class = ""

    window_name = "EcoSort UI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Variáveis de Estado do Sensor Virtual (iniciadas APÓS o warm-up e a criação da janela)
    baseline_gray = None
    prev_gray = None
    stabilization_start = 0.0
    presence_first_seen = 0.0
    is_stabilizing = False
    cooldown_until = time.time() + STARTUP_GRACE_PERIOD  # 5s reais de estabilização no loop
    waiting_plate_clear = False          # Evita re-classificar o mesmo objeto se deu "unsure" (não despejou)
    need_baseline_refresh = True         # Garante foto limpa do fundo assim que o cooldown acabar

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            current_time = time.time()
            clean_frame = frame.copy()
            h, w, _ = frame.shape

            # Define a ROI Central (60% centrais da tela - focado no prato da lixeira)
            roi_x1, roi_y1 = int(w * 0.20), int(h * 0.15)
            roi_x2, roi_y2 = int(w * 0.80), int(h * 0.85)
            roi_frame = clean_frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # Pré-processamento da ROI para o sensor de presença e movimento
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)

            key = cv2.waitKey(30)

            # Reset manual do fundo (Tecla R ou touch reset_bg.txt)
            if key in [ord('r'), ord('R')] or os.path.exists(reset_baseline_file):
                if os.path.exists(reset_baseline_file):
                    os.remove(reset_baseline_file)
                baseline_gray = gray_roi.astype("float")
                waiting_plate_clear = False
                is_stabilizing = False
                presence_first_seen = 0.0
                config.logger.info("Baseline do prato vazio recalibrado manualmente!")

            trigger_inference = False
            sensor_text = "MODO MANUAL"
            sensor_color = (0, 255, 255)

            if AUTO_TRIGGER_ENABLED:
                # Fase de Cooldown (servos em movimento ou câmera inicializando)
                if current_time < cooldown_until:
                    rem = cooldown_until - current_time
                    sensor_text = f"AGUARDANDO PRATO... ({rem:.1f}s)"
                    sensor_color = (0, 165, 255)
                    is_stabilizing = False
                    presence_first_seen = 0.0
                    prev_gray = gray_roi
                    # Absorve gradualmente a iluminação real enquanto aguarda o cooldown
                    if baseline_gray is None:
                        baseline_gray = gray_roi.astype("float")
                    else:
                        cv2.accumulateWeighted(gray_roi, baseline_gray, 0.2)

                else:
                    # Inicializa ou recalibra o baseline após os servos voltarem ao repouso
                    if baseline_gray is None or need_baseline_refresh:
                        if prev_gray is not None:
                            diff_settle = cv2.absdiff(prev_gray, gray_roi)
                            _, thresh_settle = cv2.threshold(diff_settle, MOTION_SETTLE_THRESHOLD, 255, cv2.THRESH_BINARY)
                            if cv2.countNonZero(thresh_settle) < MAX_MOTION_AREA:
                                baseline_gray = gray_roi.astype("float")
                                need_baseline_refresh = False
                                config.logger.info("Fundo do prato calibrado automaticamente.")
                        else:
                            baseline_gray = gray_roi.astype("float")
                            need_baseline_refresh = False

                    # Calcula diferença contra o prato vazio (Presença de Objeto)
                    bg_uint8 = cv2.convertScaleAbs(baseline_gray)
                    delta_bg = cv2.absdiff(bg_uint8, gray_roi)
                    thresh_bg = cv2.threshold(delta_bg, PRESENCE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                    thresh_bg = cv2.dilate(thresh_bg, None, iterations=2)
                    contours, _ = cv2.findContours(thresh_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    max_obj_area = 0
                    best_box = None
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > max_obj_area:
                            max_obj_area = area
                            best_box = cv2.boundingRect(c)

                    object_present = max_obj_area >= MIN_OBJECT_AREA

                    # Calcula diferença contra o frame anterior (Verifica movimento brusco como a mão)
                    is_moving_now = False
                    if prev_gray is not None:
                        delta_motion = cv2.absdiff(prev_gray, gray_roi)
                        thresh_motion = cv2.threshold(delta_motion, MOTION_SETTLE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                        motion_pixels = cv2.countNonZero(thresh_motion)
                        is_moving_now = motion_pixels > MAX_MOTION_AREA

                    prev_gray = gray_roi

                    # Desenha área de detecção no modo DEV
                    if config.DISPLAY_MODE == "dev":
                        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (100, 100, 100), 1)
                        if object_present and best_box is not None:
                            bx, by, bw, bh = best_box
                            cv2.rectangle(frame, (roi_x1 + bx, roi_y1 + by), (roi_x1 + bx + bw, roi_y1 + by + bh), (0, 255, 255), 2)

                    # Se o último lixo foi "unsure" (prato não inclinou), aguarda a pessoa tirar o lixo do prato
                    if waiting_plate_clear:
                        if not object_present:
                            waiting_plate_clear = False
                            presence_first_seen = 0.0
                            config.logger.info("Objeto removido do prato. Sensor reativado.")
                        else:
                            sensor_text = "REMOVA O ITEM (OU TOUCH TRIGGER)"
                            sensor_color = (0, 0, 255)

                    elif object_present:
                        if presence_first_seen == 0.0:
                            presence_first_seen = current_time

                        total_wait = current_time - presence_first_seen

                        # Se já passou do tempo máximo (3.0s) com objeto no prato, dispara mesmo oscilando
                        if total_wait >= MAX_WAIT_FOR_SETTLE:
                            trigger_inference = True
                            is_stabilizing = False
                            presence_first_seen = 0.0
                            config.logger.info(f"Disparo por tempo limite de presenca ({MAX_WAIT_FOR_SETTLE}s). Area: {int(max_obj_area)}")

                        elif is_moving_now:
                            # Movimento grande detectado (mão colocando o objeto): aguarda parar
                            stabilization_start = current_time
                            is_stabilizing = True
                            sensor_text = f"POSICIONANDO... ({MAX_WAIT_FOR_SETTLE - total_wait:.1f}s)"
                            sensor_color = (0, 255, 255)
                        else:
                            # Objeto presente e com oscilação dentro do limite tolerável
                            if not is_stabilizing:
                                is_stabilizing = True
                                stabilization_start = current_time

                            elapsed = current_time - stabilization_start
                            if elapsed >= STABILIZATION_TIME:
                                trigger_inference = True
                                is_stabilizing = False
                                presence_first_seen = 0.0
                                config.logger.info(f"Objeto estabilizado (Area: {int(max_obj_area)}). Disparando IA!")
                            else:
                                sensor_text = f"ESTABILIZANDO... {STABILIZATION_TIME - elapsed:.1f}s"
                                sensor_color = (0, 255, 0)
                    else:
                        # Prato vazio e sem movimento: adapta suavemente o fundo contra variações de luz
                        is_stabilizing = False
                        presence_first_seen = 0.0
                        sensor_text = "PRONTO (COLOQUE O RESIDUO)"
                        sensor_color = (0, 255, 0)
                        if not is_moving_now and baseline_gray is not None:
                            cv2.accumulateWeighted(gray_roi, baseline_gray, 0.02)

            # Exibe status do sensor na tela (Coordenadas Y=140 e Y=170 sem sobrepor a IA)
            cv2.putText(frame, f"Status: {sensor_text}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sensor_color, 2)
            cv2.putText(frame, "Auto-Detect Ativo | Backup: touch trigger.txt", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- DISPARO DA INFERÊNCIA (AUTOMÁTICO OU MANUAL VIA TRIGGER.TXT / ESPAÇO) ---
            manual_trigger = os.path.exists(config.TRIGGER_FILE) or key == 32
            if trigger_inference or manual_trigger:
                if os.path.exists(config.TRIGGER_FILE):
                    os.remove(config.TRIGGER_FILE)

                # Feedback visual instantâneo
                cv2.putText(frame, "CLASSIFICANDO...", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

                results = model.predict(source=clean_frame, imgsz=224, conf=0.01, verbose=False)
                res = results[0]

                if res.probs is not None:
                    class_name = res.names[res.probs.top1]
                    confidence = float(res.probs.top1conf)

                    if confidence >= config.CONFIDENCE_THRESHOLD:
                        last_class = class_name
                        text_color = (0, 255, 0) if confidence >= config.HIGH_CONF_THRESHOLD else (0, 165, 255)
                        ui_detected_class = class_name
                        show_ui_until = time.time() + config.DISPLAY_TIME_SUCCESS
                        # O prato vai despejar e voltar: agenda cooldown e recalibração do fundo
                        cooldown_until = time.time() + COOLDOWN_TIME
                        need_baseline_refresh = True
                        waiting_plate_clear = False
                    else:
                        last_class = f"Unsure ({class_name})"
                        text_color = (0, 0, 255)
                        ui_detected_class = "unsure"
                        show_ui_until = time.time() + config.DISPLAY_TIME_UNSURE
                        # Como o prato não inclina no "unsure", aguarda remover ou mexer no objeto
                        cooldown_until = time.time() + 2.5
                        waiting_plate_clear = True

                    last_conf = confidence
                    services.enqueue_event(str(uuid.uuid4()), clean_frame, class_name, confidence)

                    # Dispara os motores em paralelo
                    threading.Thread(
                        target=mechanism.classify,
                        args=(class_name, confidence, config.CONFIDENCE_THRESHOLD),
                        daemon=True
                    ).start()

            # Renderiza a interface da IA
            display_frame = interface.render(frame, last_class, last_conf, ui_detected_class, show_ui_until, text_color)

            try:
                cv2.imshow(window_name, display_frame)
            except cv2.error:
                pass

            if key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()