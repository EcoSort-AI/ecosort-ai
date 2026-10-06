import os
import sys
import time
import uuid
import cv2
import threading
from ultralytics import YOLO

import config
import services
import interface
from mechatronics import Mechatronics

# --- OPENCV VIRTUAL SENSOR SETTINGS ---
MOTION_THRESHOLD = 30       # Sensibilidade da diferença de cor (0 a 255)
MIN_AREA = 15000            # Tamanho mínimo do objeto para ignorar ruídos da câmera
STABILIZATION_TIME = 2.0    # Segundos de espera para a mão sair da tela antes da foto
COOLDOWN_TIME = 4.0         # Segundos de pausa após classificar (tempo para o prato esvaziar)

def main():
    config.logger.info(f"Initializing EcoSort - Display Mode: {config.DISPLAY_MODE.upper()}")

    threading.Thread(target=services.background_worker, daemon=True).start()
    threading.Thread(target=services.queue_worker, daemon=True).start()

    mechanism = Mechatronics()

    if os.path.exists(config.TRIGGER_FILE): os.remove(config.TRIGGER_FILE)

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

    last_class = "Waiting..."
    last_conf = 0.0
    text_color = (255, 255, 255)
    show_ui_until = 0.0
    ui_detected_class = ""

    baseline_frame = None
    is_stabilizing = False
    stabilization_start = 0.0
    cooldown_until = 0.0
    sensor_status = "WARMING UP..."
    sensor_color = (0, 255, 255)

    window_name = "EcoSort UI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    time.sleep(2)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap = cv2.VideoCapture(config.CAMERA_SOURCE)
                continue

            current_time = time.time()
            clean_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if baseline_frame is None:
                baseline_frame = gray
                continue

            frame_delta = cv2.absdiff(baseline_frame, gray)
            thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            for c in contours:
                if cv2.contourArea(c) > MIN_AREA:
                    motion_detected = True
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    break

            trigger_ai = False

            if current_time < cooldown_until:
                sensor_status = "COOLDOWN"
                sensor_color = (0, 0, 255)
                is_stabilizing = False
            elif motion_detected:
                if not is_stabilizing:
                    is_stabilizing = True
                    stabilization_start = current_time
                    config.logger.info("Movimento detectado. Aguardando estabilização...")
                    
                elapsed = current_time - stabilization_start
                if elapsed >= STABILIZATION_TIME:
                    sensor_status = "CLASSIFYING..."
                    sensor_color = (0, 255, 0)
                    trigger_ai = True
                    is_stabilizing = False
                    cooldown_until = current_time + COOLDOWN_TIME
                else:
                    sensor_status = f"STABILIZING... {STABILIZATION_TIME - elapsed:.1f}s"
                    sensor_color = (0, 165, 255)
            else:
                is_stabilizing = False
                sensor_status = "READY"
                sensor_color = (0, 255, 0)

            cv2.putText(frame, f"Sensor: {sensor_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sensor_color, 2)
            cv2.putText(frame, "[R] Reset Baseline", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            key = cv2.waitKey(30)

            if key == ord('r') or key == ord('R'):
                baseline_frame = gray
                config.logger.info("Baseline manual reset.")

            if trigger_ai or os.path.exists(config.TRIGGER_FILE) or key == 32:
                if os.path.exists(config.TRIGGER_FILE): os.remove(config.TRIGGER_FILE)

                results = model.predict(source=clean_frame, conf=0.01, verbose=False)
                res = results[0]

                if res.probs is not None:
                    class_name = res.names[res.probs.top1]
                    confidence = float(res.probs.top1conf)
                    event_uuid = str(uuid.uuid4())

                    if confidence >= config.CONFIDENCE_THRESHOLD:
                        last_class = class_name
                        text_color = (0, 255, 0) if confidence >= config.HIGH_CONF_THRESHOLD else (0, 165, 255)
                        ui_detected_class = class_name
                        show_ui_until = time.time() + config.DISPLAY_TIME_SUCCESS
                    else:
                        last_class = f"Unsure ({class_name})"
                        text_color = (0, 0, 255)
                        ui_detected_class = "unsure"
                        show_ui_until = time.time() + config.DISPLAY_TIME_UNSURE

                    last_conf = confidence
                    
                    services.enqueue_event(event_uuid, clean_frame, class_name, confidence)

                    threading.Thread(
                        target=mechanism.classify, 
                        args=(class_name, confidence, config.CONFIDENCE_THRESHOLD), 
                        daemon=True
                    ).start()

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
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()