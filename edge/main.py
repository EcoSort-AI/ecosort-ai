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

    window_name = "EcoSort UI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap = cv2.VideoCapture(config.CAMERA_SOURCE)
                continue

            key = cv2.waitKey(30)

            if os.path.exists(config.TRIGGER_FILE) or key == 32:
                if os.path.exists(config.TRIGGER_FILE): os.remove(config.TRIGGER_FILE)

                results = model.predict(source=frame, conf=0.01, verbose=False)
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
                    
                    services.enqueue_event(event_uuid, frame, class_name, confidence)

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