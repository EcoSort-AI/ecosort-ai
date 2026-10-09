import os
import sys
import time
import uuid
import cv2

# --- CONFIGURAÇÕES ---
CAMERA_SOURCE = 2
DATASET_DIR = "dataset_novo"

# Configurações do Sensor Virtual (Idênticas à Produção)
MOTION_THRESHOLD = 30
MIN_AREA = 15000
STABILIZATION_TIME = 5.0
COOLDOWN_TIME = 5.0 # Reduzido para 2s pois você fará a remoção manualmente

# Mapeamento de Teclas para Classes Oficiais
CLASSES = {
    ord('1'): 'plastic',
    ord('2'): 'metal',
    ord('3'): 'paper',
    ord('4'): 'cardboard',
    ord('5'): 'white-glass',
    ord('6'): 'brown-glass',
    ord('7'): 'green-glass'
}

def setup_folders():
    for cls in CLASSES.values():
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
    print(f"Pastas criadas em: {os.path.abspath(DATASET_DIR)}")

def main():
    setup_folders()
    
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_class = 'plastic' # Classe padrão ao iniciar
    
    baseline_frame = None
    is_stabilizing = False
    stabilization_start = 0.0
    cooldown_until = 0.0
    sensor_status = "WARMING UP..."
    sensor_color = (0, 255, 255)

    window_name = "EcoSort - Coletor de Dataset"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--- CONTROLES ---")
    print("Teclas 1 a 7: Mudar a classe do resíduo")
    print("Tecla R: Resetar o fundo (Prato Vazio)")
    print("Tecla ESC: Sair")
    print("-----------------\n")

    time.sleep(2) # Aguarda câmera ajustar luz

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            current_time = time.time()
            clean_frame = frame.copy() # Cópia sem os desenhos para salvar no dataset
            height, width, _ = frame.shape

            # --- PROCESSAMENTO DO SENSOR VIRTUAL ---
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

            take_photo = False

            # --- MÁQUINA DE ESTADOS ---
            if current_time < cooldown_until:
                sensor_status = "COOLDOWN (Retire o objeto)"
                sensor_color = (0, 0, 255)
                is_stabilizing = False
            elif motion_detected:
                if not is_stabilizing:
                    is_stabilizing = True
                    stabilization_start = current_time
                    
                elapsed = current_time - stabilization_start
                if elapsed >= STABILIZATION_TIME:
                    sensor_status = "FOTO CAPTURADA!"
                    sensor_color = (0, 255, 0)
                    take_photo = True
                    is_stabilizing = False
                    cooldown_until = current_time + COOLDOWN_TIME
                else:
                    sensor_status = f"ESTABILIZANDO... {STABILIZATION_TIME - elapsed:.1f}s"
                    sensor_color = (0, 165, 255)
            else:
                is_stabilizing = False
                sensor_status = "PRONTO (Prato Vazio)"
                sensor_color = (0, 255, 0)

            # --- RENDERIZAÇÃO DA INTERFACE (HUD) ---
            # Status do Sensor
            cv2.putText(frame, f"Sensor: {sensor_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sensor_color, 2)
            
            # Classe Atual
            cv2.putText(frame, f"Classe Alvo: {current_class.upper()}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            
            # Instruções
            instrucoes = "1:Plastic 2:Metal 3:Paper 4:Cardboard 5:W-Glass 6:B-Glass 7:G-Glass | R:Reset Fundo"
            cv2.putText(frame, instrucoes, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window_name, frame)

            # --- AÇÃO: SALVAR FOTO ---
            if take_photo:
                filename = f"{current_class}_{uuid.uuid4().hex[:8]}.jpg"
                filepath = os.path.join(DATASET_DIR, current_class, filename)
                cv2.imwrite(filepath, clean_frame)
                print(f"✅ Salvo: {filepath}")

            # --- CONTROLES DE TECLADO ---
            key = cv2.waitKey(30)
            
            if key == 27: # ESC para sair
                break
            elif key in [ord('r'), ord('R')]: # R para resetar fundo
                baseline_frame = gray
                print("Fundo resetado.")
            elif key in CLASSES: # 1 a 7 para trocar a classe
                current_class = CLASSES[key]
                print(f"Classe alterada para: {current_class}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Coleta encerrada.")

if __name__ == "__main__":
    main()