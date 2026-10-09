import cv2

print("Testando portas de câmera...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Câmera encontrada no índice: {i}")
        cap.release()
    else:
        print(f"❌ Nada no índice: {i}")