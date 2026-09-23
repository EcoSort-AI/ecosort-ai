import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# --- CONFIGURAÇÕES DOS CANAIS (0 a 15 no PCA9685) ---
CANAL_SERVO_X = 0 # Onde você espetou o motor X (Esquerda/Direita)
CANAL_SERVO_Y = 4 # Onde você espetou o motor Y (Frente/Trás)

print("Iniciando comunicação I2C com o PCA9685...")

try:
    # Inicia a comunicação I2C com os pinos SDA (GPIO 2) e SCL (GPIO 3)
    i2c = busio.I2C(board.SCL, board.SDA)
    
    # Inicia a placa PCA9685 no endereço padrão (0x40)
    pca = PCA9685(i2c)
    pca.frequency = 50 # Frequência padrão para servos analógicos (50Hz)

    # Configura os servos com os limites do DS3230 (500us a 2500us)
    servo_x = servo.Servo(pca.channels[CANAL_SERVO_X], min_pulse=500, max_pulse=2500, actuation_range=180)
    servo_y = servo.Servo(pca.channels[CANAL_SERVO_Y], min_pulse=500, max_pulse=2500, actuation_range=180)

    print("Movendo para a posição de REPOUSO (90°, 90°)...")
    servo_x.angle = 90
    servo_y.angle = 90
    
    # Aguarda 1 segundo e corta o sinal (detach) definindo o ângulo como None
    time.sleep(1.0)
    servo_x.angle = None
    servo_y.angle = None

    print("\n" + "="*40)
    print(" FERRAMENTA DE CALIBRAÇÃO (PCA9685)")
    print("="*40)
    print("Digite os ângulos no formato: X,Y")
    print("Digite 'sair' para encerrar.")
    print("="*40)

    while True:
        entrada = input("\nDigite X,Y -> ")
        
        if entrada.lower() == 'sair':
            break
            
        try:
            partes = entrada.split(',')
            if len(partes) != 2:
                print("Formato inválido! Use a vírgula. Ex: 45,90")
                continue
                
            ang_x = float(partes[0].strip())
            ang_y = float(partes[1].strip())
            
            ang_x = max(0, min(180, ang_x))
            ang_y = max(0, min(180, ang_y))
            
            print(f"Movendo prato para X: {ang_x}° | Y: {ang_y}°")
            servo_x.angle = ang_x
            servo_y.angle = ang_y
            
            time.sleep(1.0)
            
            # CORTA O SINAL (Detach no PCA9685)
            servo_x.angle = None
            servo_y.angle = None
            print("Sinal cortado (Motores relaxados).")
            
        except ValueError:
            print("rro: Digite apenas números separados por vírgula!")

except Exception as e:
    print(f"\nErro de Hardware I2C: {e}")
    print("DICA: Verifique se os cabos SDA e SCL estão bem conectados e se o I2C está habilitado no raspi-config.")
    
finally:
    print("\nEncerrando calibração. Desligando a placa PCA9685...")
    if 'pca' in locals():
        pca.deinit() # Libera o barramento I2C e desliga todos os sinais
