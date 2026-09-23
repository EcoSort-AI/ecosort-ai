import time
import logging

logger = logging.getLogger(__name__)

# --- CONFIGURAÇÕES ---
CANAL_SERVO_X = 0
CANAL_SERVO_Y = 1
POSICAO_REPOUSO = (90.0, 90.0)

# Mapeamento dos ângulos que você descobriu
CLASS_TO_TILT = {
    "metal":       (125.0, 115.0), # Superior Esquerdo
    "plastic":     (55.0, 115.0),  # Superior Direito
    "paper":       (55.0, 45.0),   # Inferior Esquerdo
    "cardboard":   (55.0, 45.0),   # Inferior Esquerdo
    "glass":       (125.0, 45.0),  # Inferior Direito
    "brown-glass": (125.0, 45.0),  # Inferior Direito
    "green-glass": (125.0, 45.0),  # Inferior Direito
    "white-glass": (125.0, 45.0),  # Inferior Direito
    "reject":      POSICAO_REPOUSO,
    "unsure":      POSICAO_REPOUSO
}

class Mecatronica:
    def __init__(self):
        self.pca = None
        self.servo_x = None
        self.servo_y = None
        
        # O try/except garante que se você rodar o código no seu PC (Windows/Mac) 
        # para testar a interface, o código não vai quebrar por falta de pinos I2C.
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
            from adafruit_motor import servo
            
            logger.info("Iniciando hardware PCA9685 via I2C...")
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50
            
            self.servo_x = servo.Servo(self.pca.channels[CANAL_SERVO_X], min_pulse=500, max_pulse=2500, actuation_range=180)
            self.servo_y = servo.Servo(self.pca.channels[CANAL_SERVO_Y], min_pulse=500, max_pulse=2500, actuation_range=180)
            
            # Centraliza e relaxa os motores na inicialização
            self.servo_x.angle = POSICAO_REPOUSO[0]
            self.servo_y.angle = POSICAO_REPOUSO[1]
            time.sleep(1.0)
            self.relaxar_motores()
            logger.info("Mecatrônica pronta e alinhada.")
            
        except Exception as e:
            logger.warning(f"Hardware mecatrônico não encontrado/desativado. Rodando apenas software. Erro: {e}")

    def relaxar_motores(self):
        """Corta o sinal PWM (detach) para acabar com a tremedeira do DS3230"""
        if self.servo_x: self.servo_x.angle = None
        if self.servo_y: self.servo_y.angle = None

    def classificar_residuo(self, class_name: str, confidence: float, threshold: float):
        """Move os motores para a posição correta, aguarda a queda e retorna ao centro"""
        if not self.pca:
            return

        alvo = class_name.lower()
        if confidence < threshold or alvo not in CLASS_TO_TILT:
            alvo = "reject"

        ang_x, ang_y = CLASS_TO_TILT[alvo]

        if (ang_x, ang_y) == POSICAO_REPOUSO:
            logger.info("⚙️ MECATRÔNICA: Lixo inconclusivo. Mantendo prato em repouso.")
            return

        logger.info(f"⚙️ MECATRÔNICA: Despejando {alvo.upper()} -> X:{ang_x}° | Y:{ang_y}°")
        
        # 1. Inclina o prato
        self.servo_x.angle = ang_x
        self.servo_y.angle = ang_y
        
        # 2. Aguarda o lixo escorregar
        time.sleep(2.5) 
        
        # 3. Retorna ao nivelamento e relaxa os motores
        logger.info("⚙️ MECATRÔNICA: Retornando ao repouso.")
        self.servo_x.angle = POSICAO_REPOUSO[0]
        self.servo_y.angle = POSICAO_REPOUSO[1]
        time.sleep(1.0)
        self.relaxar_motores()
