import speech_recognition as sr
import serial
import time

# Configura el puerto serial (ajusta el nombre del puerto y velocidad según tu ESP/Arduino)
puerto_serial = serial.Serial('COM6', 9600)  # Reemplaza 'COM3' por el puerto correcto
time.sleep(2)  # Espera a que el puerto esté listo

# Palabras clave
derecha_keywords = ['derecha', 'right']
izquierda_keywords = ['izquierda', 'left']

# Inicializa el reconocedor
r = sr.Recognizer()
print("🎙️ Habla algo...")

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

try:
    texto = r.recognize_google(audio, language="es-MX")
    print("📝 Texto detectado:", texto)

    # Convierte a minúsculas para comparación
    texto = texto.lower()

    if any(palabra in texto for palabra in derecha_keywords):
        print("➡️ Comando: DERECHA (1)")
        puerto_serial.write(b'1')
    elif any(palabra in texto for palabra in izquierda_keywords):
        print("⬅️ Comando: IZQUIERDA (0)")
        puerto_serial.write(b'0')
    else:
        print("⚠️ No se detectó una dirección válida.")

except sr.UnknownValueError:
    print("❌ No se pudo entender el audio.")
except sr.RequestError as e:
    print(f"❌ Error con el servicio de reconocimiento: {e}")
puerto_serial.close()
print("🔌 Puerto serial cerrado.")
