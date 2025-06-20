from gtts import gTTS
from IPython.display import Audio
import speech_recognition as sr

# Inicializa el reconocedor
r = sr.Recognizer()

print("🎙️ Habla algo...")

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

# Guarda el audio en un archivo WAV
with open("grabacion.wav", "wb") as f:
    f.write(audio.get_wav_data())

print("✅ Audio grabado y guardado como grabacion.wav")

try:
    texto = r.recognize_google(audio, language="es-MX")
    print("📝 Texto detectado:")
    print(texto)
except sr.UnknownValueError:
    print("❌ No se pudo entender el audio.")
except sr.RequestError as e:
    print(f"❌ Error con el servicio de reconocimiento: {e}")



# Convertir el texto detectado a voz
tts = gTTS(text=texto, lang="es", slow=False)
tts.save("respuesta.mp3")

# Reproducir el audio generado
print("🔈 Reproduciendo el texto convertido a voz:")
Audio("respuesta.mp3", autoplay=True)

