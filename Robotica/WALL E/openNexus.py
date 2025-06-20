import openai
import speech_recognition as sr
import pyttsx3
import time

# Configura tu API Key
openai.api_key = "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"

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

# Configuración de voz
engine.setProperty('rate', 160)  # velocidad de habla

# Función para convertir texto a voz
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Función para reconocer voz y convertirla en texto
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Escuchando...")
        audio = recognizer.listen(source, phrase_time_limit=8)
    try:
        text = recognizer.recognize_google(audio, language="es-MX")
        print(f"🗣️ Dijiste: {text}")
        return text
    except sr.UnknownValueError:
        print("😕 No se entendió el audio.")
        speak("No entendí, por favor repite.")
        return None
    except sr.RequestError as e:
        print(f"Error con el servicio de reconocimiento: {e}")
        speak("Hubo un error con el reconocimiento de voz.")
        return None

# Función para consultar a la API con contexto de asistente del hogar
def consulta_api(prompt_usuario):
    contexto = (
        "Eres un asistente de casa útil y amigable llamado NEXUS. "
        "Responde de forma breve (máximo 30 palabras), clara y hablada. "
        "Tu función es ayudar en dudas del hogar, entretenimiento, clima simulado, recetas, recordatorios simulados, etc. "
        "No tienes control real sobre dispositivos ni acceso a internet. Sé natural."
    )

    prompt = f"{contexto}\nUsuario: {prompt_usuario}\nAsistente:"
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=80
    )
    
    respuesta = response.choices[0].message.content.strip()
    print(f"🤖 Nexus: {respuesta}")
    return respuesta

# Bucle principal del asistente
def main():
    print("🔁 Asistente Nexus activo. Presiona ENTER o escribe 'sí' para hablar.")
    while True:
        entrada = input("¿Quieres hablar con Nexus? ").strip().lower()
        if entrada == "" or entrada == "si" or entrada == "sí":
            speak("Te escucho.")
            texto_usuario = listen()
            if texto_usuario:
                respuesta = consulta_api(texto_usuario)
                speak(respuesta)
        else:
            print("⏳ Esperando...")
        time.sleep(1)

if __name__ == "__main__":
    main()
