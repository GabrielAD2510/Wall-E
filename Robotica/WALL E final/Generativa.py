import openai
import os

# Pega tu API key de OpenAI
openai.api_key = "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"

# -------- Rutas --------
ruta_base = os.path.dirname(os.path.abspath(__file__))
archivo_trayectoria = os.path.join(ruta_base, "Trayectoria1.py")
archivo_simulador = os.path.join(ruta_base, "Brazo3d.py")

# -------- Leer los archivos existentes --------
with open(archivo_trayectoria, "r", encoding="utf-8") as f:
    codigo_trayectoria = f.read()

with open(archivo_simulador, "r", encoding="utf-8") as f:
    codigo_simulador = f.read()

# -------- Entrada del usuario --------
entrada_usuario = input("👉 ¿Qué figura deseas que fabrique el robot?: ")

# -------- Construcción del prompt (omitido aquí para evitar error visual) --------
prompt = f""""""


# -------- Enviar solicitud a la API de OpenAI --------
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.4,
    max_tokens=1000
)

# -------- Guardar resultado en archivo nuevo --------
codigo_generado = response.choices[0].message.content.strip()
nombre_figura = entrada_usuario.lower().strip().replace(" ", "_")
archivo_nuevo = os.path.join(ruta_base, "Trayectoria.py")


with open(archivo_nuevo, "w", encoding="utf-8") as f:
    f.write(codigo_generado)

print(f"\n✅ Código generado guardado en: {archivo_nuevo}")
