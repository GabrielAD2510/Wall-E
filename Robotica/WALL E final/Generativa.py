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
prompt = f"""Estás dentro de un script Python que modifica el comportamiento de un sistema robótico simulador.

Este sistema se compone de dos archivos principales:

1. **Brazo3d.py** — Este archivo contiene la lógica de simulación y la cinemática de un brazo robótico. **No debes modificar este archivo** bajo ninguna circunstancia.

2. **Trayectoria.py** — Este archivo contiene una función llamada `generar_trayectoria_fig()` que devuelve una lista de puntos `[x, y, z]` que el brazo seguirá para dibujar una figura 3D, en este ejemplo dibuja un cubo 3D, debes tomarlo y hacer lo que el ususario pidió.

Un usuario ha solicitado una nueva figura que el brazo debe generar. Tu tarea es **modificar el contenido de `Trayectoria.py` directamente para que genere esta figura en 3D**, sin crear nuevos archivos adicionales.
A continuación se te proporcionará:

=== Figura solicitada por el usuario: {entrada_usuario}.
=== Contenido actual de Trayectoria.py:{codigo_trayectoria}.
=== Contenido de Brazo3d.py (solo lectura):{codigo_simulador}

Condiciones obligatorias:

- **Todas las figuras solicitadas son en 3D.** No generes figuras planas o en 2D. Siempre debes proporcionar puntos con coordenadas X, Y y Z distintas entre sí para formar una figura en 3D con volumen o al menos variaciones en altura.
- Debes sobrescribir completamente el archivo `Trayectoria.py`, respetando su estructura general y conservando únicamente la constante `mesa_z_altura` y la función `generar_trayectoria_fig()`.
- No debes agregar bloques `if __name__ == "__main__"`, ni imprimir nada, ni ejecutar código fuera de la función.
- La salida debe ser un **archivo válido de Python**, completamente funcional, sin dependencias externas, y debe ser importable desde `Brazo3d.py` sin errores.
- No modifiques ni elimines la constante `mesa_z_altura`.
- Asegúrate de que la función `generar_trayectoria_fig()` retorne una trayectoria continua y realista que el brazo pueda seguir.
- No incluyas explicaciones, comentarios extensos, ni bloques de prueba. Solo código limpio y funcional.

Contexto adicional:
- El archivo generado será importado directamente por `Brazo3d.py`, no ejecutado por sí mismo.
- El contenido original de `Trayectoria.py` se reemplazará completamente por lo que generes.


Escribe únicamente el nuevo contenido completo para Trayectoria.py con los parametros necesarios"""


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
