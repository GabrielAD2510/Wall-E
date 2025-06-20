import random

# Texto base de entrenamiento
texto = "yo soy feliz y yo soy fuerte y yo soy valiente y yo soy capaz"

# Tokenizamos las palabras
palabras = texto.split()

# Diccionario para almacenar las relaciones de Markov
modelo_markov = {}

# Aprendemos las transiciones entre palabras
for i in range(len(palabras) - 1):
    palabra_actual = palabras[i]
    siguiente_palabra = palabras[i + 1]

    if palabra_actual not in modelo_markov:
        modelo_markov[palabra_actual] = []
    modelo_markov[palabra_actual].append(siguiente_palabra)

# Función para generar texto
def generar_texto(inicio, longitud=10):
    palabra = inicio
    resultado = [palabra]

    for _ in range(longitud - 1):
        if palabra in modelo_markov:
            palabra = random.choice(modelo_markov[palabra])
            resultado.append(palabra)
        else:
            break  # No hay palabra siguiente

    return ' '.join(resultado)

# Ejemplo de generación
print("Texto generado:")
print(generar_texto("yo", longitud=12))
