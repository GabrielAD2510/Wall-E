# Trayectoria.py

mesa_z_altura = 3.0  # altura de la mesa (NUNCA CAMBIAR)

def generar_trayectoria_fig(origen, lado):
    x0, y0, z0 = origen
    L = lado
    puntos = [
        [x0,     y0,     z0],
        [x0+L,   y0,     z0],
        [x0+L, y0,     z0+1],
        [x0,   y0,     z0+1],
        [x0,     y0,     z0],
        [x0,     y0+3,   z0],
        [x0+L,   y0+3,   z0],
        [x0+L, y0,     z0+1],
        [x0,   y0,     z0+1],
        [x0,     y0+3,   z0],
        [x0,     y0+3,   z0+1],
        [x0+L,   y0+3,   z0+1],
        [x0+L, y0,     z0+1],
        [x0+L, y0,     z0],
        [x0+L, y0+3,   z0],
        [x0+L, y0+3,   z0+1],
        [x0,   y0+3,   z0+1],
        [x0,   y0+3,   z0]
    ]
    return puntos

# Parámetros configurables
origen_fig = [4, 2, mesa_z_altura]
lado_fig = 2
trayectoria = generar_trayectoria_fig(origen_fig, lado_fig)