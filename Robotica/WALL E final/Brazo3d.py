import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Importar trayectoria desde el archivo externo
from Trayectoria import trayectoria, mesa_z_altura

# Parámetros del brazo
L1 = 6.0
L2 = 5.0

# Cinemática inversa
def calcular_angulos(x, y, z):
    if z < mesa_z_altura:
        raise ValueError("Altura por debajo de la mesa no permitida")

    theta0 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    dx = r
    dz = z

    D = (dx**2 + dz**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)
    theta2 = np.arccos(D)

    theta1 = np.arctan2(dz, dx) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))

    z1_test = L1 * np.sin(theta1)
    if z1_test < mesa_z_altura:
        theta2 = -theta2
        theta1 = np.arctan2(dz, dx) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        z1_test = L1 * np.sin(theta1)
        if z1_test < mesa_z_altura:
            raise ValueError("Eslabón primario baja debajo de la mesa")

    return theta0, theta1, theta2

# Cinemática directa
def cinemática_directa(theta0, theta1, theta2):
    x0, y0, z0 = 0, 0, 0
    x1 = L1 * np.cos(theta1) * np.cos(theta0)
    y1 = L1 * np.cos(theta1) * np.sin(theta0)
    z1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2) * np.cos(theta0)
    y2 = y1 + L2 * np.cos(theta1 + theta2) * np.sin(theta0)
    z2 = z1 + L2 * np.sin(theta1 + theta2)
    return (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)

# ---------- SIMULACIÓN ----------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

tray_x, tray_y, tray_z = [], [], []

# Coordenadas de la mesa para sombreado
mesa_x = [2, 7, 7, 2]
mesa_y = [2, 2, 7, 7]
mesa_x, mesa_y = np.meshgrid(mesa_x, mesa_y)
mesa_z = np.full_like(mesa_x, mesa_z_altura)

for punto in trayectoria:
    x, y, z = punto
    try:
        theta0, theta1, theta2 = calcular_angulos(x, y, z)
        base, codo, efector = cinemática_directa(theta0, theta1, theta2)
        tray_x.append(efector[0])
        tray_y.append(efector[1])
        tray_z.append(efector[2])

        ax.cla()
        ax.plot([base[0], codo[0]], [base[1], codo[1]], [base[2], codo[2]], 'r-', linewidth=3, label="Eslabón 1")
        ax.plot([codo[0], efector[0]], [codo[1], efector[1]], [codo[2], efector[2]], 'b-', linewidth=3, label="Eslabón 2")
        ax.scatter(*efector, c='g', s=100, label="Efector final")
        ax.plot(tray_x, tray_y, tray_z, 'k--', alpha=0.5, label="Trayectoria")

        ax.plot_surface(mesa_x, mesa_y, mesa_z, color='gray', alpha=0.4)

        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 10)
        ax.set_title("Brazo siguiendo trayectoria")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.pause(0.6)
    except Exception as e:
        print(f"⚠️ Punto omitido: {punto} -> {e}")
        continue

plt.show()
