import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# Parámetros del brazo
# -------------------------
L1 = 5.0  # Longitud primer eslabón (elevación 1)
L2 = 3.0  # Longitud segundo eslabón (elevación 2)

# -------------------------
# Ángulos (en grados)
# -------------------------
theta0_deg = 45  # Rotación de base sobre eje Z
theta1_deg = 60  # Elevación del primer eslabón desde la base
theta2_deg = 30  # Elevación del segundo eslabón desde el primero

# Convertir a radianes
theta0 = np.radians(theta0_deg)  # rotación base
theta1 = np.radians(theta1_deg)  # elevación 1
theta2 = np.radians(theta2_deg)  # elevación 2

# -------------------------
# Cinemática directa
# -------------------------
# Base (fija)
x0, y0, z0 = 0, 0, 0

# Primer eslabón (sube o baja sobre el plano vertical que gira con θ0)
x1 = L1 * np.cos(theta1) * np.cos(theta0)
y1 = L1 * np.cos(theta1) * np.sin(theta0)
z1 = L1 * np.sin(theta1)

# Segundo eslabón (sigue el plano anterior, elevación relativa)
x2 = x1 + L2 * np.cos(theta1 + theta2) * np.cos(theta0)
y2 = y1 + L2 * np.cos(theta1 + theta2) * np.sin(theta0)
z2 = z1 + L2 * np.sin(theta1 + theta2)

# -------------------------
# Dibujar brazo
# -------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Eslabones
ax.plot([x0, x1], [y0, y1], [z0, z1], 'r-', linewidth=4, label='Eslabón 1')
ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=4, label='Eslabón 2')
ax.scatter(x2, y2, z2, c='g', s=100, label='Efector final')

# Base circular (opcional, solo para referencia visual)
circle_radius = 1.0
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(circle_radius * np.cos(theta), circle_radius * np.sin(theta), zs=0, zdir='z', alpha=0.3)

# -------------------------
# Ejes
# -------------------------
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Brazo robótico 3D con base giratoria y 2 elevaciones")
ax.legend()
plt.tight_layout()
plt.show()
