import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# La función de activación lineal se usa en problemas de regresión
def linear(z, derivative=False):
  a = z
  if derivative:
    da = np.ones(z.shape, dtype=float)
    return a, da
  return a

# La función de activación logística se usa en 
# problemas de clasificación multi-etiquetas
def logistic(z, derivative=False):
  a = 1/(1 + np.exp(-z))
  if derivative:
    da = np.ones(z.shape, dtype=float)
    return a, da
  return a

# La función de activación Softmax se usa en
# problemas de clasificación multiclase con un 
# solo ganador
def softmax(z, derivative=False):
  exp=np.exp(z-np.max(z,axis=0))
  a=exp/(np.sum(exp,axis=0))
  if(derivative):
    da=np.ones(z.shape, dtype=float)
    return a,da
  return a

class OLN:
  """One-Layer Network"""
  
  def __init__(self, n_inputs, n_outputs,
               activation_funtion=linear):
    self.w = 
    self.b = 
    self.f = 
    self.err_t=[]


  def predict(self, X):
    Z=np.dot(self.w,X)+self.b
    Z=self.f(Z)
    return Z

  def fit(self, X, Y, epochs=1000,  lr=0.1):
    _,p=X.shape
    eta=lr/p
    for _ in range(epochs):
        #Calcular el error estimado
        Z=
        err=
        #Entrenar red
        self.w+=
        self.b+=

        if(_% 100 ==0):
          print(np.sqrt(np.sum(err**2)/p))
          self.err_t.append(np.sqrt(np.sum(err**2)/p))


# Generación de Conjunto de datos para clasificación

# Límites
minx = -5
maxx = 5

# Número de clases y puntos por clase
classes = 5
p_c = 20
X = np.zeros((2, classes * p_c))
Y = np.zeros((classes, classes * p_c))


for i in range(classes):
    seed = minx + (maxx - minx) * np.random.rand(2,1)
    X[:, i*p_c:(i+1)*p_c] = seed + 0.15 * np.random.randn(2, p_c)
    Y[i, i*p_c:(i+1)*p_c] = np.ones((1, p_c))


# Instancia una red neuronal con el numero de entradas, salidas
# y función de activación correctas
net =OLN(2,5,softmax)

# Entrena la red neuronal
net.fit(X,Y,epochs=1000,lr=0.1)

# Guarda las predicciones de la red de todos los datos en X
Ypred = net.predict(X)

# Colores para dibujar las clases

cm	= ['b','g','r','c','m']

# Gráfico con los datos originales
ax1=plt.subplot(1, 2, 1)
y_c =np.argmax(Y, axis=0)
for i in range(X.shape[1]):
    ax1.scatter(X[0,i], X[1,i], c=cm[y_c[i]], edgecolors='k')
ax1.axis([-5.5,5.5,-5.5,5.5])
ax1.set_title('Problema Original')
ax1.grid()

# Gráfico de las predicciones de la red
ax2=plt.subplot(1, 2, 2)
y_c =np.argmax(Ypred, axis=0)
for i in range(X.shape[1]):
    ax2.scatter(X[0,i], X[1,i], c=cm[y_c[i]], edgecolors='k')
ax2.axis([-5.5,5.5,-5.5,5.5])
ax2.set_title('Predicción de la red')
ax2.grid()

plt.figure()
plt.plot(net.err_t)
plt.show()