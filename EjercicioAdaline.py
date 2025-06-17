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

class Adaline:

    def __init__(self, n_inputs, learning_rate=0.1):
        self.w = - 1 + 2 * np.random.rand(n_inputs)
        self.b = - 1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict_proba(self, X):
        Z = np.dot(self.w, X) + self.b
        Y_est = 1/(1+np.exp(-Z))
        return Y_est
    
    def predict(self, X, umbral=0.5):
        Z = np.dot(self.w, X) + self.b
        Y_est = 1/(1+np.exp(-Z))
        return 1 * (Y_est > umbral)

    def train(self, X, Y, epochs=100):
        p = X.shape[1]
        for _ in range(epochs):
            #Escribe la actualizacion de pesos
            Y_est = self.predict_proba(X)
            self.w += (self.eta/p) * np.dot((Y - Y_est), X.T).ravel()
            self.b += (self.eta/p) * np.sum(Y - Y_est)

def MLP_binary_classification_2d(X,Y,net):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0,i]==0:
            plt.plot(X[0,i], X[1,i], '.r')
        else:
            plt.plot(X[0,i], X[1,i], '.b')
    xmin, ymin=np.min(X[0,:])-0.5, np.min(X[1,:])-0.5
    xmax, ymax=np.max(X[0,:])+0.5, np.max(X[1,:])+0.5
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,100), 
                         np.linspace(ymin,ymax,100))
    data = [xx.ravel(), yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx,yy,zz, alpha=0.8, 
                 cmap=plt.cm.RdBu)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.grid()
    plt.show()

def draw_2d_percep(model):
  w1, w2, b = model.w[0], model.w[1], model.b 
  plt.plot([-2, 2],[(1/w2)*(-w1*(-2)-b),(1/w2)*(-w1*2-b)],'--k')

def LoadLogicalDoorData(t='and'):
  X = np.array([[0, 0, 1, 1],[0, 1, 0, 1]])
  Y = np.array([[0, 0, 0, 1]])
  if(t=='or'):
    Y = np.array([[0, 1, 1, 1]])
  elif(t=='xor'):
    Y = np.array([[1, 0, 0, 1]])
  return X,Y

def LoadCVSFile():
   # Cargar CSV
  df = pd.read_csv('adeline_dataset.csv')
  X=df[['x1','x2']].to_numpy().T
  Y = np.array([[(y+1)/2 for y in df['y']]])
  return X,Y

def IMC_data(p=200):
  X = np.zeros((2,p))
  Y = np.zeros((1,p))
  for i in range(p):
    # masa aleatoria
    X[0,i] = np.random.uniform(40,120)

    # estatura aleatoria
    X[1,i] = np.random.uniform(1,2.2)

    imc = X[0,i] / X[1,i]**2

    if imc >= 25:
      Y[0,i]=1
    else:
      Y[0,i]=0

  return X,Y

# Ejemplo Compuertas logicas
#X,Y=LoadLogicalDoorData()
#X,Y=IMC_data()
X,Y=LoadCVSFile()

neuron = Adaline(2, 1)
neuron.train(X,Y)

MLP_binary_classification_2d(X,Y,neuron)