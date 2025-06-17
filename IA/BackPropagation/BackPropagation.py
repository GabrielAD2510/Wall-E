import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Funciones de activación
def linear(z, derivative=False):
    a = z
    if derivative:
        da = 1
        return a, da
    return a


def sigmoid(z, derivative=False):
    a = 1/(1+np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a


def tanh(z, derivative=False):
    a = np.tanh(z)
    if derivative:
        da = (1 - a) * (1 + a)
        return a, da
    return a


def relu(z, derivative=False):
    a = z * (z >= 0)
    if derivative:
        da = np.array(z >= 0, dtype=float)
        return a, da
    return a

class MLP:

  def __init__(self, layers_dims, 
               hidden_activation=relu,
               output_activation=sigmoid,
               learning_rate=0.1):

    self.L=len(layers_dims)-1
    self.w =[np.ndarray] * (self.L+1)
    self.b =[np.ndarray] * (self.L+1)
    self.f = [None] * (self.L+1)
    self.eta = learning_rate
    self.err_t=[]

    # Initialize weights
    for l in range(1, self.L+1):
        self.w[l]=-1+2*np.random.rand(layers_dims[l],layers_dims[l-1])
        self.b[l]=-1+2*np.random.rand(layers_dims[l],1)
        if(l==self.L):
            self.f[l]=output_activation
        else:
            self.f[l]=hidden_activation

  def batchSize(self,X):
     p=0
     if(type(X)==np.ndarray):
        p=X.shape[1]
     else:
        p=len(X[0])
     return p
  
  def predict(self, X):
     p=self.batchSize(X)
     A=X
     for l in range(1, self.L+1):
        A=(self.w[l]@A)+self.b[l]
        A=self.f[l](A)
     return A

  def fit(self, X, Y, epochs=500):

    # Number of samples
    p = X.shape[1]
    learn=self.eta/p
    #Gradient Descent
    for _ in range(epochs):

      # Initialize activations and their derivatives
      A = [None] * (self.L + 1)
      dA = [None] * (self.L + 1)
      lg = [None] * (self.L + 1)
                
      A[0]=X
      # Propagation
      for l in range(1, self.L+1):
        z=(self.w[l]@A[l-1])+self.b[l]
        A[l], dA[l]=self.f[l](z,True)

      # Backpropagation
      err=(Y-A[l])
      for l in range(self.L, 0,-1 ):
        if(l==self.L):
            lg[l]=err*dA[l]
        else:
            lg[l]=(self.w[l+1].T@lg[l+1])*dA[l]
      # Update parameters
      for l in range(1, self.L+1):
         self.w[l]+=learn*(lg[l]@A[l-1].T)
         self.b[l]+=learn*np.sum(lg[l])

      if(_% 100 ==0):
          print(np.sqrt(np.sum(err**2)/p))
          self.err_t.append(np.sqrt(np.sum(err**2)/p))


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

#XOR
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[1, 0, 0, 1]]) 

#Blobs
#df = pd.read_csv('blobs.csv')
#Circles
#df = pd.read_csv('circles.csv')
#Moons
#df = pd.read_csv('moons.csv')

# Escribe en el siguiente apartado la matriz X y Y
#X = df[['x1','x2']].to_numpy().T
#Y = df[['y']].to_numpy().T


net = MLP((2,10,20,1),learning_rate=0.2,hidden_activation=relu,output_activation=sigmoid)

net.fit(X, Y,epochs=200)
MLP_binary_classification_2d(X,Y,net)