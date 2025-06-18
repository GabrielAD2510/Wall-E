#Importar base de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons  

#pythorch
import torch
import torch.nn as nn
import torch.optim as optim
#graficas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
#Librerias
import numpy as np
import pandas as pd

is_process=False

# Crear la figura y la cuadrícula
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(4, 4, figure=fig)

class SimpleNN(nn.Module):
    er_t:list[float]
    er_d:float
    max_eph:int

    def __init__(self, inputs=2, outputs=1, est_err=0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(inputs, 16,bias=True)  # Capa oculta con 10 neuronas
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, outputs,bias=True)
        self.sigmoid = nn.Sigmoid()  
        self.criterion = nn.BCELoss()
        self.estErr=est_err
        self.er_t=[]
        self.er_d=0
        self.max_eph=0
        self.confucionM=np.array([[1,0],[0,1]])
        self.exactitud=0
        self.presicion=0
        self.sensibilidad=0
        self.F1score=0
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)
    
    def Get_errT(self):
        if(len(self.er_t)==0):
            return 0
        return self.er_t[-1]
    
    def Get_errD(self):
        return self.er_d

    def startTrain(self):
        self.er_t=[]
        self.er_d=[]
        self.max_eph=0

    def completeTrain(self):
        global is_process
        is_process=False
    
    def test(self,X,Y):
        y_pred = self(X)
        loss = self.criterion(y_pred, Y)
        self.er_d=loss.item()
        y_pred=np.array(((y_pred>0.5)*1.0))
        y_pred=y_pred.astype(int)
        Y=np.array(Y).astype(int)
        
        N=len(Y) # Numero de datos
        pp=np.count_nonzero(Y) # Cantidad de positivos en valor real
        pn=N-pp # Numero de negativos
        TP=np.sum(np.bitwise_and(y_pred,Y)) 
        TN=pp-TP
        FP=np.sum(np.bitwise_and(1-y_pred,1-Y))
        FN=pn-FP

        self.confucionM=np.array([[TP,FN],[TN,FP]])/N

        self.exactitud=(TP+TN)/N
        self.presicion=(TP)/(TP+FP)
        self.sensibilidad=TP/(TP+FN)
        self.F1score=2*((self.presicion*self.sensibilidad)/(self.presicion+self.sensibilidad))


    def fit_Gradiant(self,X,Y,epoch=200):
        optimizer_gd = optim.Adam(self.parameters(), lr=0.1)
        i=0
        for _ in range(epoch):
            optimizer_gd.zero_grad()
            y_pred = self(X)
            loss = self.criterion(y_pred, Y)

            self.er_t.append(loss.item())
            if(loss.item()<self.estErr):
               break

            loss.backward()
            optimizer_gd.step()
            i+=1
        self.max_eph=i
        print("End GM training")
        self.completeTrain()
    
     
        print("End LM training")
        self.completeTrain()

def MLP_binary_classification_2d(X,Y,net,axis):
    global fig
    axis.clear()
    for i in range(X.shape[0]):
        if Y[i]==0:
            axis.plot(X[i,0], X[i,1], '.r')
        else:
            axis.plot(X[i,0], X[i,1], '.b')
    xmin, ymin=np.min(X[:,0])-1.0, np.min(X[:,1])-1.0 
    xmax, ymax=np.max(X[:,0])+1.0, np.max(X[:,1])+1.0
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,100), 
                         np.linspace(ymin,ymax,100))
                         
    data = np.c_[xx.ravel(), yy.ravel()]
    d=torch.from_numpy(data).float()
    zz = net(d).detach().numpy()
    zz = zz.reshape(xx.shape)
    axis.contourf(xx,yy,zz, alpha=0.8, 
                 cmap=plt.cm.RdBu)
    axis.set_xlim([xmin,xmax])
    axis.set_ylim([ymin,ymax])
    

# Generar datos en forma de dos medias lunas  
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)  

# Separar datos para el entranamiento y para el test
##test size-> % de datos para la prueba
##random_state -> manejo del randomizado
X_train, X_test , y_train , y_test =train_test_split(X,y,test_size=0.2,random_state=42)

#Escalamiento
scalar=StandardScaler()
X_trainS=scalar.fit_transform(X_train)
X_testS=scalar.transform(X_test)

#Agregar Bias
#X_train=np.c_[np.ones(X_trainS.shape[0]),X_trainS]
#X_test=np.c_[np.ones(X_testS.shape[0]),X_testS]

X_t= torch.from_numpy(X_train).float()
X_d= torch.from_numpy(X_test).float()
y_t= torch.from_numpy(y_train).float().view(-1, 1)
y_d= torch.from_numpy(y_test).float().view(-1, 1)

#GD
modelGD = SimpleNN()

# Crear subgrids
sub_gs = gs[2:, :].subgridspec(1, 1)  # Región 2x2
sub_gs_1x3 = gs[:2, :4].subgridspec(1, 3)  # Región 1x3

# Agregar gráficas
ax_GDgraph = fig.add_subplot(sub_gs[0, 0])
ax_GDgraph.set_title("Back Propagation")

ax_GD_error = fig.add_subplot(sub_gs_1x3[0, 0])

data=np.array([[1,0],[0,1]])
ax_GD_conf=fig.add_subplot(sub_gs_1x3[0, 1])
ax_GD_conf.matshow(data,cmap='Blues')
ax_GD_info = fig.add_subplot(sub_gs_1x3[0, 2])

for i in range(2):
    for j in range(2):
        ax_GD_conf.text(j, i, str(data[i, j]), ha='center', va='center', color='black', fontsize=12)

for ax in [ax_GD_error, ax_GD_info,ax_GD_conf]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

def UpdateText():
    ax_GD_info.clear()
    ax_GD_conf.clear()
    ax_GD_conf.matshow(modelGD.confucionM,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax_GD_conf.text(j, i, str(modelGD.confucionM[i, j]), ha='center', va='center', color='black', fontsize=12)
    ax_GD_info.text(0.1, 0.3, 
                "Error Train:{tr:.3f}\nError Test:{tt:.3f}\n#Epocas:{e}\nExactitud:{exact:.3f}\n#Presicion:{pres:.3f}\nSensibilidad:{sens:.3f}\nF1score:{F1:.3f}".format(
                   tr=modelGD.Get_errT(), tt=modelGD.Get_errD(), e=modelGD.max_eph,exact=modelGD.exactitud,pres=modelGD.presicion,sens=modelGD.sensibilidad,F1=modelGD.F1score), 
                   ha='left', va='bottom', fontsize=8)

print("GD model")
modelGD.fit_Gradiant(X_t,y_t,1000)
modelGD.test(X_d,y_d)
MLP_binary_classification_2d(X_test,y_test,modelGD,ax_GDgraph)
ax_GD_error.clear()
ax_GD_error.plot(modelGD.er_t)
UpdateText()
fig.canvas.draw()

UpdateText()

# Mostrar la figura
plt.tight_layout()
plt.show()

# Visualizar los datos  
#MLP_binary_classification_2d(X_test,y_test,model)
