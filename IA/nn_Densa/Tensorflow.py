# Paquetería básica
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importamos Tensorflow y Keras
import tensorflow as tf
from tensorflow import keras

# Usamos funcionalidades básicas de Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow.keras.backend as K

def AutoDataSet():
    #Leer base de datos
    dataset = pd.read_csv('auto-mpg.csv')
    print(dataset.head())

    dataset.info()
    #Remplazar datos faltantes ? con nan
    np.array(dataset.horsepower)
    dataset['horsepower'].replace('?', np.nan, inplace=True)
    np.array(dataset.horsepower)
    #Remover datos con valor nan
    dataset = dataset.dropna()
    dataset.isna().sum()
    #Cambiar origen por tres columnas 
    origin = dataset.pop('origin')
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0
    #Impirme los ultimos datos de la base de datos
    print(dataset.tail())
    #Asigna valor a X y Y descartando las columnas escritas
    x = np.asanyarray(dataset.drop(columns=['mpg', 'car name']))
    y = np.asanyarray(dataset[['mpg']])
    x = StandardScaler().fit_transform(x)
    return dataset, x, y


def r2score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

def build_model():

  model = keras.Sequential()
  model.add(keras.layers.Dense(64, activation='relu', input_shape=[x.shape[1]]))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(32, activation='tanh'))
  model.add(keras.layers.Dense(1, activation='linear'))

  model.compile(loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(learning_rate=0.1),
                metrics=[keras.metrics.MAE, r2score])
  return  model

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.subplot(121)
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(hist['epoch'], hist['loss'],
           label='Train')
  plt.plot(hist['epoch'], hist['val_loss'],
           label = 'Val')
  plt.yscale('log')
  plt.legend()

  plt.subplot(122)
  plt.title('R2 Score')
  plt.xlabel('Epoch')
  plt.ylabel('r2score')
  plt.plot(hist['epoch'], hist['r2score'],
           label='Train')
  plt.plot(hist['epoch'], hist['val_r2score'],
           label = 'Val')
  plt.ylim([0,1])
  plt.legend()
  plt.show()


dataset, x, y = AutoDataSet()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=42)

model = build_model()
model.summary()

history = model.fit(xtrain, ytrain,
                    batch_size=5000, epochs=5000,
                    validation_data=(xtest, ytest), verbose=1)

plot_history(history)