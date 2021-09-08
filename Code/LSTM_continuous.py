from keras import * 
import pandas as pd
from pandas.core.arrays import integer 
import pandas_datareader.data as web
import sklearn as sk 
import numpy as np
from IPython.display import display
import tensorflow as tf 
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Auxiliary_functions import *

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#
'''
Aqui carregamos nossos dados na memória e chamamos a função definida anteriormente
'''
dataframe_path = fr"C:\Users\GuilhermeTakata\Documents\TCC\TCC\Datasets\Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv"

dataframe = web.DataReader('GC=F', 'yahoo', start = '2001-01-01', end = '2020-12-31') # Carregamos nossos dados de treino na memória usando o pandas

display(dataframe)

# scaler = MinMaxScaler(feature_range = (0,1))

# data = scaler.fit_transform(dataframe[['Close']])

data = dataframe['Close'].to_list()

print(data)

len_lags = 14 # tamanho da nossa janela

X_train, y_train = split_sequence(data, len_lags)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

'''
Parte responsável pela construção, experimentação e treinamento das redes 
'''

Model = Sequential() # Inicialização da nossa rede
Model.add(LSTM(64, activation = 'tanh', input_shape = (len_lags, 1), return_sequences = True))
Model.add(LSTM(64, activation = 'relu', input_shape = (len_lags, 1)))
# Model.add(LSTM(64))
# Model.add(LSTM(32))
Model.add(Dense(15, use_bias = False))
Model.compile(loss = 'mean_squared_error', optimizer = 'adam') # Compilação do modelo indicando qual função de perda a ser usada e o otimizador de escolha

Model.fit(X_train, y_train, epochs = 50, batch_size = 30, verbose = 1) # Chamada do treinamento e otimização da rede



#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

'''
Carregamento dos dados de validação do modelo
'''

dataframe2 = web.DataReader('GC=F', 'yahoo', start = '2021-01-01', end = '2021-08-29') # Carregamento na memória da nossa base de teste 

# data_test = dataframe2['Close'].diff()

data_test = dataframe2['Close'].values.tolist()

X_test, Y_test = split_sequence(data, len_lags)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

predictions = Model.predict(X_test)


#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

'''
Plotagem para comparação dos dados de teste e as predições feitas
'''

index = [x for x in range(len(Y_test))]


plt.plot(index , Y_test[:], label = "original")

plt.plot(index, predictions[:,0][:], label = "predição do modelo")

plt.legend(loc = 'best')
plt.show()