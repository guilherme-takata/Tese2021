from keras import *
from numpy.core.numeric import True_ 
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
from tensorflow.keras import Sequential
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#
'''
Aqui carregamos nossos dados na memória e chamamos a função definida anteriormente
'''

dataframe_train = web.DataReader('AAPL', 'yahoo', start = '2000-01-01', end = '2021-07-31') # Carregamos nossos dados de treino na memória usando o pandas

display(dataframe_train)

scaler = MinMaxScaler()

series = dataframe_train.iloc[:,3:4].values

processed_series = scaler.fit_transform(series)

len_lags = 40 # tamanho da nossa janela

X_train, Y_train = split_sequence(processed_series, len_lags)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

print(X_train.shape)

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

'''
Parte responsável pela construção, experimentação e treinamento das redes 
'''

Model = Sequential() # Inicialização da nossa rede
Model.add(LSTM(units = 85, return_sequences = True, input_shape = (len_lags,1)))
Model.add(Dropout(0.2))
# Model.add(LSTM(units = 70, return_sequences = True))
# Model.add(Dropout(0.2))
Model.add(LSTM(units = 85))
# Model.add(Dropout(0.2))
Model.add(Dense(10))
Model.add(Dense(units = 1))

Model.compile(loss = 'mean_squared_error', optimizer = 'adam') # Compilação do modelo indicando qual função de perda a ser usada e o otimizador de escolha

Model.fit(X_train, Y_train, epochs = 10, batch_size = 32, verbose = 1) # Chamada do treinamento e otimização da rede


#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

'''
Carregamento dos dados de validação do modelo
'''

dataframe2 = web.DataReader('AAPL', 'yahoo', start = '2021-08-01', end = '2021-11-17') # Carregamento na memória da nossa base de teste 

data_test = dataframe2.iloc[:,3:4].values

data_test_processed = scaler.transform(data_test)

X_test, Y_test = split_sequence(data_test_processed, len_lags)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

predictions = Model.predict(X_test)

print(Model.evaluate(X_test, Y_test))

predictions = scaler.inverse_transform(predictions)

Y_test = scaler.inverse_transform(Y_test)

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

'''
Plotagem para comparação dos dados de teste e as predições feitas
'''

fig = go.Figure(go.Scatter(x = dataframe2.index, y = Y_test[:,0], name = "Dados reais", mode = "lines+markers"))

fig.add_trace(go.Scatter(x = dataframe2.index, y = predictions[:,0], name = "Predições do modelo", mode = "lines+markers"))

fig.write_image(fr"C:\Users\GuilhermeTakata\Documents\Tese2021\Graphs and Images\LSTM_AAPL.png", format = "png", width = 1600, height = 900)

fig.show()