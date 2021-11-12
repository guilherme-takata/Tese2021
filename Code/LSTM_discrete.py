from numpy.core.numeric import False_
import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from Auxiliary_functions import *
import math
from IPython.display import display
import tensorflow as tf 
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Auxiliary_functions import *
from tensorflow.keras import Sequential
import plotly.graph_objects as go


len_lags = 6

'''
Seção para carregamento da base de dados e adequação dos nossos dados de treinamento
'''

dataframe = pd.read_csv(r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", low_memory = False, sep = ',') 

dataframe["Month"] = pd.to_datetime(dataframe["Month"], format = "%Y-%m")

dataframe.set_index("Month", inplace = True)

series = dataframe['Passengers'].values.tolist()

split_margin = math.floor(len(series) * 0.8) # Número usado para pegar 80% dos registros da nossa base

train_index = dataframe.iloc[ :split_margin + 1].index.tolist()

test_index = dataframe.iloc[split_margin: ].index.tolist()

series_train = series[ :split_margin + 1] # Série usada para o treinamento

series_test = series[split_margin: ] # Série usada para a validação


X_train, Y_train = split_sequence(series_train, len_lags)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

print(X_train)

'''
Construção da rede a ser usada e treinada no nosso conjunto de dados
'''

Model = Sequential() # Inicialização da nossa rede
Model.add(LSTM(128, input_shape = (len_lags, 1)))
# Model.add(LSTM(160, input_shape = (len_lags, 1)))
# Model.add(Dense(16, use_bias = True))
Model.add(Dense(1))
Model.compile(loss = 'mean_squared_error', optimizer = 'nadam') # Compilação do modelo indicando qual função de perda a ser usada e o otimizador de escolha

Model.fit(X_train, Y_train, epochs = 100, batch_size = 20 , verbose = 2) # Chamada do treinamento e otimização da rede


X_test, Y_test = split_sequence(series_test, len_lags)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

Test_predictions = Model.predict(X_test)

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

fig = go.Figure(data = go.Scatter(x = test_index, y = Test_predictions[:,0][:], text = "Previsões do modelo"))

fig.add_trace(go.Scatter(x = test_index, y = series_test[:], text = "Dados reais"))

fig.show()