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


len_lags = 10

'''
Seção para carregamento da base de dados e adequação dos nossos dados de treinamento
'''

dataframe = pd.read_csv(r"https://raw.githubusercontent.com/guilherme-takata/Tese2021_datasets/main/DOMensalEstadoDesde1991_agregado.csv", low_memory = False, sep = ';') 

dataframe["Mês/ano"] = pd.to_datetime(dataframe["Mês/ano"], format = "%m/%Y")

dataframe.set_index("Mês/ano", inplace = True)

series = dataframe['0'].values.tolist()

print(series)

split_margin = math.floor(len(series) * 0.8) # Número usado para pegar 80% dos registros da nossa base

train_index = dataframe.iloc[ :split_margin + 1].index.tolist()

test_index = dataframe.iloc[split_margin: ].index.tolist()

series_train = series[ :split_margin + 1] # Série usada para o treinamento

series_test = series[split_margin: ] # Série usada para a validação


X_train, Y_train = split_sequence(series_train, len_lags)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


'''
Construção da rede a ser usada e treinada no nosso conjunto de dados
'''

Model = Sequential() # Inicialização da nossa rede
Model.add(LSTM(128, input_shape = (len_lags, 1), recurrent_activation = 'tanh', return_sequences= True))
# Model.add(Dropout(0.2))
Model.add(LSTM(256, input_shape = (len_lags, 1), recurrent_activation = 'tanh'))
# Model.add(Dense(32, activation = 'relu'))
# Model.add(LSTM(16, return_sequences = True))
# Model.add(LSTM(256, return_sequences = True))
# Model.add(LSTM(128))
# Model.add(Dense(120, activation = 'relu'))
# Model.add(Dense(64, activation = 'relu'))
# Model.add(Dense(32, activation = 'relu'))
# Model.add(Dense(64))
Model.add(Dense(1))

Model.compile(loss = 'mean_squared_error', optimizer = 'adam') # Compilação do modelo indicando qual função de perda a ser usada e o otimizador de escolha

Model.fit(X_train, Y_train, epochs = 15, batch_size = 9 , verbose = 1) # Chamada do treinamento e otimização da rede


X_test, Y_test = split_sequence(series_test, len_lags)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

Test_predictions = Model.predict(X_test)

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

fig = go.Figure(data = go.Scatter(x = test_index, y = Test_predictions[:,0][:], text = "Previsões do modelo"))

fig.add_trace(go.Scatter(x = test_index, y = series_test, text = "Dados reais"))

fig.show()