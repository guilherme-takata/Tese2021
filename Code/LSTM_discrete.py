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


len_lags = 5

'''
Seção para carregamento da base de dados e adequação dos nossos dados de treinamento
'''

dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", low_memory = False, sep = ';') 
dataframe['FL_DATE'] = pd.to_datetime(dataframe['FL_DATE'], format = '%Y-%m-%d')
dataframe.set_index('FL_DATE', inplace = True)
dataframe = dataframe.groupby(pd.Grouper(freq = 'M')).agg(Num_cancelados = ('CANCELLED', 'sum'))
dataframe['Num_cancelados'] = pd.to_numeric(dataframe['Num_cancelados'], downcast = 'integer')
dataframe = dataframe[['Num_cancelados']]

series = dataframe['Num_cancelados'].values.tolist()

split_margin = math.floor(len(series) * 0.8) # Número usado para pegar 80% dos registros da nossa base

series_train = series[ :split_margin + 1] # Série usada para o treinamento

series_test = series[split_margin: ] # Série usada para a validação


X_train, y_train = split_sequence(series_train, len_lags)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


'''
Construção da rede a ser usada e treinada no nosso conjunto de dados
'''

Model = Sequential() # Inicialização da nossa rede
Model.add(LSTM(1056, input_shape = (len_lags, 1), return_sequences = True))
Model.add(LSTM(528, return_sequences = True))
Model.add(LSTM(256, return_sequences = True))
Model.add(LSTM(128))
Model.add(Dense(120, activation = 'linear'))
Model.add(Dense(64, activation = 'linear'))
Model.add(Dense(32, activation = 'linear'))
Model.add(Dense(10, activation = 'linear'))
Model.add(Dense(1, activation = 'linear'))

Model.compile(loss = 'mean_squared_error', optimizer = 'nadam') # Compilação do modelo indicando qual função de perda a ser usada e o otimizador de escolha

Model.fit(X_train, y_train, epochs = 20, batch_size = 500, verbose = 1) # Chamada do treinamento e otimização da rede



