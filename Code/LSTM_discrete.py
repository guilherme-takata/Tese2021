import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from Auxiliary_functions import *
import math


len_lags = 14

'''
Seção para carregamento da base de dados e adequação dos nossos dados de treinamento
'''

dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", low_memory = False, sep = ';').groupby('FL_DATE', as_index = True).agg(Num_cancelados = ('CANCELLED', 'sum'))

series = dataframe['Num_cancelados'].values.tolist()

split_margin = math.floor(len(series) * 0.8) # Número usado para pegar 80% dos registros da nossa base

series_train = series[ : split_margin] # Série usada para o treinamento

series_test = series[split_margin: ] # Série usada para a validação


X_train, y_train = split_sequence(series_train, len_lags)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))










