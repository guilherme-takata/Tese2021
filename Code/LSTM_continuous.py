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
from sklearn.metrics import *

__name__ = "main"

def load_prep_data(): 
    '''
    Aqui carregamos nossos dados na memória e chamamos a função definida anteriormente
    '''

    dataframe_train = web.DataReader('AAPL', 'yahoo', start = '2000-01-01', end = '2021-07-31') # Carregamos nossos dados de treino na memória usando o pandas

    display(dataframe_train)

    global scaler

    scaler = MinMaxScaler()

    series = dataframe_train.iloc[:,3:4].values

    processed_series = scaler.fit_transform(series)

    global len_lags

    len_lags = 30 # tamanho da nossa janela

    X_train, Y_train = split_sequence(processed_series, len_lags)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    return(X_train, Y_train)


def create_train_model(train_X: np.array, train_Y: np.array) -> Sequential:

    '''
    Parte responsável pela construção, experimentação e treinamento das redes 
    '''

    Model = Sequential() # Inicialização da nossa rede
    Model.add(LSTM(units = 125, return_sequences = False, input_shape = (len_lags,1)))
    Model.add(Dense(units = 10))
    Model.add(Dense(units = 1, activation = "linear"))

    Model.compile(loss = 'mean_squared_error', optimizer = 'adam') # Compilação do modelo indicando qual função de perda a ser usada e o otimizador de escolha

    Model.fit(train_X, train_Y, epochs = 20, batch_size = 30, verbose = 1) # Chamada do treinamento e otimização da rede

    return(Model)


def validation_loading():
    
    '''
    Carregamento dos dados de validação do modelo
    '''

    global dataframe2

    dataframe2 = web.DataReader('AAPL', 'yahoo', start = '2021-08-01', end = '2021-11-20') # Carregamento na memória da nossa base de teste 

    data_test = dataframe2.iloc[:,3:4].values

    data_test_processed = scaler.transform(data_test)

    X_test, Y_test = split_sequence(data_test_processed, len_lags)

    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return(X_test_reshaped, Y_test)


def model_validation(X_test, Y_test, Model):

    '''
    Plotagem para comparação dos dados de teste e as predições feitas
    '''

    predictions = Model.predict(X_test)

    predictions = scaler.inverse_transform(predictions)

    Y_test = scaler.inverse_transform(Y_test)

    fig = go.Figure(go.Scatter(x = dataframe2.index, y = Y_test[:,0], name = "Dados reais", mode = "lines+markers"))

    fig.add_trace(go.Scatter(x = dataframe2.index, y = predictions[:,0], name = "Predições do modelo", mode = "lines+markers"))

    fig.write_image(fr"C:\Users\GuilhermeTakata\Documents\Tese2021\Graphs and Images\LSTM_AAPL.png", format = "png", width = 1600, height = 900)

    fig.show()

    return(mean_squared_error(predictions, Y_test))

def main():

    X_train, Y_train = load_prep_data()
    Model = create_train_model(X_train, Y_train)
    X_test, Y_test = validation_loading()
    validation_mse = model_validation(X_test, Y_test, Model)
    print(validation_mse)

if __name__ == "main":

    main()