import pmdarima as pmd
import pandas as pd 
import numpy as np
import pandas_datareader.data as web
import math
from IPython.display import display
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing


def arima_model(series): # Funcão para achar os melhores parâmetros para o modelo de ARIMA


    autoarima = pmd.auto_arima(series, trace = True, start_p = 1, start_q = 1,max_p = 4, max_q = 4, test = 'adf', d = 1, seasonal = True, start_P = 1, start_Q = 1, start_D = 1, m = 12, stepwise = False)
    autoarima.fit(series)

    return(autoarima)


def load_dataframe(name: str):
    if name == 'Flight':

        dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", sep = ';', low_memory = False).groupby('FL_DATE', as_index = True).agg(Num_cancelados = ('CANCELLED', 'sum'))

    elif name == 'Apple':

        dataframe = web.DataReader('AAPL', 'yahoo', start = '2017-01-01', end = '2021-09-30')

    return(dataframe)   

dataset_name = 'Apple'

dataframe = load_dataframe(dataset_name)

display(dataframe)

data_series = dataframe['Close']

display(data_series)

train_data = data_series[:math.floor(len(data_series)*0.8)]

test_data = data_series[math.floor(len(data_series)*0.8):]

print(test_data)

print(test_data.index)

model = arima_model(train_data)

model.fit(train_data)

model_pred = model.predict(len(test_data))

plt.plot(train_data.index, train_data, color = 'purple', label = 'Dados de treinamento')

plt.plot(test_data.index, model_pred, color = 'blue', label = 'Predição do modelo')

plt.plot(test_data.index, test_data, color = 'red', label = 'Dados reais')

plt.show()

