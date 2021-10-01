import pmdarima as pmd
import pandas as pd 
import numpy as np
import pandas_datareader.data as web
import math
from IPython.display import display
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing


def arima_model(series ): # Funcão para achar os melhores parâmetros para o modelo de ARIMA


    autoarima = pmd.auto_arima(series, trace = True, start_p = 1, start_q = 1, test = 'adf', d = 1, seasonal = True, start_P = 1, start_Q = 1, start_D = 1, m = 12)
    autoarima.fit(series)

    return(autoarima)


def load_dataframe(name: str):
    if name == 'Flight':

        dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", sep = ';', low_memory = False).groupby('FL_DATE', as_index = True).agg(Num_cancelados = ('CANCELLED', 'sum'))

    elif name == 'Gold':

        dataframe = web.DataReader('GC=F', 'yahoo', start = '2001-01-01', end = '2021-08-27')

    return(dataframe)   

dataset_name = 'Gold'

dataframe = load_dataframe(dataset_name)

data_series = dataframe['Close']

train_data = data_series[:math.floor(len(data_series)*0.7)]

test_data = data_series[math.floor(len(data_series)*0.7):]

print(test_data)

print(test_data.index)

model = arima_model(train_data)

model.fit(train_data)

forecast = model.predict(n_periods = 1)

print(forecast)

forecast_df = pd.DataFrame(forecast, index = test_data.index, columns = ['Predições'] )

print(forecast_df)

pd.concat([test_data, forecast_df], axis = 1).plot()

plt.show()

