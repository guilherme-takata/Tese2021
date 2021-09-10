import pmdarima as pmd
import pandas as pd 
import numpy as np
import pandas_datareader.data as web
import math
from IPython.display import display
import matplotlib.pyplot as plt

dataset_name = 'Gold'


def arima_model(series, seasonality): # Funcão para achar os melhores parâmetros para o modelo de ARIMA


    autoarima = pmd.auto_arima(series, test = 'adf', trace = True, seasonal = seasonality)

    return(autoarima)


def load_dataframe(name: str):
    if name == 'Flight':

        dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", sep = ';', low_memory = False)

    elif name == 'Gold':

        dataframe = web.DataReader('GC=F', 'yahoo', start = '2001-01-01', end = '2021-08-27')

    return(dataframe)   


dataframe = load_dataframe(dataset_name)

data_series = dataframe['Close']

train_data = data_series[:math.floor(len(data_series)*0.7)]

test_data = data_series[math.floor(len(data_series)*0.7):]

print(test_data)

print(test_data.index)

model = arima_model(train_data, seasonality = True)

model.fit(train_data)

forecast = model.predict(n_periods = len(test_data))

print(forecast)

forecast_df = pd.DataFrame(forecast, index = test_data.index, columns = ['Predições'] )

print(forecast_df)

pd.concat([test_data, forecast_df], axis = 1).plot()

plt.show()

