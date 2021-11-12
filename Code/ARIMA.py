import pmdarima as pmd
import pandas as pd 
import numpy as np
import pandas_datareader.data as web
import math
from IPython.display import display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def arima_model(series): # Funcão para achar os melhores parâmetros para o modelo de ARIMA


    autoarima = pmd.auto_arima(series, trace = True, start_p = 1, start_q = 1, max_p = 5, max_q = 5, d = 1, seasonal = True, start_P = 1, start_Q = 1, D = 1, m = 12, stepwise = True, maxiter= 200)
    autoarima.fit(series)

    return(autoarima)


def load_dataframe(name: str):

    if name == 'Rio crimes':

        dataframe = pd.read_csv(r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", sep = ',', low_memory = False)

        dataframe["Month"] = pd.to_datetime(dataframe["Month"], format = "%Y-%m")

        dataframe.set_index("Month", inplace = True)

    else:

        dataframe = web.DataReader('AAPL', 'yahoo', start = '2017-01-01', end = '2021-09-30')

    return(dataframe)

dataset_name = 'Rio crimes'

dataframe = load_dataframe(dataset_name)

data_series = dataframe['Passengers']

train_data = data_series[:math.floor(len(data_series)*0.8)]

test_data = data_series[math.floor(len(data_series)*0.8):]

model = arima_model(train_data)

fitted_model = model.fit(train_data)

model_pred = fitted_model.predict(len(test_data))

fig_test = go.Figure(go.Scatter(x = dataframe.index, y = model_pred, name = "Predições do modelo", mode = "lines+markers"))

fig_test.add_trace(go.Scatter(x = dataframe.index, y = test_data, name = "Dados reais", mode = "lines+markers"))

fig_test.show()

# plt.plot(train_data.index, train_data, color = 'purple', label = 'Dados de treinamento')

# plt.plot(test_data.index, model_pred, color = 'blue', label = 'Predição do modelo')

# plt.plot(test_data.index, test_data, color = 'red', label = 'Dados reais')

# plt.show()

