import pmdarima as pmd
import pandas as pd 
import numpy as np
import pandas_datareader.data as web
import math
from IPython.display import display
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
import seaborn as sns 
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error


def arima_model(series): # Funcão para achar os melhores parâmetros para o modelo de ARIMA


    autoarima = pmd.auto_arima(series, trace = True, start_p = 1, start_q = 1,max_p = 10, max_q = 10, test = 'adf', d = 0, seasonal = True, start_P = 1, start_Q = 1, m = 12, stepwise = False, max_P = 10, max_Q = 10, maxiter = 10000, D = 2)
    
    autoarima.fit(series)

    return(autoarima)


def load_dataframe(name: str):

    if name == 'Flight':

        dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", low_memory = False, sep = ';')
        dataframe['FL_DATE'] = pd.to_datetime(dataframe['FL_DATE'], format = '%Y-%m-%d')
        dataframe.set_index('FL_DATE', inplace = True)
        dataframe = dataframe.groupby(pd.Grouper(freq = 'M')).agg(Num_cancelados = ('CANCELLED', 'sum'))
        dataframe['Num_cancelados'] = pd.to_numeric(dataframe['Num_cancelados'], downcast = 'integer')
        dataframe = dataframe['Num_cancelados']
        # dataseries = dataframe['Num_cancelados']

    elif name == 'Apple':

        dataframe = web.DataReader('AAPL', 'yahoo', start = '2017-01-01', end = '2021-09-30')

        dataframe = dataframe['Close']
        
    return(dataframe)

dataset_name = 'Flight' # Determina qual conjunto de dados iremos usar

data_series = load_dataframe(dataset_name)

display(data_series)

train_data = data_series[:math.floor(len(data_series)*0.8)+1] # Divisão dos dados em partes para treinamento e de teste

test_data = data_series[math.floor(len(data_series)*0.8):] # Parte dos dados para testar o modelo

model = arima_model(train_data)

model.fit(train_data)

model_pred = model.predict(len(test_data))

print("Erro quadrático do modelo: ",mean_squared_error(test_data.values, model_pred))


fig = go.Figure()

fig.add_trace(go.Scatter(x = test_data.index, y = model_pred, mode = 'lines+markers', name = 'Predição do modelo'))

fig.add_trace(go.Scatter(x = test_data.index, y = test_data.values, mode = 'lines+markers', name = 'Dados reais'))

fig.show()

# plt.plot(train_data.index, train_data, color = 'purple', label = 'Dados de treinamento')

# plt.plot(test_data.index, model_pred, color = 'blue', label = 'Predição do modelo')

# plt.plot(test_data.index, test_data, color = 'red', label = 'Dados reais')

# plt.legend(loc = 'upper left')

# plt.show()

