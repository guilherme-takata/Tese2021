import pmdarima as pmd
import pandas as pd 
import numpy as np


dataset_name = 'Flight'


def arima_model(series, seasonality): # Funcão para achar os melhores parâmetros para o modelo de ARIMA


    autoarima = pmd.auto_arima(series, start_p = 1, start_q = 1, test = 'adf', trace = True, seasonal = seasonality)

    return(autoarima)


def load_dataframe(name: str):
    if name == 'Flight':
        dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", sep = ';', low_memory = False)

    elif name == 'Gold':
        dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\gold_price_data.csv", sep = ',', low_memory = False)

    return(dataframe)


load_dataframe(dataset_name)

