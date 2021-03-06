from Auxiliary_functions import Collatz, split_sequence
import pandas as pd
from matplotlib import pyplot as plt
import pandas_datareader.data as web
import numpy as np
from statsmodels.tsa.stattools import pacf, acf
from IPython.display import display
import datetime as dt
from traitlets.traitlets import Int
import yfinance as yf
import plotly.graph_objects as go
import kaleido
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#
'''
Definição das funções usadas para plotar nossas séries temporais
'''


# Recebe um booleano para decidir se vamos plottar a série já diferenciada ou não
def plot_stock_prices(diff: bool, auc: bool):
	'''
	Plotta a série temporal dos preços do ouro (em dólares)

	diff : Parâmetro para determinar se plottamos a série diferenciada

	auc : Parâmetro que determina se será plotado a autocorrelação da série
	'''
	dataframe = web.DataReader(
		'AAPL', 'yahoo', start='2010-01-01', end='2021-09-30')

	series = dataframe['Close']

	suffix = ''
	if diff:

		series = series.diff()

	if auc:

		plot_acf(series, lags=100)

		plt.show()

		return()

	fig = go.Figure(data=go.Scatter(x=dataframe.index, y=series, text='Close'))

	fig.update_layout({"title": 'Preço das ações da Apple ao fechar a bolsa', "xaxis": {
					  "title": "Data"}, "yaxis": {'title': 'Preço de fechamento'}})

	fig.write_image(fr"C:\Users\GuilhermeTakata\Documents\Tese2021\Graphs and Images\AAPL_dataset.png",
					width=1600, format='png', height=900)

	fig.show()
	return()


def plot_flight_database(diff: bool, auc: bool):
	'''
	Plotta a série temporal do número de cancelamentos de voos de 2009 até 2018

	diff : Parâmetro para determinar se plottamos a série diferenciada

	auc : Parâmetro que determina se será plotado a autocorrelação da série
	'''

	dataframe = pd.read_csv(
		r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', sep=',', low_memory=False)

	dataframe["Month"] = pd.to_datetime(dataframe["Month"], format="%Y-%m")

	dataframe.set_index("Month", inplace=True)

	print(dataframe.index)

	series = dataframe['Passengers']

	if diff:

		series = series.diff()
		series = series[1:]

	if auc:
		plot_acf(series, lags=30)

		plt.savefig(
			r"C:\Users\GuilhermeTakata\Documents\Tese2021\Graphs and Images\Passengers_acf.png",)

		plt.show()
		return()


# def plot_Collatz(x0: Int, diff: Int, auc: Int):
# 	'''
# 	Plotta a sequência de Collatz começando em x0

# 	diff : Parâmetro para determinar se plottamos a série diferenciada

# 	auc : Parâmetro que determina se será plotado a autocorrelação da série
# 	'''

# 	series = pd.Series(Collatz(x0))

# 	if diff:

# 		series = series.diff()

# 		series = series[1:]

# 	if auc:

# 		plot_acf(series)


#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

plot_stock_prices(False, True)
