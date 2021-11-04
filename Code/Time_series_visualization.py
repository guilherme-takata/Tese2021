from Auxiliary_functions import Collatz, split_sequence
import pandas as pd
from matplotlib import pyplot as plt
import pandas_datareader.data as web
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from IPython.display import display
import datetime as dt
from traitlets.traitlets import Int
import yfinance as yf
import plotly.graph_objects as go
import kaleido

#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#
'''
Definição das funções usadas para plotar nossas séries temporais
'''

def plot_stock_prices(diff : bool, auc : bool): # Recebe um booleano para decidir se vamos plottar a série já diferenciada ou não

	'''
	Plotta a série temporal dos preços do ouro (em dólares)

	diff : Parâmetro para determinar se plottamos a série diferenciada

	auc : Parâmetro que determina se será plotado a autocorrelação da série
	'''
	dataframe = web.DataReader('AAPL', 'yahoo', start = '2010-01-01', end = '2021-09-30')

	series = dataframe['Close']
	suffix = ''
	if diff:

		series = series.diff()

	if auc:

		suffix = 'ACF'
		
		series = series[1:]

		plot_pacf(series)

		plt.show()

		return()

	fig = go.Figure(data = go.Scatter(x = dataframe.index, y = series, text = 'Close'))

	fig.update_layout({"title": 'Preço das ações da Apple ao fechar a bolsa', "xaxis" :{"title":"Data"}, "yaxis": {'title' :'Preço de fechamento'}})

	fig.write_image(fr"C:\Users\GuilhermeTakata\Documents\Tese2021\Graphs and Images\AAPL_dataset_{suffix}.png" , width = 1600, format = 'png', height = 900 )


	return()


def plot_flight_database(diff : bool, auc : bool):

	'''
	Plotta a série temporal do número de cancelamentos de voos de 2009 até 2018

	diff : Parâmetro para determinar se plottamos a série diferenciada

	auc : Parâmetro que determina se será plotado a autocorrelação da série
	'''

	dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", low_memory = False, sep = ';')
	dataframe['FL_DATE'] = pd.to_datetime(dataframe['FL_DATE'], format = '%Y-%m-%d')
	dataframe.set_index('FL_DATE', inplace = True)
	'''.groupby('FL_DATE', as_index = False).agg(Num_cancelados = ('CANCELLED', 'sum'))'''
	dataframe = dataframe.groupby(pd.Grouper(freq = 'M')).agg(Num_cancelados = ('CANCELLED', 'sum'))
	dataframe['Num_cancelados'] = pd.to_numeric(dataframe['Num_cancelados'], downcast = 'integer')
	print(dataframe.dtypes)

	series = dataframe['Num_cancelados']

	display(series)

	suffix = ''

	if diff:

		series = series.diff()
		series = series[1:]
		
	if auc:

		plot_acf(series)

		suffix = 'acf'

		plt.show()

		return()

	fig = go.Figure(data = go.Scatter(x = dataframe.index, y = dataframe['Num_cancelados']))

	fig.update_layout({"title": 'Número de voos cancelados por mês', "xaxis" :{"title":"Data"}, "yaxis": {'title' :'Número de voos cancelados'}})

	fig.write_image(fr"C:\Users\GuilhermeTakata\Documents\Tese2021\Graphs and Images\Flight_dataset_{suffix}.png", width = 1600, format = 'png', height = 900 )

	



def plot_Collatz(x0: Int, diff: Int, auc: Int):
	'''
	Plotta a sequência de Collatz começando em x0

	diff : Parâmetro para determinar se plottamos a série diferenciada

	auc : Parâmetro que determina se será plotado a autocorrelação da série
	'''
	
	series = pd.Series(Collatz(x0))

	if diff:

		series = series.diff()
		
		series = series[1:]
		
	if auc:

		plot_acf(series)


#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

plot_flight_database(False, True)
plot_stock_prices(False, False)
