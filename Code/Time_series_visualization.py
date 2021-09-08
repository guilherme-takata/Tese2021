from re import split
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
	dataframe = web.DataReader('GC=F', 'yahoo', start = '2001-01-01', end = '2021-08-27')

	series = dataframe['Close']

	if diff:

		series = series.diff()

	if auc:

		series = series[1:]

		plot_pacf(series)

		plt.show()

		return()

	series.plot()

	plt.show()

	return()


def plot_flight_database(diff : bool, auc : bool):

	'''
	Plotta a série temporal do número de cancelamentos de voos de 2009 até 2018

	diff : Parâmetro para determinar se plottamos a série diferenciada

	auc : Parâmetro que determina se será plotado a autocorrelação da série
	'''

	dataframe = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", low_memory = False, sep = ';').groupby('FL_DATE', as_index = True).agg(Num_cancelados = ('CANCELLED', 'sum'))

	series = dataframe['Num_cancelados']

	if diff:

		series = series.diff()
		series = series[1:]

	if auc:

		plot_pacf(series)

		plt.show()

		return()


	series.plot()

	plt.show()

	return()



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

		plot_pacf(series)



#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#

plot_flight_database(False, True)

