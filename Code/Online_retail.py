import pandas as pd
from IPython.display import display
import datetime as dt

dataframe_2009_2010 = pd.read_csv(r"C:\Users\GuilhermeTakata\Downloads\Kaggle_retail_DB\Year 2009-2010.csv", sep = ',', low_memory = False)

dataframe_2010_2011 = pd.read_csv(r"C:\Users\GuilhermeTakata\Downloads\Kaggle_retail_DB\Year 2010-2011.csv", sep = ',', low_memory = False)

display(dataframe_2009_2010)

display(dataframe_2010_2011)

dataframe_2009_2011 = pd.concat([dataframe_2009_2010, dataframe_2010_2011])

display(dataframe_2009_2011)

dataframe_2009_2011["InvoiceDate"] = pd.to_datetime(dataframe_2009_2011["InvoiceDate"])

dataframe_2009_2011["InvoiceDate"] = dataframe_2009_2011["InvoiceDate"].dt.date

display(dataframe_2009_2011)

dataframe_2009_2011.to_csv(r"D:\Tese2021_datasets\Online_Retail.csv", sep = ',', encoding = 'utf-8', index = False)