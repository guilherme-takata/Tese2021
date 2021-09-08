import pandas as pd 
import glob 
from IPython.display import display



flight_df = pd.read_csv(r"D:\TCC\Tese\Datasets\Flight_dataset_merged.csv", sep = ';', low_memory = False, encoding = 'utf-8')


display(flight_df)

print(flight_df['CANCELLED'].sum())

print(flight_df['FL_DATE'].unique())