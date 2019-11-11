import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


metadata_csv='/users/mtj507/scratch/defra_data/defra_eng_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.reset_index(drop=False)
area=metadata['Zone']
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
environment=metadata['Environment Type']
environments=pd.unique(environment)

emissions=['no2', 'no', 'pm25', 'o3']



for environment in environments:
    metadata_csv='/users/mtj507/scratch/defra_data/defra_eng_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.reset_index(drop=False)
    metadata=metadata.loc[metadata['Environment Type']==environment]
    check=pd.unique(metadata['Environment Type'])
    locations=metadata['Site Name']
    headers=locations.values.tolist()    
    headers.append('hour')  
    print(check)
    
    for i in emissions:
        defra_csv='/users/mtj507/scratch/defra_data/defra_'+i+'_eng_2019.csv'
        print(i)
        ddf=pd.read_csv(defra_csv, low_memory=False)
        ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
        ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
        ddf=ddf.dropna(axis=0)
        ddf=ddf.replace('No data', np.nan)
        ddf['hour']=ddf.index.hour
        ddf=ddf.loc['2019-09-22':'2019-11-05']
        ddf=ddf.loc[:,headers] 
        ddf=ddf.astype(float)
        ddf=ddf.dropna(axis=1,how='all')
        if (environment == 'Industrial Suburban' and i == 'pm25'):
          continue 
        ddf_mean=ddf.groupby('hour').mean()
        ddf_mean['mean']=ddf_mean.mean(axis=1)       
        ddf_std=ddf.groupby('hour').std()
        ddf_std['std']=ddf_std.mean(axis=1)
        plt.plot(ddf_mean.index, ddf_mean['mean'], label='Observation', color='blue')
        plt.fill_between(ddf_mean.index, (ddf_mean['mean']+ddf_std['std']), (ddf_mean['mean']-ddf_std['std']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')










