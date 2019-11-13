import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

#defining emission to be observed and conversion (can be found in conversion file)
emission='no2'
Emission='NO2'
conv = 1.88*10**9


#defining dates of Defra data
date1='2019-09-22'
date2='2019-11-04'

#types of environment: Background Urban , Traffic Urban , Industrial Urban , Background Rural , Industrial Suburban , Background Suburban .

environment_type='Background Urban'
data_area='Greater London'
city='London'

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['Environment Type']==environment_type]
#metadata=metadata[metadata['Site Name'].str.match(city)]
#metadata=metadata.loc[metadata['Site Name']=='London N. Kensington']
metadata=metadata.reset_index(drop=False)
area=metadata['Zone']
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
environment=metadata['Environment Type']
no_locations=len(metadata.index)


#change to UTF csv before moving across to Viking and edit doc so its easy to import by deleting first 3 rowns and moving time and date column headers into same row as locations. Delete empty rows up to 'end' at bottom and format time cells to time.
no2_defra_csv='/users/mtj507/scratch/defra_data/defra_no2_uk_2019.csv'
ddf=pd.read_csv(no2_defra_csv, low_memory=False)
ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
ddf=ddf.dropna(axis=0)
ddf=ddf.replace('No data', np.nan)
ddf['hour']=ddf.index.hour
ddf['weekday']=ddf.index.weekday
ddf['month']=ddf.index.month.astype(str)
ddf['month']=ddf['month'].str.zfill(2)
ddf['day']=ddf.index.day.astype(str)
ddf['day']=ddf['day'].str.zfill(2)
ddf['day and month']=ddf['month']+ddf['day']

ddf=ddf.loc[date1:date2]

no_defra_csv='/users/mtj507/scratch/defra_data/defra_no_uk_2019.csv'
noddf=pd.read_csv(no_defra_csv, low_memory=False)
noddf.index=pd.to_datetime(noddf['Date'], dayfirst=True)+pd.to_timedelta(noddf['Time'])
noddf=noddf.loc[:, ~noddf.columns.str.contains('^Unnamed')]
noddf=noddf.dropna(axis=0)
noddf=noddf.replace('No data', np.nan)
noddf['hour']=noddf.index.hour
noddf['weekday']=noddf.index.weekday
noddf['month']=noddf.index.month.astype(str)
noddf['month']=noddf['month'].str.zfill(2)
noddf['day']=noddf.index.day.astype(str)
noddf['day']=noddf['day'].str.zfill(2)
noddf['day and month']=noddf['month']+noddf['day']

noddf=noddf[date1:date2]




for i in np.arange(0, no_locations):
    ddf1=ddf.loc[:,['hour', f'{location[i]}']]
    noddf1=noddf.loc[:,[f'{location[i]}']]
    ddf1['no2']=ddf1[f'{location[i]}'].astype(float)
    noddf1['no']=noddf1[f'{location[i]}'].astype(float)
    ddf1['no']=noddf1['no']
    ddf1['nox']=ddf1['no']+ddf1['no2']
    ddf1=ddf1.dropna(axis=0)
    ddf_mean=ddf1.groupby('hour').mean()
    ddf_std=ddf1.groupby('hour').std()
    plt.plot(ddf_mean.index, ddf_mean['nox'], label='Observation', color='blue')
    plt.fill_between(ddf_mean.index, (ddf_mean['nox']+ddf_std['nox']), (ddf_mean['nox']-ddf_std['nox']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

    days_of_data=len(pd.unique(ddf['day and month']))
    dates=pd.unique(ddf['day and month'])
    mod_data = np.zeros((24,days_of_data))  
    
    for j in range(len(dates)):
        forecast_date=f'2019{str(dates[j]).zfill(4)}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[emission].data
        nospec=ds['no'].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model=pd.DataFrame(ds[emission].data[:,0,model_lat, model_lon])
        nodf_model=pd.DataFrame(ds['no'].data[:,0,model_lat, model_lon])
        df_model.index=ds.time.data
        nodf_model.index=ds.time.data
        df_model.columns=[emission]
        nodf_model.columns=['no']
        df_model['no2']=df_model['no2']*conv
        df_model['no']=nodf_model['no']*1.23*10**9
        df_model['nox']=df_model['no2']+df_model['no']
        df_model['Hour']=df_model.index.hour
        df_model=df_model.reset_index()
        df_model=df_model.iloc[0:24]
        df_model=df_model.sort_index() 
       
  
        for k in range(24):
            mod_data[k,j] = df_model['nox'].loc[df_model['Hour'] == k].values[0]
        

    plt.plot(range(24),np.median(mod_data,1),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=1)
    Q3=np.percentile(mod_data, 75, axis=1)
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red') 
    plt.xlabel('Hour of Day')
    plt.ylabel('NOX ug/m3')
    plt.legend()
    plt.title(location[i])
    path='/users/mtj507/scratch//obs_vs_forecast/plots/nox/full_week_diurnal/'
    plt.savefig(path+'nox'+f'_{location[i]}_diurnal.png')
    plt.close()
    print(location[i])
  















