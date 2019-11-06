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

#defining cities from whhich to extract data
city_1='London'
city_2='York'
city_3='Leeds'

date1='2019-09-22'
date2='2019-11-04'

#obtaining meta data from openaq like lat + lon
api=openaq.OpenAQ()
opendata=api.measurements(df=True, country='GB', parameter=emission, limit=10000) 
df=pd.DataFrame(opendata)
df=df.loc[df['location']=='Leeds Centre']
#df=df.loc[(df['city']==city_1)|(df['city']==city_2)|(df['city']==city_3)]
df=df.drop_duplicates(subset='location', keep='first')
df=df.reset_index(drop=False)
city=df['city']
location=df['location']
latitude=df['coordinates.latitude']
longitude=df['coordinates.longitude']
no_locations=len(df.index)  #counting number of indexes for use in np.aranges

#change to UTF csv before moving across to Viking and edit doc so its easy to import by deleting first 3 rowns and moving time and date column headers into same row as locations. Delete empty rows up to 'end' at bottom and format time cells to time.
#using defra rather than openaq for actual data
no2_defra_csv='/users/mtj507/scratch/defra_data/defra_no2_eng_2019.csv'
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

no_defra_csv='/users/mtj507/scratch/defra_data/defra_no_eng_2019.csv'
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


#weekday diurnal
for i in np.arange(0, no_locations):
    ddf1=ddf.loc[:,['hour', f'{location[i]}', 'weekday', 'day and month']]
    noddf1=noddf.loc[:,[f'{location[i]}']]
    ddf1['no2']=ddf1[f'{location[i]}'].astype(float)
    noddf1['no']=noddf1[f'{location[i]}'].astype(float)
    ddf1['no']=noddf1['no']
    ddf1['nox']=ddf1['no']+ddf1['no2']
    ddf2=ddf1.loc[(ddf['weekday'] >= 0) & (ddf['weekday'] <= 4)]
    ddf2=ddf2.dropna(axis=0)
    ddf_mean=ddf2.groupby('hour').mean()
    ddf_std=ddf2.groupby('hour').std()
    plt.plot(ddf_mean.index, ddf_mean['nox'], label='Observation', color='blue')
    plt.fill_between(ddf_mean.index, (ddf_mean['nox']+ddf_std['nox']), (ddf_mean['nox']-ddf_std['nox']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

    days_of_data=len(pd.unique(ddf2['day and month']))
    dates=pd.unique(ddf2['day and month'])
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
    path='/users/mtj507/scratch/obs_vs_forecast/plots/nox/weekend_weekday_diurnal/'
    plt.savefig(path+'nox'+f'_{location[i]}_weekday_diurnal.png')
    plt.close()
    print(location[i])
  


#weekend diurnal
for x in np.arange(0, no_locations):
    ddf1=ddf.loc[:,['hour', f'{location[x]}', 'weekday', 'day and month']]
    noddf1=noddf.loc[:,[f'{location[x]}']]
    ddf1['no2']=ddf1[f'{location[x]}'].astype(float)
    noddf1['no']=noddf1[f'{location[x]}'].astype(float)
    ddf1['no']=noddf1['no']
    ddf1['nox']=ddf1['no']+ddf1['no2']
    ddf2=ddf1.loc[(ddf['weekday'] >= 5) & (ddf['weekday'] <= 6)]
    ddf2=ddf2.dropna(axis=0)
    ddf_mean=ddf2.groupby('hour').mean()
    ddf_std=ddf2.groupby('hour').std()
    plt.plot(ddf_mean.index, ddf_mean['nox'], label='Observation', color='blue')
    plt.fill_between(ddf_mean.index, (ddf_mean['nox']+ddf_std['nox']), (ddf_mean['nox']-ddf_std['nox']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

    days_of_data=len(pd.unique(ddf2['day and month']))
    dates=pd.unique(ddf2['day and month'])
    mod_data = np.zeros((24,days_of_data))  


    for y in range(len(dates)):
        forecast_date=f'2019{str(dates[y]).zfill(4)}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[emission].data
        nospec=ds['no'].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[x]-lats))
        model_lon=np.argmin(np.abs(longitude[x]-lons))
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
       
  
        for z in range(24):
            mod_data[z,y] = df_model['nox'].loc[df_model['Hour'] == z].values[0]




    plt.plot(range(24),np.median(mod_data,1),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=1)
    Q3=np.percentile(mod_data, 75, axis=1)
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red') 
    plt.xlabel('Hour of Day')
    plt.ylabel('NOX ug/m3')
    plt.legend()
    plt.title(location[i])
    path='/users/mtj507/scratch/obs_vs_forecast/plots/nox/weekend_weekday_diurnal/'
    plt.savefig(path+'nox'+f'_{location[i]}_weekend_diurnal.png')
    plt.close()
    print(location[i])






