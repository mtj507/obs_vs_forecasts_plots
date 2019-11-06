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
emission='pm25'
nasa_emission='pm25_rh35_gcc'
Emission='PM 2.5'
conv = 1

#defining cities from whhich to extract data
city_1='London'
city_2='York'
city_3='Newcastle'

#obtaining meta data from openaq like lat + lon
api=openaq.OpenAQ()
opendata=api.measurements(df=True, country='GB', parameter=emission, limit=10000) 
df=pd.DataFrame(opendata)
#df=df.loc[df['location']=='London Westminster']
df=df.loc[(df['city']==city_1)|(df['city']==city_2)|(df['city']==city_3)]
df=df.drop_duplicates(subset='location', keep='first')
df=df.reset_index(drop=False)
city=df['city']
location=df['location']
latitude=df['coordinates.latitude']
longitude=df['coordinates.longitude']
no_locations=len(df.index)  #counting number of indexes for use in np.aranges

#change to UTF csv before moving across to Viking and edit doc so its easy to import by deleting first 3 rowns and moving time and date column headers into same row as locations. Delete empty rows up to 'end' at bottom and format time cells to time.
#using defra rather than openaq for actual data
defra_csv='/users/mtj507/scratch/defra_data/defra_pm25_eng_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
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

#selecting dates for DEFRA data to be taken from.
ddf=ddf.loc['2019-09-22':'2019-11-04']


#creating weekday diurnal
for i in np.arange(0, no_locations):
    ddf1=ddf.loc[(ddf['weekday'] >= 0) & (ddf['weekday'] <= 4)]
    ddf2=ddf1.loc[:,['hour', f'{location[i]}']]
    ddf2=ddf2.dropna(axis=0)
    ddf2['value']=ddf2[f'{location[i]}'].astype(float)
    ddf_mean=ddf2.groupby('hour').mean()
    ddf_std=ddf2.groupby('hour').std()
    plt.plot(ddf_mean.index, ddf_mean['value'], label='Observation', color='blue')
    plt.fill_between(ddf_mean.index, (ddf_mean['value']+ddf_std['value']), (ddf_mean['value']-ddf_std['value']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')


    days_of_data=len(pd.unique(ddf1['day and month']))
    dates=pd.unique(ddf1['day and month']) 
    mod_data = np.zeros((24,days_of_data)) 

    for j in range(len(dates)):
        forecast_date=f'{dates[j]}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_2019'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[nasa_emission].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model=pd.DataFrame(ds[nasa_emission].data[:,0,model_lat, model_lon])
        df_model.index=ds.time.data
        df_model.columns=[nasa_emission]
        df_model.index.name='date_time'
        time=df_model.index.hour
        df_model['Hour']=time
        df_model=df_model.reset_index()
        df_model=df_model.iloc[0:24]
        df_model=df_model.sort_index() 
        df_model[nasa_emission]=df_model[nasa_emission]*conv       
         
        for k in range(24):
            mod_data[k,j] = df_model[nasa_emission].loc[df_model['Hour'] == k].values[0]
        

    plt.plot(range(24),np.median(mod_data,1),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=1)
    Q3=np.percentile(mod_data, 75, axis=1)
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red') 
    plt.xlabel('Hour of Day')
    plt.ylabel(Emission + ' ug/m3')
    plt.legend()
    plt.title(location[i]+' Weekday Diurnal')
    path='/users/mtj507/scratch/obs_vs_forecast/plots/'+emission+'/weekend_weekday_diurnal/'
    plt.savefig(path+emission+f'_{location[i]}_weekday_diurnal.png')
    plt.close()
    print(location[i])
  

#creating weekend diurnal
for x in np.arange(0, no_locations):
    ddf1=ddf.loc[(ddf['weekday'] >= 5) & (ddf['weekday'] <= 6)]
    ddf2=ddf1.loc[:,['hour', f'{location[x]}']]
    ddf2=ddf2.dropna(axis=0)
    ddf2['value']=ddf2[f'{location[x]}'].astype(float)
    ddf_mean=ddf2.groupby('hour').mean()
    ddf_std=ddf2.groupby('hour').std()
    plt.plot(ddf_mean.index, ddf_mean['value'], label='Observation', color='blue')
    plt.fill_between(ddf_mean.index, (ddf_mean['value']+ddf_std['value']), (ddf_mean['value']-ddf_std['value']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')


    days_of_data=len(pd.unique(ddf1['day and month']))
    dates=pd.unique(ddf1['day and month']) 
    mod_data = np.zeros((24,days_of_data)) 

    for y in range(len(dates)):
        forecast_date=f'{dates[y]}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_2019'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[nasa_emission].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[x]-lats))
        model_lon=np.argmin(np.abs(longitude[x]-lons))
        df_model=pd.DataFrame(ds[nasa_emission].data[:,0,model_lat, model_lon])
        df_model.index=ds.time.data
        df_model.columns=[nasa_emission]
        df_model.index.name='date_time'
        time=df_model.index.hour
        df_model['Hour']=time
        df_model=df_model.reset_index()
        df_model=df_model.iloc[0:24]
        df_model=df_model.sort_index() 
        df_model[nasa_emission]=df_model[nasa_emission]*conv       
         
        for z in range(24):
            mod_data[z,y] = df_model[nasa_emission].loc[df_model['Hour'] == z].values[0]
        

    plt.plot(range(24),np.median(mod_data,1),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=1)
    Q3=np.percentile(mod_data, 75, axis=1)
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red') 
    plt.xlabel('Hour of Day')
    plt.ylabel(Emission + ' ug/m3')
    plt.legend()
    plt.title(location[x]+' Weekend Diurnal')
    path='/users/mtj507/scratch/obs_vs_forecast/plots/'+emission+'/weekend_weekday_diurnal/'
    plt.savefig(path+emission+f'_{location[x]}_weekend_diurnal.png')
    plt.close()
    print(location[x])
  










