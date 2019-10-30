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
city_1='York'
city_2='York'
city_3='York'

api=openaq.OpenAQ()
opendata=api.measurements(df=True, country='GB', parameter=emission, limit=10000) 
df=pd.DataFrame(opendata)
#df=df.loc[(df['city']==city_1)|(df['city']==city_2)|(df['city']==city_3)]
df=df.loc[df['location']=='York Fishergate']
df=df.drop_duplicates(subset='location', keep='first')
df=df.reset_index(drop=False)
city=df['city']
location=df['location']
latitude=df['coordinates.latitude']
longitude=df['coordinates.longitude']
no_locations=len(df.index)  #counting number of indexes for use in np.aranges


api=openaq.OpenAQ()
#openaq data has both utc and local time so use date.utc

for i in np.arange(0,no_locations):
    data=api.measurements(df=True, parameter=emission, location=f'{location[i]}', limit=10000)
    time=data['date.utc'].index.hour
    df1=pd.DataFrame(data)
    df1['time']=time
    df2=df1.drop(columns=['date.utc'])
    df_mean=df2.groupby('time').mean()
    df_std=df2.groupby('time').std()
    plt.plot(df_mean.index, df_mean['value'], label='Observation', color='blue')
    plt.fill_between(df_mean.index, (df_mean['value']+df_std['value']), (df_mean['value']-df_std['value']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')   
    
    mod_data = np.zeros((24,23))  #ensure 2nd number here is equal to number of days being used below
    
    dates=np.append(np.arange(922,930), np.arange(1001,1016))
 
    for j in range(len(dates)):
        forecast_date=f'2019{str(dates[j]).zfill(4)}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
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
    plt.title(location[i])
    path='/users/mtj507/scratch//obs_vs_forecast/plots/'+emission
    plt.savefig(path+'/'+emission+f'_{location[i]}_comparison.png')   
    plt.close()
    print(location[i])
  















