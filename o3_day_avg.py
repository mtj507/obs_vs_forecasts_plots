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
emission='o3'
Emission='O3'
conv = 2*10**9

#defining cities from whhich to extract data
city_1='Newcastle'
city_2='Newcastle'
city_3='Newcastle'

api=openaq.OpenAQ()
opendata=api.measurements(df=True, country='GB', parameter=emission, limit=10000) 
df=pd.DataFrame(opendata)
df=df.loc[(df['city']==city_1)|(df['city']==city_2)|(df['city']==city_3)]
df=df.drop_duplicates(subset='location', keep='first')
df=df.reset_index(drop=False)
city=df['city']
location=df['location']
latitude=df['coordinates.latitude']
longitude=df['coordinates.longitude']
no_locations=len(df.index)  #counting number of indexes for use in np.aranges


defra_csv='/users/mtj507/scratch/defra_data/defra_o3_eng_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
ddf=ddf.dropna(axis=0)
ddf=ddf.replace('No data', np.nan)
ddf['hour']=ddf.index.hour

ddf.loc['2019-09-22':'2019-11-03']

for i in np.arange(0, no_locations):
    ddf1=ddf.loc[:,['hour', f'{location[i]}']]
    ddf1=ddf1.dropna(axis=0)
    ddf1['value']=ddf1[f'{location[i]}'].astype(float)
    ddf_mean=ddf1.groupby('hour').mean()
    ddf_std=ddf1.groupby('hour').std()
    plt.plot(ddf_mean.index, ddf_mean['value'], label='Observation', color='blue')
    plt.fill_between(ddf_mean.index, (ddf_mean['value']+ddf_std['value']), (ddf_mean['value']-ddf_std['value']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')


#openaq data has both utc and local time so use date.utc
#for i in np.arange(0,no_locations):
#    data=api.measurements(df=True, city=f'{city[i]}', parameter=emission, location=f'{location[i]}', limit=1000) #date_from='2018-01-01')
#    time=data['date.utc'].index.hour
#    df1=pd.DataFrame(data)
#    df1['time']=time
#    df2=df1.drop(columns=['date.utc'])
#    df_mean=df2.groupby('time').mean()
#    df_std=df2.groupby('time').std()
#    plt.plot(df_mean.index, df_mean['value'], label='Observation', color='blue')
#    plt.fill_between(df_mean.index, (df_mean['value']+df_std['value']), (df_mean['value']-df_std['value']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')   
   
    
    mod_data = np.zeros((24,43))  #ensure 2nd number here is equal to number of days being used
    
    dates=np.append(np.arange(922,931), np.arange(1001,1032))
    dates=np.append(dates, np.arange(1101,1104))
 
    for j in range(len(dates)):
        forecast_date=f'2019{str(dates[j]).zfill(4)}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[emission].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model=pd.DataFrame(ds[emission].data[:,0,model_lat, model_lon])
        df_model.index=ds.time.data
        df_model.columns=[emission]
        df_model.index.name='date_time'
        time=df_model.index.hour
        df_model['Hour']=time
        df_model=df_model.reset_index()
        df_model=df_model.iloc[0:24]
        df_model=df_model.sort_index() 
        df_model[emission]=df_model[emission]*conv       
         
        for k in range(24):
            mod_data[k,j] = df_model[emission].loc[df_model['Hour'] == k].values[0]
        


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
  















