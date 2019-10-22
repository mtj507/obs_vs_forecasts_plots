import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from itertools import chain


#defining emission to be observed and conversion (can be found in conversion file)
emission='no2'
Emission='NO2'
conv = 1.88*10**9

#defining cities from whhich to extract data
city_1='York'
city_2='York'
city_3='York'

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


api=openaq.OpenAQ()
#openaq data in local time.
for i in np.arange(0,no_locations):
    data=api.measurements(df=True, city=f'{city[i]}', parameter=emission, location=f'{location[i]}', limit=1000)
    data['date.utc'] = data['date.utc']
    time=data['date.utc'].index.hour
    df1=pd.DataFrame(data)
    df1['time']=time
    df2=df1.drop(columns=['date.utc'])
    df2=df2.groupby('time').mean()
    plt.plot(df2.index, df2['value'], label='Observation')
     
 
    mod_data = np.zeros((24,23))
    
    dates=chain(range(922,930), range(1001,1016))
 
    for j in dates:
        forecast_date=f'2019{str(j).zfill(4)}'
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
         
        for dates in range(922,930):
            for k in range(24):
             mod_data[k,dates-922] = df_model[emission].loc[df_model['Hour'] == k].values[0]
        
        for dates in range(1001,1016):
            for k in range(24):
             mod_data[k,dates-1001] = df_model[emission].loc[df_model['Hour'] == k].values[0]



    plt.plot(range(24),np.mean(mod_data,1),label='Model',color='red')
    plt.xlabel('Hour of Day')
    plt.ylabel(Emission + ' ug/m3')
    plt.legend()
    plt.title(location[i])
    plt.savefig('/users/mtj507/scratch//obs_vs_forecast/plots/'+emission+f'_{location[i]}_comparison.png')
    plt.close()
    print(location[i])
  















