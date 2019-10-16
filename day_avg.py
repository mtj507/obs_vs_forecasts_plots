import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#defining emission to be observed and conversion (can be found in conversion file)
emission='no2'
Emission='NO2'
conv = 1.88*10**9

#setting up plot
fig,ax = plt.subplots()
ax.set_xlabel('Hour')
ax.set_ylabel(Emission+'/ug m-3')
ax.set_ylim([0,100])
ax.set_xlim(0, 24)
ax.grid(True)

#defining cities from whhich to extract data
city_1='York'
city_2='York'
city_3='York'

#input of locations from which to get data
openaqdata=[['London', 'London Westminster', 51.494, -0.132, 'blue'], ['London', 'London Bloomsbury', 51.522, -0.126, 'green'], ['London', 'London Marylebone Road', 51.522, -0.155, 'red'], ['London', 'London N. Kensington', 51.521, -0.214, 'pink'], ['London', 'London Eltham', 51.452, 0.07, 'gold'], ['London', 'London Bexley', 51.466, 0.184, 'teal'], ['London', 'London Teddington Bushy Park', 51.425, -0.346, 'navy'], ['London', 'London Harlington', 51.488, -0.442, 'slategrey'], ['London', 'Camden Kerbside', 51.544, -0.176, 'crimson'], ['York', 'York Bootham', 53.967, -1.087, 'orchid'], ['York', 'York Fishergate', 53.951, -1.076, 'lawngreen'], ['Newcastle', 'Newcastle Centre', 54.978, -1.611, 'black'], ['Edinburgh', 'Edinburgh St Leonards', 55.945, -3.183, 'orange']]
df=pd.DataFrame(openaqdata, columns=['city', 'location', 'latitude', 'longitude', 'color'])
df=df.loc[(df['city']==city_1)|(df['city']==city_2)|(df['city']==city_3)]
df=df.reset_index(drop=True)
city=df['city']
location=df['location']
latitude=df['latitude']
longitude=df['longitude']
color=df['color']
no_locations=len(df.index)  #counting number of indexes for use in np.aranges


#plotting openaq data
api = openaq.OpenAQ()
for i in np.arange(0, no_locations):
  data=api.measurements(df=True, city=f'{city[i]}', parameter=emission, location=f'{location[i]}', limit=1000)
  data['date.utc'] = data['date.utc']+pd.Timedelta('30 min')#accounting for 30 min delay behind model
  time=data['date.utc'].index.hour
  df1=pd.DataFrame(data)
  df1['time']=time
  df2=df1.drop(columns=['date.utc'])
  df2=df2.groupby('time').mean()
  ax.plot(df2.index, df2['value'], color=f'{color[i]}', label=Emission+f' {location[i]}')
  plt.legend() 


#turning model dataset into dataframe
for j in np.arange(0, no_locations):
  for i in np.arange(922, 930):
    forecast_date=f'2019{str(i).zfill(4)}'
    f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
    ds=xr.open_dataset(f)
    spec=ds[emission].data
    lats=ds['lat'].data
    lons=ds['lon'].data
    model_lat=np.argmin(np.abs(latitude[j]-lats))
    model_lon=np.argmin(np.abs(longitude[j]-lons))
    df_model=pd.DataFrame(ds[emission].data[:,0,model_lat, model_lon])
    df_model.index=ds.time.data
    df_model.columns=[emission]
    df_model.index.name='hour'
    df_model.iloc[0:24]
    
plt.show()













