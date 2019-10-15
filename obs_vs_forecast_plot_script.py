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

#defining emission to be observed
emission='no2'
Emission='NO2'
conv = 1

#setting up plot
fig,ax = plt.subplots()
ax.set_xlabel('Date')
ax.set_ylabel(Emission + '/ug m-3')
ax.set_ylim([0,100])
ax.set_xlim([datetime(2019, 9, 21),datetime(2019, 10, 6)  ]) 
ax.xaxis.set_major_formatter(DateFormatter("%m-%d-%y %H:%M"))
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
df

#plotting openaq data
api = openaq.OpenAQ()
for i in np.arange(0, no_locations):
  data=api.measurements(df=True, city=f'{city[i]}', parameter=emission, location=f'{location[i]}', limit=1000)
  ax.plot(data['date.utc'], data['value'], color=f'{color[i]}', label=f'PM 2.5 {location[i]}')
  plt.legend()  

#plotting forecast data for j-desired cities and i-desired dates
for j in np.arange(0, no_locations):
  for i in np.arange(922,930):
    forecast_date=f'2019{str(i).zfill(4)}'
    f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
    ds=xr.open_dataset(f)
    spec=ds[emission].data
    lats=ds['lat'].data
    lons=ds['lon'].data
    spec=ds[emission].sel(lat=f'{latitude[j]}', lon=f'{longitude[j]}', method='nearest').data*1.88*10**9
    spec=spec[0:23]
    time=ds.time.data[0:23]
    ax.plot(time, spec, linestyle= ':', color=f'{color[j]}')

#setting colour of forecast data separately otherwise multiplle iteratiosn appear in legend. 
#for i in range(len(location.unique())):
for i in np.arange(0, no_locations):
  plt.plot([],[], linestyle=':', color=f'{color[i]}',label=f'PM 2.5 Forecast {location[i]}', alpha=1)
  plt.legend()


plt.show()
#plt.close()




#plotting maps
#pdf = PdfPages ('/users/mtj507/scratch/plots/pm25_practice_plot.pdf')

#for i in np.arange(0,24):
#  print(i)
#  fig = plt.figure(figsize=(10, 10))
#  ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#  ax.set_extent([-10, 5, 48, 62], ccrs.PlateCarree())
#  lons=lons[:]
#  lats=lats[:]
#  data=(spec[i,0,:,:])*(10**9)
#  plt.suptitle('1/10/19 Hour %i' %(i+1) )

#  map=ax.contourf(lons, lats, data,
#                transform=ccrs.PlateCarree(),
#                cmap='plasma', levels=np.arange(0,40,2))
  
#  ax.coastlines(resolution='50m')


#  cbar = fig.colorbar(map,ax=ax, orientation='horizontal', fraction=.1)	
#  cbar.ax.set_xlabel('pm25 / ppb')
#  pdf.savefig()
#  plt.close()

#pdf.close()










