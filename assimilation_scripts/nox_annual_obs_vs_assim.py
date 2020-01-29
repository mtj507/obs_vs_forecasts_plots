import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

emission='no2'

environment_type='Background Urban'

week='fullweek'

date1='2019-01-01'
date2='2019-12-31'

if emission == 'no2':
  conv=1.88*10**9
  nasa_emission='no2'
  Emission='NO2'

if emission == 'no':
  conv=1.23*10**9
  nasa_emission='no'
  Emission='NO'

if emission == 'pm25':
  conv=1
  nasa_emission='pm25_rh35_gcc'
  Emission='PM 2.5'

if emission == 'o3':
  conv=2*10**9
  nasa_emission='o3'
  Emission='O3'

if week == 'fullweek':
  day1=0
  day2=6
if week == 'weekday':
  day1=0
  day2=4
if week == 'weekend':
  day1=5
  day2=6


metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
#metadata=metadata.loc[metadata['Environment Type']==environment_type]
#metadata=metadata[metadata['Site Name'].str.match(city)]
metadata=metadata.loc[metadata['Site Name']=='London Westminster']
metadata=metadata.reset_index(drop=False)
area=metadata['Zone']
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
environment=metadata['Environment Type']
no_locations=len(metadata.index)


defra_csv='/users/mtj507/scratch/defra_data/'+emission+'_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
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
ddf=ddf.loc[(ddf['weekday'] >= day1) & (ddf['weekday'] <= day2)]

no_defra_csv='/users/mtj507/scratch/defra_data/no_2019.csv'
noddf=pd.read_csv(no_defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
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
noddf=noddf.loc[(noddf['weekday'] >= day1) & (noddf['weekday'] <= day2)]


for i in np.arange(0, no_locations):
    ddf1=ddf.loc[:,['hour', f'{location[i]}']]
    noddf1=noddf.loc[:,[f'{location[i]}']]
    ddf1['no2']=ddf1[f'{location[i]}'].astype(float)
    noddf1['no']=noddf1[f'{location[i]}'].astype(float) 
    ddf1['no']=noddf1['no']
    ddf1['nox']=ddf1['no']+ddf1['no2']
    ddf1=ddf1.dropna(axis=0)
    plt.plot(ddf1.index,ddf1['nox'],label='Obs',color='dimgrey')

    days_of_data=len(pd.unique(ddf['day and month']))
    dates=pd.unique(ddf['day and month'])
    mod_data = np.zeros((24,days_of_data))

    mean_obs=ddf1['value'].mean()

    f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
    ds=xr.open_dataset(f)    
    spec=ds[nasa_emission].data
    lats=ds['lat'].data
    lons=ds['lon'].data
    model_lat=np.argmin(np.abs(latitude[i]-lats))
    model_lon=np.argmin(np.abs(longitude[i]-lons))
    df_model=pd.DataFrame(ds[nasa_emission].data[:,0,model_lat, model_lon])
    nodf_model=pd.DataFrame(ds['no'].data[:,0,model_lat, model_lon])
    df_model.index=pd.to_datetime(ds.time.data)
    nodf_model.index=pd.to_datetime(ds.time.data)
    df_model.columns=[nasa_emission]
    nodf_model.columns=['no']
    df_model[nasa_emission]=df_model[nasa_emission]*conv
    nodf_model['no']=nodf_model['no']*1.23*10**9
    df_model['nox']=nodf_model['no']+df_model['no2']
    plt.plot(df_model.index,df_model['nox'],label='Model',color='green',alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('NOx ug/m3')
    plt.legend()
    plt.title(location[i])
  
    path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
    plt.savefig(path+emission+f'_{location[i]}_2019.png')
    plt.close()







