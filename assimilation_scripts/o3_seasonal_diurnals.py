import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
from scipy.odr import *

emission='o3'
#env_type='AURN'
sites_to_test=['Narberth','Lough Navar','Rochester Stoke','Chilbolton Observatory']

seasons=['Winter','Spring','Summer','Autumn']
no_szns=len(seasons)

week='fullweek'

date1='2019-01-01'
date2='2019-12-31'

nrow=4
ncol=4
fsize=[10,10]

if emission == 'o3':
    conv=2*10**9
    nasa_emission='o3'
    Emission=r'$O_3$'

fig=plt.figure()
fig,axes=plt.subplots(ncols=ncol,nrows=nrow,figsize=fsize)

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
#metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
metadata=metadata.loc[metadata['Site Name'].isin(sites_to_test)]
metadata=metadata.reset_index(drop=True)
environment=metadata['Environment Type']
a=metadata['Site Name']

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

b=ddf.columns
locations=set(a).intersection(b)
location_list=list(locations)

metadata=metadata.loc[metadata['Site Name'].isin(location_list)]
metadata=metadata.reset_index(drop=True)
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
no_locations=len(metadata.index)


for i,ax in zip(range(0,no_locations),axes.flatten()):
    site=location[i]
    ddf1=ddf.loc[:,[site,'hour']]
    ddf1=ddf1.replace(0,np.nan)
    ddf1=ddf1.dropna(axis=0)
    ddf1=ddf1.astype(float)

    for y,ax in zip(range(no_szns),axes.flatten()[(i*4):(i*4)+4]):
        season=seasons[y]

        if season == 'Winter':
            date1='2019-01-01'
            date2='2019-03-19'
        if season == 'Spring':
            date1='2019-03-20'
            date2='2019-06-20'
        if season == 'Summer':
            date1='2019-06-21'
            date2='2019-09-22'
        if season == 'Autumn':
            date1='2019-09-23'
            date2='2019-12-20'
        if season == '2019':
            date1='2019-01-01'
            date2='2019-12-31'

        sdf=ddf1.loc[date1:date2]

        f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
        ds=xr.open_dataset(f)

        df=pd.DataFrame() 
        df=sdf.groupby('hour').median()
        df['q1']=sdf.groupby('hour').quantile(0.25)
        df['q3']=sdf.groupby('hour').quantile(0.75)

        ax.plot(df.index,df[site],label='Observation',color='dimgrey')
        ax.fill_between(df.index,df['q1'],df['q3'],alpha=0.5,facecolor='dimgrey',edgecolor='grey')

        df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
        spec=ds[nasa_emission].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model['model']=ds[nasa_emission].data[:,0,model_lat, model_lon]
        df_model['model']=df_model['model']*conv
        df_model.drop(df_model.tail(1).index,inplace=True)
        df_model['hour']=df_model.index.hour
        df_model=df_model.loc[date1:date2]
        df_model=df_model.astype(float)

        mdf=pd.DataFrame()
        mdf=df_model.groupby('hour').median()
        mdf['q1']=df_model.groupby('hour').quantile(0.25)
        mdf['q3']=df_model.groupby('hour').quantile(0.75)

        ax.plot(mdf.index,mdf['model'],label='Model',color='green')
        ax.fill_between(mdf.index,mdf['q1'],mdf['q3'],alpha=0.5,facecolor='limegreen',edgecolor='forestgreen')

        ax.set_title(site+' '+season,fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')


fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time of Day (hour)')
plt.ylabel(r'Ozone Concentration ($\mu g\:  m^{-3}$)') 
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_site_test_diurnals.png')
plt.close()


























