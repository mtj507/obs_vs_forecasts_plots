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

fig=plt.figure(figsize=[20,20])
fig,ax=plt.subplots(2,2,figsize=[8,8])
    
print('ox')
env_list=['AURN','Background Urban','Background Rural','Traffic Urban','Industrial urban','Industrial Suburban','background Suburban']
env_no=4

week='fullweek'

season='2019'

if season == 'winter':
    date1='2019-01-01'
    date2='2019-03-19'
if season == 'spring':
    date1='2019-03-20'
    date2='2019-06-20'
if season == 'summer':
    date1='2019-06-21'
    date2='2019-09-22'
if season == 'autumn':
    date1='2019-09-23'
    date2='2019-12-20'
if season == '2019':
    date1='2019-01-01'
    date2='2019-12-31'


if week == 'fullweek':
  day1=0
  day2=6
if week == 'weekday':
  day1=0
  day2=4
if week == 'weekend':
  day1=5
  day2=6

def rmse(predictions, targets):
    return np.sqrt(((predictions-targets)**2).mean())


for e in range(env_no):
    env_type=env_list[e]
    print(env_type)
    if env_type == 'AURN':
        env_type=' '

    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
    metadata=metadata.reset_index(drop=False)
    area=metadata['Zone']
    location=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    environment=metadata['Environment Type']
    no_locations=len(metadata.index)
    a=location

    defra_csv='/users/mtj507/scratch/defra_data/no2_2019.csv'
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

    oz_defra_csv='/users/mtj507/scratch/defra_data/o3_2019.csv'
    ozddf=pd.read_csv(oz_defra_csv, low_memory=False)
    ozddf.loc[ozddf['Time'] == '00:00:00','Time']='24:00:00'
    ozddf.index=pd.to_datetime(ozddf['Date'], dayfirst=True)+pd.to_timedelta(ozddf['Time'])
    ozddf=ozddf.loc[:, ~ozddf.columns.str.contains('^Unnamed')]
    ozddf=ozddf.dropna(axis=0)
    ozddf=ozddf.replace('No data', np.nan)
    ozddf['hour']=ozddf.index.hour
    ozddf['weekday']=ozddf.index.weekday
    ozddf['month']=ozddf.index.month.astype(str)
    ozddf['month']=ozddf['month'].str.zfill(2)
    ozddf['day']=ozddf.index.day.astype(str)
    ozddf['day']=ozddf['day'].str.zfill(2)
    ozddf['day and month']=ozddf['month']+ozddf['day']

    ozddf=ozddf[date1:date2]
    ozddf=ozddf.loc[(ozddf['weekday'] >= day1) & (ozddf['weekday'] <= day2)]

    b=ddf.columns
    c=set(a).intersection(b)
    d=ozddf.columns
    headers=set(c).intersection(d)
 

    ddf=ddf.loc[:,headers]
    ozddf=ozddf.loc[:,headers]
    ddf1=pd.concat([ddf,ozddf],axis=1)
    ddf1=ddf1.astype(float)
    ddf1=ddf1.groupby(ddf1.columns,axis=1).sum()
    ddf1['hour']=ddf1.index.hour

    ddf_median=ddf1.groupby('hour').median()
    ddf_median['median']=ddf_median.mean(axis=1)
    ddf_Q1=ddf1.groupby('hour').quantile(0.25)
    ddf_Q1['Q1']=ddf_Q1.mean(axis=1)
    ddf_Q3=ddf1.groupby('hour').quantile(0.75)
    ddf_Q3['Q3']=ddf_Q3.mean(axis=1)
   
    ax.ravel()[e].plot(ddf_median.index,ddf_median['median'],label='Observation',color='dimgrey')
    ax.ravel()[e].fill_between(ddf_median.index,ddf_Q1['Q1'],ddf_Q3['Q3'],alpha=0.5,facecolor='dimgrey',edgecolor='grey')


    f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
    ds=xr.open_dataset(f)
    df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
    ozdf_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))

    for i in np.arange(0,no_locations):
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model[location[i]]=ds['no2'].data[:,0,model_lat, model_lon]
        ozdf_model[location[i]]=ds['o3'].data[:,0,model_lat, model_lon]
        df_model[location[i]]=df_model[location[i]]*1.88*10**9
        ozdf_model[location[i]]=ozdf_model[location[i]]*2*10**9

    df_model=pd.concat([df_model,ozdf_model],axis=1) 
    df_model=df_model.astype(float)
    df_model=df_model.groupby(df_model.columns,axis=1).sum()
    df_model['hour']=df_model.index.hour

    df_median=df_model.groupby('hour').median()
    df_median['median']=df_median.mean(axis=1)
    df_Q1=df_model.groupby('hour').quantile(0.25)
    df_Q1['Q1']=df_Q1.mean(axis=1)
    df_Q3=df_model.groupby('hour').quantile(0.75)
    df_Q3['Q3']=df_Q3.mean(axis=1)

    ax.ravel()[e].plot(df_median.index,df_median['median'],label='Model',color='green')
    ax.ravel()[e].fill_between(df_median.index,df_Q1['Q1'],df_Q3['Q3'],alpha=0.5,facecolor='limegreen',edgecolor='forestgreen')

    if env_type == ' ':
        env_type='AURN'

    if e == 0 or e == 2:
        ax.ravel()[e].set_ylabel(r'Ox ($\mu g\:  m^{-3}$)')
    if e == 2 or e == 3:
        ax.ravel()[e].set_xlabel('Time of Day (hour)')
    if e == 1 or e == 3:
        ax.ravel()[e].set_ylabel('')
    if e == 0 or e == 1:
        ax.ravel()[e].set_xlabel('')
    if e == 1:
        ax.ravel()[e].legend(fontsize='small')
    ax.ravel()[e].set_title(env_type)
    
    if e == 3:
        ax.ravel()[e].set_visible(False)

    mod_mean=df_median['median'].mean()
    mod_mean=str(round(mod_mean,2))
    print('mod mean = '+mod_mean)

    obs_mean=ddf_median['median'].mean()
    obs_mean=str(round(obs_mean,2))
    print('obs mean = '+obs_mean)
 
    rmse_val=rmse(df_median['median'],ddf_median['median'])
    rmse_txt=str(round(rmse_val,2))
    print('RMSE = '+rmse_txt)

fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/ox/'
plt.savefig(path+'ox_2019_diurnal.png')
plt.close()













