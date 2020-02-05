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
    
print(emission)
env_list=['AURN','Background Urban','Background Rural','Traffic Urban','Industrial urban','Industrial Suburban','background Suburban']
env_no=4

week='fullweek'

seasons=['Winter','Spring','Summer','Autumn']
no_szns=len(seasons)

if emission == 'no2':
  conv=1.88*10**9
  nasa_emission='no2'
  Emission=r'$NO_2$'

if emission == 'no':
  conv=1.23*10**9
  nasa_emission='no'
  Emission='NO'

if emission == 'pm25':
  conv=1
  nasa_emission='pm25_rh35_gcc'
  Emission=r'$PM_{2.5}$'

if emission == 'o3':
  conv=2*10**9
  nasa_emission='o3'
  Emission=r'$O_3$'

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

    fig=plt.figure(figsize=[20,20])
    fig,ax=plt.subplots(2,2,figsize=[8,8])

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
    
    for y in range(no_szns):
        season=seasons[y]
        print(season)    
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

        ddf1=ddf.loc[date1:date2]
        ddf1=ddf1.loc[(ddf1['weekday'] >= day1) & (ddf1['weekday'] <= day2)]

        b=ddf1.columns
        headers=set(a).intersection(b)

        ddf1=ddf1.loc[:,headers]
        ddf1['hour']=ddf1.index.hour
        ddf1=ddf1.astype(float)
        ddf_median=ddf1.groupby('hour').median()
        ddf_median['median']=ddf_median.mean(axis=1)
        ddf_Q1=ddf1.groupby('hour').quantile(0.25)
        ddf_Q1['Q1']=ddf_Q1.mean(axis=1)
        ddf_Q3=ddf1.groupby('hour').quantile(0.75)
        ddf_Q3['Q3']=ddf_Q3.mean(axis=1)
   
        ax.ravel()[y].plot(ddf_median.index,ddf_median['median'],label='Observation',color='dimgrey')
        ax.ravel()[y].fill_between(ddf_median.index,ddf_Q1['Q1'],ddf_Q3['Q3'],alpha=0.5,facecolor='dimgrey',edgecolor='grey')


        f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
        ds=xr.open_dataset(f)
        df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))

        for i in np.arange(0,no_locations):
            spec=ds[nasa_emission].data
            lats=ds['lat'].data
            lons=ds['lon'].data
            model_lat=np.argmin(np.abs(latitude[i]-lats))
            model_lon=np.argmin(np.abs(longitude[i]-lons))
            df_model[location[i]]=ds[nasa_emission].data[:,0,model_lat, model_lon]
            df_model[location[i]]=df_model[location[i]]*conv
     
        df_model['hour']=df_model.index.hour
        df_model=df_model.loc[date1:date2]
        df_median=df_model.groupby('hour').median()
        df_median['median']=df_median.mean(axis=1)
        df_Q1=df_model.groupby('hour').quantile(0.25)
        df_Q1['Q1']=df_Q1.mean(axis=1)
        df_Q3=df_model.groupby('hour').quantile(0.75)
        df_Q3['Q3']=df_Q3.mean(axis=1)

        ax.ravel()[y].plot(df_median.index,df_median['median'],label='Model',color='green')
        ax.ravel()[y].fill_between(df_median.index,df_Q1['Q1'],df_Q3['Q3'],alpha=0.5,facecolor='limegreen',edgecolor='forestgreen')

        if env_type == ' ':
            env_type='AURN'

        if y == 0 or y == 2:
            ax.ravel()[y].set_ylabel(Emission+r'($\mu g\:  m^{-3}$)')
        if y == 2 or y == 3:
            ax.ravel()[y].set_xlabel('Time of Day (hour)')
        if y == 1 or y == 3:
            ax.ravel()[y].set_ylabel('')
        if y == 0 or y == 1:
            ax.ravel()[y].set_xlabel('')
        if y == 1:
            ax.ravel()[y].legend(fontsize='small')
        ax.ravel()[y].set_title(season)

        if emission == 'o3' and y == 3:
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
    path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
    plt.savefig(path+emission+'_'+env_type+'_seasonal_diurnals_'+'_'+week+'.png')
    plt.close()













