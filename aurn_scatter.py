import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.odr import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

emission='pm25'

week='fullweek'

date1='2019-09-22'
date2='2019-11-05'

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
environment=metadata['Environment Type']
environments=pd.unique(environment)

plt.xlabel('Observation Median  ug/m3')
plt.ylabel('Forecast Median  ug/m3')
plt.title(emission+' '+week)
xy=np.linspace(*plt.xlim())
plt.plot(xy,xy,linestyle='dashed',color='grey')

aurn_list=[]
aurn_Q1_list=[]
aurn_Q3_list=[]

obs_BU_list=[]
obs_BU_Q1_list=[]
obs_BU_Q3_list=[]
obs_TU_list=[]
obs_TU_Q1_list=[]
obs_TU_Q3_list=[]
obs_BR_list=[]
obs_BR_Q1_list=[]
obs_BR_Q3_list=[]
obs_IU_list=[]
obs_IU_Q1_list=[]
obs_IU_Q3_list=[]
obs_BS_list=[]
obs_BS_Q1_list=[]
obs_BS_Q3_list=[]
obs_IS_list=[]
obs_IS_Q1_list=[]
obs_IS_Q3_list=[]


mod_list=[]
mod_Q1_list=[]
mod_Q3_list=[]

mod_BU_list=[]
mod_BU_Q1_list=[]
mod_BU_Q3_list=[]
mod_TU_list=[]
mod_TU_Q1_list=[]
mod_TU_Q3_list=[]
mod_BR_list=[]
mod_BR_Q1_list=[]
mod_BR_Q3_list=[]
mod_IU_list=[]
mod_IU_Q1_list=[]
mod_IU_Q3_list=[]
mod_BS_list=[]
mod_BS_Q1_list=[]
mod_BS_Q3_list=[]
mod_IS_list=[]
mod_IS_Q1_list=[]
mod_IS_Q3_list=[]

for environment in environments:
    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type']==environment]
    check=pd.unique(metadata['Environment Type'])
    metadata=metadata.reset_index(drop=True)
    locations=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    headers=locations.values.tolist()
    headers.append('hour')
    print(check)
    no_locations=len(metadata.index)
    a=list(metadata['Site Name'])
   
    defra_csv='/users/mtj507/scratch/defra_data/defra_'+emission+'_uk_2019.csv'
    ddf=pd.read_csv(defra_csv, low_memory=False)
    ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
    ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
    ddf=ddf.dropna(axis=0)
    ddf=ddf.replace('No data', np.nan)
    b=list(ddf.columns)
    c=set(a).intersection(b)
    ddf=ddf[ddf.columns.intersection(c)]
    ddf['weekday']=ddf.index.weekday
    ddf['month']=ddf.index.month.astype(str)
    ddf['month']=ddf['month'].str.zfill(2)
    ddf['day']=ddf.index.day.astype(str)
    ddf['day']=ddf['day'].str.zfill(2)
    ddf['day and month']=ddf['month']+ddf['day']
    ddf['hour']=ddf.index.hour

    ddf=ddf.loc[date1:date2]
    ddf=ddf.loc[(ddf['weekday'] >= day1) & (ddf['weekday'] <= day2)]

    metadata=metadata[metadata['Site Name'].isin(c)]
    metadata=metadata.reset_index(drop=False)
    area=metadata['Zone']
    location=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    no_locations=len(metadata.index)

    for i in np.arange(0, no_locations):
        ddf1=ddf.loc[:,['hour', f'{location[i]}']]
        ddf1=ddf1.dropna(axis=0)
        ddf1['value']=ddf1[f'{location[i]}'].astype(float)
        ddf_median=ddf1.groupby('hour').median()
        ddf_Q1=ddf1.groupby('hour')['value'].quantile(0.25)
        ddf_Q3=ddf1.groupby('hour')['value'].quantile(0.75)
        obs_median=ddf_median['value'].mean()
        obs_median=float(round(obs_median,2))
        obs_Q1=ddf_Q1.mean()
        obs_Q1=float(round(obs_Q1,2))
        obs_Q3=ddf_Q3.mean()
        obs_Q3=float(round(obs_Q3,2))

        days_of_data=len(pd.unique(ddf['day and month']))
        dates=pd.unique(ddf['day and month'])
        mod_data = np.zeros((24,days_of_data))

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


        nasa_median=np.median(mod_data)
        nasa_median=float(round(nasa_median,2))
        nasa_Q1=np.percentile(mod_data,25)
        nasa_Q1=float(round(nasa_Q1,2))
        nasa_Q3=np.percentile(mod_data,75)
        nasa_Q3=float(round(nasa_Q3,2))

        mod_list.append(nasa_median)
        mod_Q1_list.append(nasa_Q1)
        mod_Q3_list.append(nasa_Q3) 

        aurn_list.append(obs_median)
        aurn_Q1_list.append(obs_Q1)
        aurn_Q3_list.append(obs_Q3)

        if environment == 'Background Urban':
            obs_BU_list.append(obs_median)
            obs_BU_Q1_list.append(obs_Q1)
            obs_BU_Q3_list.append(obs_Q3)
            mod_BU_list.append(nasa_median)
            mod_BU_Q1_list.append(nasa_Q1)
            mod_BU_Q3_list.append(nasa_Q3)

        if environment == 'Background Rural':
            obs_BR_list.append(obs_median)
            obs_BR_Q1_list.append(obs_Q1)
            obs_BR_Q3_list.append(obs_Q3)
            mod_BR_list.append(nasa_median)
            mod_BR_Q1_list.append(nasa_Q1)
            mod_BR_Q3_list.append(nasa_Q3)

        if environment == 'Traffic Urban':
            obs_TU_list.append(obs_median)
            obs_TU_Q1_list.append(obs_Q1)
            obs_TU_Q3_list.append(obs_Q3)
            mod_TU_list.append(nasa_median)
            mod_TU_Q1_list.append(nasa_Q1)
            mod_TU_Q3_list.append(nasa_Q3)

        if environment == 'Industrial Urban':
            obs_IU_list.append(obs_median)
            obs_IU_Q1_list.append(obs_Q1)
            obs_IU_Q3_list.append(obs_Q3)
            mod_IU_list.append(nasa_median)
            mod_IU_Q1_list.append(nasa_Q1)
            mod_IU_Q3_list.append(nasa_Q3)

        if environment == 'Background Suburban':
            obs_BS_list.append(obs_median)
            obs_BS_Q1_list.append(obs_Q1)
            obs_BS_Q3_list.append(obs_Q3)
            mod_BS_list.append(nasa_median)
            mod_BS_Q1_list.append(nasa_Q1)
            mod_BS_Q3_list.append(nasa_Q3)

        if environment == 'Industrial Suburban':
            obs_IS_list.append(obs_median)
            obs_IS_Q3_list.append(obs_Q1)
            obs_IS_Q3_list.append(obs_Q3)
            mod_IS_list.append(nasa_median)
            mod_IS_Q1_list.append(nasa_Q1)
            mod_IS_Q3_list.append(nasa_Q3)

    


def linear_func(p, x):
    y=p*x
    return y


aurn_data={'obs':aurn_list,'obs Q1':aurn_Q1_list,'obs Q3':aurn_Q3_list,'model':mod_list,'model Q1':mod_Q1_list,'model Q3':mod_Q3_list}
aurn_df=pd.DataFrame(aurn_data)
aurn_df=aurn_df[aurn_df > 0].dropna()
aurn_df=aurn_df.dropna()
aurn_df['obs_err']=aurn_df['obs Q3']-aurn_df['obs Q1']
aurn_df['mod_err']=aurn_df['model Q3']-aurn_df['model Q1']
aurn_df=aurn_df.reset_index(drop=True)


















