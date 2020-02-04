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
import seaborn as sns
from scipy.odr import *


emission='pm25'

fig=plt.figure(figsize=[20,20])
fig,ax=plt.subplots(2,2,figsize=[10,10],subplot_kw=dict(aspect='equal'))

print(emission)
env_list=['AURN','Background Urban','Traffic Urban','Background Rural','Industrial urban','Industrial Suburban','background Suburban']
env_no=4

if emission == 'o3':
    env_list=['AURN','Background Urban','Background Rural']
    env_no=3

week='fullweek'

date1='2019-01-01'
date2='2019-12-31'

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

def linear_func(p, x):
    y=p*x
    return y


for e in range(env_no):
    env_type=env_list[e]
    print(env_type)
    if env_type == 'AURN':
        env_type=' '


    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
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
    environment=metadata['Environment Type']
    no_locations=len(metadata.index)

    obs_list=[]
    obsQ1_list=[]
    obsQ3_list=[]
    mod_list=[]
    modQ1_list=[]
    modQ3_list=[]

    for i in np.arange(0, no_locations):
        ddf1=ddf.loc[:,[f'{location[i]}']]
        ddf1=ddf1.dropna(axis=0)
        ddf1['value']=ddf1[f'{location[i]}'].astype(float)
        if ddf1.empty:
            obs_list.append(np.nan)
            obsQ1_list.append(np.nan)
            obsQ3_list.append(np.nan)
            mod_list.append(np.nan)
            modQ1_list.append(np.nan)
            modQ3_list.append(np.nan)
            continue

        obs_median=ddf1['value'].median(axis=0)
        obs_median=float(round(obs_median,2))
        obs_Q1=ddf1.quantile(0.25,axis=0)
        obs_Q1=float(round(obs_Q1,2))
        obs_Q3=ddf1.quantile(0.75,axis=0)
        obs_Q3=float(round(obs_Q3,2))
        obs_list.append(obs_median)
        obsQ1_list.append(obs_Q1)
        obsQ3_list.append(obs_Q3)
       
        f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
        ds=xr.open_dataset(f)
        spec=ds[nasa_emission].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model=pd.DataFrame()
        df_model[location[i]]=ds[nasa_emission].data[:,0,model_lat, model_lon]
        df_model=df_model[location[i]]*conv

        mod_median=df_model.median(axis=0)
        mod_median=float(round(mod_median,2))
        mod_Q1=df_model.quantile(0.25)
        mod_Q1=float(round(mod_Q1,2))
        mod_Q3=df_model.quantile(0.75)
        mod_Q3=float(round(mod_Q3,2))
        mod_list.append(mod_median)
        modQ1_list.append(mod_Q1)
        modQ3_list.append(mod_Q3)

    gdf=pd.DataFrame(index=range(0,no_locations))
    gdf['obs']=obs_list
    gdf['obs Q1']=obsQ1_list
    gdf['obs Q3']=obsQ3_list
    gdf['obs_err']=gdf['obs Q3']-gdf['obs Q1']
    gdf['mod']=mod_list
    gdf['mod Q1']=modQ1_list
    gdf['mod Q3']=modQ3_list
    gdf['mod_err']=gdf['mod Q3']-gdf['mod Q1']
    gdf=gdf.dropna(axis=0)

    x_data=gdf['obs']
    y_data=gdf['mod']
    x_err=gdf['obs_err']
    y_err=gdf['mod_err']

    linear=Model(linear_func)
    datas=RealData(x_data,y_data,sx=x_err,sy=y_err)
    odr=ODR(datas,linear,beta0=[0])
    output=odr.run()
    output.pprint()
    beta=output.beta
    betastd=output.sd_beta

    ax.ravel()[e].plot(x_data,linear_func(beta,x_data),color='black',alpha=0.7)

    if e == 0:
        sns.scatterplot(x=x_data,y=y_data,data=gdf,ax=ax[0,0])
    if e == 1:
        sns.scatterplot(x=x_data,y=y_data,data=gdf,ax=ax[0,1])
    if e == 2:
        sns.scatterplot(x=x_data,y=y_data,data=gdf,ax=ax[1,0])
    if e == 3:
        sns.scatterplot(x=x_data,y=y_data,data=gdf,ax=ax[1,1])
 
    if env_type == ' ':
        env_type='AURN'

    if e == 0 or e == 2:
        ax.ravel()[e].set_ylabel(Emission+r' Model ($\mu g\:  m^{-3}$)')
    if e == 2 or e == 3:
        ax.ravel()[e].set_xlabel(Emission+r' Observation ($\mu g\:  m^{-3}$)')
    if e == 1 or e == 3:
        ax.ravel()[e].set_ylabel('')
    if e == 0 or e == 1:
        ax.ravel()[e].set_xlabel('')

    ax.ravel()[e].set_title(env_type)

    xy=np.linspace(*ax.ravel()[e].get_xlim())
    ax.ravel()[e].plot(xy,xy,linestyle='dashed',color='grey')

path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_site_scatter_2019.png')
plt.close()








