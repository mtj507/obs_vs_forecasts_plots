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

emission='no2'

fig=plt.figure(figsize=[20,20])
fig,ax=plt.subplots(2,2,figsize=[8,8])


print(emission)
env_list=['AURN','Background Urban','Background Rural','Traffic Urban','Industrial urban','Industrial Suburban','background Suburban']
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

    b=ddf.columns
    headers=set(a).intersection(b)

    ddf=ddf.loc[:,headers]
    ddf=ddf.astype(float)
    ddf=ddf.replace(0,np.nan)
    ddf=ddf.dropna(axis=0,thresh=5)
    df=pd.DataFrame(index=ddf.index)
    df['obs median']=ddf.median(axis=1)
    df['obs Q1']=ddf.quantile(0.25,axis=1)
    df['obs Q3']=ddf.quantile(0.75,axis=1)
    df['obs_err']=df['obs Q3']-df['obs Q1']   
    df['obs_err']=df['obs_err'].replace(0,np.nan)    
    df=df.dropna(axis=0)    

    f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
    ds=xr.open_dataset(f)
    df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))

    for i in np.arange(0,no_locations):
        f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
        ds=xr.open_dataset(f)
        spec=ds[nasa_emission].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model[location[i]]=ds[nasa_emission].data[:,0,model_lat, model_lon]
        df_model[location[i]]=df_model[location[i]]*conv

    if env_type == ' ':
        env_type='AURN'

    df_model.drop(df_model.tail(1).index,inplace=True)
    df_model=df_model.astype(float)
    mdf=pd.DataFrame(index=df_model.index)
    mdf['model median']=df_model.median(axis=1)
    mdf['model Q1']=df_model.quantile(0.25,axis=1)
    mdf['model Q3']=df_model.quantile(0.75,axis=1)
    mdf['model_err']=mdf['model Q3']-mdf['model Q1']
    mdf.index=mdf.index.round('H')
    mdf['model_err'].values[mdf['model_err'].values < 0.01] = np.nan
    mdf['model median'].values[mdf['model median'].values < 0.05] = np.nan
    mdf=mdf.dropna(axis=0)

    zdf=pd.merge(df,mdf,right_index=True,left_index=True)    
    zdf=zdf.drop(columns=['obs Q1','obs Q3','model Q1','model Q3'])

    x_data=zdf['obs median']
    y_data=zdf['model median']
    x_err=zdf['obs_err']
    y_err=zdf['model_err']

    linear=Model(linear_func)
    datas=RealData(x_data,y_data,sx=x_err,sy=y_err)
    odr=ODR(datas,linear,beta0=[0])
    output=odr.run()
    output.pprint()
    beta=(output.beta)
    betastd=(output.sd_beta)
    
    ax.ravel()[e].plot(x_data,linear_func(beta,x_data),color='black',alpha=0.7)

    rmse_val=rmse(zdf['model median'],zdf['obs median'])
    rmse_txt=str(round(rmse_val,2))

    print('Model mean = '+ str(round(zdf['model median'].mean(),2)))
    print('Obs mean = '+ str(round(zdf['obs median'].mean(),2)))    
    print('RMSE = '+rmse_txt)

    if e == 0:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[0,0]) 
    if e == 1:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[0,1])
    if e == 2:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[1,0])
    if e == 3:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[1,1])

    if e == 0 or e == 2:
        ax.ravel()[e].set_ylabel(Emission+r' Model ($\mu g\:  m^{-3}$)')
    if e == 2 or e == 3:
        ax.ravel()[e].set_xlabel(Emission+r' Observation ($\mu g\:  m^{-3}$)')
    if e == 1 or e == 3:
        ax.ravel()[e].set_ylabel('')
    if e == 0 or e == 1:
        ax.ravel()[e].set_xlabel('')
    if emission == 'o3':
        ax.ravel()[3].set_visible(False)

    ax.ravel()[e].set_title(env_type)

    xy=np.linspace(*ax.ravel()[e].get_xlim())
    ax.ravel()[e].plot(xy,xy,linestyle='dashed',color='grey')

fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_scatter_2019.png')
plt.close()


