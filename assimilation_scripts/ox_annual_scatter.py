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

fig=plt.figure(figsize=[20,20])
fig,ax=plt.subplots(2,2,figsize=[8,8])

env_list=['AURN','Background Urban','Background Rural','Traffic Urban','Industrial urban','Industrial Suburban','background Suburban']
env_no=3



week='fullweek'

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
    a=metadata['Site Name']

    if env_type == ' ':
        env_type='AURN'

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

    metadata=metadata.loc[metadata['Site Name'].isin(headers)]
    metadata=metadata.reset_index(drop=False)
    location=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    no_locations=len(location)

    ddf=ddf.loc[:,headers]
    ozddf=ozddf.loc[:,headers]
    ddf1=pd.concat([ddf,ozddf],axis=1)
    ddf1=ddf1.astype(float)
    ddf=ddf.dropna(axis=0,thresh=5)
    ddf1=ddf1.groupby(ddf1.columns,axis=1).sum()

    df=pd.DataFrame(index=ddf1.index)
    df['obs median']=ddf1.median(axis=1)
    df['obs Q1']=ddf1.quantile(0.25,axis=1)
    df['obs Q3']=ddf1.quantile(0.75,axis=1)
    df['obs_err']=df['obs Q3']-df['obs Q1']
    df['obs_err']=df['obs_err'].replace(0,np.nan)
    df=df.dropna(axis=0)

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
    mdf=pd.DataFrame(index=df_model.index)
    mdf['model median']=df_model.median(axis=1)
    mdf['model Q1']=df_model.quantile(0.25,axis=1)
    mdf['model Q3']=df_model.quantile(0.75,axis=1)
    mdf['model_err']=mdf['model Q3']-mdf['model Q1']
    mdf.index=mdf.index+pd.Timedelta('1 min')
    mdf.index=mdf.index.round('H')
    mdf['model_err'].values[mdf['model_err'].values < 0.01] = np.nan
    mdf['model median'].values[mdf['model median'].values < 0.05] = np.nan
    mdf=mdf.dropna(axis=0)

    zdf=pd.merge(df,mdf,right_index=True,left_index=True)
    zdf=zdf.drop(columns=['obs Q1','obs Q3','model Q1','model Q3'])

    x_data=zdf['obs median']
    y_data=zdf['model median']

    linear=Model(linear_func)
    datas=RealData(x_data,y_data)
    odr=ODR(datas,linear,beta0=[0])
    output=odr.run()
#    output.pprint()
    beta=(output.beta)
    betastd=(output.sd_beta)

    rmse_val=rmse(zdf['model median'],zdf['obs median'])
    rmse_txt=str(round(rmse_val,2))

    print('Model mean = '+ str(round(zdf['model median'].mean(),2)))
    print('Obs mean = '+ str(round(zdf['obs median'].mean(),2)))
    print('RMSE = '+rmse_txt)
    print('Beta = '+str(beta[0]))


    if e == 0:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[0,0],s=6,alpha=1)
    if e == 1:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[0,1],s=6,alpha=1)
    if e == 2:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[1,0],s=6,alpha=1)
    if e == 3:
        sns.scatterplot(x=x_data,y=y_data,data=zdf,ax=ax[1,1],s=6,alpha=1)

    ax.ravel()[e].plot(x_data,linear_func(beta,x_data),color='black',alpha=0.8)

    if e == 0 or e == 2:
        ax.ravel()[e].set_ylabel(r'Ox Model ($\mu g\:  m^{-3}$)')
    if e == 2 or e == 3:
        ax.ravel()[e].set_xlabel(r'Ox Observation ($\mu g\:  m^{-3}$)')
    if e == 1 or e == 3:
        ax.ravel()[e].set_ylabel('')
    if e == 0 or e == 1:
        ax.ravel()[e].set_xlabel('')

    ax.ravel()[3].set_visible(False)

    ax.ravel()[e].set_title(env_type)

    xy=np.linspace(*ax.ravel()[e].get_xlim())
    ax.ravel()[e].plot(xy,xy,linestyle='dashed',color='grey')
   


fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/ox/'
plt.savefig(path+'ox_scatter_2019.png')
plt.close()











