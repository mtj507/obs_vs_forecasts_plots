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
env_type='AURN'




week='fullweek'

date1='2019-01-01'
date2='2019-12-31'

if emission == 'o3':
    conv=2*10**9
    nasa_emission='o3'
    Emission=r'$O_3$'
    if env_type == 'AURN':
        nrow=8
        ncol=10
        fsize=[20,20]
    if env_type == 'Background Urban':
        nrow=7
        ncol=7
        fsize=[15,15]
    if env_type == 'Background Rural':
        nrow=5
        ncol=5
        fsize=[10,10]
    if env_type == 'Traffic Urban':
        nrow=1
        ncol=3
        fsize=[10,10]    
  
if week == 'fullweek':
  day1=0
  day2=6
if week == 'weekday':
  day1=0
  day2=4
if week == 'weekend':
  day1=5
  day2=6

def inter_func(B,x):
    y=B[0]*x+B[1]
    return y

def rmse(predictions, targets):
    return np.sqrt(((predictions-targets)**2).mean())

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
locations=set(a).intersection(b)
location_list=list(locations)

fig=plt.figure()
fig,axes=plt.subplots(ncols=ncol,nrows=nrow,figsize=fsize)

metric_df=pd.DataFrame(columns=['site name','obs median','mod median','RMSE','ODR gradient'])

f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
ds=xr.open_dataset(f)
    
for i,ax in zip(range(len(location_list)),axes.flatten()):
    site=location_list[i]
    ddf1=ddf.loc[:,site]
    ddf1=ddf1.astype(float)
    ddf1=ddf1.replace(0,np.nan)
    df=pd.DataFrame(index=ddf1.index)
    df['obs']=ddf1

    df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
    spec=ds[nasa_emission].data
    lats=ds['lat'].data
    lons=ds['lon'].data
    model_lat=np.argmin(np.abs(latitude[i]-lats))
    model_lon=np.argmin(np.abs(longitude[i]-lons))
    df_model['model']=ds[nasa_emission].data[:,0,model_lat, model_lon]
    df_model['model']=df_model['model']*conv
    df_model.drop(df_model.tail(1).index,inplace=True)
    df_model.index=df_model.index.round('H')
    df_model=df_model.astype(float)    

    sdf=pd.merge(df,df_model,right_index=True,left_index=True)
    sdf=sdf.dropna(axis=0)

    x_data=sdf['obs']
    y_data=sdf['model'] 

    obs_median=x_data.median()
    obs_median=str(round(obs_median,2))
    mod_median=y_data.median()
    mod_median=str(round(mod_median,2))  

    sns.scatterplot(x=x_data,y=y_data,ax=ax,s=6) 
   
    linear=Model(inter_func)
    datas=RealData(x_data,y_data)
    odr=ODR(datas,linear,beta0=[1,0])
    output=odr.run()
#    output.pprint()
    beta=(output.beta)
    beta_txt=str(round(beta[0],2))
    betastd=(output.sd_beta)

    ax.plot(x_data,inter_func(beta,x_data),color='black',alpha=0.8)

    rmse_val=rmse(y_data,x_data)
    rmse_txt=str(round(rmse_val,2))

    xy=np.linspace(*ax.get_xlim())
    ax.plot(xy,xy,linestyle='dashed',color='grey')

    ax.set_title(site,fontsize=8)
    ax.set_ylabel('')
    ax.set_xlabel('')
    
#    print(site)
#    print('Obs Median = '+obs_median)
#    print('Mod Median = '+mod_median)
#    print('RMSE = '+rmse_txt)
#    print('ODR gradient = '+beta_txt)


if env_type == 'Background Rural':
    axes[4,1].set_visible(False)
    axes[4,2].set_visible(False)
    axes[4,3].set_visible(False)
    axes[4,4].set_visible(False)
if env_type == ' ':
    axes[7,5].set_visible(False)
    axes[7,6].set_visible(False)
    axes[7,7].set_visible(False)
    axes[7,8].set_visible(False)
    axes[7,9].set_visible(False)
    env_type='AURN'
if env_type == 'Background Urban':
    axes[6,1].set_visible(False)
    axes[6,2].set_visible(False)
    axes[6,3].set_visible(False)
    axes[6,4].set_visible(False)
    axes[6,5].set_visible(False)
    axes[6,6].set_visible(False)

fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'Observations ($\mu g\:  m^{-3}$)')
plt.ylabel(r'Model ($\mu g\:  m^{-3}$)')
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_non_forced_large_scatter_'+env_type+'.png')
plt.close()





