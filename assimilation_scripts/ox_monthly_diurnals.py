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

env_type='Background Rural'

months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    
fig=plt.figure()
fig,axes=plt.subplots(3,4,figsize=[8,8])

week='fullweek'



def rmse(predictions, targets):
    return np.sqrt(((predictions-targets)**2).mean())

if env_type == 'AURN':
    env_type=' '

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
metadata=metadata.reset_index(drop=False)
a=metadata['Site Name']

defra_csv='/users/mtj507/scratch/defra_data/no2_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
ddf=ddf.dropna(axis=0)
ddf=ddf.replace('No data', np.nan)
ddf.drop(ddf.tail(1).index,inplace=True)

oz_defra_csv='/users/mtj507/scratch/defra_data/o3_2019.csv'
ozddf=pd.read_csv(oz_defra_csv, low_memory=False)
ozddf.loc[ozddf['Time'] == '00:00:00','Time']='24:00:00'
ozddf.index=pd.to_datetime(ozddf['Date'], dayfirst=True)+pd.to_timedelta(ozddf['Time'])
ozddf=ozddf.loc[:, ~ozddf.columns.str.contains('^Unnamed')]
ozddf=ozddf.dropna(axis=0)
ozddf=ozddf.replace('No data', np.nan)
ozddf.drop(ddf.tail(1).index,inplace=True)

b=ddf.columns
c=set(a).intersection(b)
d=ozddf.columns
headers=set(c).intersection(d)

metadata=metadata.loc[metadata['Site Name'].isin(headers)]
metadata=metadata.reset_index(drop=True)  
location=metadata['Site Name']
longitude=metadata['Longitude']
latitude=metadata['Latitude']
no_locations=len(metadata.index)

ddf=ddf.loc[:,headers]
ozddf=ozddf.loc[:,headers]
oxddf=pd.concat([ddf,ozddf],axis=1)
oxddf=oxddf.astype(float)
oxddf=oxddf.groupby(oxddf.columns,axis=1).sum()
oxddf['month']=oxddf.index.month
oxddf['hour']=oxddf.index.hour

for m,ax in zip(np.arange(len(months)),axes.flatten()):
    print(months[m])
    ddf1=oxddf.loc[oxddf['month']==m+1] 
    ddf_med=pd.DataFrame()
    ddf_med=ddf1.groupby('hour').median()
    ddf_med['median']=ddf_med.mean(axis=1)
    ddf_q1=ddf1.groupby('hour').quantile(0.25)
    ddf_q1['q1']=ddf_q1.mean(axis=1)
    ddf_q3=ddf1.groupby('hour').quantile(0.75)
    ddf_q3['q3']=ddf_q3.mean(axis=1)

    ax.plot(ddf_med.index,ddf_med['median'],label='Observation',color='dimgrey')
    ax.fill_between(ddf_med.index,ddf_q1['q1'],ddf_q3['q3'],alpha=0.5,facecolor='dimgrey',edgecolor='grey')
 
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
    df_model['month']=df_model.index.month
    df_model['hour']=df_model.index.hour
    df_model=df_model.loc[df_model['month']==m+1]

    df_median=df_model.groupby('hour').median()
    df_median['median']=df_median.mean(axis=1)
    df_Q1=df_model.groupby('hour').quantile(0.25)
    df_Q1['Q1']=df_Q1.mean(axis=1)
    df_Q3=df_model.groupby('hour').quantile(0.75)
    df_Q3['Q3']=df_Q3.mean(axis=1)
        
    ax.plot(df_median.index,df_median['median'],label='Model',color='green')
    ax.fill_between(df_median.index,df_Q1['Q1'],df_Q3['Q3'],alpha=0.5,facecolor='limegreen',edgecolor='forestgreen')

    if env_type == ' ':
        env_type='AURN'

    ax.set_title(months[m],fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')

fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Hour of Day')
plt.ylabel(r'Ox Concentration ($\mu g\:  m^{-3}$)')
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/ox/'
plt.savefig(path+'ox_monthly_diurnal_'+env_type+'.png')
plt.close()







