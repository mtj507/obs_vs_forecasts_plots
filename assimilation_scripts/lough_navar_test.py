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

date1='2019-01-01'
date2='2019-12-31'

#site='Lough Navar'
site='Narberth'

fig=plt.figure()
fig,axes=plt.subplots(3,3,figsize=[10,10],sharex=True,sharey=True)

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['Site Name']==site]
metadata=metadata.reset_index(drop=True)
latitude=float(metadata['Latitude'])
longitude=float(metadata['Longitude'])

defra_csv='/users/mtj507/scratch/defra_data/'+emission+'_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
ddf=ddf.dropna(axis=0)
ddf=ddf.replace('No data', np.nan)
ddf=ddf.loc[:,[site]]
ddf['hour']=ddf.index.hour 
ddf=ddf.astype(float)

df=pd.DataFrame(index=np.unique(ddf['hour']))
df['med']=ddf.groupby('hour').median()
df['q1']=ddf.groupby('hour').quantile(0.25)
df['q3']=ddf.groupby('hour').quantile(0.75)

f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
ds=xr.open_dataset(f)
lats=ds['lat'].data
lons=ds['lon'].data
model_lat=np.argmin(np.abs(latitude-lats))
model_lon=np.argmin(np.abs(longitude-lons))

positions=['North West','North','North East','West',site,'East','South West','South','South East']
no_pos=len(positions)

for i,ax in zip(np.arange(0,no_pos),axes.flatten()):
    position=positions[i]

    if i == 0 or i == 1 or i == 2:
        plat=model_lat+2
    if i == 3 or i == 4 or i == 5:
        plat=model_lat
    if i == 6 or i == 7 or i == 8:
        plat=model_lat-2

    if i == 0 or i == 3 or i == 6:
        plon=model_lon-2
    if i == 1 or i == 4 or i == 7:
        plon=model_lon
    if i == 2 or i == 5 or i == 8:
        plon=model_lon+2

#    if i == 8:
#        plat=14
#        plon=11

    print(position,lats[plat],lons[plon])

    ax.plot(df.index,df['med'],label='Observation',color='dimgrey')
    ax.fill_between(df.index,df['q1'],df['q3'],alpha=0.5,facecolor='dimgrey',edgecolor='grey')

    
    df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
    df_model['model']=ds['o3'].data[:,0,plat,plon]
    df_model['model']=df_model['model']*2*10**9
    df_model.drop(df_model.tail(1).index,inplace=True)
    df_model['hour']=df_model.index.hour
    df_model=df_model.astype(float)

    mdf=pd.DataFrame()
    mdf=df_model.groupby('hour').median()
    mdf['q1']=df_model.groupby('hour').quantile(0.25)
    mdf['q3']=df_model.groupby('hour').quantile(0.75)

    ax.plot(mdf.index,mdf['model'],label='Model',color='green')
    ax.fill_between(mdf.index,mdf['q1'],mdf['q3'],alpha=0.5,facecolor='limegreen',edgecolor='forestgreen')

    ax.set_title(position)
    ax.set_xlabel('')
    ax.set_ylabel('')


fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time of Day (hour)')
plt.ylabel(r'Ozone Concentration ($\mu g\:  m^{-3}$)')
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/'
plt.savefig(path+'o3_'+site+'_diurnals.png')
plt.close()



#    print(mdf)

#plt.show()





