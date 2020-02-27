import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy import config
import cartopy.crs as ccrs
import xarray as xr
import matplotlib as mp1
import cartopy.feature as cfeature

fig=plt.figure()
fig,axes=plt.subplots(3,3,figsize=[10,10])

mode_csv='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/ozone_modes.csv'
modes=pd.read_csv(mode_csv,low_memory=False)
modes=modes.replace(['day','night','uniform'],['day over','night over','uniform over'])

mode_df=pd.DataFrame(index=range(0,7))
mode_df['site']=['London Marylebone Road','Sheffield Devonshire Green','Northampton Spring Park','Strathvaich','High Muffles','Reading New Town','Mace Head']
mode_df['mode']=['Bad','Day Over','Good','Night Over','Night Under','Uniform Over','Uniform Under']
mode_df=mode_df.sort_values('site')
mode_df=mode_df.reset_index(drop=True)
modes=mode_df['mode']

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['Site Name'].isin(mode_df['site'])]
metadata=metadata.reset_index(drop=True)
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
no_locations=len(metadata.index)


defra_csv='/users/mtj507/scratch/defra_data/o3_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
ddf=ddf.dropna(axis=0)
ddf=ddf.replace('No data', np.nan)
ddf['hour']=ddf.index.hour

f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
ds=xr.open_dataset(f)

for i,ax in zip(range(0,no_locations),axes.flatten()):
    site=location[i]
    mode=modes[i]
    ddf1=ddf.loc[:,[site,'hour']]
    ddf1=ddf1.replace(0,np.nan)
    ddf1=ddf1.dropna(axis=0)
    ddf1=ddf1.astype(float)

    df=pd.DataFrame()
    df=ddf1.groupby('hour').median()
    df['q1']=ddf1.groupby('hour').quantile(0.25)
    df['q3']=ddf1.groupby('hour').quantile(0.75)

    ax.plot(df.index,df[site],label='Observation',color='dimgrey')
    ax.fill_between(df.index,df['q1'],df['q3'],alpha=0.5,facecolor='dimgrey',edgecolor='grey')

    df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
    lats=ds['lat'].data
    lons=ds['lon'].data
    model_lat=np.argmin(np.abs(latitude[i]-lats))
    model_lon=np.argmin(np.abs(longitude[i]-lons))
    df_model['model']=ds['o3'].data[:,0,model_lat, model_lon]
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

    ax.set_title(mode,fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
   
axes[2,2].set_visible(False)
axes[2,1].set_visible(False)
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time of Day (hour)')
plt.ylabel(r'Ozone Concentration ($\mu g\:  m^{-3}$)')
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/'
plt.savefig(path+'ozone_mode_examples.png')



















