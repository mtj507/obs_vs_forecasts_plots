import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy import config
import cartopy.crs as ccrs
import xarray as xr
import matplotlib as mp1
import cartopy.feature as cfeature


fig=plt.figure(figsize=[10,10])

emission='Ozone'
env_type='Background Urban'
metric_testing='ratio'


if env_type == 'AURN':
    env_type=' '

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['AURN Pollutants Measured'].str.contains(emission)]
metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
metadata=metadata.reset_index(drop=True)
location=metadata['Site Name']

if env_type == ' ':
    env_type='AURN'

metric_csv='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/ozone_site_metrics_'+env_type+'.csv'
metrics=pd.read_csv(metric_csv,low_memory=False)
metrics.drop(metrics.columns[0],axis=1,inplace=True)
metrics['ratio']=metrics['mod median']/metrics['obs median'
]
df=pd.merge(metadata,metrics,left_on='Site Name',right_on='site name',how='inner')

ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.OCEAN.with_scale('50m'))

x=df['Longitude']
y=df['Latitude']    
plt.scatter(x,y,c=df[metric_testing],cmap='plasma')

cbar=plt.colorbar(orientation='horizontal')
if not metric_testing == 'ODR gradient':
    cbar.ax.set_xlabel(metric_testing+r' $\mu g\: m^{-3}$')
if metric_testing == 'ODR gradient' or metric_testing == 'ratio':
    cbar.ax.set_xlabel(metric_testing)

ax.set_title(emission+' '+metric_testing+' - '+env_type)

#plt.show()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/'    
plt.savefig(path+emission+'_'+metric_testing+'_map_'+env_type+'.png')
plt.close()



