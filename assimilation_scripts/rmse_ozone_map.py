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
#metrics that can be tested:ratio of mod med:obs med ('ratio'),rmse forced through origin ('RMSE'), ODR forced through origin('ODR gradient'),odr not forced through origin (nf ODR gradient).

emission='Ozone'
env_type='Background Urban'
metric_testing='RMSE'


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
metrics['ratio']=metrics['mod median']/metrics['obs median']

df=pd.merge(metadata,metrics,left_on='Site Name',right_on='site name',how='inner')

nf_csv='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/ozone_nf_site_metrics_'+env_type+'.csv'
nf=pd.read_csv(nf_csv,low_memory=False)
nf.drop(nf.columns[0],axis=1,inplace=True)

df=pd.merge(df,nf,left_on='Site Name',right_on='site name',how='inner')

ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.OCEAN.with_scale('50m'))


m=plt.cm.ScalarMappable(cmap='cool')
m.set_clim(10,35)
cbar=plt.colorbar(m,orientation='horizontal')

x=df['Longitude']
y=df['Latitude']    
plt.scatter(x,y,c=df[metric_testing],cmap='cool',vmin=10,vmax=35)

#norm=mp1.colors.Normalize(vmin=19,vmax=37)
#cbar=plt.colorbar(orientation='horizontal',norm=norm)
#cbar.set_clim(19,37)

if not metric_testing == 'ODR gradient':
    cbar.ax.set_xlabel(metric_testing+r' $\mu g\: m^{-3}$')
if metric_testing == 'ODR gradient' or metric_testing == 'ratio':
    cbar.ax.set_xlabel(metric_testing)
if metric_testing == 'nf RMSE':
    cbar.ax.set_xlabel(r'RMSE $\mu g\: m^{-3}$')

    
ax.set_title(emission+' '+metric_testing+' - '+env_type)

#plt.show()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/'    
plt.savefig(path+emission+'_'+metric_testing+'_map_'+env_type+'.png')
plt.close()



