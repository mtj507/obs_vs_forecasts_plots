import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import pandas as pd
from cartopy import config
import cartopy.crs as ccrs
import xarray as xr
import matplotlib as mp1
import cartopy.feature as cfeature

fig=plt.figure(figsize=[12,12])

emission='Ozone'
env_type='Background Rural'

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['AURN Pollutants Measured'].str.contains(emission)]
metadata=metadata.loc[metadata['Environment Type']==env_type]
metadata=metadata.reset_index(drop=True)
latitude=metadata['Latitude']
longitude=metadata['Longitude']
site=metadata['Site Name']

ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.OCEAN.with_scale('50m'))

for i in range(len(metadata.index)):
    x=longitude[i]
    y=latitude[i]
    plt.scatter(x,y,label=site[i],color='red',marker='x')
    plt.text(x-0.1,y+0.1,site[i],fontsize=12)

#plt.show()

ax.set_title(emission+' '+env_type+' Site Map')
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/maps/'    
plt.savefig(path+emission+'_'+env_type+'_labelled_site_map.png')




