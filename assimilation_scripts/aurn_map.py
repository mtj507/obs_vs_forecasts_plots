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

fig=plt.figure()
ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)

env_list=['Background Urban','Background Rural','Traffic Urban','Industrial Urban','Industrial Suburban','Background Suburban']

emissions=['no2','no','o3','pm25']

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)

for i in range(0,len(env_list)):
    env_type=env_list[i]
    print(env_type)
    meta=metadata.loc[metadata['Environment Type']==env_type]
    meta=meta.reset_index(drop=False)  
    latitude=meta['Latitude']
    longitude=meta['Longitude']
    plt.scatter(longitude,latitude,marker='x',label=env_type)
    plt.legend()
    plt.show()
