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


env_list=['Background Urban','Background Rural','Traffic Urban','Industrial Urban','Industrial Suburban','Background Suburban']

emissions=['Nitrogen dioxide','Nitric oxide','Ozone','PM2.5']

for x in range(0,len(emissions)):
    emission=emissions[x]
    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['AURN Pollutants Measured'].str.contains(emission)]

    fig=plt.figure(figsize=[10,10])
    ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))

    for i in range(0,len(env_list)):
        env_type=env_list[i]
#        print(env_type)
        meta=metadata.loc[metadata['Environment Type']==env_type]
        meta=meta.reset_index(drop=False)  
        latitude=meta['Latitude']
        longitude=meta['Longitude']
        plt.scatter(longitude,latitude,marker='x',label=env_type)
        plt.legend()
#        plt.show()

    ax.set_title(emission+' site map')
    path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/maps/'    
    plt.savefig(path+emission+'_site_map.png')
#    plt.cla()
#    print('saved')




