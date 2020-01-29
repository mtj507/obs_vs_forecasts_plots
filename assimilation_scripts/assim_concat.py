import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
lst=[]
for x in range(12):
    g=f'/users/mtj507/scratch/nasa_assimilations/assim_{months[x]}_2019.nc'
    mds=xr.open_dataset(g)
    lst.append(mds)
ds=xr.concat(lst,dim='time')
ds.to_netcdf('/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc')

# df_name=xr.concat([df_1,df_2, df_3, df_4], dim='time')


