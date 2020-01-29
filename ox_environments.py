import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import warnings
#prevent warning when slicing ddf by headers
warnings.simplefilter(action='ignore', category=FutureWarning)

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
environment=metadata['Environment Type']
environments=pd.unique(environment)

#Ox is NO2 and O3

date1='2019-09-22'
date2='2019-11-04'


for environment in environments:
    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type']==environment]
    check=pd.unique(metadata['Environment Type'])
    metadata=metadata.reset_index(drop=True)
    locations=metadata['Site Name']
    headers=locations.values.tolist()    
    headers.append('hour')  
    print(check)
    
    no2_defra_csv='/users/mtj507/scratch/defra_data/defra_no2_uk_2019.csv'
    ddf=pd.read_csv(no2_defra_csv, low_memory=False)
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
    ddf1=ddf.loc[:,headers]
    ddf1=ddf1.astype(float)
    ddf1=ddf1.dropna(axis=1,how='all')

    oz_defra_csv='/users/mtj507/scratch/defra_data/defra_o3_uk_2019.csv'
    ozddf=pd.read_csv(oz_defra_csv, low_memory=False)
    ozddf.index=pd.to_datetime(ozddf['Date'], dayfirst=True)+pd.to_timedelta(ozddf['Time'])
    ozddf=ozddf.loc[:, ~ozddf.columns.str.contains('^Unnamed')]
    ozddf=ozddf.dropna(axis=0)
    ozddf=ozddf.replace('No data', np.nan)
    ozddf['hour']=ozddf.index.hour
    ozddf['weekday']=ozddf.index.weekday
    ozddf['month']=ozddf.index.month.astype(str)
    ozddf['month']=ozddf['month'].str.zfill(2)
    ozddf['day']=ozddf.index.day.astype(str)
    ozddf['day']=ozddf['day'].str.zfill(2)
    ozddf['day and month']=ozddf['month']+ozddf['day']
    ozddf=ozddf[date1:date2]
    ozddf1=ozddf.loc[:,headers]
    ozddf1=ozddf1.astype(float)
    ozddf1=ozddf1.dropna(axis=1,how='all')
    
    df=pd.concat([ddf1, ozddf1], axis=1)
    df=df.dropna(axis=1,how='all')
    df=df.drop('hour', axis=1)
    a=ddf1.columns
    b=ozddf1.columns
    ox_locations=set(a).intersection(b)
    df=df.groupby(df.columns, axis=1).sum()
    df=df.loc[:,ox_locations]
    df['hour']=df.index.hour
    df_mean=df.groupby('hour').mean()
    df_mean['mean']=df_mean.mean(axis=1)
    df_std=df.groupby('hour').std()
    df_std['std']=df_std.std(axis=1)
    plt.plot(df_mean.index, df_mean['mean'], label='Observation', color='blue')
    plt.fill_between(df_mean.index, (df_mean['mean']+df_std['std']), (df_mean['mean']-df_std['std']), alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')
   
    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type']==environment]
    metadata=metadata[metadata['Site Name'].isin(ox_locations)]
    metadata=metadata.reset_index(drop=True)
    locations=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']

    no_locations=len(locations)
    days_of_data=len(pd.unique(ddf['day and month']))
    dates=pd.unique(ddf['day and month'])
    mod_data = np.zeros((24,days_of_data,no_locations))
    

    for x in np.arange(0, no_locations):
        print(f'{locations[x]}')

        for j in range(len(dates)):
            forecast_date=f'2019{str(dates[j]).zfill(4)}'
            f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
            ds=xr.open_dataset(f)
            spec=ds['no2'].data
            ozspec=ds['o3'].data
            lats=ds['lat'].data
            lons=ds['lon'].data
            model_lat=np.argmin(np.abs(latitude[x]-lats))
            model_lon=np.argmin(np.abs(longitude[x]-lons))
            df_model=pd.DataFrame(ds['no2'].data[:,0,model_lat, model_lon])
            ozdf_model=pd.DataFrame(ds['o3'].data[:,0,model_lat, model_lon])
            df_model.index=ds.time.data
            ozdf_model.index=ds.time.data
            df_model.columns=['no2']
            ozdf_model.columns=['o3']
            df_model['no2']=df_model['no2']*1.88*10**9
            df_model['o3']=ozdf_model['o3']*2*10**9
            df_model['ox']=df_model['no2']+df_model['o3']
            df_model['Hour']=df_model.index.hour
            df_model=df_model.reset_index()
            df_model=df_model.iloc[0:24]
            df_model=df_model.sort_index()


            for k in range(24):
                mod_data[k,j,x] = df_model['ox'].loc[df_model['Hour'] == k].values[0]
        
    plt.plot(range(24),np.median(mod_data,axis=(1,2)),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=(1,2))
    Q3=np.percentile(mod_data, 75, axis=(1,2))
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red')
    plt.xlabel('Hour of Day')
    plt.ylabel('Ox' + ' ug/m3')
    plt.legend()
    plt.title(environment +' '+ 'Ox')
    path='/users/mtj507/scratch/obs_vs_forecast/plots/environments/'
    plt.savefig(path+environment+'_'+'ox')
    print('saved')
    plt.close() 






    
