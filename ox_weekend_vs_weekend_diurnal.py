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
warnings.simplefilter(action='ignore', category=FutureWarning)

#defining emission to be observed and conversion (can be found in conversion file)
emission='no2'
Emission='NO2'
conv = 1.88*10**9


#defining dates of Defra data
date1='2019-09-22'
date2='2019-11-04'

#types of environment: Background Urban , Traffic Urban , Industrial Urban , Background Rural , Industrial Suburban , Background Suburban .

environment_type='Background Urban'
data_area='Greater London'
city='London'

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
#metadata=metadata.loc[metadata['Environment Type']==environment_type]
#metadata=metadata[metadata['Site Name'].str.match(city)]
metadata=metadata.loc[metadata['Site Name']=='London N. Kensington']
metadata=metadata.reset_index(drop=False)
area=metadata['Zone']
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
environment=metadata['Environment Type']
no_locations=len(metadata.index)


#change to UTF csv before moving across to Viking and edit doc so its easy to import by deleting first 3 rowns and moving time and date column headers into same row as locations. Delete empty rows up to 'end' at bottom and format time cells to time.
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

for i in np.arange(0, no_locations):
    try:
        ddf1=ddf.loc[:,['hour', f'{location[i]}','weekday','day and month']]
        ozddf1=ozddf.loc[:,[f'{location[i]}']]
        ddf1['no2']=ddf1[f'{location[i]}'].astype(float)
        ozddf1['o3']=ozddf1[f'{location[i]}'].astype(float)
        ddf1['o3']=ozddf1['o3']
        ddf1['ox']=ddf1['o3']+ddf1['no2']
        ddf2=ddf1.loc[(ddf['weekday'] >= 0) & (ddf['weekday'] <= 4)]
        ddf2=ddf2.dropna(axis=0)
        ddf_median=ddf2.groupby('hour').median()
        ddf_Q1=ddf2.groupby('hour')['ox'].quantile(0.25)
        ddf_Q3=ddf2.groupby('hour')['ox'].quantile(0.75)
        plt.plot(ddf_median.index, ddf_median['ox'], label='Observation', color='blue')
        plt.fill_between(ddf_median.index, ddf_Q1, ddf_Q3, alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

        obs_median=ddf_median['ox'].mean()
        obs_median=str(round(obs_median,2))
        obs_Q1=ddf_Q1.mean()
        obs_Q1=str(round(obs_Q1,2))
        obs_Q3=ddf_Q3.mean()
        obs_Q3=str(round(obs_Q3,2))

        days_of_data=len(pd.unique(ddf2['day and month']))
        dates=pd.unique(ddf2['day and month'])
        mod_data = np.zeros((24,days_of_data))  
    except KeyError:
        continue
 
    for j in range(len(dates)):
        forecast_date=f'2019{str(dates[j]).zfill(4)}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[emission].data
        ozspec=ds['o3'].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model=pd.DataFrame(ds[emission].data[:,0,model_lat, model_lon])
        ozdf_model=pd.DataFrame(ds['o3'].data[:,0,model_lat, model_lon])
        df_model.index=ds.time.data
        ozdf_model.index=ds.time.data
        df_model.columns=[emission]
        ozdf_model.columns=['o3']
        df_model['no2']=df_model['no2']*conv
        df_model['o3']=ozdf_model['o3']*2*10**9
        df_model['ox']=df_model['no2']+df_model['o3']
        df_model['Hour']=df_model.index.hour
        df_model=df_model.reset_index()
        df_model=df_model.iloc[0:24]
        df_model=df_model.sort_index() 
  
        for k in range(24):
            mod_data[k,j] = df_model['ox'].loc[df_model['Hour'] == k].values[0]
        

    plt.plot(range(24),np.median(mod_data,1),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=1)
    Q3=np.percentile(mod_data, 75, axis=1)
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red') 
    plt.xlabel('Hour of Day')
    plt.ylabel('Ox ug/m3')
    plt.legend()
    plt.title(location[i])

    nasa_median=np.median(mod_data)
    nasa_median=str(round(nasa_median,2))
    nasa_Q1=np.percentile(mod_data,25)
    nasa_Q1=str(round(nasa_Q1,2))
    nasa_Q3=np.percentile(mod_data,75)
    nasa_Q3=str(round(nasa_Q3,2))

    text=' Obs median = ' + obs_median + ' ug/m3 \n Obs IQR = ' + obs_Q1 + ' - ' + obs_Q3 + ' ug/m3 \n Forecast median = ' + nasa_median + ' ug/m3 \n Forecast IQR = ' + nasa_Q1 + ' - ' + nasa_Q3 + ' ug/m3'
    plt.annotate(text, fontsize=7, xy=(0.01, 0.85), xycoords='axes fraction')

    path='/users/mtj507/scratch/obs_vs_forecast/plots/ox/weekend_weekday_diurnal/'
    #plt.savefig(path+'ox'+f'_{location[i]}_weekday_diurnal.png')
    #plt.close()
    print(location[i])



for x in np.arange(0, no_locations):
    try:
        ddf1=ddf.loc[:,['hour', f'{location[x]}','weekday','day and month']]
        ozddf1=ozddf.loc[:,[f'{location[x]}']]
        ddf1['no2']=ddf1[f'{location[x]}'].astype(float)
        ozddf1['o3']=ozddf1[f'{location[x]}'].astype(float)
        ddf1['o3']=ozddf1['o3']
        ddf1['ox']=ddf1['o3']+ddf1['no2']
        ddf2=ddf1.loc[(ddf['weekday'] >= 5) & (ddf['weekday'] <= 6)]
        ddf2=ddf2.dropna(axis=0)
        ddf_median=ddf2.groupby('hour').median()
        ddf_Q1=ddf2.groupby('hour')['ox'].quantile(0.25)
        ddf_Q3=ddf2.groupby('hour')['ox'].quantile(0.75)
        plt.plot(ddf_median.index, ddf_median['ox'], label='Observation', color='blue')
        plt.fill_between(ddf_median.index, ddf_Q1, ddf_Q3, alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

        obs_median=ddf_median['ox'].mean()
        obs_median=str(round(obs_median,2))
        obs_Q1=ddf_Q1.mean()
        obs_Q1=str(round(obs_Q1,2))
        obs_Q3=ddf_Q3.mean()
        obs_Q3=str(round(obs_Q3,2))

        days_of_data=len(pd.unique(ddf2['day and month']))
        dates=pd.unique(ddf2['day and month'])
        mod_data = np.zeros((24,days_of_data))  
    except KeyError:
        continue
 
    for y in range(len(dates)):
        forecast_date=f'2019{str(dates[y]).zfill(4)}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[emission].data
        ozspec=ds['o3'].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[x]-lats))
        model_lon=np.argmin(np.abs(longitude[x]-lons))
        df_model=pd.DataFrame(ds[emission].data[:,0,model_lat, model_lon])
        ozdf_model=pd.DataFrame(ds['o3'].data[:,0,model_lat, model_lon])
        df_model.index=ds.time.data
        ozdf_model.index=ds.time.data
        df_model.columns=[emission]
        ozdf_model.columns=['o3']
        df_model['no2']=df_model['no2']*conv
        df_model['o3']=ozdf_model['o3']*2*10**9
        df_model['ox']=df_model['no2']+df_model['o3']
        df_model['Hour']=df_model.index.hour
        df_model=df_model.reset_index()
        df_model=df_model.iloc[0:24]
        df_model=df_model.sort_index() 
  
        for z in range(24):
            mod_data[z,y] = df_model['ox'].loc[df_model['Hour'] == z].values[0]
        

    plt.plot(range(24),np.median(mod_data,1),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=1)
    Q3=np.percentile(mod_data, 75, axis=1)
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red') 
    plt.xlabel('Hour of Day')
    plt.ylabel('Ox ug/m3')
    plt.legend()
    plt.title(location[x])

    nasa_median=np.median(mod_data)
    nasa_median=str(round(nasa_median,2))
    nasa_Q1=np.percentile(mod_data,25)
    nasa_Q1=str(round(nasa_Q1,2))
    nasa_Q3=np.percentile(mod_data,75)
    nasa_Q3=str(round(nasa_Q3,2))

    text=' Obs median = ' + obs_median + ' ug/m3 \n Obs IQR = ' + obs_Q1 + ' - ' + obs_Q3 + ' ug/m3 \n Forecast median = ' + nasa_median + ' ug/m3 \n Forecast IQR = ' + nasa_Q1 + ' - ' + nasa_Q3 + ' ug/m3'
    plt.annotate(text, fontsize=7, xy=(0.01, 0.85), xycoords='axes fraction')

    path='/users/mtj507/scratch/obs_vs_forecast/plots/ox/weekend_weekday_diurnal/'
    #plt.savefig(path+'ox'+f'_{location[x]}_weekend_diurnal.png')
    #plt.close()
    print(location[x])
    plt.show()












