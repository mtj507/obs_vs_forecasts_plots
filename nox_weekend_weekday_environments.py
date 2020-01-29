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


date1='2019-09-22'
date2='2019-11-04'

#weekday
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
    ddf=ddf.loc[(ddf['weekday'] >= 0) & (ddf['weekday'] <= 4)]
    ddf1=ddf.loc[:,headers]
    ddf1=ddf1.astype(float)
    ddf1=ddf1.dropna(axis=1,how='all')

    no_defra_csv='/users/mtj507/scratch/defra_data/defra_no_uk_2019.csv'
    noddf=pd.read_csv(no_defra_csv, low_memory=False)
    noddf.index=pd.to_datetime(noddf['Date'], dayfirst=True)+pd.to_timedelta(noddf['Time'])
    noddf=noddf.loc[:, ~noddf.columns.str.contains('^Unnamed')]
    noddf=noddf.dropna(axis=0)
    noddf=noddf.replace('No data', np.nan)
    noddf['hour']=noddf.index.hour
    noddf['weekday']=noddf.index.weekday
    noddf['month']=noddf.index.month.astype(str)
    noddf['month']=noddf['month'].str.zfill(2)
    noddf['day']=noddf.index.day.astype(str)
    noddf['day']=noddf['day'].str.zfill(2)
    noddf['day and month']=noddf['month']+noddf['day']
    noddf=noddf[date1:date2]
    noddf=noddf.loc[(noddf['weekday'] >= 0) & (noddf['weekday'] <= 4)]
    noddf1=noddf.loc[:,headers]
    noddf1=noddf1.astype(float)
    noddf1=noddf1.dropna(axis=1,how='all')
    
    df=pd.concat([ddf1, noddf1], axis=1)
    df=df.dropna(axis=1, how='all')
    df=df.drop('hour',axis=1)
    a=ddf1.columns
    b=noddf1.columns
    nox_locations=set(a).intersection(b)
    df=df.groupby(df.columns, axis=1).sum()
    df=df.loc[:,nox_locations]
    df['hour']=df.index.hour
    df_median=df.groupby('hour').median()
    df_median['median']=df_median.mean(axis=1)
    df_Q1=df1.groupby('hour').quantile(0.25)
    df_Q1['Q1']=df_Q1.mean(axis=1)
    df_Q3=df1.groupby('hour').quantile(0.75)
    df_Q3['Q3']=df_Q3.mean(axis=1)
    plt.plot(df_median.index, df_median['median'], label='Observation', color='blue')
    plt.fill_between(df_median.index, df_Q1['Q1'], df_Q3['Q3'], alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

    obs_median=df_median['median'].mean()
    obs_median=str(round(obs_median,2))
    obs_Q1=ddf_Q1['Q1'].mean()
    obs_Q1=str(round(obs_Q1,2))
    obs_Q3=ddf_Q3['Q3'].mean()
    obs_Q3=str(round(obs_Q3,2))

    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type']==environment]
    metadata=metadata[metadata['Site Name'].isin(nox_locations)]
    metadata=metadata.reset_index(drop=True)
    locations=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    
    no_locations=len(locations)
    days_of_data=len(pd.unique(ddf['day and month']))
    dates=pd.unique(ddf['day and month'])
    mod_data = np.zeros((24,days_of_data,no_locations))


    for x in np.arange(0, no_locations):
        print(f'{locations[x]}'+' weekday')

        for j in range(len(dates)):
            forecast_date=f'2019{str(dates[j]).zfill(4)}'
            f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
            ds=xr.open_dataset(f)
            spec=ds['no2'].data
            nospec=ds['no'].data
            lats=ds['lat'].data
            lons=ds['lon'].data
            model_lat=np.argmin(np.abs(latitude[x]-lats))
            model_lon=np.argmin(np.abs(longitude[x]-lons))
            df_model=pd.DataFrame(ds['no2'].data[:,0,model_lat, model_lon])
            nodf_model=pd.DataFrame(ds['no'].data[:,0,model_lat, model_lon])
            df_model.index=ds.time.data
            nodf_model.index=ds.time.data
            df_model.columns=['no2']
            nodf_model.columns=['no']
            df_model['no2']=df_model['no2']*1.88*10**9
            df_model['no']=nodf_model['no']*1.23*10**9
            df_model['nox']=df_model['no2']+df_model['no']
            df_model['Hour']=df_model.index.hour
            df_model=df_model.reset_index()
            df_model=df_model.iloc[0:24]
            df_model=df_model.sort_index()


            for k in range(24):
                mod_data[k,j,x] = df_model['nox'].loc[df_model['Hour'] == k].values[0]
        
    plt.plot(range(24),np.median(mod_data,axis=(1,2)),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=(1,2))
    Q3=np.percentile(mod_data, 75, axis=(1,2))
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red')
    plt.xlabel('Hour of Day')
    plt.ylabel('NOx' + ' ug/m3')
    plt.legend()
    plt.title(environment +' '+ 'NOx weekday')

    nasa_median=np.median(mod_data)
    nasa_median=str(round(nasa_median,2))
    nasa_Q1=np.percentile(mod_data,25)
    nasa_Q1=str(round(nasa_Q1,2))
    nasa_Q3=np.percentile(mod_data,75)
    nasa_Q3=str(round(nasa_Q3,2))


    def rmse(predictions, targets):
       return np.sqrt(((predictions-targets)**2).mean())

    rmse_val=rmse(np.median(mod_data,axis=(1,2)),df_median['median'])
    rmse_txt=str(round(rmse_val,2))

    text=' RMSE = '+rmse_txt+' ug/m3'
#    text=' Obs median = ' + obs_median + ' ug/m3 \n Obs IQR = ' + obs_Q1 + ' - ' + obs_Q3 + ' ug/m3 \n Forecast median = ' + nasa_median + ' ug/m3 \n Forecast IQR = ' + nasa_Q1 + ' - ' + nasa_Q3 + ' ug/m3'
    plt.annotate(text, fontsize=7, xy=(0.01, 0.85), xycoords='axes fraction')

    path='/users/mtj507/scratch/obs_vs_forecast/plots/environments/weekday/'
    plt.savefig(path+environment+'_'+'nox_weekday')
    print('saved')
    plt.close() 



#weekend
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
    ddf=ddf.loc[(ddf['weekday'] >= 5) &(ddf['weekday'] <= 6)]
    ddf1=ddf.loc[:,headers]
    ddf1=ddf1.astype(float)
    ddf1=ddf1.dropna(axis=1,how='all')

    no_defra_csv='/users/mtj507/scratch/defra_data/defra_no_uk_2019.csv'
    noddf=pd.read_csv(no_defra_csv, low_memory=False)
    noddf.index=pd.to_datetime(noddf['Date'], dayfirst=True)+pd.to_timedelta(noddf['Time'])
    noddf=noddf.loc[:, ~noddf.columns.str.contains('^Unnamed')]
    noddf=noddf.dropna(axis=0)
    noddf=noddf.replace('No data', np.nan)
    noddf['hour']=noddf.index.hour
    noddf['weekday']=noddf.index.weekday
    noddf['month']=noddf.index.month.astype(str)
    noddf['month']=noddf['month'].str.zfill(2)
    noddf['day']=noddf.index.day.astype(str)
    noddf['day']=noddf['day'].str.zfill(2)
    noddf['day and month']=noddf['month']+noddf['day']
    noddf=noddf[date1:date2]
    noddf=noddf.loc[(noddf['weekday'] >= 5) & (noddf['weekday'] <= 6)]
    noddf1=noddf.loc[:,headers]
    noddf1=noddf1.astype(float)
    noddf1=noddf1.dropna(axis=1,how='all')
    
    df=pd.concat([ddf1, noddf1], axis=1)
    df=df.dropna(axis=1, how='all')
    df=df.drop('hour',axis=1)
    a=ddf1.columns
    b=noddf1.columns
    nox_locations=set(a).intersection(b)
    df=df.groupby(df.columns, axis=1).sum()
    df=df.loc[:,nox_locations]
    df['hour']=df.index.hour
    df_median=df.groupby('hour').median()
    df_median['median']=df_median.mean(axis=1)
    df_Q1=df1.groupby('hour').quantile(0.25)
    df_Q1['Q1']=df_Q1.mean(axis=1)
    df_Q3=df1.groupby('hour').quantile(0.75)
    df_Q3['Q3']=df_Q3.mean(axis=1)
    plt.plot(df_median.index, df_median['median'], label='Observation', color='blue')
    plt.fill_between(df_median.index, df_Q1['Q1'], df_Q3['Q3'], alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

    obs_median=df_median['median'].mean()
    obs_median=str(round(obs_median,2))
    obs_Q1=ddf_Q1['Q1'].mean()
    obs_Q1=str(round(obs_Q1,2))
    obs_Q3=ddf_Q3['Q3'].mean()
    obs_Q3=str(round(obs_Q3,2))

    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type']==environment]
    metadata=metadata[metadata['Site Name'].isin(nox_locations)]
    metadata=metadata.reset_index(drop=True)
    locations=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    
    no_locations=len(locations)
    days_of_data=len(pd.unique(ddf['day and month']))
    dates=pd.unique(ddf['day and month'])
    mod_data = np.zeros((24,days_of_data,no_locations))


    for y in np.arange(0, no_locations):
        print(f'{locations[y]}'+' weekend')

        for l in range(len(dates)):
            forecast_date=f'2019{str(dates[l]).zfill(4)}'
            f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
            ds=xr.open_dataset(f)
            spec=ds['no2'].data
            nospec=ds['no'].data
            lats=ds['lat'].data
            lons=ds['lon'].data
            model_lat=np.argmin(np.abs(latitude[y]-lats))
            model_lon=np.argmin(np.abs(longitude[y]-lons))
            df_model=pd.DataFrame(ds['no2'].data[:,0,model_lat, model_lon])
            nodf_model=pd.DataFrame(ds['no'].data[:,0,model_lat, model_lon])
            df_model.index=ds.time.data
            nodf_model.index=ds.time.data
            df_model.columns=['no2']
            nodf_model.columns=['no']
            df_model['no2']=df_model['no2']*1.88*10**9
            df_model['no']=nodf_model['no']*1.23*10**9
            df_model['nox']=df_model['no2']+df_model['no']
            df_model['Hour']=df_model.index.hour
            df_model=df_model.reset_index()
            df_model=df_model.iloc[0:24]
            df_model=df_model.sort_index()


            for z in range(24):
                mod_data[z,l,y] = df_model['nox'].loc[df_model['Hour'] == z].values[0]
        
    plt.plot(range(24),np.median(mod_data,axis=(1,2)),label='Model',color='maroon')
    Q1=np.percentile(mod_data, 25, axis=(1,2))
    Q3=np.percentile(mod_data, 75, axis=(1,2))
    plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red')
    plt.xlabel('Hour of Day')
    plt.ylabel('NOx' + ' ug/m3')
    plt.legend()
    plt.title(environment +' '+ 'NOx weekend')

    nasa_median=np.median(mod_data)
    nasa_median=str(round(nasa_median,2))
    nasa_Q1=np.percentile(mod_data,25)
    nasa_Q1=str(round(nasa_Q1,2))
    nasa_Q3=np.percentile(mod_data,75)
    nasa_Q3=str(round(nasa_Q3,2))

    def rmse(predictions, targets):
       return np.sqrt(((predictions-targets)**2).mean())

    rmse_val=rmse(np.median(mod_data,axis=(1,2)),df_median['median'])
    rmse_txt=str(round(rmse_val,2))

    text=' RMSE = '+rmse_txt+' ug/m3'
#    text=' Obs median = ' + obs_median + ' ug/m3 \n Obs IQR = ' + obs_Q1 + ' - ' + obs_Q3 + ' ug/m3 \n Forecast median = ' + nasa_median + ' ug/m3 \n Forecast IQR = ' + nasa_Q1 + ' - ' + nasa_Q3 + ' ug/m3'
    plt.annotate(text, fontsize=7, xy=(0.01, 0.85), xycoords='axes fraction')

    path='/users/mtj507/scratch/obs_vs_forecast/plots/environments/weekend/'
    plt.savefig(path+environment+'_'+'nox_weekend')
    print('saved')
    plt.close() 
    
