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

emissions=['no2', 'no', 'pm25', 'o3']


for environment in environments:
    metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    metadata=metadata.loc[metadata['Environment Type']==environment]
    check=pd.unique(metadata['Environment Type'])
    metadata=metadata.reset_index(drop=True)
    locations=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    headers=locations.values.tolist()    
    headers.append('hour')  
    print(check)
    no_locations=len(metadata.index)
    
#weekday 
    for i in emissions:
        defra_csv='/users/mtj507/scratch/defra_data/defra_'+i+'_uk_2019.csv'
        print(i)
        ddf=pd.read_csv(defra_csv, low_memory=False)
        ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
        ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
        ddf=ddf.dropna(axis=0)
        ddf=ddf.replace('No data', np.nan)
        ddf['weekday']=ddf.index.weekday
        ddf['month']=ddf.index.month.astype(str)
        ddf['month']=ddf['month'].str.zfill(2)
        ddf['day']=ddf.index.day.astype(str)
        ddf['day']=ddf['day'].str.zfill(2)
        ddf['day and month']=ddf['month']+ddf['day']

        ddf['hour']=ddf.index.hour
        ddf=ddf.loc['2019-09-22':'2019-11-05']
        ddf=ddf.loc[(ddf['weekday'] >= 0) & (ddf['weekday'] <= 4)]
        ddf1=ddf.loc[:,headers] 
        ddf1=ddf1.astype(float)
        ddf1=ddf1.dropna(axis=1,how='all')
        if (environment == 'Industrial Suburban' and i == 'pm25'):
          continue 
        ddf_median=ddf1.groupby('hour').median()
        ddf_median['median']=ddf_median.mean(axis=1)
        ddf_Q1=ddf1.groupby('hour').quantile(0.25)
        ddf_Q1['Q1']=ddf_Q1.mean(axis=1)
        ddf_Q3=ddf1.groupby('hour').quantile(0.75)
        ddf_Q3['Q3']=ddf_Q3.mean(axis=1)
        plt.plot(ddf_median.index, ddf_median['median'], label='Observation', color='blue')
        plt.fill_between(ddf_median.index, ddf_Q1['Q1'], ddf_Q3['Q3'], alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

        obs_median=ddf_median['median'].mean()
        obs_median=str(round(obs_median,2))
        obs_Q1=ddf_Q1['Q1'].mean()
        obs_Q1=str(round(obs_Q1,2))
        obs_Q3=ddf_Q3['Q3'].mean()
        obs_Q3=str(round(obs_Q3,2))

        days_of_data=len(pd.unique(ddf['day and month']))
        dates=pd.unique(ddf['day and month'])
        mod_data = np.zeros((24,days_of_data,no_locations))
        
        if i == 'no2': 
          conv=1.88*10**9
        if i == 'no':
          conv=1.23*10**9   
        if i == 'pm25':
          conv=1
        if i == 'o3':
          conv=2*10**9

        for x in np.arange(0, no_locations):
            print(f'{locations[x]}'+' weekday')
            

            for j in range(len(dates)):
                forecast_date=f'2019{str(dates[j]).zfill(4)}'
                f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
                ds=xr.open_dataset(f)
                if i == 'pm25':
                  i='pm25_rh35_gcc'
                spec=ds[i].data
                lats=ds['lat'].data
                lons=ds['lon'].data
                model_lat=np.argmin(np.abs(latitude[x]-lats))
                model_lon=np.argmin(np.abs(longitude[x]-lons))
                df_model=pd.DataFrame(ds[i].data[:,0,model_lat, model_lon])
                df_model.index=ds.time.data
                df_model.columns=[i]
                df_model.index.name='date_time'
                time=df_model.index.hour
                df_model['Hour']=time
                df_model=df_model.reset_index()
                df_model=df_model.iloc[0:24]
                df_model=df_model.sort_index()
                
                df_model[i]=df_model[i]*conv

                for k in range(24):
                    mod_data[k,j,x] = df_model[i].loc[df_model['Hour'] == k].values[0]
        
        if i == 'pm25_rh35_gcc':
          i = 'pm25' 
        plt.plot(range(24),np.median(mod_data,axis=(1,2)),label='Model',color='maroon')
        Q1=np.percentile(mod_data, 25, axis=(1,2))
        Q3=np.percentile(mod_data, 75, axis=(1,2))
        plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red')
        plt.xlabel('Hour of Day')
        plt.ylabel(i + ' ug/m3')
        plt.legend()
        plt.title(environment +' '+ i + ' weekday')

        nasa_median=np.median(mod_data)
        nasa_median=str(round(nasa_median,2))
        nasa_Q1=np.percentile(mod_data,25)
        nasa_Q1=str(round(nasa_Q1,2))
        nasa_Q3=np.percentile(mod_data,75)
        nasa_Q3=str(round(nasa_Q3,2))

        def rmse(predictions, targets):
            return np.sqrt(((predictions-targets)**2).mean())

        rmse_val=rmse(np.median(mod_data,axis=(1,2)),ddf_median['median'])
        rmse_txt=str(round(rmse_val,2))

        text=' RMSE = '+rmse_txt+' ug/m3'
#        text=' Obs median = ' + obs_median + ' ug/m3 \n Obs IQR = ' + obs_Q1 + ' - ' + obs_Q3 + ' ug/m3 \n Forecast median = ' + nasa_median + ' ug/m3 \n Forecast IQR = ' + nasa_Q1 + ' - ' + nasa_Q3 + ' ug/m3'
        plt.annotate(text, fontsize=7, xy=(0.01, 0.85), xycoords='axes fraction')

        path='/users/mtj507/scratch/obs_vs_forecast/plots/environments/weekday/'
        plt.savefig(path+environment+'_'+i+'_weekday')
        print('saved')
        plt.close() 



#weekend
    for m in emissions:
        defra_csv='/users/mtj507/scratch/defra_data/defra_'+m+'_uk_2019.csv'
        print(m)
        ddf=pd.read_csv(defra_csv, low_memory=False)
        ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
        ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
        ddf=ddf.dropna(axis=0)
        ddf=ddf.replace('No data', np.nan)
        ddf['weekday']=ddf.index.weekday
        ddf['month']=ddf.index.month.astype(str)
        ddf['month']=ddf['month'].str.zfill(2)
        ddf['day']=ddf.index.day.astype(str)
        ddf['day']=ddf['day'].str.zfill(2)
        ddf['day and month']=ddf['month']+ddf['day']

        ddf['hour']=ddf.index.hour
        ddf=ddf.loc['2019-09-22':'2019-11-05']
        ddf=ddf.loc[(ddf['weekday'] >= 5) & (ddf['weekday'] <= 6)]
        ddf1=ddf.loc[:,headers] 
        ddf1=ddf1.astype(float)
        ddf1=ddf1.dropna(axis=1,how='all')
        if (environment == 'Industrial Suburban' and m == 'pm25'):
          continue 
        ddf_median=ddf1.groupby('hour').median()
        ddf_median['median']=ddf_median.mean(axis=1)
        ddf_Q1=ddf1.groupby('hour').quantile(0.25)
        ddf_Q1['Q1']=ddf_Q1.mean(axis=1)
        ddf_Q3=ddf1.groupby('hour').quantile(0.75)
        ddf_Q3['Q3']=ddf_Q3.mean(axis=1)
        plt.plot(ddf_median.index, ddf_median['median'], label='Observation', color='blue')
        plt.fill_between(ddf_median.index, ddf_Q1['Q1'], ddf_Q3['Q3'], alpha=0.5, facecolor='turquoise', edgecolor='deepskyblue')

        obs_median=ddf_median['median'].mean()
        obs_median=str(round(obs_median,2))
        obs_Q1=ddf_Q1['Q1'].mean()
        obs_Q1=str(round(obs_Q1,2))
        obs_Q3=ddf_Q3['Q3'].mean()
        obs_Q3=str(round(obs_Q3,2))

        days_of_data=len(pd.unique(ddf['day and month']))
        dates=pd.unique(ddf['day and month'])
        mod_data = np.zeros((24,days_of_data,no_locations))
        
        if m == 'no2': 
          conv=1.88*10**9
        if m == 'no':
          conv=1.23*10**9   
        if m == 'pm25':
          conv=1
        if m == 'o3':
          conv=2*10**9

        for y in np.arange(0, no_locations):
            print(f'{locations[y]}'+' weekend')
            

            for l in range(len(dates)):
                forecast_date=f'2019{str(dates[l]).zfill(4)}'
                f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
                ds=xr.open_dataset(f)
                if m == 'pm25':
                  m='pm25_rh35_gcc'
                spec=ds[m].data
                lats=ds['lat'].data
                lons=ds['lon'].data
                model_lat=np.argmin(np.abs(latitude[y]-lats))
                model_lon=np.argmin(np.abs(longitude[y]-lons))
                df_model=pd.DataFrame(ds[m].data[:,0,model_lat, model_lon])
                df_model.index=ds.time.data
                df_model.columns=[m]
                df_model.index.name='date_time'
                time=df_model.index.hour
                df_model['Hour']=time
                df_model=df_model.reset_index()
                df_model=df_model.iloc[0:24]
                df_model=df_model.sort_index()
                
                df_model[m]=df_model[m]*conv

                for z in range(24):
                    mod_data[z,l,y] = df_model[m].loc[df_model['Hour'] == z].values[0]
        
        if m == 'pm25_rh35_gcc':
          m = 'pm25' 
        plt.plot(range(24),np.median(mod_data,axis=(1,2)),label='Model',color='maroon')
        Q1=np.percentile(mod_data, 25, axis=(1,2))
        Q3=np.percentile(mod_data, 75, axis=(1,2))
        plt.fill_between(range(24), Q1, Q3, alpha=0.5, facecolor='red', edgecolor='red')
        plt.xlabel('Hour of Day')
        plt.ylabel(m + ' ug/m3')
        plt.legend()
        plt.title(environment +' '+ m + ' weekend')

        nasa_median=np.median(mod_data)
        nasa_median=str(round(nasa_median,2))
        nasa_Q1=np.percentile(mod_data,25)
        nasa_Q1=str(round(nasa_Q1,2))
        nasa_Q3=np.percentile(mod_data,75)
        nasa_Q3=str(round(nasa_Q3,2))

        def rmse(predictions, targets):
            return np.sqrt(((predictions-targets)**2).mean())

        rmse_val=rmse(np.median(mod_data,axis=(1,2)),ddf_median['median'])
        rmse_txt=str(round(rmse_val,2))

        text=' RMSE = '+rmse_txt+' ug/m3'
#        text=' Obs median = ' + obs_median + ' ug/m3 \n Obs IQR = ' + obs_Q1 + ' - ' + obs_Q3 + ' ug/m3 \n Forecast median = ' + nasa_median + ' ug/m3 \n Forecast IQR = ' + nasa_Q1 + ' - ' + nasa_Q3 + ' ug/m3'
        plt.annotate(text, fontsize=7, xy=(0.01, 0.85), xycoords='axes fraction')

        path='/users/mtj507/scratch/obs_vs_forecast/plots/environments/weekend/'
        plt.savefig(path+environment+'_'+m+'_weekend')
        print('saved')
        plt.close() 
    
