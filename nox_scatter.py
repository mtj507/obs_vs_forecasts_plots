import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.odr import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#defining emission to be observed and conversion (can be found in conversion file)
emission='no2'
Emission='NO2'
conv = 1.88*10**9

environment_type='Background Urban'

week='fullweek'

if week == 'fullweek':
  day1=0
  day2=6
if week == 'weekday':
  day1=0
  day2=4
if week == 'weekend':
  day1=5
  day2=6

#defining dates of Defra data
date1='2019-09-22'
date2='2019-11-04'

#types of environment: Background Urban , Traffic Urban , Industrial Urban , Background Rural , Industrial Suburban , Background Suburban .

data_area='Greater London'
city='London'

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['Environment Type']==environment_type]
#metadata=metadata[metadata['Site Name'].str.match(city)]
#metadata=metadata.loc[metadata['Site Name']=='London Westminster']
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
ddf=ddf.loc[(ddf['weekday'] >= day1) & (ddf['weekday'] <= day2)]

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
noddf=noddf.loc[(noddf['weekday'] >= day1) & (noddf['weekday'] <= day2)]

plt.xlabel('Observation Median  ug/m3')
plt.ylabel('Forecast Median  ug/m3')
plt.title('NOX '+environment_type+' '+week)

obs_list=[]
obsQ1_list=[]
obsQ3_list=[]
nasa_list=[]
nasaQ1_list=[]
nasaQ3_list=[]

for i in np.arange(0, no_locations):
    try:
      ddf1=ddf.loc[:,['hour', f'{location[i]}']]
      noddf1=noddf.loc[:,[f'{location[i]}']]
      ddf1['no2']=ddf1[f'{location[i]}'].astype(float)
      noddf1['no']=noddf1[f'{location[i]}'].astype(float)
      ddf1['no']=noddf1['no']
      ddf1['nox']=ddf1['no']+ddf1['no2']
      ddf1=ddf1.dropna(axis=0)
      ddf_median=ddf1.groupby('hour').median()
      ddf_Q1=ddf1.groupby('hour')['nox'].quantile(0.25)
      ddf_Q3=ddf1.groupby('hour')['nox'].quantile(0.75)

      obs_median=ddf_median['nox'].mean()
      obs_median=float(round(obs_median,2))
      obs_Q1=ddf_Q1.mean()
      obs_Q1=float(round(obs_Q1,2))
      obs_Q3=ddf_Q3.mean()
      obs_Q3=float(round(obs_Q3,2))

      obs_list.append(obs_median)
      obsQ1_list.append(obs_Q1)
      obsQ3_list.append(obs_Q3)

      days_of_data=len(pd.unique(ddf['day and month']))
      dates=pd.unique(ddf['day and month'])
      mod_data = np.zeros((24,days_of_data))  
  
    except KeyError:
      continue
  
    for j in range(len(dates)):
        forecast_date=f'2019{str(dates[j]).zfill(4)}'
        f='/users/mtj507/scratch/nasa_forecasts/forecast_'+forecast_date+'.nc'
        ds=xr.open_dataset(f)
        spec=ds[emission].data
        nospec=ds['no'].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model=pd.DataFrame(ds[emission].data[:,0,model_lat, model_lon])
        nodf_model=pd.DataFrame(ds['no'].data[:,0,model_lat, model_lon])
        df_model.index=ds.time.data
        nodf_model.index=ds.time.data
        df_model.columns=[emission]
        nodf_model.columns=['no']
        df_model['no2']=df_model['no2']*conv
        df_model['no']=nodf_model['no']*1.23*10**9
        df_model['nox']=df_model['no2']+df_model['no']
        df_model['Hour']=df_model.index.hour
        df_model=df_model.reset_index()
        df_model=df_model.iloc[0:24]
        df_model=df_model.sort_index() 
       
  
        for k in range(24):
            mod_data[k,j] = df_model['nox'].loc[df_model['Hour'] == k].values[0]
        
       
    nasa_median=np.median(mod_data)
    nasa_median=float(round(nasa_median,2))
    nasa_Q1=np.percentile(mod_data,25)
    nasa_Q1=float(round(nasa_Q1,2))
    nasa_Q3=np.percentile(mod_data,75)
    nasa_Q3=float(round(nasa_Q3,2))
    
    nasa_list.append(nasa_median)
    nasaQ1_list.append(nasa_Q1)
    nasaQ3_list.append(nasa_Q3)

    print(location[i])

ax=plt.axes()
graph_data={'obs':obs_list,'obs Q1':obsQ1_list,'obs Q3':obsQ3_list,'forecast':nasa_list,'forecast Q1':nasaQ1_list,'forecast Q3':nasaQ3_list}
sdf=pd.DataFrame(graph_data)
sdf=sdf[sdf > 0].dropna()
sdf=sdf.dropna()
sdf['obs_err']=sdf['obs Q3']-sdf['obs Q1']
sdf['fcast_err']=sdf['forecast Q3']-sdf['forecast Q1']
sdf=sdf.reset_index(drop=True)

x_data=sdf['obs']
y_data=sdf['forecast']
x_err=sdf['obs_err']
y_err=sdf['fcast_err']

plt.scatter(x_data,y_data,color='red')
xy=np.linspace(*ax.get_xlim())
plt.plot(xy,xy,linestyle='dashed',color='grey')

def linear_func(p, x):
    y=p*x
    return y

linear=Model(linear_func)
datas=RealData(x_data,y_data,sx=x_err,sy=y_err)
odr=ODR(datas,linear,beta0=[0])
output=odr.run()
output.pprint()
beta=output.beta
betastd=output.sd_beta

plt.plot(x_data,linear_func(beta,x_data),color='black',alpha=0.7)

textbeta=float(output.beta)
textbeta=str(round(textbeta,3))
textbetastd=float(output.sd_beta)
textbetastd=str(round(textbetastd,3))

text=' Orthogonal Distance Regression: \n Gradient = '+textbeta+'\n Standard error = '+textbetastd
plt.annotate(text,fontsize=7,xy=(0.01,0.85),xycoords='axes fraction')

path='/users/mtj507/scratch/obs_vs_forecast/plots/scatter/'
plt.savefig(path+'nox_'+environment_type+'_'+week+'.png')
plt.close()















