import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import openaq
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

emission='no2'

env_list=['AURN','Background Urban','Traffic Urban','Background Rural','Industrial urban','Industrial Suburban','background Suburban']
 
env_type='AURN'
if env_type =='AURN':
    env_type =' '

week='fullweek'

date1='2019-01-01'
date2='2019-12-31'

if emission == 'no2':
  conv=1.88*10**9
  nasa_emission='no2'
  Emission='NO2'

if emission == 'no':
  conv=1.23*10**9
  nasa_emission='no'
  Emission='NO'

if emission == 'pm25':
  conv=1
  nasa_emission='pm25_rh35_gcc'
  Emission='PM 2.5'

if emission == 'o3':
  conv=2*10**9
  nasa_emission='o3'
  Emission='O3'

if week == 'fullweek':
  day1=0
  day2=6
if week == 'weekday':
  day1=0
  day2=4
if week == 'weekend':
  day1=5
  day2=6

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
#metadata=metadata[metadata['Site Name'].str.match(city)]
#metadata=metadata.loc[metadata['Site Name']=='London Westminster']
metadata=metadata.reset_index(drop=False)
area=metadata['Zone']
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
environment=metadata['Environment Type']
no_locations=len(metadata.index)
a=location

defra_csv='/users/mtj507/scratch/defra_data/'+emission+'_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
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

no_defra_csv='/users/mtj507/scratch/defra_data/no_2019.csv'
noddf=pd.read_csv(no_defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
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

b=ddf.columns
headers=set(a).intersection(b)

ddf=ddf.loc[:,headers]
noddf=noddf.loc[:,headers]
ddf1=pd.concat([ddf,noddf],axis=1)
ddf1.groupby(ddf1.columns,axis=1).sum()


df=pd.DataFrame(index=ddf.index,columns=['median','Q1','Q3'])
df['median']=ddf.median(axis=1)
df['Q1']=ddf.quantile(0.25,axis=1)
df['Q3']=ddf.quantile(0.75,axis=1)
plt.plot(df.index,df['median'],color='dimgrey',alpha=0.8,label='Observation')
obs_mean=df['median'].mean(axis=0)
obs_mean=str(round(obs_mean,2))


f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
ds=xr.open_dataset(f)
df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))

for i in np.arange(0,no_locations):
    f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
    ds=xr.open_dataset(f)
    spec=ds[nasa_emission].data
    lats=ds['lat'].data
    lons=ds['lon'].data
    model_lat=np.argmin(np.abs(latitude[i]-lats))
    model_lon=np.argmin(np.abs(longitude[i]-lons))
    df_model[location[i]]=ds[nasa_emission].data[:,0,model_lat, model_lon]
    df_model[location[i]]=df_model[location[i]]*conv

if env_type == ' ':
    env_type='AURN'

df_model['median']=df_model.median(axis=1)
mod_mean=df_model['median'].mean(axis=0)
mod_mean=str(round(mod_mean,2))

txt=(' mod mean='+mod_mean+' ug/m3 \n obs mean='+obs_mean+' ug/m3')
plt.annotate(txt,fontsize=8,xy=(0.35,0.85),xycoords='axes fraction')

df_model.drop(df_model.tail(1).index,inplace=True) #this is to make the length of both mod and obs equal
rmse_df=pd.DataFrame(index=range(len(df_model)))
rmse_df['model']=df_model['median']
rmse_df['obs']=df['median']

def rmse(predictions, targets):
   return np.sqrt(((predictions-targets)**2).mean())
rmse_val=rmse(rmse_df['model'],rmse_df['obs'])
rmse_txt=str(round(rmse_val,2))
text=' RMSE = '+rmse_txt+' ug/m3'
#plt.annotate(text, fontsize=7, xy=(0.01, 0.85), xycoords='axes fraction')

plt.plot(df_model.index,df_model['median'],color='green',alpha=0.8,label='Model')
plt.ylabel(Emission+' ug/m3')
plt.xlabel('Time')
plt.legend()
plt.title(env_type)

path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_'+env_type+'_2019.png')
plt.close()



