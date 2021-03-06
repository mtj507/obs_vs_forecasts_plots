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

#defining emission to be observed and conversion (can be found in conversion file)
emission='no2'
Emission='NO2'
conv = 1.88*10**9

environment_type='Urban'

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

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['Environment Type'].str.contains(environment_type)]
a=list(metadata['Site Name'])

#change to UTF csv before moving across to Viking and edit doc so its easy to import by deleting first 3 rowns and moving time and date column headers into same row as locations. Delete empty rows up to 'end' at bottom and format time cells to time.
no2_defra_csv='/users/mtj507/scratch/defra_data/defra_no2_uk_2019.csv'
ddf=pd.read_csv(no2_defra_csv, low_memory=False)
ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
ddf=ddf.dropna(axis=0)
ddf=ddf.replace('No data', np.nan)
b=list(ddf.columns)
c=set(a).intersection(b)
ddf=ddf[ddf.columns.intersection(c)]
d=list(ddf.columns)
ddf['hour']=ddf.index.hour
ddf['weekday']=ddf.index.weekday
ddf['month']=ddf.index.month.astype(str)
ddf['month']=ddf['month'].str.zfill(2)
ddf['day']=ddf.index.day.astype(str)
ddf['day']=ddf['day'].str.zfill(2)
ddf['day and month']=ddf['month']+ddf['day']

ddf=ddf.loc[date1:date2]
ddf=ddf.loc[(ddf['weekday'] >= day1) & (ddf['weekday'] <= day2)]

oz_defra_csv='/users/mtj507/scratch/defra_data/defra_o3_uk_2019.csv'
ozddf=pd.read_csv(oz_defra_csv, low_memory=False)
ozddf.index=pd.to_datetime(ozddf['Date'], dayfirst=True)+pd.to_timedelta(ozddf['Time'])
ozddf=ozddf.loc[:, ~ozddf.columns.str.contains('^Unnamed')]
ozddf=ozddf.dropna(axis=0)
ozddf=ozddf.replace('No data', np.nan)
e=list(ozddf.columns)
f=set(d).intersection(e)
ozddf=ozddf[ozddf.columns.intersection(f)]
ozddf['hour']=ozddf.index.hour
ozddf['weekday']=ozddf.index.weekday
ozddf['month']=ozddf.index.month.astype(str)
ozddf['month']=ozddf['month'].str.zfill(2)
ozddf['day']=ozddf.index.day.astype(str)
ozddf['day']=ozddf['day'].str.zfill(2)
ozddf['day and month']=ozddf['month']+ozddf['day']

ozddf=ozddf[date1:date2]
ozddf=ozddf.loc[(ozddf['weekday'] >= day1) & (ozddf['weekday'] <= day2)]

metadata=metadata[metadata['Site Name'].isin(f)]
metadata=metadata.reset_index(drop=False)
area=metadata['Zone']
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
environment=metadata['Environment Type']
no_locations=len(metadata.index)

plt.xlabel('Observation Median  ug/m3')
plt.ylabel('Forecast Median  ug/m3')
plt.title('Ox '+environment_type+' '+week)

obs_list=[]
obsQ1_list=[]
obsQ3_list=[]
nasa_list=[]
nasaQ1_list=[]
nasaQ3_list=[]

obslist1=[]
obslist2=[]
obslist3=[]

modlist1=[]
modlist2=[]
modlist3=[]

types=list(pd.unique(environment))

for i in np.arange(0, no_locations):
    ddf1=ddf.loc[:,['hour', f'{location[i]}']]
    ozddf1=ozddf.loc[:,[f'{location[i]}']]
    ddf1['no2']=ddf1[f'{location[i]}'].astype(float)
    ozddf1['o3']=ozddf1[f'{location[i]}'].astype(float)
    ddf1['o3']=ozddf1['o3']
    ddf1['ox']=ddf1['o3']+ddf1['no2']
    ddf1=ddf1.dropna(axis=0)
    ddf_median=ddf1.groupby('hour').median()
    ddf_Q1=ddf1.groupby('hour')['ox'].quantile(0.25)
    ddf_Q3=ddf1.groupby('hour')['ox'].quantile(0.75)
    env=f'{environment[i]}'
    obs_median=ddf_median['ox'].mean()
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
        

    nasa_median=np.median(mod_data)
    nasa_median=float(round(nasa_median,2))
    nasa_Q1=np.percentile(mod_data,25)
    nasa_Q1=float(round(nasa_Q1,2))
    nasa_Q3=np.percentile(mod_data,75)
    nasa_Q3=float(round(nasa_Q3,2))
    
    nasa_list.append(nasa_median)
    nasaQ1_list.append(nasa_Q1)
    nasaQ3_list.append(nasa_Q3)

    if len(types) == 1:
      if env == types[0]:
        obslist1.append(obs_median)
        label1=types[0]
        modlist1.append(nasa_median)

    if len(types) == 2:
      if env == types[0]:
        obslist1.append(obs_median)
        label1=types[0]
        modlist1.append(nasa_median)
      if env == types[1]:
        obslist2.append(obs_median)
        label2=types[1]
        modlist2.append(nasa_median)

    if len(types) >=3:
      if env == types[0]:
        obslist1.append(obs_median)
        label1=types[0]
        modlist1.append(nasa_median)
      if env == types[1]:
        obslist2.append(obs_median)
        label2=types[1]
        modlist2.append(nasa_median)
      if env == types[2]:
        obslist3.append(obs_median)
        label3=types[2]
        modlist3.append(nasa_median)

    print(location[i])

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

def linear_func(p, x):
    y=p*x
    return y

linear=Model(linear_func)
datas=RealData(x_data,y_data,sx=x_err,sy=y_err)
odr=ODR(datas,linear,beta0=[0])
output=odr.run()
#output.pprint()
beta=output.beta
betastd=output.sd_beta

plt.plot(x_data,linear_func(beta,x_data),color='black',alpha=0.7)

textbeta=float(output.beta)
textbeta=str(round(textbeta,3))
textbetastd=float(output.sd_beta)
textbetastd=str(round(textbetastd,3))
sites=str(no_locations)

text=sites+' sites \n'+'Orthogonal Distance Regression: \n Gradient = '+textbeta+'\n Standard error = '+textbetastd

plt.scatter(obslist1,modlist1,color='red',label=label1,marker='o')
if len(types) >= 2:
  plt.scatter(obslist2,modlist2,color='blue',label=label2,marker='x')
if len(types) >= 3:
  plt.scatter(obslist3,modlist3,color='green',label=label3,marker='v')

if len(types) == 1:
  plt.annotate(text,fontsize=7,xy=(0.01,0.85),xycoords='axes fraction')
if len(types) >= 2:
  plt.legend(loc='best')
  plt.annotate(text,fontsize=7,xy=(0.5,0.85),xycoords='axes fraction')

xy=np.linspace(*plt.xlim())
plt.plot(xy,xy,linestyle='dashed',color='grey')

path='/users/mtj507/scratch/obs_vs_forecast/plots/scatter/'
plt.savefig(path+'ox_'+environment_type+'_'+week+'.png')
plt.close()















