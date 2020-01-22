import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.odr import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

emission='no2'
Emission='NO2'
conv=1.88*10**9

week='fullweek'

date1='2019-09-22'
date2='2019-11-05'


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
environment=metadata['Environment Type']
environments=pd.unique(environment)

plt.xlabel('Observation Median  ug/m3')
plt.ylabel('Forecast Median  ug/m3')
plt.title('NOx '+week)

aurn_list=[]
aurn_Q1_list=[]
aurn_Q3_list=[]

obs_BU_list=[]
obs_BU_Q1_list=[]
obs_BU_Q3_list=[]
obs_TU_list=[]
obs_TU_Q1_list=[]
obs_TU_Q3_list=[]
obs_BR_list=[]
obs_BR_Q1_list=[]
obs_BR_Q3_list=[]
obs_IU_list=[]
obs_IU_Q1_list=[]
obs_IU_Q3_list=[]
obs_BS_list=[]
obs_BS_Q1_list=[]
obs_BS_Q3_list=[]
obs_IS_list=[]
obs_IS_Q1_list=[]
obs_IS_Q3_list=[]


mod_list=[]
mod_Q1_list=[]
mod_Q3_list=[]

mod_BU_list=[]
mod_BU_Q1_list=[]
mod_BU_Q3_list=[]
mod_TU_list=[]
mod_TU_Q1_list=[]
mod_TU_Q3_list=[]
mod_BR_list=[]
mod_BR_Q1_list=[]
mod_BR_Q3_list=[]
mod_IU_list=[]
mod_IU_Q1_list=[]
mod_IU_Q3_list=[]
mod_BS_list=[]
mod_BS_Q1_list=[]
mod_BS_Q3_list=[]
mod_IS_list=[]
mod_IS_Q1_list=[]
mod_IS_Q3_list=[]

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
    a=list(metadata['Site Name'])
  
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

    no_defra_csv='/users/mtj507/scratch/defra_data/defra_no_uk_2019.csv'
    noddf=pd.read_csv(no_defra_csv, low_memory=False)
    noddf.index=pd.to_datetime(noddf['Date'], dayfirst=True)+pd.to_timedelta(noddf['Time'])
    noddf=noddf.loc[:, ~noddf.columns.str.contains('^Unnamed')]
    noddf=noddf.dropna(axis=0)
    noddf=noddf.replace('No data', np.nan)
    e=list(noddf.columns)
    f=set(d).intersection(e)
    noddf=noddf[noddf.columns.intersection(f)]
    noddf['hour']=noddf.index.hour
    noddf['weekday']=noddf.index.weekday
    noddf['month']=noddf.index.month.astype(str)
    noddf['month']=noddf['month'].str.zfill(2)
    noddf['day']=noddf.index.day.astype(str)
    noddf['day']=noddf['day'].str.zfill(2)
    noddf['day and month']=noddf['month']+noddf['day']

    noddf=noddf[date1:date2]
    noddf=noddf.loc[(noddf['weekday'] >= day1) & (noddf['weekday'] <= day2)]
 
    metadata=metadata[metadata['Site Name'].isin(f)]
    metadata=metadata.reset_index(drop=False)
    area=metadata['Zone']
    location=metadata['Site Name']
    latitude=metadata['Latitude']
    longitude=metadata['Longitude']
    no_locations=len(metadata.index)

    for i in np.arange(0, no_locations):
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

        days_of_data=len(pd.unique(ddf['day and month']))
        dates=pd.unique(ddf['day and month'])
        mod_data = np.zeros((24,days_of_data))

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

        mod_list.append(nasa_median)
        mod_Q1_list.append(nasa_Q1)
        mod_Q3_list.append(nasa_Q3) 

        aurn_list.append(obs_median)
        aurn_Q1_list.append(obs_Q1)
        aurn_Q3_list.append(obs_Q3)

        if environment == 'Background Urban':
            obs_BU_list.append(obs_median)
            obs_BU_Q1_list.append(obs_Q1)
            obs_BU_Q3_list.append(obs_Q3)
            mod_BU_list.append(nasa_median)
            mod_BU_Q1_list.append(nasa_Q1)
            mod_BU_Q3_list.append(nasa_Q3)

        if environment == 'Background Rural':
            obs_BR_list.append(obs_median)
            obs_BR_Q1_list.append(obs_Q1)
            obs_BR_Q3_list.append(obs_Q3)
            mod_BR_list.append(nasa_median)
            mod_BR_Q1_list.append(nasa_Q1)
            mod_BR_Q3_list.append(nasa_Q3)

        if environment == 'Traffic Urban':
            obs_TU_list.append(obs_median)
            obs_TU_Q1_list.append(obs_Q1)
            obs_TU_Q3_list.append(obs_Q3)
            mod_TU_list.append(nasa_median)
            mod_TU_Q1_list.append(nasa_Q1)
            mod_TU_Q3_list.append(nasa_Q3)

        if environment == 'Industrial Urban':
            obs_IU_list.append(obs_median)
            obs_IU_Q1_list.append(obs_Q1)
            obs_IU_Q3_list.append(obs_Q3)
            mod_IU_list.append(nasa_median)
            mod_IU_Q1_list.append(nasa_Q1)
            mod_IU_Q3_list.append(nasa_Q3)

        if environment == 'Background Suburban':
            obs_BS_list.append(obs_median)
            obs_BS_Q1_list.append(obs_Q1)
            obs_BS_Q3_list.append(obs_Q3)
            mod_BS_list.append(nasa_median)
            mod_BS_Q1_list.append(nasa_Q1)
            mod_BS_Q3_list.append(nasa_Q3)

        if environment == 'Industrial Suburban':
            obs_IS_list.append(obs_median)
            obs_IS_Q1_list.append(obs_Q1)
            obs_IS_Q3_list.append(obs_Q3)
            mod_IS_list.append(nasa_median)
            mod_IS_Q1_list.append(nasa_Q1)
            mod_IS_Q3_list.append(nasa_Q3)

    


def linear_func(p, x):
    y=p*x
    return y


aurn_data={'obs':aurn_list,'obs Q1':aurn_Q1_list,'obs Q3':aurn_Q3_list,'mod':mod_list,'mod Q1':mod_Q1_list,'mod Q3':mod_Q3_list}
aurn_df=pd.DataFrame(aurn_data)
aurn_df=aurn_df[aurn_df > 0].dropna()
aurn_df=aurn_df.dropna()
aurn_df['obs_err']=aurn_df['obs Q3']-aurn_df['obs Q1']
aurn_df['mod_err']=aurn_df['mod Q3']-aurn_df['mod Q1']
aurn_df=aurn_df.reset_index(drop=True)

bu_data={'obs':obs_BU_list,'obs Q1':obs_BU_Q1_list,'obs Q3':obs_BU_Q3_list,'mod':mod_BU_list,'mod Q1':mod_BU_Q1_list,'mod Q3':mod_BU_Q3_list}
bu_df=pd.DataFrame(bu_data)
bu_df=bu_df[bu_df > 0].dropna()
bu_df=bu_df.dropna()
bu_df['obs_err']=bu_df['obs Q3']-bu_df['obs Q1']
bu_df['mod_err']=bu_df['mod Q3']-bu_df['mod Q1']
bu_df=bu_df.reset_index(drop=True)

br_data={'obs':obs_BR_list,'obs Q1':obs_BR_Q1_list,'obs Q3':obs_BR_Q3_list,'mod':mod_BR_list,'mod Q1':mod_BR_Q1_list,'mod Q3':mod_BR_Q3_list}
br_df=pd.DataFrame(br_data)
br_df=br_df[br_df > 0].dropna()
br_df=br_df.dropna()
br_df['obs_err']=br_df['obs Q3']-br_df['obs Q1']
br_df['mod_err']=br_df['mod Q3']-br_df['mod Q1']
br_df=br_df.reset_index(drop=True)

tu_data={'obs':obs_TU_list,'obs Q1':obs_TU_Q1_list,'obs Q3':obs_TU_Q3_list,'mod':mod_TU_list,'mod Q1':mod_TU_Q1_list,'mod Q3':mod_TU_Q3_list}
tu_df=pd.DataFrame(tu_data)
tu_df=tu_df[tu_df > 0].dropna()
tu_df=tu_df.dropna()
tu_df['obs_err']=tu_df['obs Q3']-tu_df['obs Q1']
tu_df['mod_err']=tu_df['mod Q3']-tu_df['mod Q1']
tu_df=tu_df.reset_index(drop=True)

iu_data={'obs':obs_IU_list,'obs Q1':obs_IU_Q1_list,'obs Q3':obs_IU_Q3_list,'mod':mod_IU_list,'mod Q1':mod_IU_Q1_list,'mod Q3':mod_IU_Q3_list}
iu_df=pd.DataFrame(iu_data)
iu_df=iu_df[iu_df > 0].dropna()
iu_df=iu_df.dropna()
iu_df['obs_err']=iu_df['obs Q3']-iu_df['obs Q1']
iu_df['mod_err']=iu_df['mod Q3']-iu_df['mod Q1']
iu_df=iu_df.reset_index(drop=True)

bs_data={'obs':obs_BS_list,'obs Q1':obs_BS_Q1_list,'obs Q3':obs_BS_Q3_list,'mod':mod_BS_list,'mod Q1':mod_BS_Q1_list,'mod Q3':mod_BS_Q3_list}
bs_df=pd.DataFrame(bs_data)
bs_df=bs_df[bs_df > 0].dropna()
bs_df=bs_df.dropna()
bs_df['obs_err']=bs_df['obs Q3']-bs_df['obs Q1']
bs_df['mod_err']=bs_df['mod Q3']-bs_df['mod Q1']
bs_df=bs_df.reset_index(drop=True)

is_data={'obs':obs_IS_list,'obs Q1':obs_IS_Q1_list,'obs Q3':obs_IS_Q3_list,'mod':mod_IS_list,'mod Q1':mod_IS_Q1_list,'mod Q3':mod_IS_Q3_list}
is_df=pd.DataFrame(is_data)
is_df=is_df[is_df > 0].dropna()
is_df=is_df.dropna()
is_df['obs_err']=is_df['obs Q3']-is_df['obs Q1']
is_df['mod_err']=is_df['mod Q3']-is_df['mod Q1']
is_df=is_df.reset_index(drop=True)


envs=['aurn','bu','br','tu','iu','bs','is']

for x in envs:
    if x == 'aurn':
      sdf=aurn_df
      lbl='AURN'
      clr='black'
      obslist=aurn_list
      modlist=mod_list
    if x == 'bu':
      sdf=bu_df
      lbl='Background Urban'
      clr='red'
      obslist=obs_BU_list
      modlist=mod_BU_list
    if x == 'br':
      sdf=br_df
      lbl='Background Rural'
      clr='blue'
      obslist=obs_BR_list
      modlist=mod_BR_list
    if x == 'tu':
      sdf=tu_df
      lbl='Traffic Urban'
      clr='green'
      obslist=obs_TU_list
      modlist=mod_TU_list
    if x == 'iu':
      sdf=iu_df
      lbl='Industrial Urban'
      clr='grey'
      obslist=obs_IU_list
      modlist=mod_IU_list
    if x == 'bs':
      sdf=bs_df
      lbl='Background Suburban'
      clr='brown'
      obslist=obs_BS_list
      modlist=mod_BS_list
    if x == 'is':
      sdf=is_df
      lbl='Industrial Suburban'
      clr='orange'
      obslist=obs_IS_list
      modlist=mod_IS_list

    if sdf.empty:
      continue
    else:
      x_data=sdf['obs']
      y_data=sdf['mod']
      x_err=sdf['obs_err']
      y_err=sdf['mod_err']

      linear=Model(linear_func)
      datas=RealData(x_data,y_data,sx=x_err,sy=y_err)
      odr=ODR(datas,linear,beta0=[0])
      output=odr.run()
      #output.pprint()
      beta=output.beta
      betastd=output.sd_beta

      if len(sdf) > 4:
        plt.plot(x_data,linear_func(beta,x_data),alpha=1,color=clr)

      textbeta=float(output.beta)
      textbeta=str(round(textbeta,3))
      textbetastd=float(output.sd_beta)
      textbetastd=str(round(textbetastd,3))
      sites=str(len(x_data))
      print(x) 
      print('beta = '+textbeta)
      print('std = '+textbetastd)
      print('no of sites = '+sites)

      plt.scatter(obslist,modlist,color=clr,label=lbl,marker='x',alpha=0.7) 
      plt.legend(fontsize='small')

xy=np.linspace(*plt.xlim())
plt.plot(xy,xy,linestyle='dashed',color='grey')

path='/users/mtj507/scratch/obs_vs_forecast/plots/scatter/nox_'
plt.savefig(path+week+'.png')
plt.close()










