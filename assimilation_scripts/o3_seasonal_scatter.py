import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
from scipy.odr import *

emission='o3'
#env_type='AURN'
sites_to_test=['Narberth','Lough Navar','Rochester Stoke','Chilbolton Observatory']

seasons=['Winter','Spring','Summer','Autumn']
no_szns=len(seasons)

week='fullweek'

date1='2019-01-01'
date2='2019-12-31'

nrow=4
ncol=4
fsize=[10,10]

if emission == 'o3':
    conv=2*10**9
    nasa_emission='o3'
    Emission=r'$O_3$'

fig=plt.figure()
fig,axes=plt.subplots(ncols=ncol,nrows=nrow,figsize=fsize)

def inter_func(B,x):
    y=B[0]*x+B[1]
    return y


def rmse(predictions, targets):
    return np.sqrt(((predictions-targets)**2).mean())


metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
#metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
metadata=metadata.loc[metadata['Site Name'].isin(sites_to_test)]
metadata=metadata.reset_index(drop=True)
environment=metadata['Environment Type']
a=metadata['Site Name']

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

b=ddf.columns
locations=set(a).intersection(b)
location_list=list(locations)

metadata=metadata.loc[metadata['Site Name'].isin(location_list)]
metadata=metadata.reset_index(drop=True)
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
no_locations=len(metadata.index)


for i,ax in zip(range(0,no_locations),axes.flatten()):
    site=location[i]
    ddf1=ddf.loc[:,site]
    ddf1=ddf1.replace(0,np.nan)
    ddf1=ddf1.dropna(axis=0)
    ddf1=ddf1.astype(float)

    for y,ax in zip(range(no_szns),axes.flatten()[(i*4):(i*4)+4]):
        season=seasons[y]

        if season == 'Winter':
            date1='2019-01-01'
            date2='2019-03-19'
        if season == 'Spring':
            date1='2019-03-20'
            date2='2019-06-20'
        if season == 'Summer':
            date1='2019-06-21'
            date2='2019-09-22'
        if season == 'Autumn':
            date1='2019-09-23'
            date2='2019-12-20'
        if season == '2019':
            date1='2019-01-01'
            date2='2019-12-31'

        df=ddf1.loc[date1:date2]
        df=df.astype(float)
 
        f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
        ds=xr.open_dataset(f)

        df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
        df_model.index=df_model.index+pd.Timedelta('1 min')
        spec=ds[nasa_emission].data
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model['model']=ds[nasa_emission].data[:,0,model_lat, model_lon]
        df_model['model']=df_model['model']*conv
        df_model.drop(df_model.tail(1).index,inplace=True)
        df_model.index=df_model.index.round('H')
        df_model=df_model.loc[date1:date2]
        df_model=df_model.astype(float)

        sdf=pd.merge(df,df_model,right_index=True,left_index=True)
        sdf=sdf.dropna(axis=0)

        x_data=sdf[location[i]]
        y_data=sdf['model']

        sns.scatterplot(x=x_data,y=y_data,ax=ax,s=6) 

        linear=Model(inter_func)
        datas=RealData(x_data,y_data)
        odr=ODR(datas,linear,beta0=[1,0])
        output=odr.run()
  #     output.pprint()
        beta=(output.beta)
        beta_txt=str(round(beta[0],2))
        betastd=(output.sd_beta)

        ax.plot(x_data,inter_func(beta,x_data),color='black',alpha=0.8)

        xy=np.linspace(*ax.get_xlim())
        ax.plot(xy,xy,linestyle='dashed',color='grey')




        ax.set_title(site+' '+season,fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')


fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'Observation Ozone Concentration ($\mu g\:  m^{-3}$)')
plt.ylabel(r'Model Ozone Concentration ($\mu g\:  m^{-3}$)') 
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_site_test_season_scatter.png')
plt.close()


























