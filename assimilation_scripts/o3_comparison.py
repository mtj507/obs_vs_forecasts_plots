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
conv=2*10**9

environments=['Mace Head','Background Urban','Background Rural']

months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

fig=plt.figure()
fig,axes=plt.subplots(3,4,figsize=[10,10],sharex=True,sharey=True)

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'

for i in range(0,len(environments)):
    env=environments[i]
    metadata=pd.read_csv(metadata_csv, low_memory=False)
    if i == 0:
        metadata=metadata.loc[metadata['Site Name']==env]
    if i == 1 or i == 2:
        metadata=metadata.loc[metadata['Environment Type'].str.contains(env)]
    metadata=metadata.reset_index(drop=True)
    a=metadata['Site Name']

    defra_csv='/users/mtj507/scratch/defra_data/'+emission+'_2019.csv'
    ddf=pd.read_csv(defra_csv, low_memory=False)
    ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
    ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
    ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
    ddf=ddf.dropna(axis=0)
    ddf=ddf.replace('No data', np.nan)
    ddf.drop(ddf.tail(1).index,inplace=True)

    b=ddf.columns
    headers=set(a).intersection(b)

    metadata=metadata.loc[metadata['Site Name'].isin(headers)]
    metadata=metadata.reset_index(drop=True)
    location=metadata['Site Name']
    longitude=metadata['Longitude']
    latitude=metadata['Latitude']
    no_locations=len(metadata.index)

    ddf=ddf.loc[:,headers]
    ddf['month']=ddf.index.month
    ddf['hour']=ddf.index.hour
    ddf=ddf.astype(float)

    if env == 'Mace Head':
        c='blue'
        c1='aqua'
    if env == 'Background Urban':
        c='darkred'
        c1='red'
    if env == 'Background Rural':
        c='green'
        c1='limegreen'

    for m,ax in zip(np.arange(len(months)),axes.flatten()):
        ddf1=ddf.loc[ddf['month']==m+1]
        ddf1=ddf1.drop(columns=['month'])
        df=pd.DataFrame(index=range(0,24))
        ddf_med=ddf1.groupby('hour').median()
        ddf_med['median']=ddf_med.mean(axis=1)
        ddf_q1=ddf1.groupby('hour').quantile(0.25)
        ddf_q1['q1']=ddf_q1.mean(axis=1)
        ddf_q3=ddf1.groupby('hour').quantile(0.75)
        ddf_q3['q3']=ddf_q3.mean(axis=1)
 
        ax.plot(ddf_med.index,ddf_med['median'],label=env+' Observation',color=c,linestyle=':')

        f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
        ds=xr.open_dataset(f)
        df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))

        for i in np.arange(0,no_locations):
            lats=ds['lat'].data
            lons=ds['lon'].data
            model_lat=np.argmin(np.abs(latitude[i]-lats))
            model_lon=np.argmin(np.abs(longitude[i]-lons))
            df_model[location[i]]=ds[emission].data[:,0,model_lat, model_lon]
            df_model[location[i]]=df_model[location[i]]*conv

        df_model['month']=df_model.index.month
        df_model['hour']=df_model.index.hour
        df_model=df_model.loc[df_model['month']==m+1]

        if env=='Mace Head':
            df_median=df_model.groupby('hour').median()
            df_Q1=df_model.groupby('hour').quantile(0.25)
            df_Q3=df_model.groupby('hour').quantile(0.75)
            ax.plot(df_median.index,df_median['Mace Head'],label=env+' Model',color=c)
            ax.fill_between(df_median.index,df_Q1['Mace Head'],df_Q3['Mace Head'],alpha=0.5,facecolor=c1,edgecolor=c1)

 
        if env=='Background Urban' or env=='Background Rural':
            df_median=df_model.groupby('hour').median()
            df_median['median']=df_median.mean(axis=1)
            df_Q1=df_model.groupby('hour').quantile(0.25)
            df_Q1['Q1']=df_Q1.mean(axis=1)
            df_Q3=df_model.groupby('hour').quantile(0.75)
            df_Q3['Q3']=df_Q3.mean(axis=1)
            ax.plot(df_median.index,df_median['median'],label=env+' Model',color=c)
            ax.fill_between(df_median.index,df_Q1['Q1'],df_Q3['Q3'],alpha=0.5,facecolor=c1,edgecolor=c1)

        ax.set_title(months[m],fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')

        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='best')

fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Hour of Day')
plt.ylabel(r'Ozone Concentration ($\mu g\:  m^{-3}$)')
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_macehead_env_comparison_diurnal.png')
plt.close()




