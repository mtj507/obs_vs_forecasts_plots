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

environments=['Mace Head','Background Urban','Background Rural']

months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

fig=plt.figure()
fig,axes=plt.subplots(3,4,figsize=[8,8],sharex=True,sharey=True)

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

    defra_csv='/users/mtj507/scratch/defra_data/no2_2019.csv'
    ddf=pd.read_csv(defra_csv, low_memory=False)
    ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
    ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
    ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
    ddf=ddf.dropna(axis=0)
    ddf=ddf.replace('No data', np.nan)
    ddf.drop(ddf.tail(1).index,inplace=True)

    oz_defra_csv='/users/mtj507/scratch/defra_data/o3_2019.csv'
    ozddf=pd.read_csv(oz_defra_csv, low_memory=False)
    ozddf.loc[ozddf['Time'] == '00:00:00','Time']='24:00:00'
    ozddf.index=pd.to_datetime(ozddf['Date'], dayfirst=True)+pd.to_timedelta(ozddf['Time'])
    ozddf=ozddf.loc[:, ~ozddf.columns.str.contains('^Unnamed')]
    ozddf=ozddf.dropna(axis=0)
    ozddf=ozddf.replace('No data', np.nan)
    ozddf.drop(ozddf.tail(1).index,inplace=True)
    
    if i==0:
        headers=['Mace Head']

    if i ==1 or i==2:
        b=ddf.columns
        c=set(a).intersection(b)
        d=ozddf.columns
        headers=set(c).intersection(d)

    metadata=metadata.loc[metadata['Site Name'].isin(headers)]
    metadata=metadata.reset_index(drop=True)
    location=metadata['Site Name']
    longitude=metadata['Longitude']
    latitude=metadata['Latitude']
    no_locations=len(metadata.index)    

    ddf1=pd.concat([ddf,ozddf],axis=1)
    ddf1=ddf1.loc[:,headers]
    ddf1=ddf1.astype(float)
    if i == 0:
        ddf1=ddf1.groupby(ddf1.columns,axis=1).sum()
    if i == 1 or i == 2:
        ddf1=ddf1.groupby(ddf1.columns,axis=1).sum(min_count=2)
    ddf1['hour']=ddf1.index.hour
    ddf1['month']=ddf1.index.month
    ddf1=ddf1.astype(float)

    if env == 'Mace Head':
        c='blue'
        c1='aqua'
    if env == 'Background Urban':
        c='darkred'
        c1='red'
    if env == 'Background Rural':
        c='green'
        c1='limegreen'
    print(env)
    for m,ax in zip(np.arange(len(months)),axes.flatten()):
        ddf2=ddf1.loc[ddf1['month']==m+1]
        ddf2=ddf2.drop(columns=['month'])
        df=pd.DataFrame(index=range(0,24))
        ddf_med=ddf2.groupby('hour').median()
        ddf_med['median']=ddf_med.mean(axis=1)
        ddf_q1=ddf2.groupby('hour').quantile(0.25)
        ddf_q1['q1']=ddf_q1.mean(axis=1)
        ddf_q3=ddf2.groupby('hour').quantile(0.75)
        ddf_q3['q3']=ddf_q3.mean(axis=1)

        f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
        ds=xr.open_dataset(f)
        df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
        ozdf_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))

        for i in np.arange(0,no_locations):
            lats=ds['lat'].data
            lons=ds['lon'].data
            model_lat=np.argmin(np.abs(latitude[i]-lats))
            model_lon=np.argmin(np.abs(longitude[i]-lons))
            df_model[location[i]]=ds['no2'].data[:,0,model_lat, model_lon]
            ozdf_model[location[i]]=ds['o3'].data[:,0,model_lat, model_lon]
            df_model[location[i]]=df_model[location[i]]*1.88*10**9
            ozdf_model[location[i]]=ozdf_model[location[i]]*2*10**9


        df_model=pd.concat([df_model,ozdf_model],axis=1)
        df_model=df_model.astype(float)
        df_model=df_model.groupby(df_model.columns,axis=1).sum()
        df_model['month']=df_model.index.month
        df_model['hour']=df_model.index.hour
        df_model=df_model.loc[df_model['month']==m+1]

        if env=='Mace Head':
            df_median=df_model.groupby('hour').median()
            df_Q1=df_model.groupby('hour').quantile(0.25)
            df_Q3=df_model.groupby('hour').quantile(0.75)
            bdf=pd.DataFrame(index=ddf_med.index)
            bdf['median']=df_median['Mace Head']-ddf_med['Mace Head']
            ax.plot(bdf.index,bdf['median'],label=env+' Model',color=c)

 
        if env=='Background Urban' or env=='Background Rural':
            df_median=df_model.groupby('hour').median()
            df_median['median']=df_median.mean(axis=1)
            df_Q1=df_model.groupby('hour').quantile(0.25)
            df_Q1['Q1']=df_Q1.mean(axis=1)
            df_Q3=df_model.groupby('hour').quantile(0.75)
            df_Q3['Q3']=df_Q3.mean(axis=1)
            bdf=pd.DataFrame(index=ddf_med.index)
            bdf['median']=df_median['median']-ddf_med['median']

            ax.plot(bdf.index,bdf['median'],label=env+' Model',color=c)

        ax.set_title(months[m],fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_ylim(-20,30)        

        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Hour of Day')
plt.ylabel(r'Ox Median Model Bias ($\mu g\:  m^{-3}$)')
fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/ox/'
plt.savefig(path+'ox_monthly_diurnal_bias.png')
plt.close()




