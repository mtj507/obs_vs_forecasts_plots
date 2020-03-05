import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

emission='o3'

sites=['Lough Navar','Narberth','Mace Head']#/'Yarner Wood','Aston Hill']

months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
macedata=metadata.loc[metadata['Site Name']=='Mace Head']
macedata=macedata.reset_index(drop=True)
mhlat=macedata['Latitude'].values
mhlon=macedata['Longitude'].values

defra_csv='/users/mtj507/scratch/defra_data/'+emission+'_2019.csv'
ddf=pd.read_csv(defra_csv, low_memory=False)
ddf.loc[ddf['Time'] == '00:00:00','Time']='24:00:00'
ddf.index=pd.to_datetime(ddf['Date'], dayfirst=True)+pd.to_timedelta(ddf['Time'])
ddf=ddf.loc[:, ~ddf.columns.str.contains('^Unnamed')]
ddf=ddf.dropna(axis=0)
ddf=ddf.replace('No data', np.nan)
ddf['hour']=ddf.index.hour

mh=ddf.loc[:,['Mace Head']]
mh['hour']=mh.index.hour
mh['month']=mh.index.month
mh=mh.astype(float)

f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
ds=xr.open_dataset(f)

mh_mod=pd.DataFrame(index=pd.to_datetime(ds.time.data))
lats=ds['lat'].data
lons=ds['lon'].data
model_lat=np.argmin(np.abs(mhlat[0]-lats))
model_lon=np.argmin(np.abs(mhlon[0]-lons))
mh_mod['model']=ds[emission].data[:,0,model_lat, model_lon]
mh_mod['model']=mh_mod['model']*2*10**9
mh_mod['hour']=mh_mod.index.hour
mh_mod['month']=mh_mod.index.month

metadata=metadata.loc[metadata['Site Name'].isin(sites)]
metadata=metadata.reset_index(drop=True)
location=metadata['Site Name']
latitude=metadata['Latitude']
longitude=metadata['Longitude']
no_locations=len(metadata.index)

fig=plt.figure()
fig,axes=plt.subplots(3,4,figsize=[8,8],sharex=True,sharey=True)

for i in range(0,no_locations):
    site=location[i]   
    ddf1=ddf.loc[:,[site]]
    ddf1=ddf1.replace(0,np.nan)
    ddf1=ddf1.dropna(axis=0)
    ddf1['hour']=ddf1.index.hour
    ddf1['month']=ddf1.index.month
    ddf1=ddf1.astype(float)
    
    for m,ax in zip(np.arange(len(months)),axes.flatten()):
        month=months[m]

        mh1=mh.loc[mh['month']==m+1]
        mh1=mh1.drop(columns=['month'])
        mhf=pd.DataFrame(index=range(0,24))
        mhf['obs med']=mh1.groupby('hour').median()
        mhf['obs q1']=mh1.groupby('hour').quantile(0.25)
        mhf['obs q3']=mh1.groupby('hour').quantile(0.75)

        mh_mod1=mh_mod.loc[mh_mod['month']==m+1]
        mh_mod1=mh_mod1.drop(columns=['month'])
        mhf['mod med']=mh_mod1.groupby('hour').median()
        mhf['mod q1']=mh_mod1.groupby('hour').quantile(0.25)
        mhf['mod q3']=mh_mod1.groupby('hour').quantile(0.75)

        ddf2=ddf1.loc[ddf1['month']==m+1]
        ddf2=ddf2.drop(columns=['month'])
 
        ldf=pd.DataFrame(index=range(0,24))
        ldf['med']=ddf2.groupby('hour').median()
        ldf['q1']=ddf2.groupby('hour').quantile(0.25)
        ldf['q3']=ddf2.groupby('hour').quantile(0.75)

        df=pd.DataFrame(index=range(0,24))
        df['obs med']=ldf['med']-mhf['obs med']
        df['obs q1']=ldf['q1']-mhf['obs q1']
        df['obs q3']=ldf['q3']-mhf['obs q3']

        df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
        lats=ds['lat'].data
        lons=ds['lon'].data
        model_lat=np.argmin(np.abs(latitude[i]-lats))
        model_lon=np.argmin(np.abs(longitude[i]-lons))
        df_model['model']=ds[emission].data[:,0,model_lat, model_lon]
        df_model['model']=df_model['model']*2*10**9
        df_model.drop(df_model.tail(1).index,inplace=True)
        df_model['hour']=df_model.index.hour
        df_model['month']=df_model.index.month
        df_model=df_model.loc[df_model['month']==m+1]
        df_model=df_model.drop(columns=['month'])
        df_model=df_model.astype(float)

        mdf=pd.DataFrame()
        mdf=df_model.groupby('hour').median()
        mdf['q1']=df_model.groupby('hour').quantile(0.25)
        mdf['q3']=df_model.groupby('hour').quantile(0.75)

        df['mod med']=mdf['model']-mhf['mod med']
        df['mod q1']=mdf['q1']-mhf['mod q1']
        df['mod q3']=mdf['q3']-mhf['mod q3']

        if site == 'Lough Navar':
            c='red'
        if site == 'Narberth':
            c='green'
        if site == 'Mace Head':
            c='blue'

        ax.plot(df.index,df['obs med'],linestyle=':',label=site+' observation',color=c)
        ax.plot(df.index,df['mod med'],label=site+' model',color=c)
        ax.set_title(month)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')

fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time of Day (hour)')
plt.ylabel(r'Site Ozone Concentration - Mace Head Ozone Concentration  ($\mu g\:  m^{-3}$)')

fig.tight_layout()
path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_rural_minus_mace_head.png')
plt.close()




