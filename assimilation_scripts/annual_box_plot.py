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
import seaborn as sns

fig=plt.figure()
fig,ax=plt.subplots(2,2,figsize=[6,12])
palette=sns.color_palette(['green','grey'])

emission='no'
print(emission)
env_list=['AURN','Background Urban','Background Rural','Traffic Urban','Industrial urban','Industrial Suburban','background Suburban']
env_no=4
 
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

for e in range(env_no):
    env_type=env_list[e]
    if env_type == 'AURN':
        env_type=' '
    print(env_type)

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

    b=ddf.columns
    headers=set(a).intersection(b)

    ddf=ddf.loc[:,headers]
    ddf=ddf.astype(float)
    df=pd.DataFrame(index=ddf.index,columns=['median'])
    df['median']=ddf.median(axis=1)

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
    df_model.drop(df_model.tail(1).index,inplace=True) #this is to make the length of both mod and obs equal
    rmse_df=pd.DataFrame(index=range(len(df_model)))
    rmse_df['model']=df_model['median'].values
    rmse_df['obs']=df['median'].values
    df_box=pd.melt(rmse_df)

    if e == 0 :
        sns.boxplot(y=df_box['value'],x=df_box['variable'],data=df_box,orient='v',ax=ax[0,0],palette=palette)
    if e == 1:
        sns.boxplot(y=df_box['value'],x=df_box['variable'],data=df_box,orient='v',ax=ax[0,1],palette=palette)
    if e == 2 :
        sns.boxplot(y=df_box['value'],x=df_box['variable'],data=df_box,orient='v',ax=ax[1,0],palette=palette)
    if e == 3:
        sns.boxplot(y=df_box['value'],x=df_box['variable'],data=df_box,orient='v',ax=ax[1,1],palette=palette)

#    ax.ravel()[e].boxplot(rmse_df['model'])
#    ax.ravel()[e].boxplot(rmse_df['obs'])

    if e == 0 or e == 2:
        ax.ravel()[e].set_ylabel(Emission+' ug/m3')
    if e == 1 or e == 3:
        ax.ravel()[e].set_ylabel('')
    ax.ravel()[e].set_title(env_type)
    ax.ravel()[e].set_xlabel('')

path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/'+emission+'/'
plt.savefig(path+emission+'_boxplot_2019.png')
plt.close()


