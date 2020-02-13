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

emission='Ozone'
env_type='Background Urban'

if env_type == 'AURN':
    env_type=' '


def linear_func(p, x):
    y=p*x
    return y

metadata_csv='/users/mtj507/scratch/defra_data/defra_site_metadata.csv'
metadata=pd.read_csv(metadata_csv, low_memory=False)
metadata=metadata.loc[metadata['AURN Pollutants Measured'].str.contains(emission)]
metadata=metadata.loc[metadata['Environment Type'].str.contains(env_type)]
metadata=metadata.reset_index(drop=True)

if env_type == ' ':
    env_type='AURN'

metric_csv='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/ozone_site_metrics_'+env_type+'.csv'
metrics=pd.read_csv(metric_csv,low_memory=False)
metrics.drop(metrics.columns[0],axis=1,inplace=True)
metrics['ratio']=metrics['mod median']/metrics['obs median']

df=pd.merge(metadata,metrics,left_on='Site Name',right_on='site name',how='inner')

nf_csv='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/ozone_nf_site_metrics_'+env_type+'.csv'
nf=pd.read_csv(nf_csv,low_memory=False)
nf.drop(nf.columns[0],axis=1,inplace=True)

df=pd.merge(df,nf,left_on='Site Name',right_on='site name',how='inner')
location=df['Site Name']
longitude=df['Longitude']
latitude=df['Latitude']
no_locations=len(df.index)



f='/users/mtj507/scratch/nasa_assimilations/2019_assimilation.nc'
ds=xr.open_dataset(f)
df_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))
nodf_model=pd.DataFrame(index=pd.to_datetime(ds.time.data))

#not all ozone sites record nox, so use model for nox values
for i in range(0,no_locations):
    lats=ds['lat'].data
    lons=ds['lon'].data
    model_lat=np.argmin(np.abs(latitude[i]-lats))
    model_lon=np.argmin(np.abs(longitude[i]-lons))
    df_model[location[i]]=ds['no2'].data[:,0,model_lat, model_lon]
    nodf_model[location[i]]=ds['no'].data[:,0,model_lat, model_lon]
    df_model[location[i]]=df_model[location[i]]*1.88*10**9
    nodf_model[location[i]]=nodf_model[location[i]]*1.23*10**9
        
mdf=pd.concat([df_model,nodf_model],axis=1)
mdf=mdf.astype(float)
mdf=mdf.groupby(mdf.columns,axis=1).sum()
mdf=mdf.median(axis=0)
mdf=pd.DataFrame({'Site Name':mdf.index,'nox median':mdf.values})

ddf=pd.merge(df,mdf,on='Site Name',how='inner')

x=ddf['nox median']
y=ddf['RMSE']

sns.scatterplot(x,y,data=ddf)

plt.title(env_type)
plt.xlabel(r'Model NOx median ($\mu g\:  m^{-3}$)')
plt.ylabel(r'Ozone RMSE ($\mu g\:  m^{-3}$)')
plt.xlim(0,35)
plt.ylim(0,35)

linear=Model(linear_func)
datas=RealData(x,y)
odr=ODR(datas,linear,beta0=[0])
output=odr.run()
output.pprint()
beta=(output.beta)
betastd=(output.sd_beta)

beta_val=float(beta[0])
beta_txt=str(round(beta_val,2))

#plt.plot(x,linear_func(beta,x),color='grey',alpha=0.7)

path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/'
plt.savefig(path+emission+'_vs_nox_scatter_'+env_type+'.png')
plt.close()













