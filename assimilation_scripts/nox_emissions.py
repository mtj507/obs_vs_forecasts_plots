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

nox_csv='/users/mtj507/scratch/defra_data/nox_emissions.csv'
df=pd.read_csv(nox_csv)

def inter_func(B,x):
    y=B[0]*x+B[1]
    return y

x=df['Year']
y=df['NOx Emissions']
linear=Model(inter_func)
datas=RealData(x,y)
odr=ODR(datas,linear,beta0=[-50,3200])
output=odr.run()
#output.pprint()
beta=(output.beta)
beta_txt=str(round(beta[0],2))
betastd=(output.sd_beta)

#plt.plot(x,inter_func(beta,x),color='grey',alpha=0.8)

sns.scatterplot(x=x,y=y)
plt.scatter(x=2010,y=1233,color='red')

x=np.linspace(1965,2025,500)
y=beta[1]+beta[0]*x
plt.plot(x,y,color='black',alpha=0.7)
plt.axvline(x=2019,color='grey',linestyle=':')
plt.axhline(y=808.78,color='grey',linestyle=':')
plt.ylabel('NOx Emissions (kilotonnes)')

path='/users/mtj507/scratch//obs_vs_forecast/assimilation_scripts/plots/whole_year/o3/'
plt.savefig(path+'nox_emission_inventory.png')












