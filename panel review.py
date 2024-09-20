import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.graph_objs as go
import plotly.express as px
import more_itertools as mit
import csv
import glob
import os
from datetime import datetime

###############MRA data 10/10/2023############
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\2021-10-12.csv")
fig,axis = plt.subplots(1,1,figsize=(15,6),sharey=True,sharex=False);
axis.plot(df['grade'],'blue', label='MRA data (12/10/2023)')
axis.set_xlabel('time counts',fontsize=18)
axis.set_ylabel('Cu grade (w.t%)',fontsize=18)
axis.tick_params(axis='both', which='major', labelsize=18) 
axis.legend(loc='upper right',fontsize=18)

#######Individual pseudo truck############
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\2021-10-12.csv",sep=" ")
individual_truck = df['Individual'][0].split(',')

individual_truck = [float(i) for i in individual_truck]

fig,axis = plt.subplots(1,1,figsize=(15,6),sharey=True,sharex=False);
axis.plot(individual_truck,'blue', label='Individual pseudo truck example')
axis.set_xlabel('time counts',fontsize=18)
axis.set_ylabel('Cu grade (w.t%)',fontsize=18)
axis.tick_params(axis='both', which='major', labelsize=18) 
axis.legend(loc='upper right',fontsize=18)


pseudo_truck=[]
files = os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade')
for i in files:
    pseudo_truck.append(pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\"+str(i),sep=" "))

pseudo_truck1 = []
for j in pseudo_truck:
    for i in range(len(j)):
        pseudo_truck1.append(j['Individual'][i].split(','))

pseudo_truck_new = []
for i in pseudo_truck1:
    if len(i)>75:
        pseudo_truck2 = []
        for j in i:
            pseudo_truck2.append(float(j))
        pseudo_truck_new.append(pseudo_truck2)    

df = pd.DataFrame(pseudo_truck_new)
average_df = df.mean(axis=1)
fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=True,sharex=False);
axis.hist(average_df,bins=100,color='blue', label='Individual pseudo trucks')
axis.set_xlabel('Average Cu grade (w.t%) for individual pseudo truck',fontsize=18)
axis.set_ylabel('Frequecy',fontsize=18)
axis.tick_params(axis='both', which='major', labelsize=18) 
axis.legend(loc='upper right',fontsize=18)


pio.renderers.default='browser'
df_gps = pd.read_excel('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy.xlsx')
df_gps = df_gps.drop(['DATE','END_TS','DATE1','TIME1','TIMESEC'],axis=1)
df_gps = df_gps[df_gps['LOAD_LOCATION_SNAME']!='FINGER_9']
df_gps['TIME'] = df_gps['TIME'].astype(str)
df_gps = df_gps.dropna(subset=['MID_X'])
df_gps = df_gps.dropna(subset=['TONNES'])
x1 = 3000
x2 = x1 + 1000
y1 = 12500
y2 = y1 + 500
z1 = 1200
z2 = z1+200
df_gps = df_gps.loc[(pd.to_numeric(df_gps["MID_X"], errors='coerce')>=x1) & (pd.to_numeric(df_gps["MID_X"], errors='coerce')<x2) 
            &(pd.to_numeric(df_gps["MID_Y"], errors='coerce')>=y1) & (pd.to_numeric(df_gps["MID_Y"], errors='coerce')<y2)
            &(pd.to_numeric(df_gps["MID_Z"], errors='coerce')>=z1) & (pd.to_numeric(df_gps["MID_Z"], errors='coerce')<z2)]
df_gps = df_gps.reset_index(drop=True)
df_gps = df_gps[968:]
df_gps = df_gps.reset_index(drop=True)
df_gps.insert(0, 'DATE', df_gps['TIME'].str[:10])

df_gps['TIME'] = pd.to_datetime(df_gps['TIME'])
df_gps['year'] = df_gps['TIME'].dt.year
df_gps['month'] = df_gps['TIME'].dt.month
df_gps['day'] = df_gps['TIME'].dt.day
df_gps['hour'] = df_gps['TIME'].dt.hour
df_gps['minute'] = df_gps['TIME'].dt.minute
df_gps['second'] = df_gps['TIME'].dt.second
df_gps['X'] = df_gps['MID_X']
df_gps['Y'] = df_gps['MID_Y']
df_gps['Z'] = df_gps['MID_Z']

fig = px.scatter_3d(df_gps, x="X",y="Y",z="Z",color="DATE")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  

fields = ['X','Y','Z','CU'] 
pio.renderers.default='browser'
df_bore_core = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)
df_bore_core = df_bore_core.dropna()

df_bore_core1 = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=3000)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<4000)
        & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=12500)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<13000)
        & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=1200)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<1400)]
df_bore_core1 = df_bore_core1.reset_index(drop=True)
fig = px.scatter_3d(df_bore_core1, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=2)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  



# from scipy.optimize import curve_fit
# np.random.seed(0)
# # Define the nonlinear function
# def quadratic_func(x, a, b, c, d):
#     return a + b * x + c * x**2 +d*x**3 
# popt, pcov = curve_fit(quadratic_func, x1, y1)
# Visualize the results

import pandas as pd
import numpy as np
import math
import numpy
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale.csv")
x1 = np.array(df['x'])
y1 = np.array(df['y'])
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale_mra.csv")
x2 = np.array(df['x']).reshape(-1,1)
y2 = np.array(df['y'])
fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=True,sharex=False);
axis.scatter(x1[0:1], y1[0:1], label='Bore core',color='blue')
axis.scatter(x2[0:1], y2[0:1], label='MRA',color='red')
axis.set_xlabel('Log10 (Tonnage scale)',fontsize=18)
axis.set_ylabel('log10 (Variance)',fontsize=18)
axis.tick_params(axis='both', which='major', labelsize=18) 
axis.legend(loc='upper right',fontsize=18)
axis.set_xlim(-1.5,8)
axis.set_ylim(-2.5,1)

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
from scipy import stats
import random
#### df_bore_core = bore core data ####
fields = ['X','Y','Z','CU','VEIN'] 
pio.renderers.default='browser'
df_bore_core = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)
df_bore_core = df_bore_core.dropna()

df_bore_core1 = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=3000)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<4000)
        & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=12500)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<13000)
        & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=1200)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<1400)]
df_bore_core1 = df_bore_core1.reset_index(drop=True)
# df_bore_core1 = df_bore_core1.groupby(['BHID']).filter(lambda x: len(x)>20)
fig = px.scatter_3d(df_bore_core1, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=2)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show() 


fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=True,sharex=False)
axis.hist(df_bore_core1['CU'],bins=np.arange(0,2,0.01),density=True,color='m')
axis.set_xlabel('Cu grade (w.t%)',fontsize=22)
axis.set_ylabel('Density',fontsize=22)
axis.tick_params(axis='both', which='major', labelsize=22) 
axis.legend(loc='upper right',fontsize=22)


fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=True,sharex=False)
axis.hist(df_bore_core1['STRAT'],density=True,color='m')
axis.set_xlabel('strata',fontsize=22)
axis.set_ylabel('Density',fontsize=22)
axis.tick_params(axis='both', which='major', labelsize=22) 
axis.legend(loc='upper right',fontsize=22)

fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=True,sharex=False)
axis.hist(df_bore_core1['VEIN'],density=True,color='m')
axis.set_xlabel('vein',fontsize=22)
axis.set_ylabel('Density',fontsize=22)
axis.tick_params(axis='both', which='major', labelsize=22) 
axis.legend(loc='upper right',fontsize=22)


fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
#df = df.dropna()
df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>=0.5)& (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)]

df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>=0.5) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0)& (pd.to_numeric(df["As_dh"], errors='coerce')>0)
        & (pd.to_numeric(df["X"], errors='coerce')>=16000)& (pd.to_numeric(df["X"], errors='coerce')<16500)
        & (pd.to_numeric(df["Y"], errors='coerce')>=106500)& (pd.to_numeric(df["Y"], errors='coerce')<107000)
        & (pd.to_numeric(df["Z"], errors='coerce')>=2500)& (pd.to_numeric(df["Z"], errors='coerce')<3000)]
df['X'] = round(df['X'],2)
df['Y'] = round(df['Y'],2)
df['Z'] = round(df['Z'],2)



df2 = df[['BHID','X','Y','Z','CuT_dh','Fe_dh','As_dh','LITH','AL_ALT']]


n = 100
m = 50
xx1 = np.arange(16000, 16500, n).astype('float64')
yy1 = np.arange(106500,107000, n).astype('float64')
zz1 = np.arange(2500, 3000, m).astype('float64')

blocks = []
for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df2.loc[(pd.to_numeric(df2["X"], errors='coerce')>=i) & (pd.to_numeric(df2["X"], errors='coerce')<i+n) &
                         (pd.to_numeric(df2["Y"], errors='coerce')>=j) & (pd.to_numeric(df2["Y"], errors='coerce')<j+n)
                         &(pd.to_numeric(df2["Z"], errors='coerce')>=k) & (pd.to_numeric(df2["Z"], errors='coerce')<k+m)]
            blocks.append(sub_block)
blocks1 = []
for i,j in enumerate(blocks):
    if len(j)>=5:
        blocks1.append(j)
for i, j in enumerate(blocks1):
    blocks1[i]['blocks'] = i
 
df2_new = pd.concat(blocks1)   
block_idxs1 = np.array(df2_new['blocks'])
n_blocks = len(df2_new['blocks'].unique())

from scipy import linalg, stats


df3= df2_new[df2_new['blocks']==32].sort_values(by=['CuT_dh'])
X = np.array(df3['CuT_dh'])
Y = np.array(df3['Fe_dh'])
phi_x = np.vstack([X**0, X**1])      

wML = linalg.solve(np.dot(phi_x, phi_x.T), np.dot(phi_x, Y))
Y_pred = np.dot(wML, phi_x)   ###MLE            


subdata = df2_new[df2_new['blocks']==32]
fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=True,sharex=False)
axis.scatter(subdata['CuT_dh'],subdata['Fe_dh'],color='m')
axis.plot(X, np.dot(wML, phi_x), ls='--',color='b', lw=3, label="MLE")
axis.set_xlabel('Cu w.t%',fontsize=22)
axis.set_ylabel('Fe w.t%',fontsize=22)
axis.tick_params(axis='both', which='major', labelsize=22) 
axis.legend(loc='upper right',fontsize=22)














