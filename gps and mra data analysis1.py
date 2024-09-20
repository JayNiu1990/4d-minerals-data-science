import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
import arviz as az
from scipy import stats
import random
pio.renderers.default='browser'
df2 = pd.read_excel('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy.xlsx')
df2 = df2.drop(['DATE','END_TS','DATE1','TIME1','TIMESEC'],axis=1)
df2['TIME1'] = df2['TIME'].astype(str)
#df2 = df2.dropna()
#df2 = df2[df2['LOAD_LOCATION_SNAME']=='FINGER_9']
df2 = df2.dropna(subset=['MID_X'])
df2 = df2.reset_index(drop=True)
df2.insert(0, 'date', df2.TIME1.str[0:10])
df2.insert(1, 'time', pd.Series([val.time() for val in df2['TIME']]))
df2.insert(2, 'hour', df2.TIME1.str[10:13].astype(int))
df2.insert(3, 'minute', df2.TIME1.str[14:16].astype(int))
df2.insert(4, 'second', df2.TIME1.str[17:19].astype(int))
df2.insert(5, 'timestamp', df2['hour']*3600 + df2['minute']*60 + df2['second'])
df2 = df2.drop(['hour','minute','second','TIME1'],axis=1)
df2 = df2[1348:]
#df2 = df2[2194:]
df2 = df2.reset_index(drop=True)
df2 = df2[(pd.to_numeric(df2["MID_X"], errors='coerce')>=3000)& (pd.to_numeric(df2["MID_X"], errors='coerce')<4000)
        & (pd.to_numeric(df2["MID_Y"], errors='coerce')>=12500)& (pd.to_numeric(df2["MID_Y"], errors='coerce')<13000)]
df2 = df2.reset_index(drop=True)


fig = px.scatter_3d(df2, x="MID_X",y="MID_Y",z="MID_Z",color="TCU")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=16))
fig.update_layout(scene_aspectmode='data')
fig.show()  


#### interested region - (X:3000-4000, Y:125000-130000, Z:1200-1350)

fields = ["X","Y","Z",'CU']
pio.renderers.default='browser'
df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\bore core regular.csv", skipinitialspace=True, usecols=fields)
df1 = df1.dropna()
df1 = df1[(pd.to_numeric(df1["X"], errors='coerce')>=3000)& (pd.to_numeric(df1["X"], errors='coerce')<4000)
        & (pd.to_numeric(df1["Y"], errors='coerce')>=12500)& (pd.to_numeric(df1["Y"], errors='coerce')<13500)
        & (pd.to_numeric(df1["Z"], errors='coerce')>=1280)& (pd.to_numeric(df1["Z"], errors='coerce')<1330)]
df1 = df1.reset_index(drop=True)
 
fig = px.scatter_3d(df1, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=12))
fig.update_layout(scene_aspectmode='data')
fig.show()  












