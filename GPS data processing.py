import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
from scipy import stats
import random
#### df_bore_core = bore core data ####
fields = ['X','Y','Z','CU'] 
pio.renderers.default='browser'
df_bore_core = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)
df_bore_core = df_bore_core.dropna()

df_bore_core1 = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=3000)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<4000)
        & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=12500)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<13000)
        & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=1200)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<1400)]
df_bore_core1 = df_bore_core1.reset_index(drop=True)
fig = px.scatter_3d(df_bore_core1, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  

# a = df_bore_core1[df_bore_core1['BHID']=='KRC066']

# df_bore_core1['CU'] = df_bore_core1['CU'].astype('float32') 
# np.log10(df_bore_core1['CU'].var())
         

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
from scipy import stats
import random
#### df_gps = gps data ####
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

fig = px.scatter_3d(df_gps, x="MID_X",y="MID_Y",z="MID_Z",color="DATE")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.update_layout(scene = dict(
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title=''))
fig.show()  

import datetime
import time
second_list = []
for i in range(len(df_gps)):
    dt = datetime.datetime(df_gps['year'][i], df_gps['month'][i], df_gps['day'][i] , df_gps['hour'][i], df_gps['minute'][i],df_gps['second'][i])
    second_list.append(time.mktime(dt.timetuple()))
df_gps['total_second'] = second_list
###df_mra: all mra normal data####
df_mra = []
date = list(df_gps['DATE'].unique())
for i in date:
    MRA_subdf = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\' +i +'.csv')
    df_mra.append(MRA_subdf)
df_mra = pd.concat(df_mra)   ### modified MRA all data in all days
df_mra = df_mra.loc[(pd.to_numeric(df_mra["grade"], errors='coerce')>0)] 
df_mra = df_mra.reset_index(drop=True)
df_mra['datetime'] = pd.to_datetime(df_mra['datetime'])
df_mra['year'] = df_mra['datetime'].dt.year
df_mra['month'] = df_mra['datetime'].dt.month
df_mra['day'] = df_mra['datetime'].dt.day
df_mra['hour'] = df_mra['datetime'].dt.hour
df_mra['minute'] = df_mra['datetime'].dt.minute
df_mra['second'] = df_mra['datetime'].dt.second

second_list = []
for i in range(len(df_mra)):
    dt = datetime.datetime(df_mra['year'][i], df_mra['month'][i], df_mra['day'][i] , df_mra['hour'][i], df_mra['minute'][i],df_mra['second'][i])
    second_list.append(time.mktime(dt.timetuple()))
df_mra['total_second'] = second_list

#### match gps to mra with delay time
delay = 900 # 75-300s
delay_mra_list = []
for i in range(2,3):#len(df_gps)
    gps_time = df_gps[i:i+1]['total_second']
    delay_mra = df_mra.loc[(pd.to_numeric(df_mra["total_second"], errors='coerce')>int(gps_time)+delay) & (pd.to_numeric(df_mra["total_second"], errors='coerce')<int(gps_time)+delay+round(df_gps[i:i+1]['TONNES'].values[0]/df_mra['tonnage'].mean()*3600/4))] 
    average_mra = delay_mra['grade'].mean()
    delay_mra_list.append(average_mra)
df_gps['average mra'] = delay_mra_list
df_gps = df_gps.dropna(subset=['average mra'])
#df_gps.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv')

########calculate surrounding bore core grade for gps data######
average_surrounding_bore_core_grade = []
surrounding_bore_core = []
for i in range(len(df_gps)):
    row = df_gps[i:i+1]
    x = np.array(row['MID_X'])[0]
    y = np.array(row['MID_Y'])[0]
    z = np.array(row['MID_Z'])[0]
    borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-30)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+30)
            & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-30)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+30)
            & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-10)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+10)]
    # borecoreregion = borecoreregion.astype('float')
    average_surrounding_bore_core_grade.append(borecoreregion['CU'].mean())
    surrounding_bore_core.append(borecoreregion)

surrounding_bore_core =  pd.concat(surrounding_bore_core) 
surrounding_bore_core_unique = surrounding_bore_core.drop_duplicates(keep='first')


# name = list(surrounding_bore_core_unique['BHID'].unique())
# var2 = []
# for n1 in range(1,12,1):
#     data = []
#     for i in name:
#         each_borecore = surrounding_bore_core_unique[surrounding_bore_core_unique['BHID']==i]
#         sub_data = [each_borecore[i:i+n1]['CU'] for i in range(0,len(each_borecore),n1)]
#         sub_data = [x for x in sub_data if len(x)==n1]
#         data.extend(sub_data)
#     var2.append(np.var([np.mean(j) for j in data]))





fig,axis = plt.subplots(1,1,figsize=(15,6),sharey=True,sharex=False);
axis.plot(delay_mra['grade'],'blue', label='MRA')
axis.set_xlabel('time counts',fontsize=18)
axis.set_ylabel('Cu grade (w.t%)',fontsize=18)
axis.tick_params(axis='both', which='major', labelsize=18) 
axis.legend(loc='upper right',fontsize=18)











