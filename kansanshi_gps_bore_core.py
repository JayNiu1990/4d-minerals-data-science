import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd    
import plotly.io as pio
import plotly.graph_objs as go
from scipy import stats
import random
#df_bore_core = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\bore core regular.csv", skipinitialspace=True, usecols=fields,dtype='unicode')
#### df_bore_core = bore core data ####
fields = ['X','Y','Z','CU'] 
pio.renderers.default='browser'
df_bore_core = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)
df_bore_core = df_bore_core.dropna()

df_bore_core1 = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=3000)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<4000)
        & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=12500)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<13000)
        & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=1250)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<1350)]
df_bore_core1 = df_bore_core1.reset_index(drop=True)

# df_bore_core2 = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=4700)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<5300)
#         & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=10400)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<12000)
#         & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=1200)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<1400)]
# df_bore_core2 = df_bore_core2.reset_index(drop=True)

# df_bore_core_filter = pd.concat([df_bore_core1,df_bore_core2])
# df_bore_core_filter = df_bore_core_filter.reset_index(drop=True)
# df_bore_core_filter['CU'] = df_bore_core_filter['CU'].astype('float32') 

fig = px.scatter_3d(df_bore_core1, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=2)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  

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
fig.show()  

import datetime
import time
second_list = []
for i in range(len(df_gps)):
    dt = datetime.datetime(df_gps['year'][i], df_gps['month'][i], df_gps['day'][i] , df_gps['hour'][i], df_gps['minute'][i],df_gps['second'][i])
    second_list.append(time.mktime(dt.timetuple()))
df_gps['total_second'] = second_list

# df_gps['MID_Z'].max() ### 1316.5
# df_gps['MID_Z'].min() ### 1296.5



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
for i in range(len(df_gps)):
    gps_time = df_gps[i:i+1]['total_second']
    delay_mra = df_mra.loc[(pd.to_numeric(df_mra["total_second"], errors='coerce')>int(gps_time)+delay) & (pd.to_numeric(df_mra["total_second"], errors='coerce')<int(gps_time)+delay+round(df_gps[i:i+1]['TONNES'].values[0]/df_mra['tonnage'].mean()*3600/4))] 
    average_mra = delay_mra['grade'].mean()
    delay_mra_list.append(average_mra)
df_gps['average mra'] = delay_mra_list
df_gps = df_gps.dropna(subset=['average mra'])
df_gps.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv')

########calculate surrounding bore core grade for gps data######
# average_surrounding_10x10x10_bore_core_grade = []
# surrounding_10x10_10_bore_core = []
# for i in range(len(df_gps)):
#     row = df_gps[i:i+1]
#     x = np.array(row['MID_X'])[0]
#     y = np.array(row['MID_Y'])[0]
#     z = np.array(row['MID_Z'])[0]
#     borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-10)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+10)
#             & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-10)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+10)
#             & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-10)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+10)]
#     #borecoreregion = borecoreregion.astype('float')
#     average_surrounding_10x10x10_bore_core_grade.append(borecoreregion['CU'].mean())
#     surrounding_10x10_10_bore_core.append(borecoreregion)
# df_gps['average surrounding bore core 10x10x10 grade'] = average_surrounding_10x10x10_bore_core_grade




# average_surrounding_20x20x20_bore_core_grade = []
# for i in range(len(df_gps)):
#     row = df_gps[i:i+1]
#     x = np.array(row['MID_X'])[0]
#     y = np.array(row['MID_Y'])[0]
#     z = np.array(row['MID_Z'])[0]
#     borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-20)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+20)
#             & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-20)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+20)
#             & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-20)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+20)]
#     borecoreregion = borecoreregion.astype('float')
#     average_surrounding_20x20x20_bore_core_grade.append(borecoreregion['CU'].mean())
# df_gps['average surrounding bore core 20x20x20 grade'] = average_surrounding_20x20x20_bore_core_grade

# average_surrounding_30x30x30_bore_core_grade = []
# for i in range(len(df_gps)):
#     row = df_gps[i:i+1]+ 
#     x = np.array(row['MID_X'])[0]
#     y = np.array(row['MID_Y'])[0]
#     z = np.array(row['MID_Z'])[0]
#     borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-30)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+30)
#             & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-30)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+30)
#             & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-30)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+30)]
#     borecoreregion = borecoreregion.astype('float')
#     average_surrounding_30x30x30_bore_core_grade.append(borecoreregion['CU'].mean())
# df_gps['average surrounding bore core 30x30x30 grade'] = average_surrounding_30x30x30_bore_core_grade

# average_surrounding_20x20x10_bore_core_grade = []
# for i in range(len(df_gps)):
#     row = df_gps[i:i+1]
#     x = np.array(row['MID_X'])[0]
#     y = np.array(row['MID_Y'])[0]
#     z = np.array(row['MID_Z'])[0]
#     borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-20)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+20)
#             & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-20)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+20)
#             & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-10)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+10)]
#     borecoreregion = borecoreregion.astype('float')
#     average_surrounding_20x20x10_bore_core_grade.append(borecoreregion['CU'].mean())
# df_gps['average surrounding bore core 20x20x10 grade'] = average_surrounding_20x20x10_bore_core_grade

average_surrounding_30x30x10_bore_core_grade = []
for i in range(len(df_gps)):
    row = df_gps[i:i+1]
    x = np.array(row['MID_X'])[0]
    y = np.array(row['MID_Y'])[0]
    z = np.array(row['MID_Z'])[0]
    borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-30)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+30)
            & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-30)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+30)
            & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-10)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+10)]
    average_surrounding_30x30x10_bore_core_grade.append(borecoreregion['CU'].mean())
df_gps['average surrounding bore core 30x30x10 grade'] = average_surrounding_30x30x10_bore_core_grade


df_kriging = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\bore core regular - kriging.csv')

average_surrounding_30x30x10_kriging_grade = []
for i in range(len(df_gps)):
    row = df_gps[i:i+1]
    x = np.array(row['MID_X'])[0]
    y = np.array(row['MID_Y'])[0]
    z = np.array(row['MID_Z'])[0]
    krigingregion = df_kriging[(pd.to_numeric(df_kriging["X"], errors='coerce')>=x-30)& (pd.to_numeric(df_kriging["X"], errors='coerce')<=x+30)
            & (pd.to_numeric(df_kriging["Y"], errors='coerce')>=y-30)& (pd.to_numeric(df_kriging["Y"], errors='coerce')<=y+30)
            & (pd.to_numeric(df_kriging["Z"], errors='coerce')>=z-10)& (pd.to_numeric(df_kriging["Z"], errors='coerce')<=z+10)]
    average_surrounding_30x30x10_kriging_grade.append(krigingregion['CU_kriging'].mean())
df_gps['average surrounding kriging 30x30x10 grade'] = average_surrounding_30x30x10_kriging_grade
# average_surrounding_30x30x20_bore_core_grade = []
# for i in range(len(df_gps)):
#     row = df_gps[i:i+1]
#     x = np.array(row['MID_X'])[0]
#     y = np.array(row['MID_Y'])[0]
#     z = np.array(row['MID_Z'])[0]
#     borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-30)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+30)
#             & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-30)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+30)
#             & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-20)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+20)]
#     borecoreregion = borecoreregion.astype('float')
#     average_surrounding_30x30x20_bore_core_grade.append(borecoreregion['CU'].mean())
# df_gps['average surrounding bore core 30x30x20 grade'] = average_surrounding_30x30x20_bore_core_grade

# average_surrounding_40x40x20_bore_core_grade = []
# for i in range(len(df_gps)):
#     row = df_gps[i:i+1]
#     x = np.array(row['MID_X'])[0]
#     y = np.array(row['MID_Y'])[0]
#     z = np.array(row['MID_Z'])[0]
#     borecoreregion = df_bore_core[(pd.to_numeric(df_bore_core["X"], errors='coerce')>=x-40)& (pd.to_numeric(df_bore_core["X"], errors='coerce')<=x+40)
#             & (pd.to_numeric(df_bore_core["Y"], errors='coerce')>=y-40)& (pd.to_numeric(df_bore_core["Y"], errors='coerce')<=y+40)
#             & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=z-20)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<=z+20)]
#     borecoreregion = borecoreregion.astype('float')
#     average_surrounding_40x40x20_bore_core_grade.append(borecoreregion['CU'].mean())
# df_gps['average surrounding bore core 40x40x20 grade'] = average_surrounding_40x40x20_bore_core_grade





















average_mra_day = []
average_block_model_day = []
average_bore_core_10x10x10_day = []
average_bore_core_20x20x20_day = []
average_bore_core_30x30x30_day = []
average_bore_core_20x20x10_day = []
average_bore_core_30x30x10_day_kriging = []
average_bore_core_30x30x10_day = []
average_bore_core_30x30x20_day = []
average_bore_core_40x40x20_day = []
for i in date:
    average_mra_day.append(df_gps[df_gps['DATE'] == i]['average mra'].mean())
    # average_bore_core_10x10x10_day.append(df_gps[df_gps['DATE'] == i]['average surrounding bore core 10x10x10 grade'].mean())
    # average_bore_core_20x20x20_day.append(df_gps[df_gps['DATE'] == i]['average surrounding bore core 20x20x20 grade'].mean())
    # average_bore_core_30x30x30_day.append(df_gps[df_gps['DATE'] == i]['average surrounding bore core 30x30x30 grade'].mean())
    # average_bore_core_20x20x10_day.append(df_gps[df_gps['DATE'] == i]['average surrounding bore core 20x20x10 grade'].mean())
    average_bore_core_30x30x10_day.append(df_gps[df_gps['DATE'] == i]['average surrounding bore core 30x30x10 grade'].mean())
    average_bore_core_30x30x10_day_kriging.append(df_gps[df_gps['DATE'] == i]['average surrounding kriging 30x30x10 grade'].mean())
    # average_bore_core_30x30x20_day.append(df_gps[df_gps['DATE'] == i]['average surrounding bore core 30x30x20 grade'].mean())
    # average_bore_core_40x40x20_day.append(df_gps[df_gps['DATE'] == i]['average surrounding bore core 40x40x20 grade'].mean())
    average_block_model_day.append(df_gps[df_gps['DATE'] == i]['TCU'].mean())
    

mra_average_perday = []

for j in range(len(date)):
    MRA_subdf = df_mra[df_mra['date']==date[j]].reset_index(drop=True)
    MRA_subdf = MRA_subdf[MRA_subdf.grade>0]
    mra_average_perday.append(MRA_subdf['grade'].mean())
    
fig, ax = plt.subplots(1, 1,figsize=(22,10))
#ax.plot(date,average_block_model_day, '--bo', label='block daily mean grade')
# ax.plot(date,average_bore_core_10x10x10_day, '--go', label='estimated daily mean grade from bore core 10x10x10')
# ax.plot(date,average_bore_core_20x20x20_day, '--yo', label='estimated daily mean grade from bore core 20x20x20')
# ax.plot(date,average_bore_core_30x30x30_day, '--mo', label='estimated daily mean grade from bore core 30x30x30')
# ax.plot(date,average_bore_core_40x40x40_day, '--mo', label='estimated daily mean grade from bore core 20x20x10')
#ax.plot(date,average_bore_core_10x10x10_day, '--bo', label='estimated daily mean grade from bore core 30x30x10')
ax.plot(date,average_bore_core_30x30x10_day, '--yo', label='estimated daily mean grade from bore core 30x20x10')
ax.plot(date,average_bore_core_30x30x10_day, '--yo', label='estimated daily mean grade from bore core 30x20x10')
ax.plot(date,average_mra_day, '--ro', label='mra daily mean grade')
# ax.plot(date,mra_average_perday, '--bo', label='mra daily mean grade (previous)')
ax.set_xlabel('date',fontsize=22)
ax.set_ylabel('mean grade',fontsize=22)
ax.set_ylim(0,8)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('10mins delay',fontsize=22)
plt.legend(fontsize=22)
#fig.savefig('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Journal paper2\\img3.png', bbox_inches='tight',dpi=300)
 






    
    