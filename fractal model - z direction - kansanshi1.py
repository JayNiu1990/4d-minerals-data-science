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
        & (pd.to_numeric(df_bore_core["Z"], errors='coerce')>=1250)& (pd.to_numeric(df_bore_core["Z"], errors='coerce')<1350)]
df_bore_core1 = df_bore_core1.reset_index(drop=True)
# df_bore_core1 = df_bore_core1.groupby(['BHID']).filter(lambda x: len(x)>20)
# fig = px.scatter_3d(df_bore_core1, x="X",y="Y",z="Z",color="CU")
# fig.update_traces(marker_size=3)
# fig.update_layout(font=dict(size=14))
# fig.update_layout(scene_aspectmode='data')
# fig.show() 
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

# fig = px.scatter_3d(df_gps, x="MID_X",y="MID_Y",z="MID_Z",color="DATE")
# fig.update_traces(marker_size=4)
# fig.update_layout(font=dict(size=14))
# fig.update_layout(scene_aspectmode='data')
# fig.show()  

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
delay = 600 # 75-300s
average_delay_mra_list = []
delay_mra_list = []
for i in range(len(df_gps)):
    gps_time = df_gps[i:i+1]['total_second']
    delay_mra = df_mra.loc[(pd.to_numeric(df_mra["total_second"], errors='coerce')>int(gps_time)+delay) & (pd.to_numeric(df_mra["total_second"], errors='coerce')<int(gps_time)+delay+round(df_gps[i:i+1]['TONNES'].values[0]/df_mra['tonnage'].mean()*3600/4))] 
    delay_mra_list.append(delay_mra)
    average_mra = delay_mra['grade'].mean()
    average_delay_mra_list.append(average_mra)
df_gps['average mra'] = average_delay_mra_list
df_gps = df_gps.dropna(subset=['average mra'])
df_gps.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv')

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


name = list(surrounding_bore_core_unique['BHID'].unique())
var2 = []
for n1 in range(1,12,1):
    data = []
    for i in name:
        each_borecore = surrounding_bore_core_unique[surrounding_bore_core_unique['BHID']==i]
        sub_data = [each_borecore[i:i+n1]['CU'] for i in range(0,len(each_borecore),n1)]
        sub_data = [x for x in sub_data if len(x)==n1]
        data.extend(sub_data)
    var2.append(np.var([np.mean(j) for j in data]))



#range1 = np.arange(1*6.5,11*6.5,1*6.5)       
import matplotlib.pyplot as plt

##########mra##########
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import os
mr = []
for file in os.listdir("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\")[0:52]:
    mr_df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\" + str(file), skipinitialspace=True)
    mr_df = mr_df.dropna()
    mr_df = mr_df[(pd.to_numeric(mr_df["tonnage"], errors='coerce')>=2000)]
    mr_df = mr_df[(pd.to_numeric(mr_df["grade"], errors='coerce')>0)]
    mr_df = mr_df.reset_index(drop=True)
    mr.append(mr_df)
mr = pd.concat(mr)

var3 = []
list_mean_subgroup1 = []
for j in range(50,1001,50):
    data = []
    mean = []
    subgroup1 = [mr[n:n+j] for n in range(0,len(mr),j)]
    subgroup1 = [x for x in subgroup1 if len(x)==j]
    data.extend(subgroup1)
    for sub_data in data:
        mean.append(sub_data['grade'].mean())
    var3.append(np.var(mean))


# plt.hist(mr_df1,density=True,alpha=0.5,bins=np.arange(0,5,0.1))
# plt.xlim(0,5)
average_tonnage = mr['tonnage'].mean()
scale = 4* average_tonnage/3600


mass = 3.14*(0.1**2)*1*2.65 #(T)
range1 = np.arange(1*mass,12*mass-0.001,1*mass)          
range2 = np.arange(50*scale,1001*scale-0.1,50*scale)     

df2 = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\scale\\kansanshi_largescale.csv')
range3 = np.array(df2['scale'])
var4 = np.array(df2['bore core large scale'])
var5 = np.array(df2['mra large scale'])



############kriging data############
df_kriging0 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kriging-1200-1250.csv", skipinitialspace=True)
df_kriging1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kriging-1250-1300.csv", skipinitialspace=True)
df_kriging2 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kriging-1300-1325.csv", skipinitialspace=True)
df_kriging3 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kriging-1325-1350.csv", skipinitialspace=True)
df_kriging = pd.concat([df_kriging0,df_kriging1, df_kriging2, df_kriging3], ignore_index=True)
df_kriging = df_kriging.drop(columns='Unnamed: 0')
df_kriging = df_kriging.sort_values(by="Z", ascending=True)
df_kriging = df_kriging.reset_index(drop=True)
df_kriging['log Cu'] = np.log10(df_kriging['CU_kriging'])
pio.renderers.default='browser'
fig = px.scatter_3d(df_kriging, x="X",y="Y",z="Z",color='log Cu')
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  
# df_kriging = df_kriging[(pd.to_numeric(df_kriging["Z"], errors='coerce')>=1280)& (pd.to_numeric(df_kriging["Z"], errors='coerce')<1320)]

##################### X direction################
# list_var6 = []
# kriging_Xlist = []
# subdata1 = df_kriging[(pd.to_numeric(df_kriging["Z"], errors='coerce')>=1250)& (pd.to_numeric(df_kriging["Z"], errors='coerce')<=1350)]
# for i in range(12500,13000,10): #Y
#     for j in range(1280,1320,5): #Z
#         subdata2 = subdata1[(pd.to_numeric(subdata1["Y"], errors='coerce')==i)& (pd.to_numeric(subdata1["Z"], errors='coerce')==j)]
#         subdata2 = subdata2.sort_values(by='X',ascending=True)
#         subdata2 = subdata2.reset_index(drop=True)
#         # subdata1 = subdata['CU_kriging'].values
#         kriging_Xlist.append(subdata2)

# var6 = []
# mean6 = []
# for i in range(1,11,1):
#     data = []
#     for j in range(len(kriging_Xlist)):
#         kriging_sub = kriging_Xlist[j]
#         sub_data = [kriging_sub[n:n+i]['CU_kriging']  for n in range(0,len(kriging_sub),i)]
#         sub_data = [x for x in sub_data if len(x)==i]
#         data.extend(sub_data)
#     mean6.append([np.mean(k) for k in data])
#     var6.append(np.var([np.mean(k) for k in data]))
# list_var6.append(var6)

# mass1 = 3.14*(0.1**2)*20*2.65 #(T)
# range6 = np.arange(1*mass1,11*mass1-0.001,1*mass1)   
# fig,axis = plt.subplots(1,1,figsize=(12,8))
# axis.scatter(np.log10(range1),np.log10(var2),label='bore core',color='r')
# # #plt.scatter(np.log10(range1),np.log10(var2),label='random status')
# axis.scatter(np.log10(range2),np.log10(var3),label='mra',color='b')
# axis.scatter(range3,var4,color='r')
# axis.scatter(range3,var5,color='b')
# axis.scatter(np.log10(range6),np.log10(var6),color='m',label='kriging - X direction' )
# # axis.scatter(np.log10(250),np.log10(np.var(onetruck_mean_list)),color='green',s=160,label='pseudo truck')
# axis.set_xlabel('log10(Tonnage)',fontsize=20)
# axis.set_ylabel('log10(Variance)',fontsize=20)
# axis.legend(loc='lower left',fontsize=20)


# ##################### Y direction################
# list_var6 = []
# kriging_Xlist = []
# subdata1 = df_kriging[(pd.to_numeric(df_kriging["Z"], errors='coerce')>=1250)& (pd.to_numeric(df_kriging["Z"], errors='coerce')<=1350)]
# for i in range(3000,4000,20): #X
#     for j in range(1280,1320,5): #Z
#         subdata2 = subdata1[(pd.to_numeric(subdata1["X"], errors='coerce')==i)& (pd.to_numeric(subdata1["Z"], errors='coerce')==j)]
#         subdata2 = subdata2.sort_values(by='X',ascending=True)
#         subdata2 = subdata2.reset_index(drop=True)
#         # subdata1 = subdata['CU_kriging'].values
#         kriging_Xlist.append(subdata2)

# var6 = []
# mean6 = []
# for i in range(1,11,1):
#     data = []
#     for j in range(len(kriging_Xlist)):
#         kriging_sub = kriging_Xlist[j]
#         sub_data = [kriging_sub[n:n+i]['CU_kriging']  for n in range(0,len(kriging_sub),i)]
#         sub_data = [x for x in sub_data if len(x)==i]
#         data.extend(sub_data)
#     mean6.append([np.mean(k) for k in data])
#     var6.append(np.var([np.mean(k) for k in data]))
# list_var6.append(var6)

# mass1 = 3.14*(0.1**2)*20*2.65 #(T)
# range6 = np.arange(1*mass1,11*mass1-0.001,1*mass1)   
# fig,axis = plt.subplots(1,1,figsize=(12,8))
# axis.scatter(np.log10(range1),np.log10(var2),label='bore core',color='r')
# # #plt.scatter(np.log10(range1),np.log10(var2),label='random status')
# axis.scatter(np.log10(range2),np.log10(var3),label='mra',color='b')
# axis.scatter(range3,var4,color='r')
# axis.scatter(range3,var5,color='b')
# axis.scatter(np.log10(range6),np.log10(var6),color='m',label='kriging - Y direction' )
# # axis.scatter(np.log10(250),np.log10(np.var(onetruck_mean_list)),color='green',s=160,label='pseudo truck')
# axis.set_xlabel('log10(Tonnage)',fontsize=20)
# axis.set_ylabel('log10(Variance)',fontsize=20)
# axis.legend(loc='lower left',fontsize=20)


##################### Z direction################
list_var6 = []
kriging_Xlist = []
subdata1 = df_kriging[(pd.to_numeric(df_kriging["Z"], errors='coerce')>=1250)& (pd.to_numeric(df_kriging["Z"], errors='coerce')<=1350)]
for i in range(3000,4000,20): #X
    for j in range(12500,13000,10): #Y
        subdata2 = subdata1[(pd.to_numeric(subdata1["X"], errors='coerce')==i)& (pd.to_numeric(subdata1["Y"], errors='coerce')==j)]
        subdata2 = subdata2.sort_values(by='X',ascending=True)
        subdata2 = subdata2.reset_index(drop=True)
        # subdata1 = subdata['CU_kriging'].values
        kriging_Xlist.append(subdata2)

var6 = []
mean6 = []
for i in range(1,11,1):
    data = []
    for j in range(len(kriging_Xlist)):
        kriging_sub = kriging_Xlist[j]
        sub_data = [kriging_sub[n:n+i]['CU_kriging']  for n in range(0,len(kriging_sub),i)]
        sub_data = [x for x in sub_data if len(x)==i]
        data.extend(sub_data)
    mean6.append([np.mean(k) for k in data])
    var6.append(np.var([np.mean(k) for k in data]))
list_var6.append(var6)


mass1 = 3.14*(0.1**2)*5*2.65 #(T)
range6 = np.arange(1*mass1,11*mass1-0.001,1*mass1)   
fig,axis = plt.subplots(1,1,figsize=(12,8))
axis.scatter(np.log10(range1),np.log10(var2),label='bore core',color='r')
# #plt.scatter(np.log10(range1),np.log10(var2),label='random status')
axis.scatter(np.log10(range2),np.log10(var3),label='mra',color='b')
axis.scatter(range3,var4,color='r')
axis.scatter(range3,var5,color='b')
axis.scatter(np.log10(range6),np.log10(var6),color='m',label='kriging - Z direction' )
# axis.scatter(np.log10(250),np.log10(np.var(onetruck_mean_list)),color='green',s=160,label='pseudo truck')
axis.set_xlabel('log10(Tonnage)',fontsize=20)
axis.set_ylabel('log10(Variance)',fontsize=20)
axis.legend(loc='lower left',fontsize=20)


























