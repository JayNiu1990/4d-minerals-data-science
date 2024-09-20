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
pio.renderers.default='browser'
path = r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Kansanshi MRA Time'
list_sorted_path_ordered_csv1 = []
list_sorted_path_ordered_csv2 = []
###2021 10-12
for k in range(1,2,1):      
    for j in range(1,2,1):
        #for i in range(1,10,1):
        for i in range(0,3,1):
            if j==1 and i>2:
                break
            else:
                pattern = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.csv'
                pathall = str('%s%s'%(path,pattern))
                sorted_path = glob.glob(pathall)
                print(k,j,i)
                #print(sorted_path)
                list_sorted_path_ordered_csv1.extend(sorted_path)
                list_sorted_path_ordered_csv2.extend(sorted_path)
                
list_sorted_path_ordered_csv2 = [w.replace('Kansanshi MRA Time','Kansanshi MRA Time1') for w in list_sorted_path_ordered_csv1]
df1_sub = []
for i,j in zip(list_sorted_path_ordered_csv1,list_sorted_path_ordered_csv2):
    df1 = pd.read_csv(str(i),names=['hour','minute','second','tonnage','grade'])
    df1['timestamp'] = df1['hour']*3600 + df1['minute']*60 + df1['second']
    df1['time'] = pd.to_datetime(df1["timestamp"], unit='s').dt.strftime("%H:%M:%S")
    df1['datetime'] = i[-9:-5] + '-' + i[-12:-10]  + '-' + i[-15:-13] + ' ' + df1['time']
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    df1['date'] = i[-9:-5] + '-' + i[-12:-10]  + '-' + i[-15:-13]
    df1 = df1.drop(['hour','minute','second'],axis=1)
    df1 = df1.iloc[:,[4,5,3,2,0,1]]
    df1['datetime'] = df1['datetime'].astype(str)
    df1_sub.append(df1)
    df1.to_csv(j,index=False)
df1_alldata = pd.concat(df1_sub)
df1_date = df1_alldata['date'].unique()


####stockpile
df2 = pd.read_excel('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy.xlsx')
df2 = df2.drop(['DATE','END_TS','DATE1','TIME1','TIMESEC'],axis=1)
df2['TIME1'] = df2['TIME'].astype(str)
#df2 = df2.dropna()
#df2 = df2[df2['LOAD_LOCATION_SNAME']=='FINGER_9']
df2 = df2.reset_index(drop=True)
#df2 = df2.dropna(subset=['MID_X'])
df2.insert(0, 'date', df2.TIME1.str[0:10])
df2.insert(1, 'time', pd.Series([val.time() for val in df2['TIME']]))
df2.insert(2, 'hour', df2.TIME1.str[10:13].astype(int))
df2.insert(3, 'minute', df2.TIME1.str[14:16].astype(int))
df2.insert(4, 'second', df2.TIME1.str[17:19].astype(int))
df2.insert(5, 'timestamp', df2['hour']*3600 + df2['minute']*60 + df2['second'])
df2 = df2.drop(['hour','minute','second','TIME1'],axis=1)
#df2 = df2[1341:]
df2 = df2[2194:]
df2 = df2.reset_index(drop=True)
df2_sub = []
df2_date = df2['date'].unique()
for i in df2_date:
    sub = df2[df2['date']==i]
    df2_sub.append(sub)
    
df1_date = pd.DataFrame(df1_date)    
df2_date = pd.DataFrame(df2_date)
date = df2_date.merge(df1_date)[0].tolist()
df11 = []  #MR
df22 = []  #truck
for i in date:
    sub1 = df1_alldata[df1_alldata['date']==i]
    sub2 = df2[df2['date']==i]
    df11.append(sub1)
    df22.append(sub2)

# delay = [600]#5mins(300s) 10mins(600s) 15mins(900s) 20mins(1200s) 
# #tonnage = 300 #250T = 6mins
# for n in delay:   
#     list2 = []
#     locations_all = []
#     for df_a, df_b in zip(df11,df22):
#         df_a = df_a.reset_index(drop=True)
#         df_b = df_b.reset_index(drop=True)
#         list1 = []
#         locations = []
#         for i in range(df_b.shape[0]):
#             timestamp = df_b['timestamp'][i:i+1].values[0]
#             location =  df_a[(pd.to_numeric(df_a["timestamp"], errors='coerce')>(timestamp)) & (pd.to_numeric(df_a["timestamp"], errors='coerce')<(timestamp+3000))]
#             locations.append(location)
#             if location.shape[0]>0:
#                 list1.append(i)
#         list2.append(list1)
#         locations_all.append(locations)


# MRA = pd.DataFrame(locations_all[0][94]['tonnage']).reset_index(drop=True)
# fig, ax = plt.subplots(1, 1,figsize=(18,6))
# ax.plot(MRA['tonnage'],'b')
# ax.set_xlabel('data count',fontsize=18)
# ax.set_ylabel('Cu tonnage',fontsize=18)
# ax.tick_params(axis='both', which='major', labelsize=18)

#plt.plot(np.diff(MRA['grade']))
fig = px.scatter_3d(df2, x="MID_X",y="MID_Y",z="MID_Z",color="TCU")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=16))
fig.update_layout(scene_aspectmode='data')
fig.show()  
# df2_1 = df2[['date','time','BLOCK_SNAME','TCU']]
#locations_all[0][94].reset_index(drop=True)[699:700]
#date1 = date[0:3] + date[5:21] + date[23:35] + date[36:43] ##remove abnormal data
date1 = date[8:14]
date2 = date1[20:21] + date1[23:24]  + date1[25:38]   ##select date which needs to be adjusted for tonnage
############revise tonnage for date2##############




# import more_itertools as mit
# total_tonnage_mra = []
# total_tonnage_truck = []
# for j in range(len(date1)):
#     MRA_subdf = df1_alldata[df1_alldata['date']==date1[j]].reset_index(drop=True)
#     # idx = np.where(MRA_subdf['tonnage']>1500)
#     # idx_list = []
#     # for i in mit.consecutive_groups(idx[0]):
#     #     idx_list.append(list(i))
#     # tonnage_list = []
#     # for i in range(len(idx_list)):
#     #     tonnage_list.append(MRA_subdf.loc[MRA_subdf.index[idx_list[i]]])
#     # for i in range(len(tonnage_list)):
#     #     tonnage_list[i]
#     ###total tonnage for Oct 10
#     total_tonnage_mra.append(len(MRA_subdf[MRA_subdf.tonnage>500])* 4* (MRA_subdf[MRA_subdf.tonnage>500]['tonnage'].mean()/3600))
#     total_tonnage_truck.append(df2[df2['date']==date1[j]]['TONNES'].sum())

# print(np.sum(total_tonnage_mra))
# print(np.sum(total_tonnage_truck))

sub_df = df1_alldata[df1_alldata['date']==date[36]]
fig, ax = plt.subplots(1, 1,figsize=(18,6))
ax.plot(sub_df['tonnage'],'b')
ax.set_xlabel('data count',fontsize=18)
ax.set_ylabel('Cu tonnage',fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)

####change name###
for i in list_sorted_path_ordered_csv2:
    year = i[-9:-5]
    month = i[-12:-10]
    day = i[-15:-13]
    os.rename(i, 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time2\\' + str(year)+'-'+str(month)+'-'+str(day)+'.csv' )
#####change tonnage for date2####
for i in date2:
    df3 = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time2\\' +i +'.csv')
    df3['tonnage'] = df3['tonnage']*2
    df3['grade'] = df3['grade']/2
    df3.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time2\\' +i +'.csv')


####filter truck gps date Oct 20-25###
df2 = df2[53:13043].reset_index(drop=True)
#######MRA tonnage match truck tonnage in total#####
import more_itertools as mit
total_tonnage_mra = []
total_tonnage_truck = []
for j in range(len(date1)):
    MRA_subdf = df1_alldata[df1_alldata['date']==date1[j]].reset_index(drop=True)
    if j==0:
        MRA_subdf = MRA_subdf[57:].reset_index(drop=True)
        total_tonnage_mra.append(len(MRA_subdf[MRA_subdf.tonnage>1000])* 4* (MRA_subdf[MRA_subdf.tonnage>1000]['tonnage'].mean()/3600))
        total_tonnage_truck.append(df2[df2['date']==date1[j]]['TONNES'].sum())
    else:
        total_tonnage_mra.append(len(MRA_subdf[MRA_subdf.tonnage>1000])* 4* (MRA_subdf[MRA_subdf.tonnage>1000]['tonnage'].mean()/3600))
        total_tonnage_truck.append(df2[df2['date']==date1[j]]['TONNES'].sum())
    # idx = np.where(MRA_subdf['tonnage']>1500)
    # idx_list = []
    # for i in mit.consecutive_groups(idx[0]):
    #     idx_list.append(list(i))
    # tonnage_list = []
    # for i in range(len(idx_list)):
    #     tonnage_list.append(MRA_subdf.loc[MRA_subdf.index[idx_list[i]]])
    # for i in range(len(tonnage_list)):
    #     tonnage_list[i]
    ###total tonnage for Oct 10
    
    
print(np.sum(total_tonnage_mra))
print(np.sum(total_tonnage_truck))

Oct_20_25_mra = []
for i in date1:
   df3 = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time2\\' +i +'.csv')
   Oct_20_25_mra.append(df3)
Oct_20_25_mra = pd.concat(Oct_20_25_mra)
Oct_20_25_mra = Oct_20_25_mra[57:].reset_index(drop=True)
Oct_20_25_mra = Oct_20_25_mra[Oct_20_25_mra.tonnage>1000].reset_index(drop=True)

Oct_20_25_mra['time period'] = (Oct_20_25_mra.index+1)*4

#MRA_subdf = df3_alldata[df3_alldata['date']==date1[0]].reset_index(drop=True)
Oct_20_25_truck = df2[2908:4475].reset_index(drop=True)
average_tonnage = Oct_20_25_mra['tonnage'].mean()

Oct_20_25_truck['Time by truck(s)'] = (Oct_20_25_truck['TONNES']*3600)/average_tonnage ##second (unit)
Oct_20_25_truck['Time by truck(4s)'] = np.round(Oct_20_25_truck['Time by truck(s)'] /4).astype('int32')
Oct_20_25_truck['Time by truck(4s) sum'] = Oct_20_25_truck['Time by truck(4s)'].cumsum()

mra_segment = []
delay=146 
for i in range(len(Oct_20_25_truck['Time by truck(4s) sum'])):
    if i==0:
        mra_segment.append(Oct_20_25_mra[delay:(delay + Oct_20_25_truck['Time by truck(4s) sum'][i])])
    else:
        mra_segment.append(Oct_20_25_mra[(delay+Oct_20_25_truck['Time by truck(4s) sum'][i-1]):(delay+Oct_20_25_truck['Time by truck(4s) sum'][i])])
mra_segment_average_grade = []
for i in mra_segment:
    mra_segment_average_grade.append(i['grade'].mean())
    
Oct_20_25_truck['grade by mra'] = mra_segment_average_grade    
plt.scatter(Oct_20_25_truck['TCU'],Oct_20_25_truck['grade by mra']) 
plt.xlabel('block grade')
plt.ylabel('mra mean grade')

plt.hist(Oct_20_25_truck['TCU'],bins=100,label='block grade')
plt.hist(Oct_20_25_truck['grade by mra'],bins=100,label='mra mean grade')
plt.legend()

# mra_segment = []
# for i in range(len(Oct_20_25_truck['Time by truck(4s) sum'])):
#     if i==0:
#         mra_segment.append(Oct_20_25_mra[0:(0 + Oct_20_25_truck['Time by truck(4s) sum'][i])])
#     else:
#         mra_segment.append(Oct_20_25_mra[(0+Oct_20_25_truck['Time by truck(4s) sum'][i-1]):(0+Oct_20_25_truck['Time by truck(4s) sum'][i])])

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
#fields = ['X','Y','Z','CU','BHID']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df = df.dropna()
df = df[(pd.to_numeric(df["X"], errors='coerce')>=3150)& (pd.to_numeric(df["X"], errors='coerce')<=3250)
        & (pd.to_numeric(df["Y"], errors='coerce')>=12600)& (pd.to_numeric(df["Y"], errors='coerce')<=12700)
        & (pd.to_numeric(df["Z"], errors='coerce')>=1300)& (pd.to_numeric(df["Z"], errors='coerce')<=1320)]





