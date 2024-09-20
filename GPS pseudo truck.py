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
df1_alldata = pd.concat(df1_sub)       ### old MRA all data in all days
df1_date = list(df1_alldata['date'].unique())[0:51]
df1_alldata = df1_alldata.reset_index(drop=True)
df1_alldata['datetime'] = pd.to_datetime(df1_alldata['datetime'])
df1_alldata['year'] = df1_alldata['datetime'].dt.year
df1_alldata['month'] = df1_alldata['datetime'].dt.month
df1_alldata['day'] = df1_alldata['datetime'].dt.day
df1_alldata['hour'] = df1_alldata['datetime'].dt.hour
df1_alldata['minute'] = df1_alldata['datetime'].dt.minute
df1_alldata['second'] = df1_alldata['datetime'].dt.second
import datetime
import time
second_list = []
for i in range(len(df1_alldata)):
    dt = datetime.datetime(df1_alldata['year'][i], df1_alldata['month'][i], df1_alldata['day'][i] , df1_alldata['hour'][i], df1_alldata['minute'][i],df1_alldata['second'][i])
    second_list.append(time.mktime(dt.timetuple()))
df1_alldata['total_second'] = second_list










####gps data
df2 = pd.read_excel('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy.xlsx')
df2 = df2.drop(['DATE','END_TS','DATE1','TIME1','TIMESEC'],axis=1)
df2['TIME1'] = df2['TIME'].astype(str)
df2 = df2.dropna()
#df2 = df2[df2['LOAD_LOCATION_SNAME']=='FINGER_9']
df2 = df2.reset_index(drop=True)
df2 = df2.dropna(subset=['MID_X'])
df2.insert(0, 'date', df2.TIME1.str[0:10])
df2.insert(1, 'time', pd.Series([val.time() for val in df2['TIME']]))
df2.insert(2, 'hour', df2.TIME1.str[10:13].astype(int))
df2.insert(3, 'minute', df2.TIME1.str[14:16].astype(int))
df2.insert(4, 'second', df2.TIME1.str[17:19].astype(int))
df2.insert(5, 'timestamp', df2['hour']*3600 + df2['minute']*60 + df2['second'])
df2 = df2.drop(['hour','minute','second','TIME1'],axis=1)
#df2 = df2[1341:]
#df2 = df2[2194:]
df2 = df2.reset_index(drop=True)
df2_sub = []
df2_date = df2['date'].unique()
for i in df2_date:
    sub = df2[df2['date']==i]
    df2_sub.append(sub)
    
df1_date = pd.DataFrame(df1_date)    ###day list for MRA
df2_date = pd.DataFrame(df2_date)    ###day list for gps
#date = df2_date.merge(df1_date)[0].tolist()
date = list(df2['date'].unique())
df11 = []                       ###data for each day in MRA
df22 = []                       ###data for each day in gps
for i in date:
    sub1 = df1_alldata[df1_alldata['date']==i]
    sub2 = df2[df2['date']==i]
    df11.append(sub1)
    df22.append(sub2)





#plt.plot(np.diff(MRA['grade']))
fig = px.scatter_3d(df2, x="MID_X",y="MID_Y",z="MID_Z",color="TCU")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=16))
fig.update_layout(scene_aspectmode='data')
fig.show()  





# df2_1 = df2[['date','time','BLOCK_SNAME','TCU']]
#locations_all[0][94].reset_index(drop=True)[699:700]
date_modified = date[28:29] + date[32:35] + date[36:51] ##select date which needs to be adjusted for tonnage
############revise tonnage for date2##############

import more_itertools as mit
total_tonnage_mra = []
total_tonnage_truck = []
for j in range(len(date)):
    MRA_subdf = df1_alldata[df1_alldata['date']==date[j]].reset_index(drop=True)
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
    total_tonnage_mra.append(len(MRA_subdf[MRA_subdf.tonnage>500])* 4* (MRA_subdf[MRA_subdf.tonnage>500]['tonnage'].mean()/3600))
    total_tonnage_truck.append(df2[df2['date']==date[j]]['TONNES'].sum())
print(np.sum(total_tonnage_mra))
print(np.sum(total_tonnage_truck))
total_tonnage = pd.DataFrame(total_tonnage_mra,columns=['mra total'])
total_tonnage['truck total'] = total_tonnage_truck
# sub_df = df1_alldata[df1_alldata['date']==date[36]]
# fig, ax = plt.subplots(1, 1,figsize=(18,6))
# ax.plot(sub_df['tonnage'],'b')
# ax.set_xlabel('data count',fontsize=18)
# ax.set_ylabel('Cu tonnage',fontsize=18)
# ax.tick_params(axis='both', which='major', labelsize=18)

####change name###
for i in list_sorted_path_ordered_csv2:
    year = i[-9:-5]
    month = i[-12:-10]
    day = i[-15:-13]
    os.rename(i, 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time2\\' + str(year)+'-'+str(month)+'-'+str(day)+'.csv' )
#####change tonnage for date2####
for i in date_modified:
    df3 = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\' +i +'.csv')
    df3['tonnage'] = df3['tonnage']*2
    df3['grade'] = df3['grade']/2
    df3.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\' +i +'.csv',index=False)
###load all normal data####
df2_alldata = []
for i in date:
    MRA_subdf = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi MRA Time3\\' +i +'.csv')
    df2_alldata.append(MRA_subdf)
df2_alldata = pd.concat(df2_alldata)   ### modified MRA all data in all days


#######MRA tonnage match truck tonnage in total#####

import more_itertools as mit
total_tonnage_mra = []
total_tonnage_truck = []
for j in range(len(date)):
    MRA_subdf = df2_alldata[df2_alldata['date']==date[j]].reset_index(drop=True)
    if j==0:
        MRA_subdf = MRA_subdf[57:].reset_index(drop=True)
        total_tonnage_mra.append(len(MRA_subdf[MRA_subdf.tonnage>500])* 4* (MRA_subdf[MRA_subdf.tonnage>500]['tonnage'].mean()/3600))
        total_tonnage_truck.append(df2[df2['date']==date[j]]['TONNES'].sum())
    else:
        total_tonnage_mra.append(len(MRA_subdf[MRA_subdf.tonnage>500])* 4* (MRA_subdf[MRA_subdf.tonnage>500]['tonnage'].mean()/3600))
        total_tonnage_truck.append(df2[df2['date']==date[j]]['TONNES'].sum())
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
total_tonnage1 = pd.DataFrame(total_tonnage_mra,columns=['mra total'])
total_tonnage1['truck total'] = total_tonnage_truck

######estimate average grade for gps and mra data########
mra_average_perday = []
block_average_perday = []
#date1 =date[27:] 
for j in range(len(date)):
    MRA_subdf = df2_alldata[df2_alldata['date']==date[j]].reset_index(drop=True)
    MRA_subdf = MRA_subdf[MRA_subdf.grade>0]
    mra_average_perday.append(MRA_subdf['grade'].mean())
    block_average_perday.append(df2[df2['date']==date[j]]['TCU'].mean())  ###df2 gps data for all days
df_mean = pd.DataFrame()
df_mean['date'] = date
df_mean['block_average_perday'] = block_average_perday
df_mean['mra_average_perday'] = mra_average_perday




fig, ax = plt.subplots(1, 1,figsize=(22,6))
ax.plot(date,block_average_perday, '--bo', label='block daily mean grade')
ax.plot(date,mra_average_perday, '--ro', label='mra daily mean grade')
ax.set_xlabel('date',fontsize=13)
ax.set_ylabel('mean grade',fontsize=18)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=18)




Oct_20_25_mra = []
for i in date:
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
df = df[(pd.to_numeric(df["X"], errors='coerce')>=5060)& (pd.to_numeric(df["X"], errors='coerce')<=5085)
        & (pd.to_numeric(df["Y"], errors='coerce')>=10860)& (pd.to_numeric(df["Y"], errors='coerce')<=10890)
        & (pd.to_numeric(df["Z"], errors='coerce')>=1240)& (pd.to_numeric(df["Z"], errors='coerce')<=1250)]





