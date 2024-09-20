import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import more_itertools as mit
import csv
import glob
import os
from datetime import datetime
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
    #df1.to_csv(j,index=False)
df1_alldata = pd.concat(df1_sub)
df1_date = df1_alldata['date'].unique()



df2 = pd.read_excel('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy.xlsx')
df2 = df2.drop(['DATE','END_TS','DATE1','TIME1','TIMESEC'],axis=1)
df2['TIME1'] = df2['TIME'].astype(str)
df2 = df2.dropna(subset=['MID_X'])
df2.insert(0, 'date', df2.TIME1.str[0:10])
df2.insert(1, 'time', pd.Series([val.time() for val in df2['TIME']]))
df2.insert(2, 'hour', df2.TIME1.str[10:13].astype(int))
df2.insert(3, 'minute', df2.TIME1.str[14:16].astype(int))
df2.insert(4, 'second', df2.TIME1.str[17:19].astype(int))
df2.insert(5, 'timestamp', df2['hour']*3600 + df2['minute']*60 + df2['second'])
df2 = df2.drop(['hour','minute','second','TIME1'],axis=1)
df2 = df2[2194:]
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
    

list2 = []
locations_all = []
for df_a, df_b in zip(df11,df22):
    df_a = df_a.reset_index(drop=True)
    df_b = df_b.reset_index(drop=True)
    list1 = []
    locations = []
    for i in range(df_b.shape[0]):
        timestamp = df_b['timestamp'][i:i+1].values[0]
        location =  df_a[(pd.to_numeric(df_a["timestamp"], errors='coerce')>(timestamp+300)) & (pd.to_numeric(df_a["timestamp"], errors='coerce')<(timestamp+300+360))]
        locations.append(location)
        if location.shape[0]>0:
            list1.append(i)
    list2.append(list1)
    locations_all.append(locations)











