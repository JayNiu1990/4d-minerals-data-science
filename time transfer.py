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
    #df1.to_csv(j,index=False)
df1_alldata = pd.concat(df1_sub)
df1_date = df1_alldata['date'].unique()


####stockpile
df2 = pd.read_excel('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy.xlsx')
df2 = df2.drop(['DATE','END_TS','DATE1','TIME1','TIMESEC'],axis=1)
df2['TIME1'] = df2['TIME'].astype(str)
df2 = df2[df2['LOAD_LOCATION_SNAME']=='FINGER_9']
df2 = df2.reset_index(drop=True)
#df2 = df2.dropna(subset=['MID_X'])
df2.insert(0, 'date', df2.TIME1.str[0:10])
df2.insert(1, 'time', pd.Series([val.time() for val in df2['TIME']]))
df2.insert(2, 'hour', df2.TIME1.str[10:13].astype(int))
df2.insert(3, 'minute', df2.TIME1.str[14:16].astype(int))
df2.insert(4, 'second', df2.TIME1.str[17:19].astype(int))
df2.insert(5, 'timestamp', df2['hour']*3600 + df2['minute']*60 + df2['second'])
df2 = df2.drop(['hour','minute','second','TIME1'],axis=1)
#df2 = df2[2194:]
df2 = df2[833:]
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


    
delay = [300,600,900,1200]#5mins(300s) 10mins(600s) 15mins(900s) 20mins(1200s) 
#tonnage = 300 #250T = 6mins
for n in delay:   
    list2 = []
    locations_all = []
    for df_a, df_b in zip(df11,df22):
        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)
        list1 = []
        locations = []
        for i in range(df_b.shape[0]):
            timestamp = df_b['timestamp'][i:i+1].values[0]
            location =  df_a[(pd.to_numeric(df_a["timestamp"], errors='coerce')>(timestamp+n)) & (pd.to_numeric(df_a["timestamp"], errors='coerce')<(timestamp+n+300))]
            locations.append(location)
            if location.shape[0]>0:
                list1.append(i)
        list2.append(list1)
        locations_all.append(locations)
    exec(f'mean_truck_grade_{n}s = []')
    for i in range(len(locations_all)):
        for j in locations_all[i]:
            if len(j)>0:
                if j[j['grade']>0]['grade'].mean()>0:
                    globals()['mean_truck_grade_'+str(n)+'s'].append(j[j['grade']>0]['grade'].mean())
            
                
            
plt.figure(figsize=(20,12))                
plt.hist(mean_truck_grade_300s,histtype='step',bins=50,label='300s delay')
plt.hist(mean_truck_grade_600s,histtype='step',bins=50,label='600s delay')
plt.hist(mean_truck_grade_900s,histtype='step',bins=50,label='900s delay')
plt.hist(mean_truck_grade_1200s,histtype='step',bins=50,label='1200s delay')
plt.legend(fontsize=24)
plt.xlabel('mean grade',fontsize=24)
plt.ylabel('frequency',fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24) 






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


####stockpile
df2 = pd.read_excel('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy.xlsx')

df2 = df2.drop(['DATE','END_TS','DATE1','TIME1','TIMESEC'],axis=1)
df2['TIME1'] = df2['TIME'].astype(str)
df2 = df2.reset_index(drop=True)
df2 = df2.dropna(subset=['MID_X'])
df2.insert(0, 'date', df2.TIME1.str[0:10])
df2.insert(1, 'time', pd.Series([val.time() for val in df2['TIME']]))
df2.insert(2, 'hour', df2.TIME1.str[10:13].astype(int))
df2.insert(3, 'minute', df2.TIME1.str[14:16].astype(int))
df2.insert(4, 'second', df2.TIME1.str[17:19].astype(int))
df2.insert(5, 'timestamp', df2['hour']*3600 + df2['minute']*60 + df2['second'])
df2 = df2.drop(['hour','minute','second','TIME1'],axis=1)
df2 = df2[2194:]
# df2 = df2[(pd.to_numeric(df2["MID_X"], errors='coerce')>=3000)& (pd.to_numeric(df2["MID_X"], errors='coerce')<4000)
#         & (pd.to_numeric(df2["MID_Y"], errors='coerce')>=12500)& (pd.to_numeric(df2["MID_Y"], errors='coerce')<13000)]
df2 = df2[(pd.to_numeric(df2["TCU"], errors='coerce')>0)]
df2 = df2.reset_index(drop=True)

fig = px.scatter_3d(df2, x="MID_X",y="MID_Y",z="MID_Z",color="TCU")
fig.update_traces(marker_size=4)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show()  



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


    
delay = [300,600,900,1200]#5mins(300s) 10mins(600s) 15mins(900s) 20mins(1200s) 
#tonnage = 300 #250T = 6mins
for n in delay:   
    list2 = []
    locations_all = []
    for df_a, df_b in zip(df11,df22):
        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)
        list1 = []
        locations = []
        for i in range(df_b.shape[0]):
            timestamp = df_b['timestamp'][i:i+1].values[0]
            location =  df_a[(pd.to_numeric(df_a["timestamp"], errors='coerce')>(timestamp+n)) & (pd.to_numeric(df_a["timestamp"], errors='coerce')<(timestamp+n+300))]
            locations.append(location)
            if location.shape[0]>0:
                list1.append(i)
        list2.append(list1)
        locations_all.append(locations)
    exec(f'mean_truck_grade_{n}s = []')
    for i in range(len(locations_all)):
        for j in locations_all[i]:
            if len(j)>0:
                if j[j['grade']>0]['grade'].mean()>0:
                    globals()['mean_truck_grade_'+str(n)+'s'].append(j[j['grade']>0]['grade'].mean())
#####block data
df2_copy = df2
df2_copy = df2_copy.drop(['time','timestamp','TIME','WET_TONNES','TONNES'],axis=1)
df2_copy_unique = df2_copy.drop_duplicates(subset=['BLOCK_SNAME'],keep='first')


path1 = list(os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\'))
path2 = [ x[:-4] for x in path1]

def common_elements(list1, list2):
    result = []
    for element in list1:
        if element in list2:
            result.append(element)
    return result

same_list = common_elements(path2,list(df2_date[0]))
####pseudo truck data
pseudo_truck = []
for i in same_list:
    print('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\'+i+'.csv')
    pseudo_truck.append(pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\'+i+'.csv',sep=" "))
pseudo_truck1 = []
for j in pseudo_truck:
    for i in range(len(j)):
        pseudo_truck1.append(j['Individual'][i].split(','))
pseudo_truck3 = []

for i in pseudo_truck1:
    if len(i)>75:
        pseudo_truck2 = []
        for j in i:
            pseudo_truck2.append(float(j))
        pseudo_truck3.append(pseudo_truck2)    
pseudo_truck3_mean = [np.mean(x) for x in pseudo_truck3]        

a1 = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\2021-10-26.csv',sep=" ")


variance_pseudo_truck3_mean = np.var(pseudo_truck3_mean) #truck scale - 200-300T




import itertools
pseudo_truck3_mra = list(itertools.chain(*pseudo_truck3))


plt.figure(figsize=(14,8))                
# plt.hist(mean_truck_grade_300s,histtype='step', density=True,color = 'orange',bins=50,label='MRA - 300s delay')
# plt.hist(mean_truck_grade_600s,histtype='step', density=True,color = 'b',bins=50,label='MRA - 600s delay')
# plt.hist(mean_truck_grade_1200s,histtype='step', density=True,color = 'k',bins=50,label='MRA - 1200s delay')
#plt.hist(mean_truck_grade_900s,histtype='step', density=True,color = 'g',bins=50,label='MRA - 900s delay')
plt.hist(pseudo_truck3_mean,histtype='step', density=True,color = 'r',bins=50,label='pseudo truck')
# plt.hist(df2_copy_unique['TCU'],histtype='step', density=True,color = 'm',bins=50,label='block')
plt.hist(pseudo_truck3_mra,histtype='step', density=True,color = 'c',bins=50,label='MRA 4s')
plt.legend(fontsize=24)
plt.xlabel('mean grade',fontsize=24)
plt.ylabel('frequency',fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24) 


















