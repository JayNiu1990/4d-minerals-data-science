import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import math

with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia East\\CE_Collarsmod.txt') as f:
    lines1 = f.readlines()
list1 = []
for line1 in lines1[1:]:
    line = line1.split()
    row = np.array(line[0:12])
    list1.append(row)
data1 = pd.DataFrame(list1,columns=['NAME','REGION','DRILLHOLE','X','Y','Z','DEPTH','DATE1','DATE2','D','AZIMUTH','DIP'])

# str_list = ["UE035","UE041","UE040","UE055","UE054","UE056","UE100","UE101","UE099",
#  "UE102","UE051","UE049","UE050","UE048","UE047","UE103","UE097","UE104",
#  "UE096","UE018","UE017","UE042","UE043","UE044","UE045","UE046","UE092",
#  "UE095","UE113","UE090","UE091A","UE094","UE013","UE011","UE009","UE010",
#  "UE036","UE019A","UE037","UE020","UE022","UE021","UE023","UE024","UE025",
#  "UE026","UE027","UE028","UE029","UE014","UE012","UE015"]
str_list = list(data1['NAME'].unique())
#str_list.sort()

data_list = []
for _ in str_list:
    str1 = _
    AZIMUTH = list(data1[data1['NAME']==str1]['AZIMUTH'])[0].astype('float64')
    DIP = list(data1[data1['NAME']==str1]['DIP'])[0].astype('float64')
    X = list(data1[data1['NAME']==str1]['X'])[0].astype('float64')
    Y = list(data1[data1['NAME']==str1]['Y'])[0].astype('float64')
    Z = list(data1[data1['NAME']==str1]['Z'])[0].astype('float64')
    
    with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia East\\all_data.txt') as f:
        lines2 = f.readlines()
    
    list2 = []
    for line2 in lines2[1:]:
        line = line2.split()
        row = np.concatenate((np.array(line[0:6]),np.array(line[11:12])))
        list2.append(row)
        
    data2 = pd.DataFrame(list2,columns=['SAMPLE','HOLEID','PROJECTCODE','FROM','TO','AU_ppm','CU_ppm'])
    data2 = data2.dropna()
    data2 = data2[data2['HOLEID']==str1]
    data_list.append(data2)
    data2['X'] = round(X + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.sin(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
    data2['Y'] = round(Y + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.cos(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
    data2['Z'] = round(Z + ((data2['FROM'].astype('float64')+data2['TO'].astype('float64'))*0.5 * math.sin(math.radians(DIP))),3)
data = pd.concat(data_list)

data = data[(data['HOLEID']!='UE011') & (data['HOLEID']!='UE010')& (data['HOLEID']!='UE009')]

data = data[(pd.to_numeric(data["AU_ppm"], errors='coerce')>0) & (pd.to_numeric(data["CU_ppm"], errors='coerce')>0)]
data['AU_ppm'] = data['AU_ppm'].astype('float')
data['CU_ppm'] = data['CU_ppm'].astype('float')
data['CU_wt'] = data['CU_ppm']/10000

data = data[(pd.to_numeric(data["AU_ppm"], errors='coerce')>0) & (pd.to_numeric(data["CU_wt"], errors='coerce')>=0)]


pio.renderers.default='browser'
# data = data[(pd.to_numeric(data["X"], errors='coerce')>15500) & (pd.to_numeric(data["X"], errors='coerce')<16000) &
#             (pd.to_numeric(data["Y"], errors='coerce')>21500) & (pd.to_numeric(data["Y"], errors='coerce')<22000) &
#             (pd.to_numeric(data["Z"], errors='coerce')>5000) & (pd.to_numeric(data["Z"], errors='coerce')<5500)]
# data = data.reset_index(drop=True) 
  

# data['AU_ppm'] = data['AU_ppm'].astype('float')
# data['CU_ppm'] = data['CU_ppm'].astype('float')
# data['CU_wt'] = data['CU_ppm']/10000
data['log Cu_wt'] = np.log10(data['CU_wt'])
data['log AU_ppm'] = np.log10(data['AU_ppm'])

fig = px.scatter_3d(data, x="X",y="Y",z="Z",color='log Cu_wt')
fig.update_traces(marker_size=2)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show() 


borehole = data['HOLEID'].unique()






df_filter = data.groupby(['HOLEID']).filter(lambda x: len(x)>=20)
bore_hole = data['HOLEID'].unique()


var1 = []
for n1 in range(1,11,1):
    data = []
    mean = []
    for name in df_filter['HOLEID'].unique():
        sub_df = df_filter[df_filter['HOLEID'] == name]
        sub_df = sub_df.sort_values(by=['Z'])
        sub_df = sub_df.reset_index(drop=True)
        subgroup = [sub_df[i:i+n1] for i in range(0,len(sub_df),1)]
        subgroup = [x for x in subgroup if len(x)==n1]
        data.extend(subgroup)
    for sub_data in data:
        mean.append(sub_data['CU_ppm'].mean())
    var1.append(np.var(mean))

mass = 3.14*(0.1**2)*1*2.65 #(T)
range1 = np.arange(1*mass,11*mass-0.001,1*mass)                
        


df_kriging = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Cadia_East\\Cadia_East_kriging.csv", skipinitialspace=True)














