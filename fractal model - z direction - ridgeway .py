import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import math
pio.renderers.default='browser'
with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\ridgewaydeeps.txt') as f:
    lines1 = f.readlines()
list1 = []
for line1 in lines1[1:]:
    line = line1.split()
    row = np.array(line[0:7])
    list1.append(row)
borecore = pd.DataFrame(list1,columns=['SAMPLEID','HOLEID','PROJECTCODE','SAMPLEFROM','SAMPLETO','Au_ppm_BEST','Cu_ppm_BEST'])
borecore['Cu_ppm_BEST'] =borecore['Cu_ppm_BEST'].astype('float')
borecore['CU_wt'] = borecore['Cu_ppm_BEST']/10000
borecore = borecore.groupby(['HOLEID']).filter(lambda x: len(x)>=0)

borecore.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\ridgewaydeeps.csv')
borecore = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\ridgewaydeeps.csv')
name = list(borecore['HOLEID'].unique())



# var2 = []
# for n1 in range(1,151,1):
#     data = []
#     mean = []
#     subgroup = [borecore[i:i+n1] for i in range(0,len(borecore),1)]
#     subgroup = [x for x in subgroup if len(x)==n1]
#     data.extend(subgroup)
#     for sub_data in data:
#         mean.append(sub_data['CU_wt'].mean())
#     var2.append(np.var(mean))


var2 = []
for n1 in range(1,151,1):
    data = []
    for i in name:
        each_borecore = borecore[borecore['HOLEID']==i]
        sub_data = [each_borecore[i:i+n1]['CU_wt'] for i in range(0,len(each_borecore),n1)]
        data.extend(sub_data)
    var2.append(np.var([np.mean(j) for j in data]))
        
mra = pd.read_excel("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\fs03a.xlsx")
mra.to_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\fs03a.csv",index=False)

mra = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\fs03a.csv")
mra = mra* 0.3448
mra_grade = []
for i in range(1,30,1):
    sub_data = list(mra[str(i)])
    sub_data = [x for x in sub_data if str(x) != 'nan']
    mra_grade.extend(sub_data)

import matplotlib.pyplot as plt    
# plt.hist(np.log10(borecore['CU_wt']),bins=100,label='bore core',density=True)
# plt.hist(np.log10(mra_grade),bins=100,label='mra',density=True)
# plt.legend()

mra = pd.DataFrame(mra_grade,columns=['grade'])
var3 = []
list_mean_subgroup1 = []
for j in range(20,1001,20):
    data = []
    mean = []
    subgroup1 = [mra[n:n+j] for n in range(0,len(mra),j)]
    subgroup1 = [x for x in subgroup1 if len(x)==j]
    data.extend(subgroup1)
    for sub_data in data:
        mean.append(sub_data['grade'].mean())
    var3.append(np.var(mean))
    
import matplotlib.pyplot as plt    
scale = 20* 2000/3600   #(T) 
mass = 3.14*(0.1**2)*1*2.65 #(T)


range2 = np.arange(1*mass,151*mass-0.001,1*mass)       
range3 = np.arange(20*scale,1001*scale-0.1,20*scale)   
fig,axis = plt.subplots(1,1,figsize=(12,8))
axis.scatter(np.log10(range3),np.log10(var3),label='mra',color='b')
axis.scatter(np.log10(range2),np.log10(var2),label='bore core',color='r')
axis.set_xlabel('log10(Tonnage)',fontsize=20)
axis.set_ylabel('log10(variance)',fontsize=20)
axis.tick_params(axis='both', which='major', labelsize=20)
axis.legend(loc='upper right',fontsize=28)

# import numpy as np
# from sklearn.linear_model import LinearRegression

# X1 = np.log10(range3).reshape(-1, 1)
# Y1 = np.log10(var3)
# model = LinearRegression()
# model.fit(X1, Y1)
# Y1_pred = model.predict(X1)


# X2 = np.log10(range2).reshape(-1, 1)
# Y2 = np.log10(var2)
# model = LinearRegression()
# model.fit(X2, Y2)
# Y2_pred = model.predict(X2)


# plt.scatter(np.log10(range3),np.log10(var3),label='mra')
# plt.plot(X1, Y1_pred, color='red', label="Linear Fit")
# plt.plot(X2, Y2_pred, color='green', label="Linear Fit")
# plt.scatter(np.log10(range2),np.log10(var2),label='bore core')
# plt.xlabel('log10(T)')
# plt.ylabel('log10(variance)')
# plt.legend()


###########coordinate transform###########
import pandas as pd
import numpy as np
import math
with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\rwcollarstotal.txt') as f:
    lines1 = f.readlines()

list1 = []
for line1 in lines1:
    line = line1.split()
    row = np.array(line[0:12])
    list1.append(row)
data1 = pd.DataFrame(list1,columns=['NAME','X','Y','Z','DEPTH','AZIMUTH','DIP'])

borecore = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Ridgeway\\ridgewaydeeps.csv')
borecore
name = borecore['HOLEID'].unique()
data_list = []

borecore = borecore.drop('Unnamed: 0',axis=1)

for str1 in name:
    data2 = borecore[borecore['HOLEID']==str1]
    
    AZIMUTH = list(data1[data1['NAME']==str1]['AZIMUTH'])[0].astype('float64')
    DIP = list(data1[data1['NAME']==str1]['DIP'])[0].astype('float64')
    X = list(data1[data1['NAME']==str1]['X'])[0].astype('float64')
    Y = list(data1[data1['NAME']==str1]['Y'])[0].astype('float64')
    Z = list(data1[data1['NAME']==str1]['Z'])[0].astype('float64')
    
    
    data2['X'] = round(X + ((data2['SAMPLEFROM'].astype('float64')+data2['SAMPLETO'].astype('float64'))*0.5 * math.sin(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
    data2['Y'] = round(Y + ((data2['SAMPLEFROM'].astype('float64')+data2['SAMPLETO'].astype('float64'))*0.5 * math.cos(math.radians(AZIMUTH)) * math.cos(math.radians(DIP))),3)
    data2['Z'] = round(Z + ((data2['SAMPLEFROM'].astype('float64')+data2['SAMPLETO'].astype('float64'))*0.5 * math.sin(math.radians(DIP))),3)
    data_list.append(data2)
df = pd.concat(data_list)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import math
pio.renderers.default='browser'

fig = px.scatter_3d(df, x="X",y="Y",z="Z",color='CU_wt')
fig.update_traces(marker_size=2)
fig.update_layout(font=dict(size=14))
fig.update_layout(scene_aspectmode='data')
fig.show() 








