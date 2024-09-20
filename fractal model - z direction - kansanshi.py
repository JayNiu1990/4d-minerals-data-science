import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
#fields = ['X','Y','Z','CU','BHID']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df = df.dropna()

df = df[(pd.to_numeric(df["X"], errors='coerce')>=3000)& (pd.to_numeric(df["X"], errors='coerce')<4000)
        & (pd.to_numeric(df["Y"], errors='coerce')>=12500)& (pd.to_numeric(df["Y"], errors='coerce')<13000)
        & (pd.to_numeric(df["Z"], errors='coerce')>=1200)& (pd.to_numeric(df["Z"], errors='coerce')<=1400)]
df = df.reset_index(drop=True)
df = df.sample(frac=1)
df = df.reset_index(drop=True)
df_filter = df.groupby(['BHID']).filter(lambda x: len(x)>=1)
df_filter = df_filter.reset_index(drop=True)
df_filter['logCu'] = np.log10(df_filter['CU'])

name = df_filter['BHID'].unique()

# fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CU")
# fig.update_traces(marker_size=2)
# fig.update_layout(font=dict(size=14))
# fig.update_layout(scene_aspectmode='data')
# fig.show()  

#############RANDOM#####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
# var2 = []
# for n1 in range(1,2,1):
#     data = []
#     mean = []
#     subgroup = [df_filter[i:i+n1] for i in range(0,len(df_filter),1)]
#     subgroup = [x for x in subgroup if len(x)==n1]
#     data.extend(subgroup)
#     for sub_data in data:
#         mean.append(sub_data['CU'].mean())
#     var2.append(np.var(mean))
name = list(df_filter['BHID'].unique())
var2 = []
for n1 in range(1,16,1):
    data = []
    for i in name:
        each_borecore = df_filter[df_filter['BHID']==i]
        sub_data = [each_borecore[i:i+n1]['CU'] for i in range(0,len(each_borecore),n1)]
        sub_data = [x for x in sub_data if len(x)==n1]
        data.extend(sub_data)
    var2.append(np.var([np.mean(j) for j in data]))
 








##########mra##########
import matplotlib.pyplot as plt
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
for j in range(1,1001,5):
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
np.log10(scale)
mass =  3.14*(0.1**2)*1*2.8 #(T)# 0.785#
range1 = np.arange(1*mass,16*mass-0.001,1*mass)          
range2 = np.arange(1*scale,1001*scale-0.1,5*scale)     
log_range2 = np.log10(range2)
log_var3 = np.log10(var3)
df2 = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\scale\\kansanshi_largescale.csv')
range3 = np.array(df2['scale'])
var4 = np.array(df2['bore core large scale'])
var5 = np.array(df2['mra large scale'])

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
from scipy import stats
df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")

#df1 = stats.zscore(df1)
onetruck_list = []
onetruck_mean_list = []
for i in range(len(df1)):
    onetruck1 = list(df1[i:i+1].values[0])
    onetruck2 = [x for x in onetruck1 if str(x) != 'nan']
    onetruck_list.append(onetruck2)
    onetruck_mean_list.append(np.mean(onetruck2))

range1_log = np.log10(range1)
range2_log = np.log10(range2)
var2_log = np.log10(var2)
var3_log = np.log10(var3)



fig,axis = plt.subplots(1,1,figsize=(12,8))
axis.scatter(np.log10(range1),np.log10(var2),label='bore core',color='r')
# #plt.scatter(np.log10(range1),np.log10(var2),label='random status')
#axis.scatter(np.log10(range2),np.log10(var3),label='mra',color='b')
# axis.scatter(range3,var4,color='r')
# axis.scatter(range3,var5,color='b')
# axis.scatter(np.log10(250),np.log10(np.var(onetruck_mean_list)),color='green',s=160,label='pseudo truck')
axis.set_xlabel('log10(Tonnage)',fontsize=20)
axis.set_ylabel('log10(Variance)',fontsize=20)
axis.legend(loc='upper right',fontsize=28)

scale = np.log10(range1)
variance = np.log10(var2)
# fig,axis = plt.subplots(1,1,figsize=(12,8))
# axis.scatter(range1,var2,label='bore core',color='r')
# axis.scatter(range2,var3,label='mra',color='b')
# # axis.scatter(range3,var4,color='r')
# # axis.scatter(range3,var5,color='b')
# axis.set_xlabel('Tonnage',fontsize=20)
# axis.set_ylabel('Variance',fontsize=20)
# axis.legend(loc='upper right',fontsize=28)

name = df_filter['BHID'].unique()
fig,axis = plt.subplots(1,1,figsize=(16,10))
axis.plot(df_filter[df_filter['BHID'] == name[30]]['CU'].reset_index(drop=True))
axis.tick_params(axis='both', which='major', labelsize=24)
axis.set_xlabel('length',fontsize=24)
axis.set_ylabel('grade',fontsize=24)
axis.set_title('Bore core ID:' + name[0],fontsize=24)


x11 = np.log10(range2)
y11 = np.log10(var3)












