# if __name__ == '__main__':
#     import pymc3 as pm
#     import numpy as np
#     import theano
#     import arviz as az
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     theano.config.compute_test_value = "ignore"
#     az.style.use("arviz-darkgrid")
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio
fields = ['BHID', 'X','Y','Z','CuT_dh','Fe_dh','LITH','As_dh']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Escondida Trucks\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
#df = df.dropna()
df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02)]
df['LITH'] = df['LITH'].astype(int)
df["CuT_dh"] = df["CuT_dh"].astype("float")
df["Fe_dh"] = df["Fe_dh"].astype("float")
df["As_dh"] = df["As_dh"].astype("float")
#each_name = [item for item, count in collections.Counter(df["LITH"]).items() if count > 50]
group = df.groupby('BHID')
plt.hist(list(group.size()),bins =np.arange(0,600,1))

fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="LITH")
fig.update_traces(marker_size=3)
fig.update_layout(legend=dict(font=dict(size=38)))
fig.show()

#plt.hist(df.loc[(df['LITH']==each_name[1])]['Fe_dh'],bins=np.arange(0,10,0.1),label='LITH 50',color='black',histtype='step',density=True)

# fig = px.scatter_3d(df[(df['LITH']==50) | (df['LITH']==31)], x="X",y="Y",z="Z",color="LITH")
# fig.update_traces(marker_size=3)
# # #fig.update_layout(legend=dict(font=dict(size=18)))
# fig.show()

# fig = px.scatter_3d(df[(df['LITH']!=50) & (df['LITH']!=31)], x="X",y="Y",z="Z",color="LITH")
# fig.update_traces(marker_size=3)
# # #fig.update_layout(legend=dict(font=dict(size=18)))
# fig.show()
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio

fields = ['BHID', 'X','Y','Z','CU']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\kmp_ddh_mra.csv", skipinitialspace=True, usecols=fields)
#df = df.dropna()
df = df.loc[(pd.to_numeric(df["CU"], errors='coerce')>0.02)&(pd.to_numeric(df["CU"], errors='coerce')<3)]
group = df.groupby('BHID')
plt.hist(list(group.size()),bins =np.arange(0,200,1))
plt.xlabel('numbers of points in each bore hole',fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

df = df.groupby('BHID').filter(lambda x:len(x)>50)
# df['LITH'] = df['LITH'].astype(int)
df["CU"] = df["CU"].astype("float")
#each_name = [item for item, count in collections.Counter(df["BHID"]).items() if count > 50]

fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=3)
fig.update_layout(font=dict(size=18))
fig.show()

df1 = np.array(df.drop(['BHID','CU'],axis=1))
df1 = df1[0:10]
np.set_printoptions(suppress=True)



import numpy as np

xx = np.linspace(5495, 5507, 13)
yy = np.linspace(11255, 11259, 5)
zz = np.linspace(1348, 1356, 9)
import matplotlib.pyplot as plt
coordinate = np.vstack(np.meshgrid(xx, yy,zz, indexing='ij')).reshape(3,-1).T
from scipy.spatial import KDTree
tree = KDTree(coordinate)
distances, points = tree.query(df1[0], k=1)














plt.hist(df[df['LITH']==50]['CuT_dh'],bins=np.arange(0,6,0.1),label='LITH 50',color='black',histtype='step',density=True)
plt.hist(df[df['LITH']==31]['CuT_dh'],bins=np.arange(0,6,0.1),label='LITH 31',color='blue',histtype='step',density=True)
#plt.hist(df[(df['LITH']!=50) & (df['LITH']!=31)]['CuT_dh'],bins=np.arange(0,6,0.1),label='Other LITH',color='red',histtype='step',density=True)
plt.hist(df['CuT_dh'],bins=np.arange(0,6,0.1),label='All LITH',color='m',histtype='step',density=True)
plt.title('CuT histogram')
plt.xlabel('CuT')
plt.ylabel('density')
plt.legend()
plt.show()

plt.hist(df[df['LITH']==50]['Fe_dh'],bins=np.arange(0,20,0.1),label='LITH 50',color='black',histtype='step',density=True)
plt.hist(df[df['LITH']==31]['Fe_dh'],bins=np.arange(0,20,0.1),label='LITH 31',color='blue',histtype='step',density=True)
#plt.hist(df[(df['LITH']!=50) & (df['LITH']!=31)]['CuT_dh'],bins=np.arange(0,20,0.1),label='Other LITH',color='red',histtype='step',density=True)
plt.hist(df['Fe_dh'],bins=np.arange(0,20,0.1),label='All LITH',color='m',histtype='step',density=True)
plt.title('Fe histogram')
plt.xlabel('Fe')
plt.ylabel('density')
plt.legend()
plt.show()

plt.hist(df[df['LITH']==50]['As_dh'],bins=np.arange(0,100,1),label='LITH 50',color='black',histtype='step',density=True)
plt.hist(df[df['LITH']==31]['As_dh'],bins=np.arange(0,100,1),label='LITH 31',color='blue',histtype='step',density=True)
#plt.hist(df[(df['LITH']!=50) & (df['LITH']!=31)]['CuT_dh'],bins=np.arange(0,100,1),label='Other LITH',color='red',histtype='step',density=True)
plt.hist(df['As_dh'],bins=np.arange(0,100,1),label='All LITH',color='m',histtype='step',density=True)
plt.title('As histogram')
plt.xlabel('As')
plt.ylabel('density')
plt.legend()
plt.show()





for n in range(int((len(each_name))/6)+1):
    selection = each_name[6*n:6*n+6]
    fig,axis = plt.subplots(3,2,figsize=(16,16))
    axis = axis.ravel()
    for i,w in enumerate(selection):
        sp = axis[i].hist(df[df['LITH']==int(w)]['Fe_dh'],bins=np.arange(0,df[df['LITH']==int(w)]['Fe_dh'].max(),0.1))
        axis[i].set_title('lithology label:' + str(w) + '\n' + 'mean grade:'  + '{:.3f}'.format(df[df['LITH']==int(w)]['Fe_dh'].mean()))





for i in range(0,38,1):
    fields = ['BHID', 'X','Y','Z','CuT_dh','Fe_dh','LITH','As_dh']
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Escondida Trucks\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
    df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02)
                &(pd.to_numeric(df["X"], errors='coerce')>16000)&(pd.to_numeric(df["X"], errors='coerce')<18000)
                &(pd.to_numeric(df["Y"], errors='coerce')>106000)&(pd.to_numeric(df["Y"], errors='coerce')<108000)]
    #df = df.dropna()
    # fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="LITH")
    # fig.update_traces(marker_size=3)
    # fig.update_layout(legend=dict(font=dict(size=18)))
    # fig.show()
    
    each_name = [item for item, count in collections.Counter(df["LITH"]).items() if count > 500]
    df = df.loc[(df['LITH']==each_name[i])]

    plt.hist(df['CuT_dh'],bins=np.arange(0,df['CuT_dh'].max(),0.1))
    plt.xlim(0,df['CuT_dh'].max())
    plt.show()



# meangrade = []
# for i in each_name:
#     df1 = df.loc[(df['LITH']==i)]
#     meangrade.append(df1['CuT_dh'].mean())
fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="BHID")
fig.update_traces(marker_size=3)
fig.show()

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio
fields = ['BHID', 'X','Y','Z','CU','STRAT']
pio.renderers.default='browser'
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core\\kmp_ddh_mra.csv", skipinitialspace=True, usecols=fields)
df = df.dropna()
fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="CU")
fig.update_traces(marker_size=3)
fig.show()


df['As_dh_%'] = df['As_dh']/10000
df['As/Cu'] = df['As_dh_%']/df['CuT_dh']

df['CuT_dh_log'] = np.log(df['CuT_dh'])
df['Fe_dh_log'] = np.log(df['Fe_dh'])
df['As_dh_log'] = np.log(df['As_dh'])
each_name = [item for item, count in collections.Counter(df["LITH"]).items() if count > 1]
df1 = df.loc[(df['LITH']==each_name[9])]










each_name = [item for item, count in collections.Counter(df["LITH"]).items() if count > 1]
df_ = df.loc[(df['LITH']==each_name[10])]
plt.hist(df_['CuT_dh'],bins=1000)
plt.xlim(0,5)



plt.scatter(df_['CuT_dh_log'],df_['Fe_dh_log'])
for n in range(int((len(each_name))/6)+1):
    selection = each_name[6*n:6*n+6]
    fig,axis = plt.subplots(3,2,figsize=(10,10))
    axis = axis.ravel()
    for i,w in enumerate(selection):
        sp = axis[i].scatter(df[df['LITH']==w]['CuT_dh_log'],df[df['LITH']==w]['Fe_dh_log'],c= df[df['LITH']==w]['Z'])
        axis[i].set_title(str(w))
        legend1 = fig.colorbar(sp,ax=axis[i])
        legend1.set_label('elevation')
    fig.text(0.3, 0.08, 'Log CuT', ha='center', va='center') 
    fig.text(0.7, 0.08, 'log CuT', ha='center', va='center')     
    fig.text(0.04, 0.5, 'Log Fe', va='center', rotation='vertical')














x = np.log(df_['Fe_dh_log'])
y = np.log(df_['CuT_dh_log'])
#df = df.groupby('BHID').filter(lambda x:len(x)>100)
#df['ratio_CU_ASCU'] = df['CU']/df['ASCU']
df = df.loc[(df['CU']>0.02)& (df['X']>4500) & (df['X']<5000)& (df['Y']>11000) & (df['Y']<11500)]
bins=np.arange(0,5,0.05)







plt.hist(df1['CU'],bins=bins,density=True,alpha=0.5)
# for i in each_name:
#     df1 = df.loc[(df['STRAT']==i)]
#     print(len(df1),df1['CU'].mean(0))
plt.scatter(df1['CU'],df1['ASCU'])

df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\all_grade_over_2000tonnage.csv")
df2 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
df2["average_grade"] = df2.mean(axis=1)
bins=np.arange(0,10,0.05)
plt.hist(df['CU'],bins=bins,density=True,alpha=0.5,label='bore core, small scale')
plt.hist(df1['all grade over 2000tonnage'],bins=bins,density=True,alpha=0.5,label='individual truck, large scale')
plt.hist(df2['average_grade'],bins=bins,density=True,alpha=0.5,label='belt, median scale')
plt.xlim(0,5)
plt.legend()




fig = px.scatter_3d(df, x="X",y="Y",z="Z",color="LITH")
fig.update_traces(marker_size=3)
fig.show()







fields = ['BHID', 'X','Y','Z']
data = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
each_name = [item for item, count in collections.Counter(data["BHID"]).items() if count > 1]
index_pos = []
for stri in each_name:
    index_pos.append(list(data["BHID"]).index(stri))
list1 = []
for i in index_pos:
    list1.append(data[i:i+1])
list1 = pd.concat(list1)

# import plotly.express as px
# from plotly.offline import plot
# fig = px.scatter_3d(data,x="X",y="Y",z="Z",color='BHID')
# fig.update_traces(marker_size=3)
# plot(fig)
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio
fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z"]
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
df = df.groupby('BHID').filter(lambda x:len(x)>500)


df_CuT_mean = df.groupby('BHID')['CuT_dh'].mean().to_frame()
df_Fe_mean = df.groupby('BHID')['Fe_dh'].mean().to_frame()
for n in range(int((len(each_name))/6)+1):
    selection = each_name[6*n:6*n+6]
    fig,axis = plt.subplots(3,2,figsize=(10,10))
    axis = axis.ravel()
    for i,w in enumerate(selection):
        colormap1 = df[df['BHID']==w]['Z']
        sp = axis[i].scatter(df[df['BHID']==w]['CuT_dh_log'],df[df['BHID']==w]['Fe_dh_log'],c= colormap1)
        axis[i].set_title(str(w))
        legend1 = fig.colorbar(sp,ax=axis[i])
        legend1.set_label('Elevation')
    fig.text(0.3, 0.08, 'Log CuT', ha='center', va='center') 
    fig.text(0.7, 0.08, 'log CuT', ha='center', va='center')     
    fig.text(0.04, 0.5, 'Log Fe', va='center', rotation='vertical')

df_list = []
for i in range(2000,3000,100):
    df_new = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0.1)
            & (pd.to_numeric(df["Z"], errors='coerce')>i)& (pd.to_numeric(df["Z"], errors='coerce')<(i+100))
            & (pd.to_numeric(df["X"], errors='coerce')>15000)& (pd.to_numeric(df["X"], errors='coerce')<17000)
            & (pd.to_numeric(df["Y"], errors='coerce')>106500)& (pd.to_numeric(df["Y"], errors='coerce')<109500)]
    df_new["CuT_dh"] = df_new["CuT_dh"].astype("float")
    df_new["Fe_dh"] = df_new["Fe_dh"].astype("float")
    df_new["As_dh"] = df_new["As_dh"].astype("float")
    df_new['CuT_dh_log'] = np.log(df_new['CuT_dh'])
    df_new['Fe_dh_log'] = np.log(df_new['Fe_dh'])
    df_new['As_dh_log'] = np.log(df_new['As_dh'])
    df_list.append(df_new)
    #plt.scatter(df_new["CuT_dh_log"],df_new["Fe_dh_log"])
fig,axis = plt.subplots(5,2,figsize=(13,13))
axis = axis.ravel()
for i in range(len(df_list)):
    colormap1 = df_list[i]['CuT_dh_log']
    sp = axis[i].scatter(df_list[i]['Fe_dh_log'],df_list[i]['As_dh_log'],c= colormap1)
    legend1 = fig.colorbar(sp,ax=axis[i])
    legend1.set_label('Elevation')
fig.text(0.3, 0.01, 'Log CuT', ha='center', va='center',fontsize=12) 
fig.text(0.7, 0.01, 'log CuT', ha='center', va='center',fontsize=12)     
fig.text(0.01, 0.5, 'Log Fe', va='center', rotation='vertical',fontsize=12)







# import matplotlib.pyplot as plt
# import plotly.express as px
# import numpy as np
# import pandas as pd
# import collections
# import plotly.io as pio
# fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z"]
# df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
# df = df.groupby('BHID').filter(lambda x:len(x)>500)
# df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0.1)
#         & (pd.to_numeric(df["Z"], errors='coerce')>1000)& (pd.to_numeric(df["Z"], errors='coerce')<1100)
#         & (pd.to_numeric(df["X"], errors='coerce')>15000)& (pd.to_numeric(df["X"], errors='coerce')<17000)
#         & (pd.to_numeric(df["Y"], errors='coerce')>106500)& (pd.to_numeric(df["Y"], errors='coerce')<109500)]
# df["CuT_dh"] = df["CuT_dh"].astype("float")
# df["Fe_dh"] = df["Fe_dh"].astype("float")
# df["As_dh"] = df["As_dh"].astype("float")
# df['CuT_dh_log'] = np.log(df['CuT_dh'])
# df['Fe_dh_log'] = np.log(df['Fe_dh'])
# colormap1 = df['Z']
# plt.scatter(df["CuT_dh_log"],df["Fe_dh_log"],c= colormap1)
# plt.colorbar(label='elevation')









each_name = [item for item, count in collections.Counter(df["BHID"]).items() if count > 1]
df1 = []
for stri in each_name:
    df1.append(df[df["BHID"] =="%s"%(stri)])
from sklearn.linear_model import LinearRegression
r_squared_list = []
for i in range(len(df1)):
    model = LinearRegression()
    x,y = np.array(df1[i]["Fe_dh"]), np.array(df1[i]["CuT_dh"])
    model.fit(x.reshape(-1,1), y)
    r_squared = model.score(x.reshape(-1,1), y)
    if r_squared >0:
        r_squared_list.append(r_squared)
        df1[i]['r2'] = r_squared
    else:
        r_squared_list.append(0)
        
df2 = pd.concat(df1)    
df2 = df2.loc[df2['r2']>0.5]       

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.scatter_3d(df2, x="X",y="Y",z="Z",color="r2")
fig.update_traces(marker_size=3)
fig.show()  

for n in range(int((len(each_name))/6)+1):
    selection = each_name[6*n:6*n+6]
    fig,axis = plt.subplots(3,2,figsize=(10,10))
    axis = axis.ravel()
    for i,w in enumerate(selection):
        colormap1 = df[df['BHID']==w]['Z']
        sp = axis[i].scatter(df[df['BHID']==w]['CuT_dh_log'],df[df['BHID']==w]['Fe_dh_log'],c= colormap1)
        axis[i].set_title(str(w))
        legend1 = fig.colorbar(sp,ax=axis[i])
        legend1.set_label('Elevation')
    fig.text(0.3, 0.08, 'Log CuT', ha='center', va='center') 
    fig.text(0.7, 0.08, 'log CuT', ha='center', va='center')     
    fig.text(0.04, 0.5, 'Log Fe', va='center', rotation='vertical')













# index_pos = []
# for stri in each_name:
#     index_pos.append(list(data["BHID"]).index(stri))
# list1 = []
# for i in index_pos:
#     list1.append(data[i:i+1])
# list1 = pd.concat(list1)


# list_average_fe = []
# for i in range(len(data1)):
#     list_average_fe.append(data1[i]["Fe_dh"].mean())

data1_low_copper = []
data1_median_copper = []
data1_high_copper = []
data1_copper = []
for i in range(len(data1)):
    data1_copper.append(data1[i])
        
for i in range(len(data1)):
    if data1[i]["CuT_dh"].mean() <= 0.5:
        data1_low_copper.append(data1[i])
    elif data1[i]["CuT_dh"].mean() > 0.5 and data1[i]["CuT_dh"].mean() < 1:
        data1_median_copper.append(data1[i])
    elif data1[i]["CuT_dh"].mean() >= 1:
        data1_high_copper.append(data1[i])
    else:
        pass
data1_low_copper = pd.concat(data1_low_copper)
data1_median_copper = pd.concat(data1_median_copper)
data1_high_copper = pd.concat(data1_high_copper)



    
        
list1["R2_Fe"] = r_squared_list
# data1_low_copper["label"] = pd.Series(1,index=data1_low_copper.index)
# data1_median_copper["label"] = pd.Series(2,index=data1_median_copper.index)
# data1_high_copper["label"] = pd.Series(3,index=data1_high_copper.index)
# data1_low_copper_shuffle = data1_low_copper.sample(frac=1)
# data1_median_copper_shuffle = data1_median_copper.sample(frac=1)
# data1_high_copper_shuffle = data1_high_copper.sample(frac=1)
# for i in range(len(data1))
list2 = list1[["BHID",'X',"Y","R2_Fe"]].copy()
list2 = list2.reset_index()

list2_high_r2 = list2[list2["R2_Fe"]>=0.7]
list2_median_r2 = list2[(list2["R2_Fe"]<0.7) & (list2["R2_Fe"]>=0.4)]
list2_low_r2 = list2[list2["R2_Fe"]<0.4]
plt.scatter(list(list2_high_r2['X']),list(list2_high_r2['Y']))
plt.scatter(list(list2_median_r2['X']),list(list2_median_r2['Y']))
plt.scatter(list(list2_low_r2['X']),list(list2_low_r2['Y']))
data1_high_r2 = []
each_name_high_r2 = [item for item, count in collections.Counter(list2_high_r2["BHID"]).items() if count > 0]
for stri in each_name_high_r2:
    data1_high_r2.append(data[data["BHID"] =="%s"%(stri)])




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
df = pd.read_csv(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\includedcores.csv")
df = df.loc[(df['CuT_dh']>0.05) & (df['Fe_dh']>0.1)& (df['Z']>2500)& (df['Z']<2600)]
df['CuT_dh_log'] = np.log10(df['CuT_dh'])
df['CuS_dh_log'] = np.log10(df['CuS_dh'])
df['Fe_dh_log'] = np.log10(df['Fe_dh'])
#df['CuIS_dh_log'] = np.log10(df['CuIS_dh'])



each_name = [item for item, count in collections.Counter(df["BHID"]).items() if count > 1]

df1 = df.groupby('BHID')['CuT_dh'].mean().to_frame().reset_index()
df1 = df1.loc[df1['CuT_dh']<0.4]
each_name_low = [item for item, count in collections.Counter(df1["BHID"]).items() if count > 0]

df1 = df.groupby('BHID')['CuT_dh'].mean().to_frame().reset_index()
df1 = df1.loc[(df1['CuT_dh']>=0.4) & (df1['CuT_dh']<1)]
each_name_median = [item for item, count in collections.Counter(df1["BHID"]).items() if count > 0]

df1 = df.groupby('BHID')['CuT_dh'].mean().to_frame().reset_index()
df1 = df1.loc[df1['CuT_dh']>=1]
each_name_high = [item for item, count in collections.Counter(df1["BHID"]).items() if count > 0]


df_low = df[df['BHID'].isin(each_name_low)]
df_median = df[df['BHID'].isin(each_name_median)]
df_high = df[df['BHID'].isin(each_name_high)]

for n in range(int((len(each_name))/6)+1):
    selection = each_name[6*n:6*n+6]
    fig,axis = plt.subplots(3,2,figsize=(10,10))
    axis = axis.ravel()
    for i,w in enumerate(selection):
        colormap1 = df[df['BHID']==w]['Z']
        sp = axis[i].scatter(df[df['BHID']==w]['CuT_dh_log'],df[df['BHID']==w]['Fe_dh_log'],c= colormap1)
        axis[i].set_title(str(w))
        legend1 = fig.colorbar(sp,ax=axis[i])
        legend1.set_label('Elevation')
    fig.text(0.3, 0.08, 'Log CuT', ha='center', va='center') 
    fig.text(0.7, 0.08, 'log CuT', ha='center', va='center')     
    fig.text(0.04, 0.5, 'Log Fe', va='center', rotation='vertical')
    
    
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import collections
import plotly.io as pio
fields = ['BHID', 'X','Y','Z']
pio.renderers.default='browser'
data_coordinate = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\includedcores.csv", skipinitialspace=True, usecols=fields)
fig = px.scatter_3d(data_coordinate, x="X",y="Y",z="Z",color="BHID")
fig.update_traces(marker_size=3)
fig.show()   
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fields = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z"]
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
df = df.groupby('BHID').filter(lambda x:len(x)>500)

df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02) & (pd.to_numeric(df["Fe_dh"], errors='coerce')>0.1)
        & (pd.to_numeric(df["Z"], errors='coerce')>1900)& (pd.to_numeric(df["Z"], errors='coerce')<(2000))
        & (pd.to_numeric(df["X"], errors='coerce')>15000)& (pd.to_numeric(df["X"], errors='coerce')<17000)
        & (pd.to_numeric(df["Y"], errors='coerce')>106500)& (pd.to_numeric(df["Y"], errors='coerce')<109500)]
df["CuT_dh"] = df["CuT_dh"].astype("float")
df["Fe_dh"] = df["Fe_dh"].astype("float")
df["As_dh"] = df["As_dh"].astype("float")
df['CuT_dh_log'] = np.log(df['CuT_dh'])
df['Fe_dh_log'] = np.log(df['Fe_dh'])
df['As_dh_log'] = np.log(df['As_dh'])   
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
sns.regplot(
    df["CuT_dh_log"],
    df["As_dh_log"],
    scatter_kws={"color": "b"},
    line_kws={"color": "k"},
    ax=axs[0],
)


#####################variogram##########################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.express as px
import collections
fields = ['BHID', 'X','Y','Z','CuT_dh','Fe_dh','LITH','As_dh']
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Escondida Trucks\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)
df = df.loc[(pd.to_numeric(df["CuT_dh"], errors='coerce')>0.02) &(pd.to_numeric(df["Fe_dh"], errors='coerce')>0)
            &(pd.to_numeric(df["As_dh"], errors='coerce')>0)]
df['LITH'] = df['LITH'].astype(int)
each_name1 = [item for item, count in collections.Counter(df["LITH"]).items() if count > 500]
df = df.loc[(df['LITH']==each_name1[2])]


each_name2 = [item for item, count in collections.Counter(df["BHID"]).items() if count > 500]

df1 = df.loc[(df['BHID']==each_name2[5])]
plt.plot(df1['Z'],df1['CuT_dh'])































