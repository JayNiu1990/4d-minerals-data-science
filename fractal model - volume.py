import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio


fields = ['BHID','CU',"X","Y","Z"]
pio.renderers.default='browser'
df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df1 = df1.dropna()

df1 = df1[(pd.to_numeric(df1["X"], errors='coerce')>=3000)& (pd.to_numeric(df1["X"], errors='coerce')<=4000)
        & (pd.to_numeric(df1["Y"], errors='coerce')>=12500)& (pd.to_numeric(df1["Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df1["Z"], errors='coerce')>=1200)& (pd.to_numeric(df1["Z"], errors='coerce')<=1400)]
df1 = df1.reset_index(drop=True)
df1["CU_log"] = np.log10(df1['CU'])


import matplotlib.pyplot as plt
# fig,axis = plt.subplots(1,1,figsize=(12,8),sharey=False,sharex=False); 
# axis.hist(df1['CU'], bins=np.linspace(0,max(df1['CU']),500), density=True, alpha=0.6, color='b')
# axis.set_title("Histogram of Cu grade",fontsize=24)
# axis.tick_params(axis='both', which='major', labelsize=24)
# axis.set_xlim([0, 10])
# axis.set_xlabel('Cu grade',fontsize=24)
# axis.set_ylabel('Density',fontsize=24)
# axis.legend(loc='upper right',fontsize=24)





var_list1 = []
block_list = []
for n1,n2,n3 in zip([20,30,40,50,100,150,200],[20,30,40,50,100,150,200],[20,30,40,50,100,150,200]):
    # n1 = 50
    # n2 = 50
    # n3 = 50
    print(n1,n2,n3)
    xx1 = np.arange(round(df1["X"].min(),0), round(df1["X"].max(),0), n1).astype('float64')
    yy1 = np.arange(round(df1["Y"].min(),0), round(df1["Y"].max(),0), n2).astype('float64')
    #zz1 = np.arange(round(df1["Z"].min(),0), round(df1["Z"].max(),0), n3).astype('float64')
    
    blocks = []
    # for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df1.loc[(pd.to_numeric(df1["X"], errors='coerce')>=i) & (pd.to_numeric(df1["X"], errors='coerce')<i+n1) &
                         (pd.to_numeric(df1["Y"], errors='coerce')>=j) & (pd.to_numeric(df1["Y"], errors='coerce')<j+n2)]
                             #&(pd.to_numeric(df1["Z"], errors='coerce')>=k) & (pd.to_numeric(df1["Z"], errors='coerce')<k+n3)]
            blocks.append(sub_block)
    blocks1 = []
    for i,j in enumerate(blocks):
        if len(j)>=1:
            blocks1.append(j)
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
    block_list.append(blocks1)
    mean_grade_list = []
    for i in range(len(blocks1)):
        mean_grade_list.append(blocks1[i]['CU'].mean())
    var_list1.append(np.log10(np.var(mean_grade_list)))
    
import matplotlib.pyplot as plt
range3 = [(20**3)*2800,(30**3)*2800,(40**3)*2800,(50**3)*2800,(100**3)*2800,(150**3)*2800,(200**3)*2800]
# df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\fractal_model_result.csv")
# range1 = np.arange(20,20*21-0.1,20)
# range2 = np.arange(2600,2600*10001-0.1,2600)
# plt.scatter(np.log10(range1), df['bore core order status'].dropna(),label='bore core order status')
# plt.scatter(np.log10(range1), df['bore core random status'].dropna(),label='bore core random status')
# plt.scatter(np.log10(range2), df['mra'].dropna(),label='mra')
# plt.scatter(np.log10(range3), var_list1,label='volume')
# plt.xlabel('log10(kg)')
# plt.ylabel('log10(variance)')
# plt.legend()


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
fields = ["MID_X","MID_Y","MID_Z","average mra"]
pio.renderers.default='browser'
df2 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv", skipinitialspace=True)#, usecols=fields)
df2 = df2.dropna()
df2 = df2[(pd.to_numeric(df2["MID_X"], errors='coerce')>=3000)& (pd.to_numeric(df2["MID_X"], errors='coerce')<=4000)
        & (pd.to_numeric(df2["MID_Y"], errors='coerce')>=12500)& (pd.to_numeric(df2["MID_Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df2["MID_Z"], errors='coerce')>=1200)& (pd.to_numeric(df2["MID_Z"], errors='coerce')<=1400)]
df2 = df2.reset_index(drop=True)
var_list2 = []
block1_list = []
mean_grade_list = []
variance_list = []
for n1,n2,n3 in zip([20,30,40,50,100,150,200],[20,30,40,50,100,150,200],[20,30,40,50,100,150,200]):
    # n1 = 50
    # n2 = 50
    # n3 = 50
    print(n1,n2,n3)
    
    xx1 = np.arange(round(df2["MID_X"].min(),0), round(df2["MID_X"].max(),0), n1).astype('float64')
    yy1 = np.arange(round(df2["MID_Y"].min(),0), round(df2["MID_Y"].max(),0), n2).astype('float64')
    zz1 = np.arange(round(df2["MID_Z"].min(),0), round(df2["MID_Z"].max(),0), n3).astype('float64')
    
    blocks = []
    for k in zz1:
        for j in yy1:
            for i in xx1:
                sub_block = df2.loc[(pd.to_numeric(df2["MID_X"], errors='coerce')>=i) & (pd.to_numeric(df2["MID_X"], errors='coerce')<i+n1) &
                             (pd.to_numeric(df2["MID_Y"], errors='coerce')>=j) & (pd.to_numeric(df2["MID_Y"], errors='coerce')<j+n2)
                            &(pd.to_numeric(df2["MID_Z"], errors='coerce')>=k) & (pd.to_numeric(df2["MID_Z"], errors='coerce')<k+n3)]
            blocks.append(sub_block)
    blocks1 = []
    for i,j in enumerate(blocks):
        if len(j)>=1:
            blocks1.append(j)
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
    block1_list.append(blocks1)    
    mean_grade = []
    for i in range(len(blocks1)):
        mean_grade.append(blocks1[i]['average mra'].mean())
    
    mean_grade_list.append(mean_grade)
    var_list2.append(np.log10(np.var(mean_grade)))


    ###last one
    num = []
    for i in blocks1:
        num.append(len(i))
    variance1 = np.log10(sum(((mean_grade - np.mean(mean_grade))**2)*num/sum(num)))
    variance_list.append(variance1)


plt.scatter(np.log10(range3), var_list1,label='bore core')
plt.scatter(np.log10(range3), variance_list,label='mra')
plt.xlabel('log10(kg)')
plt.ylabel('log10(variance)')
plt.legend()













####################large scale################
####################bore core##############
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
fields1 = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
pio.renderers.default='browser'
df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df1 = df1.dropna()
df1 = df1[(pd.to_numeric(df1["X"], errors='coerce')>=3000)& (pd.to_numeric(df1["X"], errors='coerce')<=4000)
        & (pd.to_numeric(df1["Y"], errors='coerce')>=12500)& (pd.to_numeric(df1["Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df1["Z"], errors='coerce')>=1200)& (pd.to_numeric(df1["Z"], errors='coerce')<=1400)]
df1 = df1.reset_index(drop=True)
var_list1 = []
block_list = []

fields2 = ["MID_X","MID_Y","MID_Z","average mra"]
pio.renderers.default='browser'
df2 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv", skipinitialspace=True)#, usecols=fields)
df2 = df2.dropna()
df2 = df2[(pd.to_numeric(df2["MID_X"], errors='coerce')>=3000)& (pd.to_numeric(df2["MID_X"], errors='coerce')<=4000)
        & (pd.to_numeric(df2["MID_Y"], errors='coerce')>=12500)& (pd.to_numeric(df2["MID_Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df2["MID_Z"], errors='coerce')>=1200)& (pd.to_numeric(df2["MID_Z"], errors='coerce')<=1400)]
df2 = df2.reset_index(drop=True)

var_list2 = []
block1_list = []
mean_grade_list1 = []
variance_list1 = []
range1 = [5,10,20,30,40,50,100,150,200]
#range1 = [5]
for n1,n2,n3 in zip(range1,range1,range1):
    # n1 = 50
    # n2 = 50
    # n3 = 50
    print(n1,n2,n3)
    xx1 = np.arange(round(df1["X"].min(),0), round(df1["X"].max(),0), n1).astype('float64')
    yy1 = np.arange(round(df1["Y"].min(),0), round(df1["Y"].max(),0), n2).astype('float64')
    zz1 = np.arange(round(df1["Z"].min(),0), round(df1["Z"].max(),0), n3).astype('float64')
    
    blocks11 = []
    #for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df1.loc[(pd.to_numeric(df1["X"], errors='coerce')>=i) & (pd.to_numeric(df1["X"], errors='coerce')<i+n1) &
                         (pd.to_numeric(df1["Y"], errors='coerce')>=j) & (pd.to_numeric(df1["Y"], errors='coerce')<j+n2)]
                            # &(pd.to_numeric(df1["Z"], errors='coerce')>=k) & (pd.to_numeric(df1["Z"], errors='coerce')<k+n3)]
            blocks11.append(sub_block)
            
    blocks22 = []
    #for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df2.loc[(pd.to_numeric(df2["MID_X"], errors='coerce')>=i) & (pd.to_numeric(df2["MID_X"], errors='coerce')<i+n1) &
                         (pd.to_numeric(df2["MID_Y"], errors='coerce')>=j) & (pd.to_numeric(df2["MID_Y"], errors='coerce')<j+n2)]
                        #&(pd.to_numeric(df2["MID_Z"], errors='coerce')>=k) & (pd.to_numeric(df2["MID_Z"], errors='coerce')<k+n3)]
            blocks22.append(sub_block)        

    blocks1 = []
    for i,j in zip(blocks11,blocks22):
        if len(i)>=1 and len(j)>=1:
            blocks1.append(i)
            
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
    
    mean_grade = []
    for i in range(len(blocks1)):
        mean_grade.append(blocks1[i]['CU'].mean())
    
    mean_grade_list1.append(mean_grade)
    var_list1.append(np.log10(np.var(mean_grade)))

    num = []
    for i in blocks1:
        num.append(len(i))
    variance1 = np.log10(sum(((mean_grade - np.mean(mean_grade))**2)*num/sum(num)))
    #variance1 = sum(((mean_grade - np.mean(mean_grade))**2)*num/sum(num))
    variance_list1.append(variance1)
############################mra new######################################## 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
fields1 = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
pio.renderers.default='browser'
df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df1 = df1.dropna()
df1 = df1[(pd.to_numeric(df1["X"], errors='coerce')>=3000)& (pd.to_numeric(df1["X"], errors='coerce')<=4000)
        & (pd.to_numeric(df1["Y"], errors='coerce')>=12500)& (pd.to_numeric(df1["Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df1["Z"], errors='coerce')>=1200)& (pd.to_numeric(df1["Z"], errors='coerce')<=1400)]
df1 = df1.reset_index(drop=True)
var_list1 = []
block_list = []

fields2 = ["MID_X","MID_Y","MID_Z","average mra"]
pio.renderers.default='browser'
df2 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv", skipinitialspace=True)#, usecols=fields)
df2 = df2.dropna()
df2 = df2[(pd.to_numeric(df2["MID_X"], errors='coerce')>=3000)& (pd.to_numeric(df2["MID_X"], errors='coerce')<=4000)
        & (pd.to_numeric(df2["MID_Y"], errors='coerce')>=12500)& (pd.to_numeric(df2["MID_Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df2["MID_Z"], errors='coerce')>=1200)& (pd.to_numeric(df2["MID_Z"], errors='coerce')<=1400)]
df2 = df2.reset_index(drop=True)
var_list2 = []
block1_list = []
mean_grade_list2 = []
variance_list2 = []
range1 = [5,10,20,30,40,50,100,150,200]
#range1 = [5]
for n1,n2,n3 in zip(range1,range1,range1):
    # n1 = 50
    # n2 = 50
    # n3 = 50
    print(n1,n2,n3)
    xx1 = np.arange(round(df1["X"].min(),0), round(df1["X"].max(),0), n1).astype('float64')
    yy1 = np.arange(round(df1["Y"].min(),0), round(df1["Y"].max(),0), n2).astype('float64')
    #zz1 = np.arange(round(df1["Z"].min(),0), round(df1["Z"].max(),0), n3).astype('float64')
    
    blocks11 = []
    #for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df1.loc[(pd.to_numeric(df1["X"], errors='coerce')>=i) & (pd.to_numeric(df1["X"], errors='coerce')<i+n1) &
                         (pd.to_numeric(df1["Y"], errors='coerce')>=j) & (pd.to_numeric(df1["Y"], errors='coerce')<j+n2)]
                             #&(pd.to_numeric(df1["Z"], errors='coerce')>=k) & (pd.to_numeric(df1["Z"], errors='coerce')<k+n3)]
            blocks11.append(sub_block)
        
    blocks22 = []
    #for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df2.loc[(pd.to_numeric(df2["MID_X"], errors='coerce')>=i) & (pd.to_numeric(df2["MID_X"], errors='coerce')<i+n1) &
                         (pd.to_numeric(df2["MID_Y"], errors='coerce')>=j) & (pd.to_numeric(df2["MID_Y"], errors='coerce')<j+n2)]
                            #&(pd.to_numeric(df2["MID_Z"], errors='coerce')>=k) & (pd.to_numeric(df2["MID_Z"], errors='coerce')<k+n3)]
            blocks22.append(sub_block)        

    blocks1 = []
    for i,j in zip(blocks11,blocks22):
        if len(i)>=1 and len(j)>=1:
            blocks1.append(j)
            
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
    block_list.append(blocks1)
    
    mean_grade = []
    for i in range(len(blocks1)):
        mean_grade.append(blocks1[i]['average mra'].mean())
    
    mean_grade_list2.append(mean_grade)
    var_list1.append(np.log10(np.var(mean_grade)))

    num = []
    for i in blocks1:
        num.append(len(i))
    variance2 = np.log10(sum(((mean_grade - np.mean(mean_grade))**2)*num/sum(num)))
    #variance2 = sum(((mean_grade - np.mean(mean_grade))**2)*num/sum(num))
    variance_list2.append(variance2)
    
range3 = [(200*5**2)*2.8,(200*10**2)*2.8,(200*20**2)*2.8,(200*30**2)*2.8,(200*40**2)*2.8,(200*50**2)*2.8,(200*100**2)*2.8,(200*150**2)*2.8,(200*200**2)*2.8]
fig,axis = plt.subplots(1,1,figsize=(12,8))
axis.scatter(np.log10(range3)[2:], variance_list1[2:],label='bore core',color='r')
#axis.scatter(np.log10(range3)[2:], variance_list2[2:],label='mra',color='b')
axis.set_xlabel('log10(tonnage)',fontsize=20)
axis.set_ylabel('log10(variance)',fontsize=20)
axis.legend(loc='upper right',fontsize=28)

scale = np.log10(range3)
variance = variance_list1

# fig,axis = plt.subplots(1,1,figsize=(12,8))
# axis.scatter(range3[2:], variance_list1[2:],label='bore core',color='r')
# axis.scatter(range3[2:], variance_list2[2:],label='mra',color='b')
# axis.set_xlabel('log10(tonnage)',fontsize=20)
# axis.set_ylabel('log10(variance)',fontsize=20)
# axis.legend(loc='upper right',fontsize=28)

dataframe = pd.DataFrame()
dataframe['scale'] = np.log10(range3)[2:]
dataframe['bore core large scale'] = variance_list1[2:]
dataframe['mra large scale'] = variance_list2[2:]
dataframe.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\scale\\kansanshi_largescale.csv')

dataframe = pd.DataFrame()
#dataframe['scale'] = np.log10(range3)[2:]
dataframe['scale'] = range3[2:]
dataframe['bore core large scale'] = variance_list1[2:]
dataframe['mra large scale'] = variance_list2[2:]
dataframe.to_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\scale\\kansanshi_largescale_nolog.csv')





np.log10(df1['CU'].var())

####################small scale################
####################bore core##############
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
fields1 = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
pio.renderers.default='browser'
df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df1 = df1.dropna()
df1 = df1[(pd.to_numeric(df1["X"], errors='coerce')>=3000)& (pd.to_numeric(df1["X"], errors='coerce')<=4000)
        & (pd.to_numeric(df1["Y"], errors='coerce')>=12500)& (pd.to_numeric(df1["Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df1["Z"], errors='coerce')>=1200)& (pd.to_numeric(df1["Z"], errors='coerce')<=1400)]
df1 = df1.reset_index(drop=True)
var_list1 = []
block_list = []

fields2 = ["MID_X","MID_Y","MID_Z","average mra"]
pio.renderers.default='browser'
df2 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv", skipinitialspace=True)#, usecols=fields)
df2 = df2.dropna()
df2 = df2[(pd.to_numeric(df2["MID_X"], errors='coerce')>=3000)& (pd.to_numeric(df2["MID_X"], errors='coerce')<=4000)
        & (pd.to_numeric(df2["MID_Y"], errors='coerce')>=12500)& (pd.to_numeric(df2["MID_Y"], errors='coerce')<=13000)
        & (pd.to_numeric(df2["MID_Z"], errors='coerce')>=1200)& (pd.to_numeric(df2["MID_Z"], errors='coerce')<=1400)]
df2 = df2.reset_index(drop=True)

var_list2 = []
block1_list = []
mean_grade_list1 = []
variance_list1 = []
range1 = [5,10,20,30,40,50,100,150,200]

for n1,n2,n3 in zip(range1,range1,range1):

    print(n1,n2,n3)
    xx1 = np.arange(round(df1["X"].min(),0), round(df1["X"].max(),0), n1).astype('float64')
    yy1 = np.arange(round(df1["Y"].min(),0), round(df1["Y"].max(),0), n2).astype('float64')
    
    blocks11 = []
    for j in yy1:
        for i in xx1:
            sub_block = df1.loc[(pd.to_numeric(df1["X"], errors='coerce')>=i) & (pd.to_numeric(df1["X"], errors='coerce')<i+n1) &
                         (pd.to_numeric(df1["Y"], errors='coerce')>=j) & (pd.to_numeric(df1["Y"], errors='coerce')<j+n2)]
                            # &(pd.to_numeric(df1["Z"], errors='coerce')>=k) & (pd.to_numeric(df1["Z"], errors='coerce')<k+n3)]
            blocks11.append(sub_block)
            
    blocks22 = []
    #for k in zz1:
    for j in yy1:
        for i in xx1:
            sub_block = df2.loc[(pd.to_numeric(df2["MID_X"], errors='coerce')>=i) & (pd.to_numeric(df2["MID_X"], errors='coerce')<i+n1) &
                         (pd.to_numeric(df2["MID_Y"], errors='coerce')>=j) & (pd.to_numeric(df2["MID_Y"], errors='coerce')<j+n2)]
                        #&(pd.to_numeric(df2["MID_Z"], errors='coerce')>=k) & (pd.to_numeric(df2["MID_Z"], errors='coerce')<k+n3)]
            blocks22.append(sub_block)        

    blocks1 = []
    blocks2 = []
    for i,j in zip(blocks11,blocks22):
        if len(i)>=1 and len(j)>=1:
            blocks1.append(i)
            blocks2.append(j)
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
    for i, j in enumerate(blocks2):
        blocks2[i]['blocks'] = i
    mean_grade = []
    for i in range(len(blocks1)):
        mean_grade.append(blocks1[i]['CU'].mean())
    
    mean_grade_list1.append(mean_grade)
    var_list1.append(np.log10(np.var(mean_grade)))

    num = []
    for i in blocks1:
        num.append(len(i))
    variance1 = np.log10(sum(((mean_grade - np.mean(mean_grade))**2)*num/sum(num)))
    variance_list1.append(variance1)




############# scale 20 bore core#############
df_borecore_smallscale = blocks1[12]
df_gps_smallscale = blocks2[12]
df_gps_smallscale = df_gps_smallscale.drop(columns=['average mra'])

df_mra = []
date = list(df_gps_smallscale['DATE'].unique())
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
import datetime
import time
second_list = []
for i in range(len(df_mra)):
    dt = datetime.datetime(df_mra['year'][i], df_mra['month'][i], df_mra['day'][i] , df_mra['hour'][i], df_mra['minute'][i],df_mra['second'][i])
    second_list.append(time.mktime(dt.timetuple()))
df_mra['total_second'] = second_list

# delay = 900 # 75-300s
# delay_mra_list = []
# for i in range(len(df_gps_smallscale)):
#     gps_time = df_gps_smallscale[i:i+1]['total_second']
#     delay_mra = df_mra.loc[(pd.to_numeric(df_mra["total_second"], errors='coerce')>int(gps_time)+delay) & (pd.to_numeric(df_mra["total_second"], errors='coerce')<int(gps_time)+delay+round(df_gps_smallscale[i:i+1]['TONNES'].values[0]/df_mra['tonnage'].mean()*3600/4))] 
#     #average_mra = delay_mra['grade'].mean()
#     delay_mra_list.append(delay_mra)

var_list3 = []
block1_list3 = []
mean_grade_list3 = []
block_list3 = []
for n in [1,2,3,4,5,6]:
    zz1 = np.arange(round(df_borecore_smallscale["Z"].min(),0), round(df_borecore_smallscale["Z"].max(),0), n).astype('float64')
    blocks = []
    # for k in zz1:
    for i in zz1:
        sub_block = df_borecore_smallscale.loc[(pd.to_numeric(df_borecore_smallscale["Z"], errors='coerce')>=i) & (pd.to_numeric(df_borecore_smallscale["Z"], errors='coerce')<i+n)]
        blocks.append(sub_block)

    blocks1 = []
    for i,j in enumerate(blocks):
        if len(j)>=1:
            blocks1.append(j)
            
    for i, j in enumerate(blocks1):
        blocks1[i]['blocks'] = i
    block_list3.append(blocks1)
    mean_grade_list3 = []
    for i in range(len(blocks1)):
        mean_grade_list3.append(blocks1[i]['CU'].mean())
    var_list3.append(np.log10(np.var(mean_grade_list3)))
    





    
    
    
    
    
    
    
    
    
    
    
    
    