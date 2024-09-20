import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import random
pio.renderers.default='browser'


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
fields1 = ['BHID','Fe_dh','As_dh','CuT_dh',"X","Y","Z","LITH","AL_ALT"]
pio.renderers.default='browser'
df3 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
df3 = df3.dropna()
df3 = df3[(pd.to_numeric(df3["X"], errors='coerce')>=3000)& (pd.to_numeric(df3["X"], errors='coerce')<=4000)
        & (pd.to_numeric(df3["Y"], errors='coerce')>=12500)& (pd.to_numeric(df3["Y"], errors='coerce')<=13000)]
df3 = df3.reset_index(drop=True)
var_list1 = []
block_list = []

fields2 = ["MID_X","MID_Y","MID_Z","average mra"]
pio.renderers.default='browser'
df4 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\TO GYRO 2 - Trucks with polygon x y z midpoint for Oct and Nov 2021 - Copy - new.csv", skipinitialspace=True)#, usecols=fields)
df4 = df4.dropna()
df4 = df4[(pd.to_numeric(df4["MID_X"], errors='coerce')>=3000)& (pd.to_numeric(df4["MID_X"], errors='coerce')<=4000)
        & (pd.to_numeric(df4["MID_Y"], errors='coerce')>=12500)& (pd.to_numeric(df4["MID_Y"], errors='coerce')<=13000)]
df4 = df4.reset_index(drop=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
# name = list(df_filter['BHID'].unique())
range1_list = []
range2_list = []
range3_list = []
var1_list = []
var2_list = []
var3_list = []
var4_list = []
for num in range(0,1,1):
    df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Kansanshi Bore Core and GPS\\kmp_ddh_mra.csv", skipinitialspace=True)#, usecols=fields)
    df = df.dropna()
    df = df[(pd.to_numeric(df["X"], errors='coerce')>=3000)& (pd.to_numeric(df["X"], errors='coerce')<4000)
            & (pd.to_numeric(df["Y"], errors='coerce')>=12500)& (pd.to_numeric(df["Y"], errors='coerce')<13000)]
    df = df.reset_index(drop=True)
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    df_filter = df.groupby(['BHID']).filter(lambda x: len(x)>=1)
    df_filter = df_filter.reset_index(drop=True)
    df_filter['logCu'] = np.log10(df_filter['CU'])
    name = df_filter['BHID'].unique()
    percentage_name =  random.sample(list(name), int(len(name)*0.75))
    df_filter = df_filter[df_filter['BHID'].isin(percentage_name)]
    
    var1 = []
    for n1 in range(1,16,1):
        data1 = []
        for i in percentage_name:
            each_borecore = df_filter[df_filter['BHID']==i]
            sub_data1 = [each_borecore[i:i+n1]['CU'] for i in range(0,len(each_borecore),n1)]
            sub_data1 = [x for x in sub_data1 if len(x)==n1]
            data1.extend(sub_data1)
            
        var1.append(np.var([np.mean(j) for j in data1]))
    
    var2 = []
    list_mean_subgroup1 = []
    for j in range(50,1001,50):
        data2 = []
        mean = []
        subgroup1 = [mr[n:n+j] for n in range(0,len(mr),j)]
        subgroup1 = [x for x in subgroup1 if len(x)==j]
        data2.extend(subgroup1)
        for sub_data2 in data2:
            mean.append(sub_data2['grade'].mean())
        var2.append(np.var(mean))
    
    average_tonnage = mr['tonnage'].mean()
    scale = 4* average_tonnage/3600
    mass =  3.14*(0.1**2)*1*2.65 #(T)# 0.785#
    range1 = np.arange(1*mass,16*mass-0.001,1*mass)          
    range2 = np.arange(50*scale,1001*scale-0.1,50*scale)     
    
    ###########volume###########
    ####################large scale################
    ####################bore core##############
    
    var_list2 = []
    block1_list = []
    mean_grade_list1 = []
    var3 = []
    range3 = [5,10,20,30,40,50,100,150,200]
    #range3 = [5]
    for n1,n2,n3 in zip(range3,range3,range3):
        # n1 = 50
        # n2 = 50
        # n3 = 50
        print(n1,n2,n3)
        xx1 = np.arange(round(df3["X"].min(),0), round(df3["X"].max(),0), n1).astype('float64')
        yy1 = np.arange(round(df3["Y"].min(),0), round(df3["Y"].max(),0), n2).astype('float64')
        zz1 = np.arange(round(df3["Z"].min(),0), round(df3["Z"].max(),0), n3).astype('float64')
        
        blocks11 = []
        for j in yy1:
            for i in xx1:
                sub_block = df3.loc[(pd.to_numeric(df3["X"], errors='coerce')>=i) & (pd.to_numeric(df3["X"], errors='coerce')<i+n1) &
                             (pd.to_numeric(df3["Y"], errors='coerce')>=j) & (pd.to_numeric(df3["Y"], errors='coerce')<j+n2)]
                                # &(pd.to_numeric(df1["Z"], errors='coerce')>=k) & (pd.to_numeric(df1["Z"], errors='coerce')<k+n3)]
                blocks11.append(sub_block)
                
        blocks22 = []
        for j in yy1:
            for i in xx1:
                sub_block = df4.loc[(pd.to_numeric(df4["MID_X"], errors='coerce')>=i) & (pd.to_numeric(df4["MID_X"], errors='coerce')<i+n1) &
                             (pd.to_numeric(df4["MID_Y"], errors='coerce')>=j) & (pd.to_numeric(df4["MID_Y"], errors='coerce')<j+n2)]
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
        var3.append(variance1)
    
    ############################mra new######################################## 
    var_list1 = []
    block_list = []
    
    var_list2 = []
    block1_list = []
    mean_grade_list2 = []
    var4 = []
    range3 = [5,10,20,30,40,50,100,150,200]
    #range3 = [5]
    for n1,n2,n3 in zip(range3,range3,range3):
        # n1 = 50
        # n2 = 50
        # n3 = 50
        print(n1,n2,n3)
        xx1 = np.arange(round(df3["X"].min(),0), round(df3["X"].max(),0), n1).astype('float64')
        yy1 = np.arange(round(df3["Y"].min(),0), round(df3["Y"].max(),0), n2).astype('float64')
        #zz1 = np.arange(round(df1["Z"].min(),0), round(df1["Z"].max(),0), n3).astype('float64')
        
        blocks11 = []
        #for k in zz1:
        for j in yy1:
            for i in xx1:
                sub_block = df3.loc[(pd.to_numeric(df3["X"], errors='coerce')>=i) & (pd.to_numeric(df3["X"], errors='coerce')<i+n1) &
                             (pd.to_numeric(df3["Y"], errors='coerce')>=j) & (pd.to_numeric(df3["Y"], errors='coerce')<j+n2)]
                                 #&(pd.to_numeric(df1["Z"], errors='coerce')>=k) & (pd.to_numeric(df1["Z"], errors='coerce')<k+n3)]
                blocks11.append(sub_block)
            
        blocks22 = []
        #for k in zz1:
        for j in yy1:
            for i in xx1:
                sub_block = df4.loc[(pd.to_numeric(df4["MID_X"], errors='coerce')>=i) & (pd.to_numeric(df4["MID_X"], errors='coerce')<i+n1) &
                             (pd.to_numeric(df4["MID_Y"], errors='coerce')>=j) & (pd.to_numeric(df4["MID_Y"], errors='coerce')<j+n2)]
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
        var4.append(variance2)
        
    range3 = [(200*5**2)*2.5,(200*10**2)*2.5,(200*20**2)*2.5,(200*30**2)*2.5,(200*40**2)*2.5,(200*50**2)*2.5,(200*100**2)*2.5,(200*150**2)*2.5,(200*200**2)*2.5]
    fig,axis = plt.subplots(1,1,figsize=(12,8))
    axis.scatter(np.log10(range1),np.log10(var1),label='bore core',color='r')
    #plt.scatter(np.log10(range1),np.log10(var2),label='random status')
    axis.scatter(np.log10(range2),np.log10(var2),label='mra',color='b')
    axis.scatter(np.log10(range3), var3,label='bore core',color='r')
    axis.scatter(np.log10(range3), var4,label='mra',color='b')
    # axis.scatter(np.log10(250),np.log10(np.var(onetruck_mean_list)),color='green',s=160,label='pseudo truck')
    axis.set_xlabel('log10(Tonnage)',fontsize=20)
    axis.set_ylabel('log10(Variance)',fontsize=20)
    axis.legend(loc='upper right',fontsize=28)
    range1_list.append(range1)
    range2_list.append(range2)
    range3_list.append(range3)
    var1_list.append(var1)
    var2_list.append(var2)
    var3_list.append(var3)
    var4_list.append(var4)

x = list(np.log10(range1_list[0]))+list(np.log10(range2_list[0]))+list(np.log10(range3_list[0][2:]))
y = list(np.log10(var1_list[0]))+list(np.log10(var2_list[0]))+list(var3_list[0][2:])
mymodel1 = np.poly1d(np.polyfit(x, y, 3))

x = list(np.log10(range1_list[1]))+list(np.log10(range2_list[1]))+list(np.log10(range3_list[1][2:]))
y = list(np.log10(var1_list[1]))+list(np.log10(var2_list[1]))+list(var3_list[1][2:])
mymodel2 = np.poly1d(np.polyfit(x, y, 3))

x = list(np.log10(range1_list[2]))+list(np.log10(range2_list[2]))+list(np.log10(range3_list[2][2:]))
y = list(np.log10(var1_list[2]))+list(np.log10(var2_list[2]))+list(var3_list[2][2:])
mymodel3 = np.poly1d(np.polyfit(x, y, 3))

myline = np.linspace(-1.1, 8, 100)
fig,axis = plt.subplots(1,1,figsize=(12,8))
axis.scatter(x,y,label='bore core',color='r')
axis.plot(myline, mymodel1(myline),color='black')
axis.plot(myline, mymodel2(myline),color='black')
axis.plot(myline, mymodel3(myline),color='black')
axis.set_xlabel('log10(Tonnage)',fontsize=20)
axis.set_ylabel('log10(Variance)',fontsize=20)

 
df5 = pd.DataFrame()
df5['x'] = x
df5['y'] = y
# df1 = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv")
# onetruck_list = []
# onetruck_mean_list = []
# for i in range(len(df1)):
#     onetruck1 = list(df1[i:i+1].values[0])
#     onetruck2 = [x for x in onetruck1 if str(x) != 'nan']
#     onetruck_list.append(onetruck2)
#     onetruck_mean_list.append(np.mean(onetruck2))
# range1_log = np.log10(range1)
# range2_log = np.log10(range2)
# var2_log = np.log10(var2)
# var3_log = np.log10(var3)




